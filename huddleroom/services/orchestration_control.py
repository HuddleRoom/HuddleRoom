"""Durable pause acceptance ordered by SQLite's transaction writer (SPR230/SPR231)."""
import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.schemas.orchestration import ControlSnapshot, PauseReceipt


def _conflict(message: str) -> HTTPException:
    return HTTPException(409, detail={"conflict": "orchestration_control_conflict", "message": message})


async def acquire_control_writer(db: AsyncSession, run_id: uuid.UUID) -> None:
    # Suppress pending ORM status writes until the authoritative fence has been read.
    with db.no_autoflush:
        await db.execute(update(OrchestrationRun).where(OrchestrationRun.id == run_id).values(
            control_version=OrchestrationRun.control_version, updated_at=OrchestrationRun.updated_at,
        ).execution_options(synchronize_session=False))


async def read_control(db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID,
                       run_id: uuid.UUID) -> ControlSnapshot:
    with db.no_autoflush:
        row = (await db.execute(select(
            OrchestrationGoal.project_id, OrchestrationGoal.id.label("goal_id"),
            OrchestrationRun.id.label("run_id"), OrchestrationRun.control_version,
            OrchestrationRun.pause_requested, OrchestrationGoal.status.label("goal_status"),
            OrchestrationRun.status.label("run_status"),
        ).join(OrchestrationRun, OrchestrationRun.goal_id == OrchestrationGoal.id).where(
            OrchestrationGoal.project_id == project_id, OrchestrationGoal.id == goal_id,
            OrchestrationRun.id == run_id,
        ))).mappings().one_or_none()
    if row is None:
        raise HTTPException(404, "Orchestration control target not found")
    return ControlSnapshot.model_validate(dict(row))


async def _receipt(db: AsyncSession, action: OrchestrationAction) -> PauseReceipt:
    request = action.request
    snapshot = await read_control(db, uuid.UUID(request["project_id"]),
                                  uuid.UUID(request["goal_id"]), action.run_id)
    with db.no_autoflush:
        current = await db.scalar(select(OrchestrationRun.id).where(
            OrchestrationRun.goal_id == snapshot.goal_id,
        ).order_by(OrchestrationRun.started_at.desc(), OrchestrationRun.id.desc()).limit(1))
    state = "applied" if snapshot.pause_requested and snapshot.run_status == "paused" else "accepted"
    if current != action.run_id or snapshot.run_status in {"completed", "cancelled"}:
        state = "superseded"
    elif snapshot.control_version != request["accepted_control_version"] or not snapshot.pause_requested:
        state = "resumed"
    return PauseReceipt(
        project_id=snapshot.project_id, goal_id=snapshot.goal_id, run_id=action.run_id,
        request_id=uuid.UUID(request["request_id"]), accepted_at=request["accepted_at"],
        accepted_control_version=request["accepted_control_version"],
        current_control_version=snapshot.control_version, state=state,
        goal_status=snapshot.goal_status, run_status=snapshot.run_status,
    )


async def read_pause_receipt(db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID,
                             request_id: uuid.UUID, actor_user_id: uuid.UUID) -> PauseReceipt | None:
    with db.no_autoflush:
        exists = await db.scalar(select(OrchestrationGoal.id).where(
            OrchestrationGoal.id == goal_id, OrchestrationGoal.project_id == project_id,
        ))
        if exists is None:
            raise HTTPException(404, "Orchestration goal not found")
        actions = (await db.scalars(select(OrchestrationAction).join(
            OrchestrationRun, OrchestrationRun.id == OrchestrationAction.run_id,
        ).join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id).where(
            OrchestrationGoal.project_id == project_id, OrchestrationGoal.id == goal_id,
            OrchestrationAction.action_type == "pause_control",
        ).execution_options(populate_existing=True))).all()
    for action in actions:
        if action.request.get("request_id") == str(request_id):
            if action.request.get("actor_user_id") != str(actor_user_id):
                raise _conflict("Pause request belongs to another actor")
            return await _receipt(db, action)
    return None


async def accept_pause(db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID,
                       run_id: uuid.UUID, expected_control_version: int, request_id: uuid.UUID,
                       actor_user_id: uuid.UUID, *, commit: bool = True) -> PauseReceipt:
    transaction = db.sync_session.get_transaction()
    owns = commit and transaction is None
    try:
        return await _accept_pause(db, project_id, goal_id, run_id, expected_control_version,
                                   request_id, actor_user_id, commit=owns)
    except OperationalError as exc:
        if getattr(exc.orig, "sqlite_errorcode", None) != sqlite3.SQLITE_BUSY_SNAPSHOT:
            if owns:
                await db.rollback()
            raise
        if not owns:
            raise HTTPException(409, detail={
                "conflict": "orchestration_control_snapshot_retry",
                "message": "Retry pause in a fresh caller transaction",
            }) from exc
        await db.rollback()
        try:
            return await _accept_pause(db, project_id, goal_id, run_id, expected_control_version,
                                       request_id, actor_user_id, commit=True)
        except BaseException:
            await db.rollback()
            raise
    except BaseException:
        if owns:
            await db.rollback()
        raise


async def _accept_pause(db, project_id, goal_id, run_id, expected_control_version,
                        request_id, actor_user_id, *, commit):
    await acquire_control_writer(db, run_id)
    replay = await read_pause_receipt(db, project_id, goal_id, request_id, actor_user_id)
    if replay is not None:
        action = await db.scalar(select(OrchestrationAction).where(
            OrchestrationAction.run_id == replay.run_id,
            OrchestrationAction.idempotency_key == f"run:{replay.run_id}:pause:{request_id}",
        ))
        if replay.run_id != run_id or action.request["expected_control_version"] != expected_control_version:
            raise _conflict("Pause request differs from its accepted target/version")
        if commit:
            await db.commit()
        return replay
    snapshot = await read_control(db, project_id, goal_id, run_id)
    current = await db.scalar(select(OrchestrationRun.id).where(
        OrchestrationRun.goal_id == goal_id, OrchestrationRun.status.in_(("running", "blocked", "paused")),
    ).order_by(OrchestrationRun.started_at.desc(), OrchestrationRun.id.desc()).limit(1))
    if current != run_id or snapshot.control_version != expected_control_version:
        raise _conflict("Pause target or control version changed")
    if snapshot.goal_status not in {"active", "blocked"} or snapshot.run_status not in {"running", "blocked"}:
        raise _conflict("Goal cannot be paused in its current state")
    accepted_at = datetime.now(timezone.utc)
    await db.execute(update(OrchestrationRun).where(OrchestrationRun.id == run_id).values(
        pause_requested=True, control_version=expected_control_version + 1, status="paused",
        baseline_authorized=False,
    ))
    await db.execute(update(OrchestrationGoal).where(OrchestrationGoal.id == goal_id).values(status="paused"))
    action = OrchestrationAction(run_id=run_id, action_type="pause_control", status="completed",
        idempotency_key=f"run:{run_id}:pause:{request_id}", target_type="orchestration_goal", target_id=goal_id,
        request={"project_id": str(project_id), "goal_id": str(goal_id), "run_id": str(run_id),
                 "request_id": str(request_id), "actor_user_id": str(actor_user_id),
                 "expected_control_version": expected_control_version,
                 "accepted_control_version": expected_control_version + 1, "accepted_at": accepted_at.isoformat()},
        dispatch_contract={"accepted_at": accepted_at.isoformat(), "control_version": expected_control_version + 1})
    db.add(action)
    await db.flush()
    receipt = await _receipt(db, action)
    if commit:
        await db.commit()
    return receipt


async def apply_pause_status(db: AsyncSession, run_id: uuid.UUID) -> ControlSnapshot:
    await acquire_control_writer(db, run_id)
    with db.no_autoflush:
        target = (await db.execute(select(OrchestrationGoal.project_id, OrchestrationGoal.id).join(
            OrchestrationRun, OrchestrationRun.goal_id == OrchestrationGoal.id,
        ).where(OrchestrationRun.id == run_id))).one_or_none()
    if target is None:
        raise HTTPException(404, "Orchestration run not found")
    snapshot = await read_control(db, target[0], target[1], run_id)
    if snapshot.pause_requested and snapshot.run_status not in {"completed", "cancelled"}:
        await db.execute(update(OrchestrationRun).where(OrchestrationRun.id == run_id).values(status="paused"))
        if snapshot.goal_status not in {"completed", "cancelled"}:
            await db.execute(update(OrchestrationGoal).where(
                OrchestrationGoal.id == snapshot.goal_id,
            ).values(status="paused"))
    return await read_control(db, target[0], target[1], run_id)
