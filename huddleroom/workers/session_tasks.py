import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError
from fastapi import HTTPException
from huddleroom.database import AsyncSessionLocal
from huddleroom.models.session import Session as SessionModel
from huddleroom.models.task import Task
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.services.event_bus import emit_event
from huddleroom.services.project_service import ProjectService
from huddleroom.services.session_sync import sync_task_from_session

logger = logging.getLogger(__name__)

_TASK_DISPATCH_ACTIONS = frozenset(("create_delegation_task", "request_plan", "request_plan_revision"))
_RECOVERY_DISPATCH_ACTIONS = frozenset(("retry_task", "reassign_task"))


async def orchestration_lineage_state(db, session: SessionModel) -> bool | None:
    """Return True for durable orchestration work, None for manual work, False for invalid lineage."""
    session_metadata = session.metadata_ if isinstance(session.metadata_, dict) else {}
    session_orchestration = session_metadata.get("orchestration")
    if not isinstance(session_orchestration, dict):
        session_orchestration = None
    if session.task_id is None:
        return False if session_orchestration is not None else None
    task = await db.get(Task, session.task_id)
    task_metadata = task.metadata_ if task is not None and isinstance(task.metadata_, dict) else {}
    task_orchestration = task_metadata.get("orchestration")
    if not isinstance(task_orchestration, dict):
        return False if session_orchestration is not None else None
    try:
        action_id = uuid.UUID(str(task_orchestration["action_id"]))
        run_id = uuid.UUID(str(task_orchestration["run_id"]))
    except (KeyError, TypeError, ValueError):
        return False
    if session_orchestration is not None and str(session_orchestration.get("action_id")) != str(action_id):
        return False
    action = await db.get(OrchestrationAction, action_id)
    run = await db.get(OrchestrationRun, run_id)
    goal = await db.get(OrchestrationGoal, run.goal_id) if run is not None else None
    if (task is None or action is None or run is None or goal is None or action.run_id != run.id
            or goal.project_id != session.project_id
            or task.project_id != session.project_id or task.assigned_to != session.agent_id
            or action.status != "completed"):
        return False
    request = action.request if isinstance(action.request, dict) else {}
    contract = action.dispatch_contract if isinstance(action.dispatch_contract, dict) else {}
    if action.action_type in _RECOVERY_DISPATCH_ACTIONS:
        return bool(
            str(request.get("task_id")) == str(task.id)
            and contract.get("owner") == "orchestration_recovery"
            and action.target_type == "session" and action.target_id == session.id
        )
    return bool(
        action.action_type in _TASK_DISPATCH_ACTIONS
        and action.target_type == "task" and action.target_id == task.id
        and not contract
    )


def is_project_not_runnable_conflict(exc: HTTPException) -> bool:
    detail = exc.detail if isinstance(exc.detail, dict) else {}
    return exc.status_code == 409 and detail.get("code") == "project_not_runnable"


_SQLITE_CLAIM_RETRIES = 3


def _is_sqlite_lock_error(exc: OperationalError) -> bool:
    return "database is locked" in str(exc).lower()


async def _require_runnable_session(db, session_id: uuid.UUID, runner_task_id: str | None = None) -> bool:
    """Atomically claim a runnable session, retrying bounded SQLite write contention."""
    sqlite = db.bind is not None and db.bind.dialect.name == "sqlite"
    if not sqlite:
        result = await _require_runnable_session_once(db, session_id, runner_task_id)
        if isinstance(result, HTTPException):
            await db.commit()
            raise result
        if result:
            await db.commit()
        return result
    for attempt in range(_SQLITE_CLAIM_RETRIES):
        try:
            result = await _require_runnable_session_once(db, session_id, runner_task_id)
            break
        except OperationalError as exc:
            if not _is_sqlite_lock_error(exc):
                raise
            await db.rollback()
            if attempt == _SQLITE_CLAIM_RETRIES - 1:
                raise
    if isinstance(result, HTTPException):
        await db.commit()
        raise result
    if result:
        await db.commit()
    else:
        await db.rollback()
    return result


async def _require_runnable_session_once(db, session_id: uuid.UUID, runner_task_id: str | None = None) -> bool:
    session = (await db.execute(select(SessionModel).where(SessionModel.id == session_id)
                                .execution_options(populate_existing=True))).scalar_one_or_none()
    if session is None:
        return False
    if session.status != "pending":
        return False

    lineage = await orchestration_lineage_state(db, session)
    if lineage is False:
        return False
    if lineage is True and not runner_task_id:
        return False

    try:
        await ProjectService().require_runnable_project(db, session.project_id)
    except HTTPException as exc:
        if not is_project_not_runnable_conflict(exc):
            raise
        detail = exc.detail
        failure_error = f"project_not_runnable: {detail['reason']}"
        values = {"status": "failed", "error": failure_error, "ended_at": datetime.now(timezone.utc)}
        where = [SessionModel.id == session.id, SessionModel.status == "pending"]
        if runner_task_id:
            metadata = dict(session.metadata_ or {})
            metadata["attempt"] = {
                "claimed_runner_task_id": runner_task_id,
                "claimed_at": datetime.now(timezone.utc).isoformat(),
                "attempt_version": 1,
                "effect_state": "not_started",
                "usage_complete": False,
                "result_status": "failed",
                "result_recorded_at": datetime.now(timezone.utc).isoformat(),
                "provider_session_id": session.provider_session_id,
            }
            values["metadata_"] = metadata
            where.extend((SessionModel.runner_task_id == runner_task_id, SessionModel.started_at.is_(None)))
        result = await db.execute(update(SessionModel).where(*where).values(**values))
        if not result.rowcount:
            return False
        await db.refresh(session)
        await emit_event(db, session.project_id, "session.failed", {
            "session_id": str(session.id),
            "task_id": str(session.task_id) if session.task_id else None,
            "error": session.error,
            "project_id": str(session.project_id),
        })
        await sync_task_from_session(db, session)
        return exc
    if runner_task_id is None:
        claim = await db.execute(
            update(SessionModel).where(
                SessionModel.id == session_id, SessionModel.status == "pending", SessionModel.started_at.is_(None),
            ).values(status="running", started_at=datetime.now(timezone.utc))
        )
        if not claim.rowcount:
            return False
        await db.refresh(session)
        await emit_event(db, session.project_id, "session.started", {
            "session_id": str(session.id), "task_id": str(session.task_id) if session.task_id else None,
            "project_id": str(session.project_id),
        })
        return True
    if runner_task_id is not None:
        marker = dict(session.metadata_ or {})
        marker["attempt"] = {"claimed_runner_task_id": runner_task_id,
                             "claimed_at": datetime.now(timezone.utc).isoformat(),
                             "attempt_version": 1,
                             "effect_state": "not_started", "usage_complete": False}
        claim = await db.execute(
            update(SessionModel).where(
                SessionModel.id == session_id, SessionModel.status == "pending",
                SessionModel.runner_task_id == runner_task_id, SessionModel.started_at.is_(None),
            ).values(status="running", started_at=datetime.now(timezone.utc), metadata_=marker)
        )
        if not claim.rowcount:
            return False
        await db.refresh(session)
        await emit_event(db, session.project_id, "session.started", {
            "session_id": str(session.id), "task_id": str(session.task_id) if session.task_id else None,
            "project_id": str(session.project_id),
        })
    result = lineage is not True or runner_task_id is not None
    return result


async def mark_attempt_effect_started(db, session_id: uuid.UUID, runner_task_id: str | None) -> bool:
    session = await db.get(SessionModel, session_id)
    if session is None or not runner_task_id:
        return False
    metadata = dict(session.metadata_ or {})
    attempt = dict(metadata.get("attempt") or {})
    version = attempt.get("attempt_version")
    if attempt.get("claimed_runner_task_id") != runner_task_id or not isinstance(version, int):
        return False
    if attempt.get("effect_state") == "started":
        return (await db.scalar(select(SessionModel.id).where(
            SessionModel.id == session_id,
            SessionModel.status == "running",
            SessionModel.runner_task_id == runner_task_id,
            SessionModel.metadata_["attempt"]["claimed_runner_task_id"].as_string() == runner_task_id,
            SessionModel.metadata_["attempt"]["effect_state"].as_string() == "started",
            SessionModel.metadata_["attempt"]["attempt_version"].as_integer() == version,
        ))) is not None
    if attempt.get("effect_state") != "not_started":
        return False
    attempt["effect_state"] = "started"
    attempt["effect_started_at"] = datetime.now(timezone.utc).isoformat()
    attempt["attempt_version"] = version + 1
    metadata["attempt"] = attempt
    result = await db.execute(
        update(SessionModel).where(
            SessionModel.id == session_id,
            SessionModel.status == "running",
            SessionModel.runner_task_id == runner_task_id,
            SessionModel.metadata_["attempt"]["claimed_runner_task_id"].as_string() == runner_task_id,
            SessionModel.metadata_["attempt"]["effect_state"].as_string() == "not_started",
            SessionModel.metadata_["attempt"]["attempt_version"].as_integer() == version,
        ).values(metadata_=metadata)
    )
    if not result.rowcount:
        db.expire(session)
        return False
    await db.refresh(session)
    await db.commit()
    return True


async def _mark_attempt_result(db, session_id: uuid.UUID, runner_task_id: str | None) -> bool:
    """Persist only facts the adapter completed after the committed effect boundary."""
    session = await db.get(SessionModel, session_id)
    if session is None or session.status not in {"completed", "failed", "cancelled"}:
        return False
    metadata = dict(session.metadata_ or {})
    attempt = dict(metadata.get("attempt") or {})
    if runner_task_id and attempt.get("claimed_runner_task_id") != runner_task_id:
        return False
    attempt["result_status"] = session.status
    attempt["result_recorded_at"] = datetime.now(timezone.utc).isoformat()
    attempt["provider_session_id"] = session.provider_session_id
    # Adapters that enforce a budget expose an explicit completeness bit; all
    # other adapters have no partial usage ledger to carry across a resume.
    usage_complete = metadata.get("token_usage_complete")
    attempt["usage_complete"] = usage_complete if isinstance(usage_complete, bool) else None
    attempt["token_count_in"] = metadata.get("token_count_in", 0)
    attempt["token_count_out"] = metadata.get("token_count_out", 0)
    metadata["attempt"] = attempt
    if not runner_task_id:
        session.metadata_ = metadata
        await db.flush()
        return True
    version = attempt.get("attempt_version")
    if not isinstance(version, int):
        return False
    attempt["attempt_version"] = version + 1
    metadata["attempt"] = attempt
    with db.no_autoflush:
        result = await db.execute(
            update(SessionModel).where(
                SessionModel.id == session_id,
                SessionModel.status == "running",
                SessionModel.runner_task_id == runner_task_id,
                SessionModel.metadata_["attempt"]["claimed_runner_task_id"].as_string() == runner_task_id,
                SessionModel.metadata_["attempt"]["attempt_version"].as_integer() == version,
            ).values(
                status=session.status, error=session.error, resumable=session.resumable, output=session.output,
                provider_session_id=session.provider_session_id, ended_at=session.ended_at, metadata_=metadata,
            )
        )
    if not result.rowcount:
        db.expire(session)
        return False
    with db.no_autoflush:
        db.expire(session)
        await db.refresh(session)
    return True


async def mark_attempt_project_not_runnable(db, session_id: uuid.UUID, runner_task_id: str, error: str) -> bool:
    """Fence a post-claim workspace rejection to the worker that still owns it."""
    session = await db.get(SessionModel, session_id)
    if session is None:
        return False
    metadata = dict(session.metadata_ or {})
    attempt = dict(metadata.get("attempt") or {})
    version = attempt.get("attempt_version")
    effect_state = attempt.get("effect_state")
    if (attempt.get("claimed_runner_task_id") != runner_task_id or not isinstance(version, int)
            or not isinstance(effect_state, str)):
        return False
    attempt.update({
        "attempt_version": version + 1, "result_status": "failed",
        "result_recorded_at": datetime.now(timezone.utc).isoformat(),
        "provider_session_id": session.provider_session_id, "usage_complete": None,
    })
    metadata["attempt"] = attempt
    result = await db.execute(
        update(SessionModel).where(
            SessionModel.id == session_id, SessionModel.status == "running",
            SessionModel.runner_task_id == runner_task_id,
            SessionModel.metadata_["attempt"]["claimed_runner_task_id"].as_string() == runner_task_id,
            SessionModel.metadata_["attempt"]["effect_state"].as_string() == effect_state,
            SessionModel.metadata_["attempt"]["attempt_version"].as_integer() == version,
        ).values(status="failed", error=error, resumable=False, ended_at=datetime.now(timezone.utc), metadata_=metadata)
    )
    if not result.rowcount:
        return False
    await db.refresh(session)
    return True


async def execute_api_session(session_id: str, runner_task_id: str) -> None:
    """Run an API adapter session."""
    from huddleroom.adapters.api_adapter import ApiAdapter

    sid = uuid.UUID(session_id)
    # The creating HTTP request may not have committed yet when this task starts.
    # Poll until the session is visible or we time out.
    for _ in range(20):
        async with AsyncSessionLocal() as probe:
            r = await probe.execute(select(SessionModel).where(SessionModel.id == sid))
            if r.scalar_one_or_none() is not None:
                break
        await asyncio.sleep(0.1)

    async with AsyncSessionLocal() as db:
        try:
            if not await _require_runnable_session(db, sid, runner_task_id):
                return
            await ApiAdapter().run(sid, db, runner_task_id)
            await _mark_attempt_result(db, sid, runner_task_id)
            await db.commit()
        except Exception as e:
            logger.error("API session %s failed: %s", session_id, e)
            await db.rollback()
            raise


async def execute_cli_session(session_id: str, runner_task_id: str) -> None:
    """Run a CLI adapter session."""
    from huddleroom.adapters.cli_adapter import CliAdapter

    sid = uuid.UUID(session_id)
    # The creating HTTP request may not have committed yet when this task starts.
    # Poll until the session is visible or we time out.
    for _ in range(20):
        async with AsyncSessionLocal() as probe:
            r = await probe.execute(select(SessionModel).where(SessionModel.id == sid))
            if r.scalar_one_or_none() is not None:
                break
        await asyncio.sleep(0.1)

    async with AsyncSessionLocal() as db:
        try:
            if not await _require_runnable_session(db, sid, runner_task_id):
                return
            await CliAdapter().run(sid, db, runner_task_id)
            await _mark_attempt_result(db, sid, runner_task_id)
            await db.commit()
        except Exception as e:
            logger.error("CLI session %s failed: %s", session_id, e)
            await db.rollback()
            raise


try:
    from huddleroom.workers.celery_app import app

    if app is not None:
        def _run_async(coro):
            loop = asyncio.new_event_loop()
            async def _run_with_relay_cleanup():
                try:
                    return await coro
                finally:
                    from huddleroom.services.agent_response_relay import close_agent_response_relay
                    await close_agent_response_relay()
            try:
                return loop.run_until_complete(_run_with_relay_cleanup())
            finally:
                loop.close()

        @app.task(bind=True, max_retries=3, name="rally.workers.session_tasks.run_api_session")
        def run_api_session(self, session_id: str):
            """Run an API adapter session via Celery."""
            runner_task_id = getattr(getattr(self, "request", None), "id", None)
            if not runner_task_id:
                return
            _run_async(execute_api_session(session_id, runner_task_id))

        @app.task(bind=True, max_retries=0, name="rally.workers.session_tasks.run_cli_session")
        def run_cli_session(self, session_id: str):
            """Run a CLI adapter session via Celery. No retry — subprocess state is not safely repeatable."""
            runner_task_id = getattr(getattr(self, "request", None), "id", None)
            if not runner_task_id:
                return
            _run_async(execute_cli_session(session_id, runner_task_id))

except ImportError:
    pass
