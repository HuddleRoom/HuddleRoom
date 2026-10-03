from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy.exc import OperationalError

from huddleroom.config import settings
from huddleroom.database import AsyncSessionLocal
from huddleroom.models.meeting import Meeting, MeetingAgendaItem
from huddleroom.models.agent import Agent
from huddleroom.services.meeting_service import MeetingService
from huddleroom.services.meeting_context import MeetingContextService
from huddleroom.services.project_service import ProjectService

logger = logging.getLogger(__name__)


start_meeting = None
run_meeting_turn = None
resume_meeting_turn = None
finalize_meeting = None
meeting_timeout = None
_meeting_task_app = None
_running_meeting_tasks: dict[str, asyncio.Task] = {}
_meeting_task_projects: dict[str, str] = {}
PROJECT_CANCELLATION_TIMEOUT_SECONDS = 5
CELERY_SHUTDOWN_POLL_SECONDS = 0.05
_IN_PROCESS_TIMEOUT_GRACE_SECONDS = 5
_SQLITE_LOCK_RETRY_SECONDS = 1


def _log_in_process_failure(task_name: str, task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error("In-process meeting task %s failed: %s", task_name, exc)


def get_running_meeting_tasks() -> dict[str, asyncio.Task]:
    return _running_meeting_tasks


def _track_in_process(task_name: str, project_id: str | uuid.UUID, coroutine) -> None:
    task_id = str(uuid.uuid4())
    scheduled = asyncio.create_task(coroutine, name=f"meeting:{task_name}")
    _running_meeting_tasks[task_id] = scheduled
    _meeting_task_projects[task_id] = str(project_id)
    def cleanup(task: asyncio.Task) -> None:
        _running_meeting_tasks.pop(task_id, None)
        _meeting_task_projects.pop(task_id, None)
        _log_in_process_failure(task_name, task)
    scheduled.add_done_callback(cleanup)


def _dispatch_task(task_name: str, task, async_fn, project_id: str | uuid.UUID, *args) -> None:
    if settings.is_sqlite or task is None:
        _track_in_process(task_name, project_id, async_fn(*args))
        return

    try:
        task.s(*args).stamp(project_id=str(project_id)).apply_async()
    except Exception as exc:
        logger.warning("Failed to dispatch Celery task %s: %s", getattr(task, "name", task_name), exc)
        raise


def dispatch_start_meeting(meeting_id: str, project_id: str | uuid.UUID) -> None:
    _dispatch_task("start", start_meeting, start_meeting_async, project_id, meeting_id)


def dispatch_run_meeting_turn(meeting_id: str, project_id: str | uuid.UUID) -> None:
    _dispatch_task("run_turn", run_meeting_turn, run_meeting_turn_async, project_id, meeting_id)


def dispatch_finalize_meeting(meeting_id: str, project_id: str | uuid.UUID) -> None:
    _dispatch_task("finalize", finalize_meeting, _run_finalize_in_process, project_id, meeting_id)


def dispatch_resume_meeting_turn(meeting_id: str, project_id: str | uuid.UUID) -> None:
    _dispatch_task("resume_turn", resume_meeting_turn, resume_meeting_turn_async, project_id, meeting_id)


def dispatch_meeting_timeout(meeting_id: str, project_id: str | uuid.UUID) -> None:
    _dispatch_task("timeout", meeting_timeout, meeting_timeout_async, project_id, meeting_id)


def _schedule_in_process_timeout(meeting_id: str, project_id: str | uuid.UUID, countdown_seconds: int) -> None:
    async def _delayed_timeout() -> None:
        await asyncio.sleep(countdown_seconds + _IN_PROCESS_TIMEOUT_GRACE_SECONDS)
        await meeting_timeout_async(meeting_id)

    _track_in_process("timeout", project_id, _delayed_timeout())


def _is_sqlite_lock_error(exc: Exception) -> bool:
    return "database is locked" in str(exc).lower()


def _is_project_not_runnable(exc: HTTPException) -> bool:
    return exc.status_code == 409 and isinstance(exc.detail, dict) and exc.detail.get("code") == "project_not_runnable"


def _schedule_in_process_turn_retry(meeting_id: str, project_id: str | uuid.UUID, countdown_seconds: int) -> None:
    async def _delayed_retry() -> None:
        await asyncio.sleep(max(0, countdown_seconds))
        dispatch_run_meeting_turn(meeting_id, project_id)

    _track_in_process("retry-turn", project_id, _delayed_retry())


def schedule_meeting_timeout(meeting_id: str, project_id: str | uuid.UUID, countdown_seconds: int) -> None:
    countdown_seconds = max(0, countdown_seconds)
    if settings.is_sqlite or meeting_timeout is None:
        _schedule_in_process_timeout(meeting_id, project_id, countdown_seconds)
        return

    try:
        meeting_timeout.s(meeting_id).stamp(project_id=str(project_id)).apply_async(countdown=countdown_seconds)
    except Exception as exc:
        logger.warning("Failed to schedule meeting timeout for %s: %s", meeting_id, exc)
        _schedule_in_process_timeout(meeting_id, project_id, countdown_seconds)


async def cancel_project_meeting_tasks(project_id: str | uuid.UUID) -> int:
    project_key = str(project_id)
    selected = [(task_id, task) for task_id, task in _running_meeting_tasks.items() if _meeting_task_projects.get(task_id) == project_key]
    for _, task in selected:
        task.cancel()
    if selected:
        _, pending = await asyncio.wait((task for _, task in selected), timeout=PROJECT_CANCELLATION_TIMEOUT_SECONDS)
        if pending:
            raise TimeoutError(f"{len(pending)} project meeting task(s) did not stop before reset timeout")
    for task_id, _ in selected:
        _running_meeting_tasks.pop(task_id, None)
        _meeting_task_projects.pop(task_id, None)
    return len(selected)


def revoke_project_meeting_tasks(project_id: str | uuid.UUID) -> None:
    if not settings.is_sqlite and _meeting_task_app is not None:
        _meeting_task_app.control.revoke_by_stamped_headers({"project_id": str(project_id)}, terminate=False)


def _reply_workers(replies) -> set[str]:
    return {worker for reply in replies or [] for worker in reply}


def _project_stamped_jobs(snapshot: dict, project_id: str) -> list[dict]:
    matches = []
    for tasks in snapshot.values():
        for item in tasks:
            task = item.get("request", item)
            stamped_project = (task.get("stamps") or {}).get("project_id")
            values = stamped_project if isinstance(stamped_project, list) else [stamped_project]
            if project_id in {str(value) for value in values if value is not None}:
                matches.append(task)
    return matches


def _revoke_and_await_project_meeting_tasks(project_id: str) -> None:
    deadline = time.monotonic() + PROJECT_CANCELLATION_TIMEOUT_SECONDS
    control = _meeting_task_app.control
    workers = _reply_workers(control.ping(timeout=PROJECT_CANCELLATION_TIMEOUT_SECONDS))
    if not workers:
        raise TimeoutError("No Celery worker acknowledged project meeting shutdown")

    remaining = max(0.01, deadline - time.monotonic())
    replies = control.revoke_by_stamped_headers(
        {"project_id": project_id},
        terminate=False,
        reply=True,
        timeout=remaining,
    )
    if _reply_workers(replies) != workers:
        raise TimeoutError("Not all Celery workers acknowledged project meeting shutdown")

    while True:
        pending = []
        for state in ("active", "reserved", "scheduled"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Project meeting Celery tasks did not stop before reset timeout")
            inspector = control.inspect(destination=sorted(workers), timeout=remaining)
            snapshot = getattr(inspector, state)()
            if snapshot is None or set(snapshot) != workers:
                raise RuntimeError(f"Celery {state} inspection did not receive all worker replies")
            pending.extend(_project_stamped_jobs(snapshot, project_id))
        if not pending:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Project meeting Celery tasks did not stop before reset timeout")
        time.sleep(min(CELERY_SHUTDOWN_POLL_SECONDS, remaining))


async def revoke_and_await_project_meeting_tasks(project_id: str | uuid.UUID) -> None:
    if settings.is_sqlite:
        return
    if _meeting_task_app is None:
        raise RuntimeError("Celery meeting task app is unavailable")
    await asyncio.to_thread(_revoke_and_await_project_meeting_tasks, str(project_id))


def _meeting_timeout_seconds(meeting: Meeting) -> int:
    return max(0, int((meeting.max_duration_minutes or 0) * 60))


def _seconds_until_finalize(meeting: Meeting) -> int:
    if meeting.status != "concluding" or not meeting.concluding_started_at:
        return 0

    concluding_started_at = meeting.concluding_started_at
    if concluding_started_at.tzinfo is None:
        concluding_started_at = concluding_started_at.replace(tzinfo=timezone.utc)

    finalize_at = concluding_started_at + timedelta(hours=meeting.veto_window_hours)
    remaining = int((finalize_at - datetime.now(timezone.utc)).total_seconds())
    return max(0, remaining)


async def start_meeting_async(meeting_id: str) -> None:
    svc = MeetingService()
    ctx_svc = MeetingContextService()
    project_svc = ProjectService()

    async with AsyncSessionLocal() as db:
        try:
            meeting = await svc.claim_scheduled_meeting(db, uuid.UUID(meeting_id))
            if not meeting:
                logger.warning("start_meeting: meeting %s not in scheduled state", meeting_id)
                return

            contexts: dict[str, str | dict[str, str]] = {}
            for aid_str in meeting.participant_agent_ids:
                try:
                    await project_svc.require_runnable_project(db, meeting.project_id)
                    agent = await db.get(Agent, uuid.UUID(aid_str))
                    if agent:
                        ctx = await ctx_svc.build_initial_context(db=db, meeting=meeting, agent=agent)
                        contexts[aid_str] = (
                            {"initial_ctx": ctx} if agent.adapter_type == "cli" else ctx
                        )
                except Exception as e:
                    logger.warning("Failed to build context for agent %s: %s", aid_str, e)

            meeting.participant_contexts = contexts

            await project_svc.require_runnable_project(db, meeting.project_id)
            await svc.transition_to_active(db=db, meeting=meeting)
            await db.commit()
            logger.info("Meeting %s transitioned to active", meeting_id)
            schedule_meeting_timeout(meeting_id, meeting.project_id, _meeting_timeout_seconds(meeting))
            dispatch_run_meeting_turn(meeting_id, meeting.project_id)
        except HTTPException as exc:
            await db.rollback()
            if _is_project_not_runnable(exc):
                logger.info("Meeting %s remains scheduled: project is not runnable", meeting_id)
                return
            raise
        except Exception as e:
            await db.rollback()
            logger.error("Error in start_meeting_async: %s", e)
            raise


async def meeting_timeout_async(meeting_id: str) -> None:
    from huddleroom.services.meeting_service import MeetingTransitionError
    from sqlalchemy import select
    from huddleroom.models.meeting import MeetingTurn

    svc = MeetingService()

    async with AsyncSessionLocal() as db:
        try:
            meeting = await db.get(Meeting, uuid.UUID(meeting_id))
            if not meeting or meeting.status in ("concluding", "concluded", "cancelled"):
                return

            result = await db.execute(
                select(MeetingAgendaItem).where(
                    MeetingAgendaItem.meeting_id == meeting.id,
                    MeetingAgendaItem.status.in_(["pending", "active"]),
                )
            )
            open_items = result.scalars().all()
            active_items = [item for item in open_items if item.status == "active"]
            for item in active_items:
                if not _should_preserve_timeout_outcome(meeting, item):
                    continue
                turn_result = await db.execute(
                    select(MeetingTurn)
                    .where(
                        MeetingTurn.meeting_id == meeting.id,
                        MeetingTurn.agenda_item_id == item.id,
                    )
                    .order_by(MeetingTurn.turn_number)
                )
                turns = turn_result.scalars().all()
                if not turns:
                    continue
                participants_heard: list[str] = []
                for turn in turns:
                    if turn.speaker_agent_id:
                        speaker = str(turn.speaker_agent_id)
                        if speaker not in participants_heard:
                            participants_heard.append(speaker)
                item.status = "unresolved"
                item.is_deadlocked = True
                item.resolution_kind = "human_intervention"
                item.resolution_summary = (
                    f"{item.title} timed out before a final decision. Human organizer review is required."
                )
                item.required_followup = (
                    "Human organizer must review the partial discussion and decide whether to resolve, revisit, or close the item."
                )
                item.participants_heard = participants_heard
                await svc.log_event(
                    db,
                    meeting.id,
                    "human_intervention_required",
                    {
                        "agenda_item_id": str(item.id),
                        "trigger": "meeting_timeout",
                        "summary": item.resolution_summary,
                    },
                )
                await svc.log_event(
                    db,
                    meeting.id,
                    "agenda_item_completed",
                    {
                        "agenda_item_id": str(item.id),
                        "resolution": item.status,
                        "resolution_kind": item.resolution_kind,
                        "resolution_summary": item.resolution_summary,
                        "required_followup": item.required_followup,
                        "participants_heard": item.participants_heard,
                    },
                )

            for item in open_items:
                if item.status in ("pending", "active"):
                    item.status = "abandoned"

            meeting.is_partial = True

            try:
                if meeting.status == "active":
                    await svc.transition_to_concluding(db=db, meeting=meeting)
            except MeetingTransitionError as e:
                logger.warning("Timeout transition error for %s: %s", meeting_id, e)

            await svc.log_event(db, meeting.id, "timeout", {"reason": "max_duration_exceeded"})
            await db.commit()
            dispatch_finalize_meeting(meeting_id, meeting.project_id)
            logger.info("Meeting %s moved to concluding on timeout", meeting_id)
            return
        except Exception as e:
            await db.rollback()
            logger.error("Error in meeting_timeout_async: %s", e)
            raise


def _should_preserve_timeout_outcome(meeting: Meeting, item: MeetingAgendaItem) -> bool:
    return (
        item.status == "active"
        and meeting.meeting_type == "decision"
        and meeting.turn_strategy == "organizer_controlled"
        and meeting.deadlock_strategy == "human_intervention"
    )


async def _run_meeting_turn_tail(db, runner, svc, meeting, project_svc, meeting_id, turn) -> None:
    """Shared tail for normal and resumed turns: evaluate round, dispatch next."""
    item = await svc.get_current_agenda_item(db=db, meeting_id=meeting.id)
    if item:
        await project_svc.require_runnable_project(db, meeting.project_id)
        await runner.evaluate_round_if_complete(db=db, meeting=meeting, item=item)
    current_item = await svc.get_current_agenda_item(db=db, meeting_id=meeting.id)

    await db.refresh(meeting)
    should_finalize = meeting.status == "concluding"
    should_continue = (
        meeting.status == "active"
        and current_item is not None
        and (
            turn is not None
            or meeting.turn_strategy in {"organizer_controlled", "moderated"}
        )
    )

    await db.commit()

    if should_finalize:
        dispatch_finalize_meeting(meeting_id, meeting.project_id)
    elif should_continue:
        dispatch_run_meeting_turn(meeting_id, meeting.project_id)


async def run_meeting_turn_async(meeting_id: str) -> None:
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.event_bus import get_event_bus

    runner = MeetingRunner(bus=get_event_bus())
    svc = MeetingService()
    project_svc = ProjectService()

    async with AsyncSessionLocal() as db:
        try:
            meeting = await db.get(Meeting, uuid.UUID(meeting_id))
            if not meeting or meeting.status != "active":
                return

            # Gate: if a turn is parked, don't dispatch until resumed
            if meeting.resume_state and meeting.resume_state.get("failed"):
                await db.commit()
                return

            # Enforce max_duration_minutes for in-process mode (no external scheduler)
            if meeting.active_started_at and meeting.max_duration_minutes:
                active_started = meeting.active_started_at
                if active_started.tzinfo is None:
                    active_started = active_started.replace(tzinfo=timezone.utc)
                elapsed_minutes = (datetime.now(timezone.utc) - active_started).total_seconds() / 60
                if elapsed_minutes >= meeting.max_duration_minutes:
                    logger.info("Meeting %s exceeded max_duration_minutes=%s, triggering timeout", meeting_id, meeting.max_duration_minutes)
                    await db.commit()
                    dispatch_meeting_timeout(meeting_id, meeting.project_id)
                    return

            await project_svc.require_runnable_project(db, meeting.project_id)
            turn = await runner.run_next_turn(db=db, meeting_id=meeting.id)

            # Re-check failure gate after run_next_turn: if parked, stop before evaluating round
            await db.refresh(meeting)
            if meeting.resume_state and meeting.resume_state.get("failed"):
                await db.commit()
                return

            await _run_meeting_turn_tail(db, runner, svc, meeting, project_svc, meeting_id, turn)
        except HTTPException as exc:
            await db.rollback()
            if _is_project_not_runnable(exc):
                logger.info("Meeting %s turn skipped: project is not runnable", meeting_id)
                return
            raise
        except Exception as e:
            await db.rollback()
            if settings.is_sqlite and isinstance(e, OperationalError) and _is_sqlite_lock_error(e):
                logger.warning(
                    "Transient SQLite lock during run_meeting_turn_async for %s; retrying in %ss",
                    meeting_id,
                    _SQLITE_LOCK_RETRY_SECONDS,
                )
                _schedule_in_process_turn_retry(meeting_id, meeting.project_id, _SQLITE_LOCK_RETRY_SECONDS)
                return
            logger.error("Error in run_meeting_turn_async for %s: %s", meeting_id, e)
            raise


async def resume_meeting_turn_async(meeting_id: str) -> None:
    from huddleroom.services.meeting_runner import MeetingRunner
    from huddleroom.services.event_bus import get_event_bus

    runner = MeetingRunner(bus=get_event_bus())
    svc = MeetingService()
    project_svc = ProjectService()

    async with AsyncSessionLocal() as db:
        try:
            meeting = await db.get(Meeting, uuid.UUID(meeting_id))
            if not meeting or meeting.status != "active":
                return

            # Only resume if there's a parked turn
            if not meeting.resume_state or not meeting.resume_state.get("failed"):
                await db.commit()
                return

            # Atomically claim the parked state to prevent concurrent resumes
            resume_state = dict(meeting.resume_state or {})
            if resume_state.get("resuming"):
                # Another resume is already in progress, bail
                await db.commit()
                return
            resume_state["resuming"] = True
            meeting.resume_state = resume_state
            await db.flush()

            await project_svc.require_runnable_project(db, meeting.project_id)
            turn = await runner.resume_failed_turn(db=db, meeting=meeting)

            # Run the shared tail (evaluate round, continue/finalize, commit, dispatch)
            await _run_meeting_turn_tail(db, runner, svc, meeting, project_svc, meeting_id, turn)
        except HTTPException as exc:
            await db.rollback()
            if _is_project_not_runnable(exc):
                logger.info("Resume meeting %s turn skipped: project is not runnable", meeting_id)
                return
            raise
        except Exception as e:
            await db.rollback()
            # Clear the resuming flag on failure so a later legitimate retry still works
            try:
                meeting = await db.get(Meeting, uuid.UUID(meeting_id))
                if meeting and meeting.resume_state:
                    resume_state = dict(meeting.resume_state)
                    resume_state.pop("resuming", None)
                    meeting.resume_state = resume_state if resume_state else {}
                    await db.commit()
            except Exception as cleanup_err:
                logger.warning("Failed to clear resuming flag for %s: %s", meeting_id, cleanup_err)

            if settings.is_sqlite and isinstance(e, OperationalError) and _is_sqlite_lock_error(e):
                logger.warning(
                    "Transient SQLite lock during resume_meeting_turn_async for %s; retrying in %ss",
                    meeting_id,
                    _SQLITE_LOCK_RETRY_SECONDS,
                )
                _schedule_in_process_turn_retry(meeting_id, meeting.project_id, _SQLITE_LOCK_RETRY_SECONDS)
                return
            logger.error("Error in resume_meeting_turn_async for %s: %s", meeting_id, e)
            raise


async def finalize_meeting_async(meeting_id: str) -> int | None:
    from huddleroom.services.meeting_outcome import MeetingOutcomeService

    outcome_svc = MeetingOutcomeService()

    async with AsyncSessionLocal() as db:
        try:
            meeting = await db.get(Meeting, uuid.UUID(meeting_id))
            if not meeting or meeting.status != "concluding":
                logger.warning(
                    "finalize_meeting: meeting %s not in concluding state (status=%s)",
                    meeting_id,
                    meeting.status if meeting else "missing",
                )
                return

            seconds_until_finalize = _seconds_until_finalize(meeting)
            if seconds_until_finalize > 0:
                logger.info(
                    "Meeting %s still in veto window; deferring finalization for %s seconds",
                    meeting_id,
                    seconds_until_finalize,
                )
                return seconds_until_finalize

            concluded, pending = await outcome_svc.finalize_meeting(db=db, meeting=meeting)
            if concluded:
                from huddleroom.services.event_bus import emit_event, get_event_bus
                await emit_event(
                    db=db,
                    project_id=meeting.project_id,
                    event_type="meeting.concluded",
                    payload={"meeting_id": meeting_id},
                    _bus=get_event_bus(),
                )
            await db.commit()

            if concluded:
                logger.info("Meeting %s finalized and concluded", meeting_id)
                return None
            if pending is None or (pending.payload or {}).get("reviewer_kind") == "organizer_user":
                return None

            try:
                decisions_made, decisions_clear, action_items_needed, action_items = (
                    await outcome_svc.ask_final_reviewer(db=db, meeting=meeting, pending=pending)
                )
                await db.commit()
                completed = await outcome_svc.complete_final_review(
                    db=db,
                    meeting=meeting,
                    decisions_made=decisions_made,
                    decisions_clear=decisions_clear,
                    action_items_needed=action_items_needed,
                    action_items=action_items,
                    actor_agent_id=(
                        uuid.UUID(pending.payload["reviewer_id"])
                        if pending.payload.get("reviewer_kind") == "organizer_agent"
                        else None
                    ),
                )
                await db.commit()
                if completed:
                    dispatch_finalize_meeting(meeting_id, meeting.project_id)
            except Exception as exc:
                await db.rollback()
                logger.warning("Automatic final review failed for %s: %s", meeting_id, exc)
            return None
        except Exception as e:
            await db.rollback()
            logger.error("Error in finalize_meeting_async for %s: %s", meeting_id, e)
            raise


async def _run_finalize_in_process(meeting_id: str) -> None:
    countdown = await finalize_meeting_async(meeting_id)
    while countdown:
        await asyncio.sleep(max(1, countdown))
        countdown = await finalize_meeting_async(meeting_id)


def register_tasks(app):
    """Register Celery tasks for meeting workflow."""
    global start_meeting, run_meeting_turn, resume_meeting_turn, finalize_meeting, meeting_timeout, _meeting_task_app

    try:
        from celery.exceptions import MaxRetriesExceededError, Retry as CeleryRetry

        if app is not None:
            _meeting_task_app = app

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

            @app.task(name="rally.meeting.start", bind=True, max_retries=3)
            def start_meeting(self, meeting_id: str):  # pylint: disable=function-redefined
                """Start a meeting and transition it to active."""
                try:
                    _run_async(start_meeting_async(meeting_id))
                except Exception as exc:
                    logger.error("start_meeting task failed for %s: %s", meeting_id, exc)
                    raise self.retry(exc=exc, countdown=10)

            @app.task(name="rally.meeting.run_turn", bind=True, max_retries=3)
            def run_meeting_turn(self, meeting_id: str):  # pylint: disable=function-redefined
                """Execute the next agent turn in a meeting."""
                try:
                    _run_async(run_meeting_turn_async(meeting_id))
                except Exception as exc:
                    logger.error("run_meeting_turn task failed for %s: %s", meeting_id, exc)
                    raise self.retry(exc=exc, countdown=5)

            @app.task(name="rally.meeting.resume_turn", bind=True, max_retries=3)
            def resume_meeting_turn(self, meeting_id: str):  # pylint: disable=function-redefined
                """Resume a parked meeting turn."""
                try:
                    _run_async(resume_meeting_turn_async(meeting_id))
                except Exception as exc:
                    logger.error("resume_meeting_turn task failed for %s: %s", meeting_id, exc)
                    raise self.retry(exc=exc, countdown=5)

            @app.task(name="rally.meeting.finalize", bind=True, max_retries=2)
            def finalize_meeting(self, meeting_id: str):  # pylint: disable=function-redefined
                """Finalize a concluding meeting."""
                try:
                    countdown = _run_async(finalize_meeting_async(meeting_id))
                    if countdown:
                        raise self.retry(countdown=countdown)
                except CeleryRetry:
                    raise
                except MaxRetriesExceededError:
                    logger.error("finalize_meeting exceeded retries for %s", meeting_id)
                    raise
                except Exception as exc:
                    logger.error("finalize_meeting task failed for %s: %s", meeting_id, exc)
                    raise self.retry(exc=exc, countdown=30)

            @app.task(name="rally.meeting.timeout", bind=True)
            def meeting_timeout(self, meeting_id: str):  # pylint: disable=function-redefined
                """Force-conclude a meeting on timeout."""
                try:
                    _run_async(meeting_timeout_async(meeting_id))
                except Exception as exc:
                    logger.error("meeting_timeout task failed for %s: %s", meeting_id, exc)

    except ImportError:
        pass
