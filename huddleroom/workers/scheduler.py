import logging
from collections import deque
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.events import EVENT_JOB_SUBMITTED
from huddleroom.config import settings

logger = logging.getLogger(__name__)

scheduler: AsyncIOScheduler | None = None   # pylint: disable=invalid-name
_recovery_submission_times: deque[datetime] = deque()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def evaluate_cron_triggers_job() -> None:
    from huddleroom.workers.trigger_tasks import evaluate_cron_triggers_async
    try:
        await evaluate_cron_triggers_async()
    except Exception as e:
        logger.error("Cron trigger evaluation failed: %s", e)


async def archive_old_events_job() -> None:
    """Delete EventLog entries older than 7 days."""
    from huddleroom.database import AsyncSessionLocal
    from huddleroom.models.event_log import EventLog
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import delete

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                delete(EventLog).where(EventLog.emitted_at < cutoff)
            )
            await db.commit()
            if result.rowcount:
                logger.info("Archived %d old event_log entries", result.rowcount)
    except Exception as e:
        logger.error("Event log archival failed: %s", e)


async def process_graph_timeouts_job() -> None:
    from huddleroom.database import AsyncSessionLocal
    from huddleroom.services.graph_engine import GraphEngineService

    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                await GraphEngineService().process_timeouts(db)
    except Exception as e:
        logger.error("Graph timeout processing failed: %s", e)


async def check_due_meetings_job() -> None:
    from huddleroom.database import AsyncSessionLocal
    from huddleroom.models.meeting import Meeting
    from huddleroom.workers.meeting_tasks import dispatch_start_meeting
    from sqlalchemy import select
    from datetime import datetime, timezone

    try:
        async with AsyncSessionLocal() as db:
            now = datetime.now(timezone.utc)
            result = await db.execute(
                select(Meeting).where(
                    Meeting.status == "scheduled",
                    Meeting.auto_start.is_(True),
                    Meeting.scheduled_at <= now,
                )
            )
            due = result.scalars().all()
            for m in due:
                logger.info("Dispatching start for due meeting %s", m.id)
                dispatch_start_meeting(str(m.id), m.project_id)
    except Exception as e:
        logger.error("check_due_meetings failed: %s", e)


async def check_blocked_tasks_for_meetings() -> None:
    from huddleroom.database import AsyncSessionLocal
    from huddleroom.models.task import Task
    from huddleroom.models.meeting import Meeting
    from huddleroom.services.event_bus import emit_event_once
    from huddleroom.services.meeting_service import MeetingService
    from sqlalchemy import select
    from datetime import datetime, timezone, timedelta

    blocked_escalation_hours = 4

    try:
        async with AsyncSessionLocal() as db:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=blocked_escalation_hours)
            result = await db.execute(
                select(Task).where(
                    (Task.status == "blocked") & (Task.updated_at < cutoff),
                )
            )
            blocked_tasks = result.scalars().all()

            if not blocked_tasks:
                return

            svc = MeetingService()
            for task in blocked_tasks:
                # Skip if active meeting already exists for this task
                existing = await db.execute(
                    select(Meeting).where(
                        Meeting.source_task_id == task.id,
                        Meeting.status.in_(["scheduled", "preparing", "active", "concluding"]),
                    )
                )
                if existing.scalar_one_or_none():
                    continue

                participant_ids = []
                if task.assigned_to:
                    participant_ids.append(str(task.assigned_to))

                meeting = await svc.create_meeting(
                    db=db,
                    project_id=task.project_id,
                    title=f"Unblock: {task.title[:80]}",
                    meeting_type="adhoc",
                    participant_agent_ids=participant_ids,
                    agenda_items=[{
                        "order": 1,
                        "title": f"Resolve blocker for: {task.title[:80]}",
                        "description": f"Task {task.id} has been blocked for over {blocked_escalation_hours} hours.",
                        "max_rounds": 2,
                    }],
                    auto_start=True,
                    created_by_trigger=True,
                    trigger_reason=f"Task {task.id} blocked for >{blocked_escalation_hours}h",
                    source_task_id=task.id,
                )
                await emit_event_once(
                    db,
                    task.project_id,
                    "meeting.scheduled",
                    {
                        "meeting_id": str(meeting.id),
                        "auto_start": meeting.auto_start,
                    },
                    source="scheduler",
                    dedup_key=f"meeting.scheduled:meeting:{meeting.id}",
                )
                logger.info("Auto-scheduled adhoc meeting for blocked task %s", task.id)
            await db.commit()
    except Exception as e:
        logger.error("check_blocked_tasks_for_meetings failed: %s", e)


async def recover_orphaned_sessions_job() -> None:
    from huddleroom.database import AsyncSessionLocal
    from huddleroom.services.session_service import SessionService

    try:
        async with AsyncSessionLocal() as db:
            recovered = await SessionService().recover_orphaned_sessions(
                db,
                timeout_seconds=3600,
                redispatch_pending=False,
            )
            await db.commit()
            if recovered:
                logger.warning("Recovered %d stale sessions during watchdog pass", recovered)
    except Exception as e:
        logger.error("Session recovery watchdog failed: %s", e)


async def reconcile_orchestration_runs_async(db) -> int:
    from sqlalchemy import select
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
    from huddleroom.services.orchestration_service import OrchestrationService

    rows = await db.execute(
        select(OrchestrationRun.id)
        .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
        .where(
            OrchestrationRun.status.in_(("running", "blocked")),
            OrchestrationGoal.status.in_(("active", "blocked")),
            ~((OrchestrationGoal.goal_type == "continuous") & (OrchestrationRun.phase == "waiting_activation")),
        )
    )
    run_ids = [row[0] for row in rows.all()]
    service = OrchestrationService()
    reconciled = 0
    for run_id in run_ids:
        try:
            await service.tick(db, run_id)
            reconciled += 1
        except Exception as exc:  # one bad run must not stop the sweep
            logger.error("Orchestration reconcile failed for run %s: %s", run_id, exc)
            await db.rollback()
    return reconciled


async def reconcile_continuous_goals_async(db, now=None) -> int:
    from sqlalchemy import select
    from huddleroom.models.orchestration import OrchestrationGoal
    from huddleroom.services.orchestration_continuous_service import OrchestrationContinuousService
    from huddleroom.services.orchestration_service import OrchestrationService

    goal_ids = list((await db.scalars(select(OrchestrationGoal.id).where(
        OrchestrationGoal.goal_type == "continuous",
        OrchestrationGoal.status.in_(("active", "blocked", "paused")),
        OrchestrationGoal.continuous_state.is_not(None),
    ).order_by(OrchestrationGoal.id))).all())
    service = OrchestrationContinuousService(OrchestrationService())
    claimed = 0
    for goal_id in goal_ids:
        try:
            if await service.claim_due_cycle(db, goal_id, now) is not None:
                claimed += 1
        except Exception as exc:
            logger.error("Continuous reconcile failed for goal %s: %s", goal_id, exc)
            await db.rollback()
    return claimed


async def reconcile_orchestration_runs_job() -> None:
    from huddleroom.database import AsyncSessionLocal
    try:
        async with AsyncSessionLocal() as db:
            await reconcile_continuous_goals_async(db)
            await reconcile_orchestration_runs_async(db)
    except Exception as e:
        logger.error("Orchestration reconcile job failed: %s", e)


async def supervise_orchestration_job() -> None:
    from huddleroom.workers.orchestration_tasks import supervise_orchestration_async
    try:
        await supervise_orchestration_async()
    except Exception as exc:
        logger.error("Orchestration supervision sweep failed: %s", exc)


async def recover_orchestration_job(invoked_at: datetime | None = None) -> None:
    from huddleroom.workers.orchestration_recovery_tasks import recover_orchestration_async
    try:
        await recover_orchestration_async(invoked_at or _utcnow())
    except Exception as exc:
        logger.error("Orchestration recovery sweep failed: %s", exc)


def _capture_recovery_submission(event) -> None:
    if event.job_id == "recover_orchestration" and event.scheduled_run_times:
        _recovery_submission_times.append(event.scheduled_run_times[0])


async def scheduled_recover_orchestration_job() -> None:
    """Use APScheduler's submission time, before delayed coroutine execution."""
    await recover_orchestration_job(_recovery_submission_times.popleft() if _recovery_submission_times else _utcnow())


def create_scheduler() -> AsyncIOScheduler:
    global scheduler
    # MemoryJobStore: job state lost on restart. Acceptable for single recurring job in MVP.
    scheduler = AsyncIOScheduler(
        jobstores={"default": MemoryJobStore()},
    )
    scheduler.add_listener(_capture_recovery_submission, EVENT_JOB_SUBMITTED)
    scheduler.add_job(
        evaluate_cron_triggers_job,
        trigger="interval",
        seconds=60,
        id="evaluate_cron_triggers",
        replace_existing=True,
    )
    scheduler.add_job(
        archive_old_events_job,
        trigger="interval",
        hours=24,
        id="archive_old_events",
        replace_existing=True,
    )
    scheduler.add_job(
        process_graph_timeouts_job,
        trigger="interval",
        seconds=30,
        id="process_graph_timeouts",
        replace_existing=True,
    )
    scheduler.add_job(
        check_due_meetings_job,
        trigger="interval",
        seconds=30,
        id="check_due_meetings",
        replace_existing=True,
    )
    scheduler.add_job(
        check_blocked_tasks_for_meetings,
        trigger="interval",
        minutes=10,
        id="check_blocked_tasks_meetings",
        replace_existing=True,
    )
    scheduler.add_job(
        recover_orphaned_sessions_job,
        trigger="interval",
        minutes=5,
        id="recover_orphaned_sessions",
        replace_existing=True,
    )
    scheduler.add_job(
        reconcile_orchestration_runs_job,
        trigger="interval",
        seconds=settings.orchestration_reconcile_interval_seconds,
        id="reconcile_orchestration_runs",
        replace_existing=True,
    )
    scheduler.add_job(
        supervise_orchestration_job,
        trigger="interval",
        seconds=settings.orchestration_reconcile_interval_seconds,
        id="supervise_orchestration",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        scheduled_recover_orchestration_job,
        trigger="interval",
        seconds=settings.orchestration_reconcile_interval_seconds,
        id="recover_orchestration",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    return scheduler


def start_scheduler() -> AsyncIOScheduler:
    sched = create_scheduler()
    sched.start()
    scheduler_ready_at = _utcnow()
    sched.add_job(recover_orchestration_job, args=[scheduler_ready_at], id="recover_orchestration_immediate", replace_existing=True)
    logger.info("APScheduler started — cron trigger evaluation every 60s")
    return sched


def stop_scheduler() -> None:
    global scheduler
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=True)
        logger.info("APScheduler stopped")
        scheduler = None
