import logging
from datetime import datetime, timezone, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from huddleroom.database import AsyncSessionLocal
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.session import SessionCreate
from huddleroom.services.session_service import SessionService

logger = logging.getLogger(__name__)


async def evaluate_cron_triggers_async(session_factory=None) -> tuple[int, int]:
    """Evaluate all cron triggers. Returns (evaluated, enqueued) counts."""
    session_service = SessionService()
    factory = session_factory or AsyncSessionLocal

    evaluated = 0
    enqueued = 0

    async with factory() as db:
        result = await db.execute(
            select(Task).where(
                Task.status.in_(["ready", "in_progress"]),
                Task.assigned_to.isnot(None),
                Task.trigger.isnot(None),
            )
        )
        tasks = result.scalars().all()

        cron_tasks = [t for t in tasks if isinstance(t.trigger, dict) and t.trigger.get("type") == "cron"]
        evaluated = len(cron_tasks)

        now = datetime.now(timezone.utc)

        for task in cron_tasks:
            try:
                from croniter import croniter
                cron_spec = task.trigger.get("spec", "* * * * *")
                # Check if the most recent scheduled fire falls within the last 120s.
                # Using 120s window (not 60s) to tolerate scheduler jitter.
                cron = croniter(cron_spec, now - timedelta(seconds=120))
                prev_fire_naive = cron.get_next(datetime)
                prev_fire = prev_fire_naive.replace(tzinfo=timezone.utc)
                delta = (now - prev_fire).total_seconds()
                if 0 <= delta <= 120:
                    existing = await db.execute(
                        select(Session).where(
                            Session.task_id == task.id,
                            Session.status.in_(["pending", "running"]),
                        )
                    )
                    if existing.scalar_one_or_none() is not None:
                        continue
                    session_data = SessionCreate(
                        agent_id=task.assigned_to,
                        task_id=task.id,
                        project_id=task.project_id,
                        origin="trigger",
                    )
                    await session_service.create(db, session_data)
                    await db.flush()
                    enqueued += 1
                    logger.info("Cron trigger fired for task %s", task.id)
            except Exception as e:
                logger.warning("Error evaluating cron trigger for task %s: %s", task.id, e)

    logger.info("Cron trigger evaluation: %d evaluated, %d sessions enqueued", evaluated, enqueued)

    return evaluated, enqueued


try:
    import asyncio
    from huddleroom.workers.celery_app import app

    if app is not None:
        @app.task(name="rally.workers.trigger_tasks.evaluate_cron_triggers")
        def evaluate_cron_triggers():
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(evaluate_cron_triggers_async())
            finally:
                loop.close()

except ImportError:
    pass
