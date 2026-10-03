"""Shared asyncio/Celery entry points for orchestration supervision."""
from __future__ import annotations

import asyncio
import logging
from datetime import timedelta, timezone
from uuid import UUID
from sqlalchemy import select

from huddleroom.database import AsyncSessionLocal
from huddleroom.services.event_bus import get_event_bus
from huddleroom.services.orchestration_supervision_scheduler import (
    SUPPORTED_EVENTS,
    OrchestrationSupervisionScheduler,
)
from huddleroom.config import settings
from huddleroom.models.event_log import EventLog

logger = logging.getLogger(__name__)
CELERY_APP = None
PENDING_WAKEUP: asyncio.Task | None = None


def request_supervision_wakeup(event_id) -> None:
    """Schedule after commit only; scheduling failure never rolls back the event."""
    if settings.is_sqlite:
        global PENDING_WAKEUP
        if PENDING_WAKEUP is None or PENDING_WAKEUP.done():
            PENDING_WAKEUP = asyncio.get_running_loop().create_task(evaluate_supervision_event_async(event_id))
    elif CELERY_APP is not None:
        evaluate_supervision_event.delay(str(event_id))


def request_supervision_evaluation(run_id) -> None:
    """Enqueue provider work only after its local sweep transaction commits."""
    if settings.is_sqlite:
        asyncio.get_running_loop().create_task(evaluate_supervision_run_async(run_id))
    elif CELERY_APP is not None:
        evaluate_supervision_run.delay(str(run_id))


async def evaluate_supervision_run_async(run_id) -> int:
    async with AsyncSessionLocal() as db:
        try:
            count = await OrchestrationSupervisionScheduler().evaluate_run(db, UUID(str(run_id)))
            await db.commit()
            return count
        except Exception:
            await db.rollback()
            raise


async def evaluate_supervision_event_async(event_id) -> int:
    await asyncio.sleep(settings.orchestration_event_coalesce_seconds)
    async with AsyncSessionLocal() as db:
        try:
            event = await db.scalar(select(EventLog).where(EventLog.id == UUID(str(event_id))))
            if event is None or event.event_type not in SUPPORTED_EVENTS:
                return 0
            scheduler = OrchestrationSupervisionScheduler()
            run_ids = await scheduler.record_event(
                db, event.event_type, event.payload, event_id=event.id,
                now=(
                    event.emitted_at.replace(tzinfo=timezone.utc)
                    if event.emitted_at.tzinfo is None
                    else event.emitted_at.astimezone(timezone.utc)
                ),
                project_id=event.project_id,
            )
            await db.commit()
            evaluated_at = event.emitted_at + timedelta(
                seconds=settings.orchestration_event_coalesce_seconds
            )
            count = sum([await scheduler.evaluate_run(db, run_id, now=evaluated_at) for run_id in run_ids])
            await db.commit()
            return count
        except Exception:
            await db.rollback()
            raise


async def supervise_orchestration_async() -> int:
    async with AsyncSessionLocal() as db:
        try:
            # Import here so deployments/tests can replace the durable scheduler.
            scheduler = OrchestrationSupervisionScheduler()
            due_run_ids = await scheduler.sweep(
                db, goal_limit=settings.orchestration_sweep_goal_limit,
                timeout_seconds=settings.orchestration_sweep_seconds_limit, collect_due=True,
            )
            await db.commit()
            for run_id in due_run_ids:
                request_supervision_evaluation(run_id)
            return len(due_run_ids)
        except Exception:
            await db.rollback()
            raise


async def run_orchestration_event_supervisor() -> None:
    """Consume only committed supported events; the durable sweep is the backstop."""
    delayed = None

    async def evaluate_after_coalesce() -> None:
        nonlocal pending_run_ids
        await asyncio.sleep(settings.orchestration_event_coalesce_seconds)
        async with AsyncSessionLocal() as db:
            try:
                run_ids, pending_run_ids = set(pending_run_ids), set()
                for run_id in run_ids:
                    await OrchestrationSupervisionScheduler().evaluate_run(db, run_id)
                await db.commit()
            except Exception:
                await db.rollback()
                logger.exception("Failed to evaluate coalesced orchestration supervision events")

    pending_run_ids = set()
    async for event in get_event_bus().subscribe(event_types=list(SUPPORTED_EVENTS)):
        async with AsyncSessionLocal() as db:
            try:
                pending_run_ids.update(await OrchestrationSupervisionScheduler().record_event(
                    db, event.event_type, event.payload, event_id=event.id, now=event.emitted_at,
                    project_id=event.project_id,
                ))
                await db.commit()
                if delayed is None or delayed.done():
                    delayed = asyncio.create_task(evaluate_after_coalesce())
            except Exception:
                await db.rollback()
                logger.exception("Failed to mark orchestration supervision event %s", event.id)


try:
    from huddleroom.workers.celery_app import app as CELERY_APP

    if CELERY_APP is not None:
        @CELERY_APP.task(name="rally.workers.orchestration_tasks.evaluate_supervision_event")
        def evaluate_supervision_event(event_id):
            return asyncio.run(evaluate_supervision_event_async(event_id))

        @CELERY_APP.task(name="rally.workers.orchestration_tasks.supervise_orchestration")
        def supervise_orchestration():
            return asyncio.run(supervise_orchestration_async())

        @CELERY_APP.task(name="rally.workers.orchestration_tasks.evaluate_supervision_run")
        def evaluate_supervision_run(run_id):
            return asyncio.run(evaluate_supervision_run_async(run_id))
except ImportError:
    pass
