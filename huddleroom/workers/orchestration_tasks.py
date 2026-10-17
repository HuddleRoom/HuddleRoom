"""Shared asyncio/Celery entry points for orchestration supervision."""
from __future__ import annotations

from huddleroom.services.orchestration_baseline_runtime import control_operation_owner

from sqlalchemy.orm import SessionTransactionOrigin

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


async def _drain_answer_event(db, event_type, payload):
    if event_type != "authority.decision_resolved" or not payload.get("decision_id"):
        return
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_answer_receipts import drain_answer_applications
    from huddleroom.services.orchestration_baseline_runtime import owns_control_transaction
    if not owns_control_transaction(db):
        return  # A reusable event helper cannot publish supplied caller work.
    with db.no_autoflush:
        decision = await db.get(OrchestrationAuthorityDecision, UUID(str(payload["decision_id"])))
        run_id = decision.run_id if decision else None
        if decision is not None and decision.continuation_action_id is not None:
            from huddleroom.models.orchestration import OrchestrationAction
            action = await db.get(OrchestrationAction, decision.continuation_action_id)
            if action is not None and action.action_type == "apply_authority_answer":
                run_id = action.run_id
        if run_id is None and decision is not None and decision.source_process_run_id is not None:
            from huddleroom.models.orchestration_process import OrchestrationProcessRun
            source = await db.get(OrchestrationProcessRun, decision.source_process_run_id)
            run_id = source.run_id if source else None
        await db.commit()  # Ownership was proven at the actual worker session boundary.
        if run_id:
            await drain_answer_applications(db, run_id)
            await db.commit()


async def evaluate_supervision_run_async(run_id) -> int:
    async with AsyncSessionLocal() as db, control_operation_owner(db):
        try:
            count = await OrchestrationSupervisionScheduler().evaluate_run(db, UUID(str(run_id)))
            await db.commit()
            return count
        except Exception:
            await db.rollback()
            raise


async def evaluate_supervision_event_async(event_id) -> int:
    from huddleroom.services.orchestration_service import GOAL_RESUMED_EVENT_TYPE, OrchestrationService

    await asyncio.sleep(settings.orchestration_event_coalesce_seconds)
    async with AsyncSessionLocal() as db, control_operation_owner(db):
        try:
            event = await db.scalar(select(EventLog).where(EventLog.id == UUID(str(event_id))))
            if event is None or event.event_type not in SUPPORTED_EVENTS:
                return 0
            if event.event_type == GOAL_RESUMED_EVENT_TYPE:
                # Resume needs one decision-capable full tick, not a local judgment sweep.
                if not event.payload.get("run_id"):
                    return 0
                transaction = db.sync_session.get_transaction()
                if (not db.in_nested_transaction() and (transaction is None
                        or transaction.origin == SessionTransactionOrigin.AUTOBEGIN)):
                    await db.commit()  # The worker owns this event-read transaction.
                await OrchestrationService().tick(
                    db, UUID(str(event.payload["run_id"])), local_only=False
                )
                await db.commit()
                return 1
            await _drain_answer_event(db, event.event_type, event.payload)
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
    async with AsyncSessionLocal() as db, control_operation_owner(db):
        try:
            from huddleroom.models.orchestration import OrchestrationAction
            from huddleroom.services.orchestration_answer_receipts import drain_answer_applications
            pending_answers = list(await db.scalars(select(OrchestrationAction.run_id).where(
                OrchestrationAction.action_type == "apply_authority_answer", OrchestrationAction.status == "reserved"
            ).distinct().limit(settings.orchestration_sweep_goal_limit)))
            await db.commit()
            for run_id in pending_answers:
                await drain_answer_applications(db, run_id)
                await db.commit()
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


async def run_orchestration_event_supervisor(
    *, ready: asyncio.Event | None = None, execution_ready: asyncio.Event | None = None,
) -> None:
    """Consume only committed supported events; the durable sweep is the backstop."""
    from huddleroom.services.orchestration_service import GOAL_RESUMED_EVENT_TYPE, OrchestrationService

    delayed: asyncio.Task | None = None
    pending_run_ids: set[UUID] = set()
    resumed_run_ids: set[UUID] = set()

    async def evaluate_after_coalesce() -> None:
        if execution_ready is not None:
            await execution_ready.wait()
        while pending_run_ids or resumed_run_ids:
            await asyncio.sleep(settings.orchestration_event_coalesce_seconds)
            run_ids, resumed = set(pending_run_ids), set(resumed_run_ids)
            pending_run_ids.clear()
            resumed_run_ids.clear()
            async with AsyncSessionLocal() as db, control_operation_owner(db):
                for run_id in resumed:
                    try:
                        transaction = db.sync_session.get_transaction()
                        if (not db.in_nested_transaction() and (transaction is None
                                or transaction.origin == SessionTransactionOrigin.AUTOBEGIN)):
                            await db.commit()  # End owned preflight before a new transition lock.
                        await OrchestrationService().tick(db, run_id, local_only=False)
                        await db.commit()
                    except Exception:
                        await db.rollback()
                        logger.exception("Failed to resume orchestration run %s", run_id)
                for run_id in run_ids - resumed:
                    try:
                        await OrchestrationSupervisionScheduler().evaluate_run(db, run_id)
                        await db.commit()
                    except Exception:
                        await db.rollback()
                        logger.exception("Failed to evaluate orchestration run %s", run_id)

    subscription_options = {"ready": ready} if ready is not None else {}
    subscription = get_event_bus().subscribe(event_types=list(SUPPORTED_EVENTS), **subscription_options)
    try:
        async for event in subscription:
            if event.event_type == GOAL_RESUMED_EVENT_TYPE:
                # Resume needs one decision-capable full tick, not a local judgment sweep.
                if event.payload.get("run_id"):
                    resumed_run_ids.add(UUID(str(event.payload["run_id"])))
                    if delayed is None or delayed.done():
                        delayed = asyncio.create_task(evaluate_after_coalesce())
                continue
            async with AsyncSessionLocal() as db, control_operation_owner(db):
                try:
                    await _drain_answer_event(db, event.event_type, event.payload)
                    run_ids = await OrchestrationSupervisionScheduler().record_event(
                        db, event.event_type, event.payload, event_id=event.id, now=event.emitted_at,
                        project_id=event.project_id,
                    )
                    await db.commit()
                    pending_run_ids.update(run_ids)
                    if delayed is None or delayed.done():
                        delayed = asyncio.create_task(evaluate_after_coalesce())
                except Exception:
                    await db.rollback()
                    logger.exception("Failed to mark orchestration supervision event %s", event.id)
    finally:
        if delayed is not None:
            delayed.cancel()
            await asyncio.gather(delayed, return_exceptions=True)
        await subscription.aclose()


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
