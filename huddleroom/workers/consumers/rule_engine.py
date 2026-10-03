from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.database import AsyncSessionLocal
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.session import SessionCreate
from huddleroom.services.event_bus import BusEvent, get_event_bus
from huddleroom.services.session_service import SessionService

logger = logging.getLogger(__name__)


async def _active_task_ids(db: AsyncSession, task_ids: list[uuid.UUID]) -> set[uuid.UUID]:
    if not task_ids:
        return set()
    result = await db.execute(
        select(Session.task_id).where(
            Session.task_id.in_(task_ids),
            Session.status.in_(["pending", "running"]),
        )
    )
    return set(result.scalars().all())


async def _create_trigger_session(db: AsyncSession, task: Task) -> None:
    session_data = SessionCreate(
        agent_id=task.assigned_to,
        task_id=task.id,
        project_id=task.project_id,
        origin="trigger",
    )
    await SessionService().create(db, session_data)
    logger.info("Trigger session created for task %s", task.id)


async def evaluate_event_triggers(db: AsyncSession, event: BusEvent) -> int:
    """Evaluate event and task_status triggers for the given event.

    Returns count of sessions created.
    """
    created = 0

    result = await db.execute(
        select(Task).where(
            Task.project_id == event.project_id,
            Task.status == "ready",
            Task.assigned_to.isnot(None),
            Task.trigger.isnot(None),
        )
    )
    tasks = list(result.scalars().all())

    # Single bulk check for active sessions across all candidate tasks
    active_ids = await _active_task_ids(db, [t.id for t in tasks])

    for task in tasks:
        trigger = task.trigger
        if not isinstance(trigger, dict):
            continue
        if task.id in active_ids:
            continue

        trigger_type = trigger.get("type")

        if trigger_type == "event":
            if trigger.get("event_type") == event.event_type:
                await _create_trigger_session(db, task)
                active_ids.add(task.id)  # prevent duplicates within this loop run
                created += 1

        elif trigger_type == "task_status" and event.event_type == "task.status_changed":
            watched_id = trigger.get("task_id")
            target_status = trigger.get("target_status")
            payload_task_id = event.payload.get("task_id")
            payload_status = event.payload.get("status")
            if (
                watched_id
                and target_status
                and str(watched_id) == str(payload_task_id)
                and target_status == payload_status
            ):
                await _create_trigger_session(db, task)
                active_ids.add(task.id)  # prevent duplicates within this loop run
                created += 1

    return created


async def run_rule_engine() -> None:
    """Consumer task: evaluates event and task_status triggers on every event."""
    bus = get_event_bus()
    logger.info("rule_engine consumer started")
    async for event in bus.subscribe(project_id=None):
        try:
            async with AsyncSessionLocal() as db:
                count = await evaluate_event_triggers(db, event)
                await db.commit()
                if count:
                    logger.info("rule_engine: %d trigger session(s) created for %s", count, event.event_type)
        except Exception as exc:
            logger.error("rule_engine error processing %s: %s", event.event_type, exc)
