"""Recovery worker: snapshot whole goals, observe outside DB, then apply."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from huddleroom.database import AsyncSessionLocal
from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.services.orchestration_recovery_service import OrchestrationRecoveryService, RunnerObservation


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def observe_runner(runner_task_id: str | None, *, is_sqlite: bool) -> str:
    if not runner_task_id:
        return "unknown"
    if is_sqlite:
        from huddleroom.workers.task_runner import is_task_live
        return "active" if is_task_live(runner_task_id) else "unknown"
    try:
        from huddleroom.workers.celery_app import app
        inspector = app and app.control.inspect()
        for method, state in (("active", "active"), ("reserved", "reserved"), ("scheduled", "reserved")):
            queues = getattr(inspector, method)() or {}
            if any(any((item.get("id") if method != "scheduled" else (item.get("request") or {}).get("id")) == runner_task_id
                       for item in items) for items in queues.values()):
                return state
        raw = app.AsyncResult(runner_task_id).state if app else "PENDING"
        return {"RECEIVED": "reserved", "STARTED": "started", "RETRY": "retrying", "SUCCESS": "terminal_success",
                "FAILURE": "terminal_failure", "REVOKED": "terminal_revoked", "PENDING": "unknown"}.get(raw, "unknown")
    except Exception:
        return "unknown"


async def recover_orchestration_async(scheduler_ready_at: datetime) -> int:
    """The caller owns readiness time; recovery must never invent it."""
    if scheduler_ready_at is None or scheduler_ready_at.tzinfo is None:
        raise ValueError("recovery requires an aware invocation timestamp")
    from huddleroom.config import settings
    async with AsyncSessionLocal() as db:
        goal_ids = list((await db.scalars(select(OrchestrationGoal.id).where(
            OrchestrationGoal.status.in_(("active", "blocked", "paused"))))).all())
    recovered = 0
    cumulative_observation_duration = 0.0
    service = OrchestrationRecoveryService()
    for goal_id in goal_ids:
        try:
            async with AsyncSessionLocal() as snapshot_db:
                snapshot = await service.build_goal_snapshot(snapshot_db, goal_id, scheduler_ready_at)
            if snapshot is None:
                continue
            observation_started = _utcnow()
            observation_duration = 0.0
            observations = {}
            for item in snapshot.sessions:
                probe_started = _utcnow()
                state = observe_runner(item.runner_task_id, is_sqlite=settings.is_sqlite)
                observed_at = _utcnow()
                observation_duration += (observed_at - probe_started).total_seconds()
                observations[item.session_id] = RunnerObservation(item.runner_task_id, state, observed_at)
            observation_ended = _utcnow()
            cumulative_observation_duration += observation_duration
            assessment_at = _utcnow()
            timing = {
                "scheduler_ready_at": scheduler_ready_at.isoformat(),
                "observation_started_at": observation_started.isoformat(),
                "observation_ended_at": observation_ended.isoformat(),
                "observation_duration_seconds": observation_duration,
                "assessment_at": assessment_at.isoformat(),
                "excluded_duration_seconds": cumulative_observation_duration,
                "within_2r": (assessment_at - scheduler_ready_at).total_seconds() - cumulative_observation_duration
                <= 2 * settings.orchestration_reconcile_interval_seconds,
            }
            async with AsyncSessionLocal() as apply_db:
                result = await service.apply_goal_recovery(apply_db, snapshot, observations, recovery_timing=timing)
                await apply_db.commit()
            recovered += result is not None
        except Exception:
            # A corrupt/contended goal cannot starve other independent goals.
            continue
    return recovered


def register_celery_task(app):
    @app.task(bind=True, name="rally.workers.orchestration_recovery_tasks.recover_orchestration")
    def recover_orchestration(task, scheduler_ready_at: str | None = None):
        import asyncio
        scheduler_ready_at = scheduler_ready_at or (task.request.headers or {}).get("orchestration_recovery_ready_at")
        if not scheduler_ready_at:
            raise ValueError("recovery requires an invocation timestamp")
        ready_at = datetime.fromisoformat(scheduler_ready_at)
        if ready_at.tzinfo is None:
            raise ValueError("recovery requires an aware invocation timestamp")
        return asyncio.run(recover_orchestration_async(ready_at))

    return recover_orchestration


try:
    from huddleroom.workers.celery_app import app as _app
    if _app is not None:
        register_celery_task(_app)
except ImportError:
    pass
