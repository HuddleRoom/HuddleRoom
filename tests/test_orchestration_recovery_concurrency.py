"""Concurrent public recovery ticks must be prompt, clean, and exactly once."""
from __future__ import annotations

import asyncio
from collections import Counter
from time import monotonic

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.orchestration import OrchestrationAction
from tests.test_orchestration_recovery_execution import (
    Executors,
    retry_task_call,
    seed,
    verify_call,
    approved,
)
from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
from huddleroom.services.orchestration_service import OrchestrationService


@pytest.mark.asyncio
async def test_concurrent_public_ticks_serialize_recovery_through_commit(
    db_session, test_engine, test_project, test_user, monkeypatch, caplog,
):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows), retry_task_call(rows))
    executors = Executors(monkeypatch, rows)
    await db_session.commit()

    # Isolate the confirmed recovery race from baseline model/project work while
    # retaining both real public tick recovery passes.
    async def baseline_tick(self, db, run_id, *, local_only=False):
        return {"run_id": str(run_id), "status": "probe_baseline_stub"}

    monkeypatch.setattr(OrchestrationService, "_tick_baseline_impl", baseline_tick)

    # Rendezvous outside the real driver, before any lifecycle lock. The first resume
    # yields once so both unprotected sessions enter it on the unfixed implementation.
    original_drive = OrchestrationService._drive_recovery
    entered = 0
    both_entered = asyncio.Event()

    async def synchronized_drive(service, db, run_id, **kwargs):
        nonlocal entered
        if not kwargs.get("diagnose", True):
            entered += 1
            if entered == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=5)
        return await original_drive(service, db, run_id, **kwargs)

    monkeypatch.setattr(OrchestrationService, "_drive_recovery", synchronized_drive)
    original_resume = OrchestrationBlockerRecoveryService.resume
    active = maximum_active = 0

    async def observed_resume(service, db, proposal_id):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            await asyncio.sleep(0.05)
            return await original_resume(service, db, proposal_id)
        finally:
            active -= 1

    monkeypatch.setattr(OrchestrationBlockerRecoveryService, "resume", observed_resume)
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def actual_tick():
        async with factory() as session:
            try:
                return await OrchestrationService().tick(session, rows.run.id)
            except Exception as exc:
                await session.rollback()
                return f"{type(exc).__name__}: {exc}"

    started = monotonic()
    outcomes = await asyncio.gather(actual_tick(), actual_tick())
    elapsed = monotonic() - started
    async with factory() as check:
        proposal = await check.get(OrchestrationAction, flow.proposal.id, populate_existing=True)
        recovery_rows = (await check.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == rows.run.id,
            OrchestrationAction.idempotency_key.startswith(f"recovery:{flow.pid}:"),
        ).execution_options(populate_existing=True))).all()
        by_type = Counter(row.action_type for row in recovery_rows)
        logs = [record.getMessage() for record in caplog.records if "recovery step failed" in record.getMessage()]
        result = {
            "elapsed_seconds": elapsed,
            "maximum_active_resumes": maximum_active,
            "tick_outcomes": outcomes,
            "proposal_status": proposal.status,
            "wrapper_counts": dict(by_type),
            "recovery_step_failure_logs": logs,
            "executor_request_verification_calls": len(executors.keys("request_verification")),
        }
        print(f"PUBLIC_TICK_RECOVERY_CONCURRENCY={result}")
        assert all(isinstance(item, dict) and item["status"] == "probe_baseline_stub" for item in outcomes), result
        assert logs == [], result
        assert maximum_active == 1, result
        assert elapsed < 5, result
        assert proposal.status == "completed", result
        assert by_type["recovery_apply"] == 2, result
        assert by_type["recovery_verify"] == 2, result
        assert by_type["recovery_retry"] == 1, result
        effect_rows = [row for row in recovery_rows if row.idempotency_key.endswith(":effect")]
        assert len(effect_rows) == 3, result
        assert all(row.status == "completed" for row in effect_rows), result
        assert len(set(executors.keys())) == len(executors.keys()) == 3, result


@pytest.mark.asyncio
async def test_concurrent_ticks_without_recovery_are_clean_control(
    db_session, test_engine, test_project, monkeypatch, caplog,
):
    rows = await seed(db_session, test_project)
    await db_session.commit()

    async def baseline_tick(self, db, run_id, *, local_only=False):
        return {"run_id": str(run_id), "status": "probe_baseline_stub"}

    monkeypatch.setattr(OrchestrationService, "_tick_baseline_impl", baseline_tick)
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def actual_tick():
        async with factory() as session:
            try:
                return await OrchestrationService().tick(session, rows.run.id)
            except Exception as exc:
                await session.rollback()
                return f"{type(exc).__name__}: {exc}"

    started = monotonic()
    outcomes = await asyncio.gather(actual_tick(), actual_tick())
    elapsed = monotonic() - started
    logs = [record.getMessage() for record in caplog.records if "recovery step failed" in record.getMessage()]
    result = {"elapsed_seconds": elapsed, "tick_outcomes": outcomes, "recovery_step_failure_logs": logs}
    print(f"PUBLIC_TICK_NO_RECOVERY_CONTROL={result}")
    assert all(isinstance(item, dict) and item["status"] == "probe_baseline_stub" for item in outcomes), result
    assert logs == [], result
    assert elapsed < 5, result


@pytest.mark.asyncio
async def test_recovery_driver_preserves_explicit_caller_transaction(
    db_session, test_engine, test_project, test_user, monkeypatch,
):
    rows = await seed(db_session, test_project)
    flow = await approved(db_session, rows, test_user, verify_call(rows))
    Executors(monkeypatch, rows)
    await db_session.commit()
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async with factory() as session:
        transaction = await session.begin()
        # sqlite legacy transaction mode otherwise RELEASE of the first savepoint
        # commits despite the SQLAlchemy outer transaction. Start the physical one.
        await session.execute(text("BEGIN"))
        await OrchestrationService()._drive_recovery(session, rows.run.id, diagnose=False)
        assert session.get_transaction() is transaction
        proposal = await session.get(OrchestrationAction, flow.proposal.id, populate_existing=True)
        assert proposal.status == "completed"
        await transaction.rollback()

    async with factory() as check:
        proposal = await check.get(OrchestrationAction, flow.proposal.id, populate_existing=True)
        assert proposal.status == "reserved"
        effects = list(await check.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == rows.run.id,
            OrchestrationAction.idempotency_key.startswith(f"recovery:{flow.pid}:"),
            OrchestrationAction.idempotency_key.endswith(":effect"),
        )))
        assert effects == []
