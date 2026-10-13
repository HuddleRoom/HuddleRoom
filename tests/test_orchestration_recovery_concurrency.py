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


async def _diagnosis_ticks(db_session, test_engine, conversation_goal_run, monkeypatch, completion):
    from tests.test_orchestration_recovery_persistence import trigger_for
    from huddleroom.services.orchestration_recovery_analyzer import RecoveryAnalyzer
    goal, run = conversation_goal_run
    run.status = "running"
    trigger = await OrchestrationBlockerRecoveryService().create_trigger(db_session, trigger_for(goal, run))
    await db_session.commit()
    real_service = OrchestrationBlockerRecoveryService
    monkeypatch.setattr(
        "huddleroom.services.orchestration_blocker_recovery.OrchestrationBlockerRecoveryService",
        lambda: real_service(analyzer=RecoveryAnalyzer(completion_fn=completion)),
    )
    async def baseline(self, db, run_id, *, local_only=False):
        return {"status": "probe_baseline_stub"}
    monkeypatch.setattr(OrchestrationService, "_tick_baseline_impl", baseline)
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async def tick():
        async with factory() as session:
            return await OrchestrationService().tick(session, run.id)
    return factory, run.id, trigger.id, tick


@pytest.mark.asyncio
@pytest.mark.parametrize("contract_failure", [False, True])
async def test_public_ticks_enter_real_analyzer_once_before_completion(
    db_session, test_engine, conversation_goal_run, test_agent, monkeypatch, contract_failure,
):
    import json
    from tests.test_orchestration_recovery_persistence import gap_payload
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0
    async def completion(**kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        payload = {"kind": "proposal"} if contract_failure else gap_payload(test_agent)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}
    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion,
    )
    first = asyncio.create_task(tick())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        second = asyncio.create_task(tick())
        # The losing public pass skips promptly; it must neither queue nor spend a call.
        await asyncio.wait_for(asyncio.shield(second), 1)
        assert calls == 1
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
    from huddleroom.models.orchestration import OrchestrationDecision
    async with factory() as check:
        exchanges = list(await check.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery",
        )))
        assert len(exchanges) == 1
        if contract_failure:
            assert calls == 3
            assert exchanges[0].rejection_reason.startswith("analysis_contract_rejected")
    if contract_failure:
        await tick()
        assert calls == 6, "typed failure must release the flight for a later logical diagnosis"


@pytest.mark.asyncio
async def test_public_ticks_exhaust_three_contract_refusals_without_provider_backoff(
    db_session, test_engine, conversation_goal_run, monkeypatch,
):
    from unittest.mock import AsyncMock
    from huddleroom.models.orchestration import OrchestrationDecision
    completion = AsyncMock(return_value={"choices": [{"message": {"content": '{"kind":"proposal"}'}}]})
    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion,
    )
    for attempt in range(3):
        await tick()
        assert completion.await_count == (attempt + 1) * 3
        async with factory() as check:
            trigger = await check.get(OrchestrationAction, trigger_id)
            assert not trigger.dispatch_contract.get("last_diagnosis_error_at")
    await tick()
    await tick()
    assert completion.await_count == 9
    async with factory() as check:
        exchanges = list(await check.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery",
        )))
        rejected = [row for row in exchanges if row.validator_status == "rejected"]
        assert len(rejected) == 3
        assert all(row.rejection_reason.startswith("analysis_contract_rejected") for row in rejected)
        assert len([row for row in exchanges if row.validator_status == "accepted"]) == 1


@pytest.mark.asyncio
async def test_public_tick_cancellation_keeps_flight_until_actual_analyzer_stops(
    db_session, test_engine, conversation_goal_run, test_agent, monkeypatch,
):
    import json
    from tests.test_orchestration_recovery_persistence import gap_payload
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0
    async def completion(**kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # A provider adapter may take time to acknowledge cancellation. Ownership
            # follows its real coroutine lifetime, not the outer cancel request.
            cancelled.set()
            await release.wait()
            raise
        return {"choices": [{"message": {"content": json.dumps(gap_payload(test_agent))}}]}
    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion,
    )
    first = asyncio.create_task(tick())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        first.cancel()
        await asyncio.wait_for(cancelled.wait(), 5)
        second = asyncio.create_task(tick())
        await asyncio.wait_for(asyncio.shield(second), 1)
        assert calls == 1
        assert not first.done()
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
    # Actual acknowledgement releases the gate and a fresh public tick can diagnose.
    await tick()
    assert calls == 2


@pytest.mark.asyncio
async def test_public_provider_outage_backs_off_without_consuming_refusal_budget(
    db_session, test_engine, conversation_goal_run, monkeypatch,
):
    from unittest.mock import AsyncMock
    from huddleroom.models.orchestration import OrchestrationDecision
    completion = AsyncMock(side_effect=OSError("provider unavailable"))
    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion,
    )
    for attempt in range(4):
        await tick()
        await tick()  # backoff means the immediate subsequent public tick spends nothing
        assert completion.await_count == attempt + 1
        async with factory() as check:
            trigger = await check.get(OrchestrationAction, trigger_id)
            assert trigger.dispatch_contract.get("last_diagnosis_error_at")
            trigger.dispatch_contract = {**trigger.dispatch_contract, "last_diagnosis_error_at": None}
            await check.commit()
    async with factory() as check:
        rows = list(await check.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery",
        )))
        assert len(rows) == 4
        assert all(row.rejection_reason.startswith("analysis_provider_failed") for row in rows)
        proposals = list(await check.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "recovery_proposal",
        )))
        assert proposals == []


@pytest.mark.asyncio
async def test_public_diagnosis_reset_invalidation_releases_flight_and_publishes_nothing(
    db_session, test_engine, conversation_goal_run, test_agent, monkeypatch,
):
    import json
    from tests.test_orchestration_recovery_persistence import gap_payload
    from huddleroom.services import orchestration_baseline_runtime as runtime
    from huddleroom.models.orchestration import OrchestrationDecision
    goal, run = conversation_goal_run
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0
    async def completion(**kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return {"choices": [{"message": {"content": json.dumps(gap_payload(test_agent))}}]}
    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion,
    )
    first = asyncio.create_task(tick())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        runtime._resetting[goal.id] = runtime._resetting.get(goal.id, 0) + 1
        release.set()
        await asyncio.wait_for(first, 5)
        async with factory() as check:
            assert list(await check.scalars(select(OrchestrationDecision).where(
                OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery",
            ))) == []
    finally:
        runtime._resetting.pop(goal.id, None)
        release.set()
        await asyncio.gather(first, return_exceptions=True)
    # Remove only the test's temporary reset reservation; the still-current trigger
    # can then diagnose, proving the gate was cleared on invalidation.
    async with factory() as check:
        trigger = await check.get(OrchestrationAction, trigger_id)
        trigger.dispatch_contract = {**trigger.dispatch_contract, "last_diagnosis_error_at": None}
        await check.commit()
    await tick()
    assert calls == 2


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
