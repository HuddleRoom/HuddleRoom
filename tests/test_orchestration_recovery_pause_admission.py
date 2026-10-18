"""Pause fences new recovery work while preserving an already admitted response."""

import asyncio
import inspect
import json

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationRun
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_recovery_evidence import RecoveryEvidenceService
from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
from huddleroom.services.orchestration_recovery_analyzer import RecoveryAnalyzer
from tests.test_orchestration_recovery_concurrency import _diagnosis_ticks
from tests.test_orchestration_recovery_persistence import gap_payload


@pytest.mark.asyncio
@pytest.mark.parametrize("response_kind", ["evidence_query", "malformed", "terminal", "fallback"])
async def test_public_recovery_pause_stops_next_admission_without_reclassifying_failure(
    db_session, test_engine, conversation_goal_run, test_agent, monkeypatch, response_kind,
):
    """Removing per-call lifecycle admission must spend more calls or publish after pause."""
    from huddleroom.services import orchestration_recovery_analyzer as analyzer_module

    entered, release = asyncio.Event(), asyncio.Event()
    goal, run = conversation_goal_run
    goal_id, project_id = goal.id, goal.project_id
    calls, evidence_queries, events = [], [], []
    usage = {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20}
    original_query = RecoveryEvidenceService.query
    original_invocation = analyzer_module.AgentResponseInvocation

    async def observed_query(service, db, trigger, query):
        evidence_queries.append(query.model_dump(mode="json"))
        return await original_query(service, db, trigger, query)

    async def publish(event):
        events.append(event)

    def invocation(context):
        return original_invocation(context, publish=publish)

    monkeypatch.setattr(RecoveryEvidenceService, "query", observed_query)
    monkeypatch.setattr(analyzer_module, "AgentResponseInvocation", invocation)

    async def completion(**values):
        calls.append(values)
        entered.set()
        await release.wait()
        if response_kind == "fallback":
            raise TypeError("unexpected keyword argument 'stream'")
        payload = gap_payload(test_agent)
        if response_kind == "evidence_query":
            payload = {"kind": "evidence_query", "query": {"kind": "waits"}}
        elif response_kind == "malformed":
            payload = {"kind": "proposal"}
        return {"choices": [{"message": {"content": json.dumps(payload)}}], "usage": usage}

    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion,
    )
    pending = asyncio.create_task(tick())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        async with factory() as pausing:
            paused_goal, paused_run = await OrchestrationService().pause_goal(pausing, project_id, goal_id)
            assert paused_goal.status == paused_run.status == "paused"
        release.set()
        await asyncio.wait_for(pending, 5)
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)

    # A completion admitted before Pause keeps its successful response and real usage,
    # even when its parsed content would otherwise cause evidence or structured repair.
    terminal = [event for event in events if event.event_type == "agent_response.terminal"]
    first_terminal = terminal[0]
    if response_kind == "fallback":
        assert first_terminal.payload["status"] == "failed"
        assert first_terminal.payload["metadata"]["reason"] == "streaming_unsupported"
    else:
        assert first_terminal.payload["status"] == "completed"
        assert first_terminal.payload["final_response"]["usage"] == usage

    async with factory() as check:
        trigger = await check.get(OrchestrationAction, trigger_id)
        persisted_run = await check.get(OrchestrationRun, run_id)
        exchanges = list(await check.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery",
        )))
        observed = {
            "provider_calls": len(calls), "evidence_queries": len(evidence_queries),
            "exchanges": len(exchanges), "trigger_status": trigger.status,
            "backoff": trigger.dispatch_contract.get("last_diagnosis_error_at"),
            "feedback": trigger.dispatch_contract.get("analysis_feedback"), "run_status": persisted_run.status,
        }
        assert observed == {
            "provider_calls": 1, "evidence_queries": 0, "exchanges": int(response_kind != "fallback"), "trigger_status": "reserved",
            "backoff": None, "feedback": None, "run_status": "paused",
        }

    # Another public pass while paused consumes no provider allowance.
    await tick()
    assert len(calls) == 1

    # Supported resume reuses the same trigger and proves admission ownership cleared.
    async with factory() as resuming:
        await OrchestrationService().resume_goal(resuming, project_id, goal_id)
        # Resume keeps the baseline paused; explicit re-authorization admits the next provider call.
        await OrchestrationService().authorize_baseline(resuming, project_id, goal_id, actor="test")
        await resuming.commit()
    response_kind = "terminal"
    await tick()
    assert len(calls) == 2
    async with factory() as check:
        trigger = await check.get(OrchestrationAction, trigger_id)
        exchanges = list(await check.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery",
        )))
        assert trigger.status == "completed"
        accepted = [row for row in exchanges if row.validator_status == "accepted"]
        assert len(accepted) == 1


@pytest.mark.asyncio
async def test_pause_committed_during_final_admission_read_commit_prevents_provider_start(
    db_session, test_engine, conversation_goal_run, test_agent, monkeypatch,
):
    """A status read before an awaited commit cannot authorize the following provider call."""
    goal, _run = conversation_goal_run
    goal_id, project_id = goal.id, goal.project_id
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def completion(**request):
        calls.append(request)
        return {"choices": [{"message": {"content": json.dumps(gap_payload(test_agent))}}]}

    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion,
    )
    original_commit = AsyncSession.commit
    admission_commits = 0

    async def gated_commit(session):
        nonlocal admission_commits
        # Narrow the deterministic rendezvous to the runnable-check callback's
        # read commit. Its second call immediately precedes the physical provider
        # boundary; the first is the evidence-round admission.
        caller = inspect.currentframe().f_back
        is_admission = caller.f_code.co_name == "before_completion"
        del caller
        if is_admission:
            admission_commits += 1
            if admission_commits == 2:
                entered.set()
                await release.wait()
        await original_commit(session)

    monkeypatch.setattr(AsyncSession, "commit", gated_commit)
    pending = asyncio.create_task(tick())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert calls == []
        async with factory() as pausing:
            await OrchestrationService().pause_goal(pausing, project_id, goal_id)
        release.set()
        await asyncio.wait_for(pending, 5)
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)

    assert calls == [], "Pause returned before the first provider invocation, but it was still admitted"
    async with factory() as check:
        result = await check.get(OrchestrationAction, trigger_id)
        assert result.status == "reserved"
        assert not result.dispatch_contract.get("last_diagnosis_error_at")
        assert list(await check.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery",
        ))) == []


@pytest.mark.asyncio
async def test_direct_diagnosis_of_paused_run_spends_no_provider_call(
    db_session, test_engine, conversation_goal_run, test_agent, monkeypatch,
):
    goal, _run = conversation_goal_run
    calls = []

    async def completion(**request):
        calls.append(request)
        return {"choices": [{"message": {"content": json.dumps(gap_payload(test_agent))}}]}

    factory, run_id, trigger_id, _tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion,
    )
    async with factory() as pausing:
        await OrchestrationService().pause_goal(pausing, goal.project_id, goal.id)
    async with factory() as caller:
        service = OrchestrationBlockerRecoveryService(analyzer=RecoveryAnalyzer(completion_fn=completion))
        with pytest.raises(HTTPException) as caught:
            await service.diagnose(caller, trigger_id)
        assert caught.value.status_code == 409
        assert calls == []
    async with factory() as check:
        result = await check.get(OrchestrationAction, trigger_id)
        assert result.status == "reserved"
        assert not result.dispatch_contract.get("last_diagnosis_error_at")
        assert list(await check.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery",
        ))) == []


@pytest.mark.asyncio
async def test_diagnosis_admission_preserves_explicit_caller_transaction_rollback(
    db_session, test_engine, conversation_goal_run, test_agent, monkeypatch,
):
    from sqlalchemy import update
    from huddleroom.models.project import Project
    calls = []
    async def completion(**_request):
        calls.append(True)
        return {"choices": [{"message": {"content": json.dumps(gap_payload(test_agent))}}]}
    factory, run_id, trigger_id, _tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion)
    project_id = conversation_goal_run[0].project_id
    async with factory() as fresh:
        original_name = await fresh.scalar(select(Project.name).where(Project.id == project_id))
    async with factory() as caller:
        transaction = await caller.begin()
        await caller.execute(update(Project).where(Project.id == project_id).values(name="caller-owned staging"))
        service = OrchestrationBlockerRecoveryService(analyzer=RecoveryAnalyzer(completion_fn=completion))
        with pytest.raises(HTTPException) as conflict:
            await service.diagnose(caller, trigger_id)
        assert conflict.value.status_code == 409
        assert calls == []
        assert caller.get_transaction() is transaction and caller.in_transaction()
        assert await caller.scalar(select(Project.name).where(Project.id == project_id)) == "caller-owned staging"
        await transaction.rollback()
    async with factory() as check:
        assert await check.scalar(select(Project.name).where(Project.id == project_id)) == original_name
        assert (await check.get(OrchestrationAction, trigger_id)).status == "reserved"
        assert list(await check.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery"))) == []


@pytest.mark.asyncio
async def test_public_tick_supports_injected_analyzer_with_original_keyword_signature(
    db_session, test_engine, conversation_goal_run, test_agent, monkeypatch,
):
    """Adding the optional guard must preserve existing injected analyzer contracts."""
    from huddleroom.services.orchestration_recovery_analyzer import parse_recovery_assessment

    calls = []

    async def forbidden_completion(**_request):
        pytest.fail("The injected analyzer must not start a provider completion")

    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, forbidden_completion,
    )

    async def analyze(_analyzer, snapshot, *, project_id, evidence_query, on_raw=None):
        calls.append(snapshot)
        return parse_recovery_assessment(gap_payload(test_agent))

    monkeypatch.setattr(RecoveryAnalyzer, "analyze", analyze)
    await tick()
    async with factory() as check:
        trigger = await check.get(OrchestrationAction, trigger_id)
        exchanges = list(await check.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery",
        )))
        observed = {"calls": len(calls), "trigger_status": trigger.status,
                    "decisions": [(row.validator_status, row.parsed_decision.get("kind")) for row in exchanges]}
        assert observed == {"calls": 1, "trigger_status": "completed", "decisions": [("accepted", "capability_gap")]}


@pytest.mark.asyncio
async def test_spr230_recovery_pause_wins_after_runnable_preflight(db_session, test_engine, conversation_goal_run, test_agent, monkeypatch):
    """SPR230/SPR231 final provider admission follows awaited lifecycle preparation."""
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    goal, _ = conversation_goal_run
    async def completion(**request):
        calls.append(request)
        return {"choices": [{"message": {"content": json.dumps(gap_payload(test_agent))}}]}
    factory, _, _, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion)
    original = OrchestrationBlockerRecoveryService._require_diagnosis_runnable
    checks = 0
    async def gated(service, db, goal_id, run_id):
        nonlocal checks
        await original(service, db, goal_id, run_id)
        frame = inspect.currentframe().f_back
        physical_preflight = frame.f_code.co_name == "before_completion"
        del frame
        if physical_preflight:
            checks += 1
            if checks == 2:
                entered.set()
                await release.wait()
    monkeypatch.setattr(OrchestrationBlockerRecoveryService, "_require_diagnosis_runnable", gated)
    pending = asyncio.create_task(tick())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert calls == []
        async with factory() as pause:
            await OrchestrationService().pause_goal(pause, goal.project_id, goal.id)
        release.set()
        await asyncio.wait_for(pending, 5)
        assert calls == [], "recovery provider started after Pause won the final preflight race"
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_spr230_recovery_admission_is_recorded_before_blocked_provider(db_session, test_engine, conversation_goal_run, test_agent, monkeypatch):
    """SPR230/SPR231 recovery must share durable attempt admission with other control models."""
    entered, release = asyncio.Event(), asyncio.Event()
    goal, _ = conversation_goal_run
    usage = {"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18}
    payload = gap_payload(test_agent)
    async def completion(**request):
        entered.set()
        await release.wait()
        return {"choices": [{"message": {"content": json.dumps(payload)}}], "usage": usage}
    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion)
    pending = asyncio.create_task(tick())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        async with factory() as fresh:
            admissions = list(await fresh.scalars(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "admit_control_call")))
            assert len(admissions) == 1, "actual recovery provider has no committed admission token"
        async with factory() as pause:
            await OrchestrationService().pause_goal(pause, goal.project_id, goal.id)
        release.set()
        await asyncio.wait_for(pending, 5)
        async with factory() as fresh:
            admissions = list(await fresh.scalars(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "admit_control_call")))
            assert len(admissions) == 1
            admission = admissions[0]
            assert admission.request["source_action_id"] == str(trigger_id)
            assert admission.dispatch_contract["provider_completed"] is True
            assert "raw_content" not in admission.dispatch_contract
            assert admission.dispatch_contract["usage"] == usage
            audits = list(await fresh.scalars(select(OrchestrationDecision).where(
                OrchestrationDecision.run_id == run_id,
                OrchestrationDecision.decision_type == "recovery")))
            paused_audits = [row for row in audits if row.rejection_reason == "paused_after_admitted_result"]
            assert len(paused_audits) == 1
            audit = paused_audits[0]
            assert audit.input_snapshot["trigger_action_id"] == str(trigger_id)
            assert len(audit.llm_output) == 1 and json.loads(audit.llm_output[0]) == payload
            assert audit.parsed_decision["kind"] == payload["kind"]
            assert (await fresh.get(OrchestrationRun, run_id)).status == "paused"
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_spr230_paused_recovery_audit_is_bounded_redacted_and_token_compact(
    db_session, test_engine, conversation_goal_run, test_agent, monkeypatch,
):
    """SPR230/SPR231 retain native audit safely without a full duplicate token payload."""
    from huddleroom.services.orchestration_recovery_analyzer import RAW_OUTPUT_MAX_CHARS
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
    payload = {**gap_payload(test_agent), "diagnosis": secret + " " + "x" * 20_000}
    entered, release = asyncio.Event(), asyncio.Event()
    goal, _ = conversation_goal_run
    calls = []
    async def completion(**request):
        calls.append(request)
        entered.set()
        await release.wait()
        return {"choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"total_tokens": 18, "raw_text": secret, "nested": {"payload": "x" * 20_000}}}
    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion)
    pending = asyncio.create_task(tick())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        async with factory() as pause:
            await OrchestrationService().pause_goal(pause, goal.project_id, goal.id)
        release.set()
        await asyncio.wait_for(pending, 5)
        assert len(calls) == 1, "structured repair ignored accepted Pause"
        async with factory() as fresh:
            audit = await fresh.scalar(select(OrchestrationDecision).where(
                OrchestrationDecision.run_id == run_id,
                OrchestrationDecision.decision_type == "recovery"))
            assert audit.input_snapshot["trigger_action_id"] == str(trigger_id)
            assert audit.rejection_reason == "paused_after_admitted_result"
            assert len(audit.llm_output) == 1
            assert len(audit.llm_output[0]) <= RAW_OUTPUT_MAX_CHARS
            assert secret not in audit.llm_output[0] and "[REDACTED]" in audit.llm_output[0]
            token = await fresh.scalar(select(OrchestrationAction).where(
                OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "admit_control_call"))
            assert token.dispatch_contract["usage"] == {"total_tokens": 18}
            assert "raw_content" not in token.dispatch_contract
            assert len(json.dumps(token.dispatch_contract)) < 1_000
            assert secret not in json.dumps(token.dispatch_contract)
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_spr230_repeated_pause_does_not_spend_recovery_refusal_budget(
    db_session, test_engine, conversation_goal_run, test_agent, monkeypatch,
):
    """SPR230/SPR231 retained paused output is an admission stop, not model refusal."""
    from huddleroom.services.orchestration_blocker_recovery import MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS
    goal, _ = conversation_goal_run
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def completion(**request):
        calls.append(request)
        entered.set()
        await release.wait()
        return {"choices": [{"message": {"content": json.dumps(gap_payload(test_agent))}}]}
    factory, run_id, trigger_id, tick = await _diagnosis_ticks(
        db_session, test_engine, conversation_goal_run, monkeypatch, completion)
    for _ in range(MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS):
        entered, release = asyncio.Event(), asyncio.Event()
        pending = asyncio.create_task(tick())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            async with factory() as pause:
                await OrchestrationService().pause_goal(pause, goal.project_id, goal.id)
            release.set()
            await asyncio.wait_for(pending, 5)
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
        async with factory() as resume:
            await OrchestrationService().resume_goal(resume, goal.project_id, goal.id)
            # Resume keeps the baseline paused; explicit re-authorization admits the next provider call.
            await OrchestrationService().authorize_baseline(resume, goal.project_id, goal.id, actor="test")
            await resume.commit()
    await tick()
    assert len(calls) == MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS + 1
    async with factory() as fresh:
        trigger = await fresh.get(OrchestrationAction, trigger_id)
        assert trigger.status == "completed", trigger.error
        audits = list(await fresh.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run_id, OrchestrationDecision.decision_type == "recovery")))
        assert sum(row.rejection_reason == "paused_after_admitted_result" for row in audits) == MAX_RECOVERY_ATTEMPTS_WITHOUT_PROGRESS
        assert sum(row.validator_status == "accepted" for row in audits) == 1
