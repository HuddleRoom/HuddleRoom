"""An unchanged failed action cannot spend an unbounded recovery evidence loop."""

import copy
import json

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning
from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
from huddleroom.services.orchestration_recovery_analyzer import RecoveryAnalyzer
from huddleroom.services.orchestration_service import OrchestrationService
from tests.test_orchestration_recovery_evidence import _seed, _trigger


@pytest.mark.asyncio
async def test_public_ticks_bound_unchanged_failed_action_queries_without_manufacturing_human_gap(
    db_session, test_engine, test_project, test_agent, monkeypatch,
):
    """Removing evidence-budget feedback or analyzer-failure closure must fail this scenario."""
    goal, run, source, task, _gate, process, _wait, _warning = await _seed(db_session, test_project, test_agent)
    trigger = await OrchestrationBlockerRecoveryService().create_trigger(db_session, _trigger(goal, run, source.id))
    immutable_source = copy.deepcopy(source.request), source.status, source.error
    queries = [
        {"kind": "action", "action_id": str(source.id)},
        {"kind": "task", "task_id": str(task.id)},
        {"kind": "process", "process_type": process.process_type},
        {"kind": "waits"},
        {"kind": "run_state", "run_id": str(run.id)},
    ]
    requests = []

    async def completion(**request):
        index = len(requests) % len(queries)
        requests.append(copy.deepcopy(request))
        payload = {"kind": "evidence_query", "query": queries[index]}
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    real_service = OrchestrationBlockerRecoveryService
    monkeypatch.setattr(
        "huddleroom.services.orchestration_blocker_recovery.OrchestrationBlockerRecoveryService",
        lambda: real_service(analyzer=RecoveryAnalyzer(completion_fn=completion)),
    )

    async def baseline(_service, _db, _run_id, *, local_only=False, owns_transaction=False):
        assert owns_transaction is True  # SPR230: public tick owns its fresh-session operation.
        return {"status": "probe_baseline_stub"}

    monkeypatch.setattr(OrchestrationService, "_tick_baseline_impl", baseline)
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    await db_session.commit()

    async def tick():
        async with factory() as caller:
            return await OrchestrationService().tick(caller, run.id)

    for attempt in range(3):
        await tick()
        assert len(requests) == 5 * (attempt + 1)

    # Exhaustion of analyzer output produces a concrete failed recovery trigger.
    # It does not assert that a human capability is missing or authorize any fix.
    await tick()
    await tick()
    assert len(requests) == 15
    async with factory() as check:
        result = await check.get(OrchestrationAction, trigger.id)
        original = await check.get(OrchestrationAction, source.id)
        exchanges = list(await check.scalars(select(OrchestrationDecision).where(
            OrchestrationDecision.run_id == run.id, OrchestrationDecision.decision_type == "recovery",
        )))
        assert result.status == "failed", {
            "trigger_status": result.status, "error": result.error,
            "assessments": [row.parsed_decision.get("kind") for row in exchanges],
        }
        assert result.error.startswith("analysis_evidence_limit")
        assert not result.dispatch_contract.get("last_diagnosis_error_at")
        assert len(exchanges) == 3
        assert all(row.validator_status == "rejected" for row in exchanges)
        recorded = [row.input_snapshot["evidence_queries"] for row in exchanges]
        assert all(len(items) == 4 for items in recorded)
        assert all([item["query"] for item in items] == queries[:4] for items in recorded)
        assert [item["result"] for item in recorded[0]] == [item["result"] for item in recorded[1]]
        assert [item["result"] for item in recorded[1]] == [item["result"] for item in recorded[2]]
        assert (original.request, original.status, original.error) == immutable_source
        approvals = list(await check.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == goal.id,
            OrchestrationAuthorityDecision.decision_key.startswith("recovery_gap:"),
        )))
        assert approvals == []
        effects = list(await check.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id,
            OrchestrationAction.action_type.in_(("recovery_proposal", "recovery_apply", "recovery_retry", "recovery_verify")),
        )))
        assert effects == []
        warnings = list(await check.scalars(select(OrchestrationWarning).where(
            OrchestrationWarning.goal_id == goal.id,
            OrchestrationWarning.related_action_id == trigger.id,
        )))
        assert len(warnings) == 1
        assert "recovery analysis failed" in warnings[0].message
        for attempt in range(3):
            first_input = json.loads(requests[attempt * 5]["messages"][1]["content"])
            if attempt:
                prior = first_input["snapshot"]["prior_attempt"]
                assert prior["rejection"] == "analysis_evidence_limit"
                assert len(prior["evidence"]["items"]) == 4
            else:
                assert "prior_attempt" not in first_input["snapshot"]
            final_input = json.loads(requests[attempt * 5 + 4]["messages"][1]["content"])
            assert final_input["queries_remaining"] == 0
            assert len(final_input["evidence"]) == 4
            assert "do NOT return an evidence_query" in requests[attempt * 5 + 4]["messages"][0]["content"]
        print("UNCHANGED_RECOVERY_EVIDENCE: 15 provider completions, 12 scoped evidence reads, "
              "3 refusals, failed trigger, no human gap or effect, unchanged source")
