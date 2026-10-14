"""typed recovery projection on decision routes and the recovery approval answer branch.

Fake completions only; the HTTP client shares ``db_session`` (no rollback on error), so tests that
prove durability commit their setup and read the outcome from a fresh session.
"""

import json
import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm.attributes import flag_modified

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGate, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
from huddleroom.models.task import Task
from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
from huddleroom.services.orchestration_service import OrchestrationService

# Reuse the recovery (fixtures are imported by name so pytest registers them in this module).
from tests.test_orchestration_recovery_authority import (  # noqa: F401  pylint: disable=unused-import
    actions_of, diagnosed, fail_on_effect, goal_run, proposal_payload, service_for,
)
from tests.test_orchestration_recovery_persistence import gap_payload
from tests.test_orchestration_recovery_persistence import service_for as gap_service_for

SECRET = "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWX"
SECRET_PAIR = "password=hunter2hunter2"
RESPONSE_KEYS = {
    "kind", "proposal_id", "version", "status", "problem", "general_updates", "after_approval",
    "global_effect", "exact_details", "evidence", "error", "source_blocker_id", "source_action_id",
    "source_decision_id",
}


def base(goal):
    return f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/decisions"


async def committed(db):
    await db.commit()


async def fresh(test_engine):
    return async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)()


async def get_decision(client, headers, goal, decision, path=""):
    response = await client.get(base(goal) + path, headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    items = body["items"] if isinstance(body, dict) else body
    return next(item for item in items if item["id"] == str(decision.id)), response


async def answer(client, headers, goal, decision, **payload):
    return await client.post(f"{base(goal)}/{decision.id}/answer", json=payload, headers=headers)


async def wrapper(db, run, proposal, action_type, key_tail, status):
    row = await OrchestrationService().reserve_action(
        db, run.id, f"recovery:{proposal.request['proposal_id']}:{key_tail}", action_type,
        {"proposal_action_id": str(proposal.id)},
    )
    row.status = status
    await db.flush()
    return row


# ---- projection ----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pending_proposal_is_typed_plain_and_exposes_no_executable_request(
    client, db_session, auth_headers, goal_run, test_agent,
):
    goal, run = goal_run
    _, trigger, proposal, decision = await diagnosed(db_session, goal, run, test_agent)

    item, _ = await get_decision(client, auth_headers, goal, decision)
    recovery = item["recovery"]

    assert set(recovery) == RESPONSE_KEYS
    assert recovery["kind"] == "recovery_proposal_v1" and recovery["status"] == "pending_approval"
    assert recovery["proposal_id"] == proposal.request["proposal_id"]
    assert recovery["problem"] == "Planning cannot start."
    assert recovery["general_updates"] == proposal.request["general_updates"]
    assert all(isinstance(text, str) and text.strip() for text in recovery["general_updates"])
    assert "retries the blocked work once" in recovery["after_approval"]
    assert recovery["global_effect"] is False and recovery["error"] is None
    assert recovery["source_blocker_id"] == proposal.request["source"]["id"]
    assert [option["key"] for option in item["options"]] == ["approve_fix", "reject"]
    # Exact details are a read-only disclosure: no executable request or analyzer prompt/diagnosis.
    for forbidden in ("calls_request", "request", "prompt", "system_prompt", "messages"):
        assert forbidden not in recovery
    assert "dispatch_contract" not in item and "dispatch_contract" not in json.dumps(recovery)


@pytest.mark.asyncio
async def test_checkpoint_route_projects_recovery_too(client, db_session, auth_headers, goal_run, test_agent):
    goal, run = goal_run
    _, _, _, decision = await diagnosed(db_session, goal, run, test_agent)
    item, _ = await get_decision(client, auth_headers, goal, decision, "/checkpoint")
    assert item["recovery"]["status"] == "pending_approval"


@pytest.mark.asyncio
async def test_spr210_projection_preserves_safe_source_and_contract(
    client, db_session, auth_headers, goal_run, test_agent, monkeypatch,
):
    goal, run = goal_run
    _, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    expected_source_id = proposal.request["source"]["id"]
    evidence_task = Task(
        project_id=goal.project_id, title="Validated recovery evidence task", status="blocked", metadata_={},
    )
    evidence_gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="validated-evidence", gate_type="work_completed",
        required_evidence={},
    )
    evidence_process = OrchestrationProcessRun(
        goal_id=goal.id, run_id=run.id, process_type="authority_interview", status="waiting_decision",
        trigger_reason="validated recovery evidence", input_snapshot={}, outputs={},
    )
    evidence_action = OrchestrationAction(
        run_id=run.id, idempotency_key=f"spr210:evidence:{uuid.uuid4()}", action_type="request_plan",
        request={},
    )
    raw_agent = Agent(
        name=f"RAW RECOVERY AGENT MUST STAY HIDDEN {uuid.uuid4()}", role="manager",
        provider="openai", model="gpt-4o-mini", adapter_type="api",
        capabilities=["coordination"], config={}, is_active=True,
    )
    db_session.add_all([raw_agent, evidence_task, evidence_gate, evidence_process, evidence_action])
    await db_session.flush()
    proposal.request = {
        **proposal.request,
        "source": {"kind": "agent", "id": str(raw_agent.id), "agent_id": str(raw_agent.id)},
    }
    proposal.dispatch_contract = {
        **(proposal.dispatch_contract or {}),
        "tool_receipt_v1": {"target_refs": [{"kind": "agent", "id": str(raw_agent.id)}]},
    }
    flag_modified(proposal, "request")
    flag_modified(proposal, "dispatch_contract")
    unbound_receipt = await wrapper(
        db_session, run, proposal, "recovery_apply", "unbound-reference", "reserved",
    )
    unbound_receipt.dispatch_contract = {
        "tool_receipt_v1": {"target_refs": [{"kind": "agent", "id": str(raw_agent.id)}]},
    }
    flag_modified(unbound_receipt, "dispatch_contract")
    recovery_context = json.loads(decision.context)
    recovery_context["problem"] = f"Planning cannot start. {SECRET}"
    recovery_context["evidence"] = [
        {"label": kind, "value": {}, "source_kind": kind, "source_id": str(record_id)}
        for kind, record_id in (
            ("action", evidence_action.id), ("task", evidence_task.id), ("gate", evidence_gate.id),
            ("process", evidence_process.id), ("agent", test_agent.id),
        )
    ]
    decision.context = json.dumps(recovery_context)
    await db_session.flush()
    provider_call = AsyncMock(side_effect=AssertionError("read-only projection must not call a provider"))
    monkeypatch.setattr("litellm.acompletion", provider_call)
    source_snapshot = {
        "id": str(decision.id), "decision_key": decision.decision_key, "title": decision.title,
        "question": decision.question, "options": decision.options, "recommendation": decision.recommendation,
        "consequences": decision.consequences, "related_action_id": str(decision.related_action_id),
    }
    session_state = (set(db_session.new), set(db_session.dirty), set(db_session.deleted))

    item, _ = await get_decision(client, auth_headers, goal, decision)

    assert item["context"] is None
    assert {key: item[key] for key in (
        "id", "decision_key", "title", "question", "options", "recommendation", "consequences",
        "related_action_id",
    )} == source_snapshot
    assert item["recovery"]["proposal_id"] == proposal.request["proposal_id"]
    assert item["recovery"]["source_blocker_id"] == expected_source_id
    assert SECRET not in json.dumps(item)
    provider_call.assert_not_awaited()
    assert (set(db_session.new), set(db_session.dirty), set(db_session.deleted)) == session_state
    assert "needs_you_references" in item, "safe recovery projection must append typed source references"
    references = {(reference["kind"], reference["id"]): reference for reference in item["needs_you_references"]}
    assert references[("action", str(evidence_action.id))]["label"] == evidence_action.action_type
    assert references[("task", str(evidence_task.id))]["label"] == evidence_task.title
    assert references[("gate", str(evidence_gate.id))]["label"] == evidence_gate.success_criterion_key
    assert references[("process", str(evidence_process.id))]["label"] == evidence_process.process_type
    assert references[("agent", str(test_agent.id))]["label"] == test_agent.name
    assert ("agent", str(raw_agent.id)) not in references
    assert raw_agent.name not in json.dumps(item)


@pytest.mark.asyncio
async def test_decision_api_projects_recovery_from_linked_actions(
    client, db_session, auth_headers, goal_run, test_agent, test_user, fail_on_effect,
):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)

    async def status():
        item, _ = await get_decision(client, auth_headers, goal, decision)
        return item["recovery"]["status"]

    assert await status() == "applying"
    (apply_row,) = await actions_of(db_session, run, "recovery_apply")
    apply_row.status = "completed"
    verify = await wrapper(db_session, run, proposal, "recovery_verify", "verify:check-1", "reserved")
    assert await status() == "verifying"
    verify.status = "completed"
    await db_session.flush()
    assert await status() == "retrying"
    retry = await wrapper(db_session, run, proposal, "recovery_retry", "retry:1", "reserved")
    assert await status() == "retrying"  # never resolved before the proposal itself completed
    retry.status = "completed"
    await db_session.flush()
    assert await status() == "retrying"
    proposal.status = "completed"
    await db_session.flush()
    assert await status() == "resolved"


@pytest.mark.asyncio
async def test_failed_attempt_projects_failed_with_safe_error(client, db_session, auth_headers, goal_run, test_agent, test_user):
    goal, run = goal_run
    svc, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    await svc.reserve_answer(db_session, decision, selected_option="approve_fix", reason=None, user_id=test_user.id)
    (apply_row,) = await actions_of(db_session, run, "recovery_apply")
    apply_row.status = "failed"
    apply_row.error = f"tool exploded with {SECRET} and {SECRET_PAIR}"
    proposal.status = "failed"
    proposal.error = apply_row.error
    await db_session.flush()

    item, response = await get_decision(client, auth_headers, goal, decision)
    assert item["recovery"]["status"] == "failed"
    assert SECRET not in response.text and "hunter2hunter2" not in response.text


@pytest.mark.asyncio
async def test_secret_shaped_values_never_appear_in_response_body(
    client, db_session, auth_headers, goal_run, test_agent,
):
    goal, run = goal_run
    _, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    # Defence in depth: even if unredacted text reached storage, no route may return it.
    context = json.loads(decision.context)
    context["exact_details"] = {"calls": [{"arguments": {"note": SECRET}}], "extra": SECRET_PAIR}
    context["evidence"][0]["value"] = {"leak": SECRET, "also": SECRET_PAIR}
    context["error"] = SECRET
    decision.context = json.dumps(context)
    request = json.loads(json.dumps(proposal.request))
    request["evidence"][0]["value"] = {"leak": SECRET}
    request["calls"][0]["arguments"]["note"] = SECRET
    proposal.request = request
    flag_modified(proposal, "request")
    await db_session.flush()

    for path in ("", "/checkpoint"):
        response = await client.get(base(goal) + path, headers=auth_headers)
        assert response.status_code == 200
        assert SECRET not in response.text and "hunter2hunter2" not in response.text, path


@pytest.mark.asyncio
async def test_goal_detail_pending_direction_never_exposes_raw_recovery_context(
    client, db_session, auth_headers, goal_run, test_agent,
):
    goal, run = goal_run
    _, _, _, decision = await diagnosed(db_session, goal, run, test_agent)
    context = json.loads(decision.context)
    context["exact_details"] = {"note": SECRET, "extra": SECRET_PAIR}
    decision.context = json.dumps(context)
    await db_session.flush()

    response = await client.get(f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}", headers=auth_headers)
    assert response.status_code == 200, response.text
    assert SECRET not in response.text and "hunter2hunter2" not in response.text
    supervision = response.json()["supervision"]
    assert supervision["condition"] == "needs_you"
    direction = supervision["pending_direction"]
    assert direction["id"] == str(decision.id)
    assert direction["context"] is None
    assert direction["recovery"]["status"] == "pending_approval"


# ---- answering -----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_reserves_first_wrapper_and_runs_no_effects(
    client, db_session, auth_headers, goal_run, test_agent, fail_on_effect,
):
    goal, run = goal_run
    _, _, _, decision = await diagnosed(db_session, goal, run, test_agent)

    first = await answer(client, auth_headers, goal, decision, selected_option="approve_fix")
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["decision"]["status"] == "answered" and body["decision"]["selected_option"] == "approve_fix"
    assert body["decision"]["recovery"]["status"] == "applying"  # not resolved before retry progress
    assert body["process"] is None
    assert len(await actions_of(db_session, run, "recovery_apply")) == 1

    # One POST disables duplicate application by identity: replay is idempotent, a flip conflicts.
    again = await answer(client, auth_headers, goal, decision, selected_option="approve_fix")
    assert again.status_code == 200
    assert len(await actions_of(db_session, run, "recovery_apply")) == 1
    conflicting = await answer(client, auth_headers, goal, decision, selected_option="reject", reason="changed my mind")
    assert conflicting.status_code == 409


@pytest.mark.asyncio
async def test_reject_fix_requires_reason_and_preserves_history(
    client, db_session, auth_headers, goal_run, test_agent, fail_on_effect,
):
    goal, run = goal_run
    _, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)

    missing_reason = await answer(client, auth_headers, goal, decision, selected_option="reject")
    blank_reason = await answer(client, auth_headers, goal, decision, selected_option="reject", reason="  ")
    assert missing_reason.status_code == 400 and blank_reason.status_code == 400
    await db_session.refresh(decision)
    assert decision.status == "pending"

    rejected = (await answer(client, auth_headers, goal, decision, selected_option="reject", reason="Too risky")).json()
    assert rejected["decision"]["recovery"]["status"] == "rejected"
    assert rejected["decision"]["reason"] == "Too risky"
    await db_session.refresh(proposal)
    assert proposal.status == "failed" and proposal.error == "rejected_by_human"
    assert await actions_of(db_session, run, "recovery_apply") == []
    listed, _ = await get_decision(client, auth_headers, goal, decision)  # history stays readable
    assert listed["status"] == "answered" and listed["recovery"]["status"] == "rejected"


@pytest.mark.asyncio
@pytest.mark.parametrize("option", ["edit", "capability_provided", "approve", "bogus"])
async def test_recovery_proposal_accepts_only_approve_fix_or_reject(
    client, db_session, auth_headers, goal_run, test_agent, option,
):
    goal, run = goal_run
    _, _, _, decision = await diagnosed(db_session, goal, run, test_agent)
    response = await answer(client, auth_headers, goal, decision, selected_option=option, reason="x")
    assert response.status_code == 400
    await db_session.refresh(decision)
    assert decision.status == "pending"
    assert await actions_of(db_session, run, "recovery_apply") == []


@pytest.mark.asyncio
async def test_generic_cancel_still_refuses_recovery_decisions(client, db_session, auth_headers, goal_run, test_agent):
    goal, run = goal_run
    _, _, _, decision = await diagnosed(db_session, goal, run, test_agent)
    response = await client.post(f"{base(goal)}/{decision.id}/cancel", json={"reason": "no"}, headers=auth_headers)
    assert response.status_code == 400
    await db_session.refresh(decision)
    assert decision.status == "pending"


@pytest.fixture
async def real_db_client(test_engine, test_user):
    """Client whose sessions behave like production get_db: commit on success, roll back on any exception."""
    from httpx import ASGITransport, AsyncClient
    from huddleroom.dependencies import get_current_user
    from huddleroom.database import get_db
    from huddleroom.main import create_app

    app = create_app()
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: test_user
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


async def _tamper_run_inactive(db, proposal, run):
    run.status = "completed"


async def _tamper_proposal_text(db, proposal, run):
    request = json.loads(json.dumps(proposal.request))
    request["problem"] = "A different problem than the human approved."
    proposal.request = request
    flag_modified(proposal, "request")


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", [_tamper_run_inactive, _tamper_proposal_text])
async def test_stale_approval_returns_409_and_commits_stale_writes(
    real_db_client, db_session, goal_run, test_agent, test_engine, fail_on_effect, drift,
):
    goal, run = goal_run
    _, _, proposal, decision = await diagnosed(db_session, goal, run, test_agent)
    await drift(db_session, proposal, run)
    await committed(db_session)
    decision_id, proposal_id, run_id = decision.id, proposal.id, run.id

    response = await answer(real_db_client, {}, goal, decision, selected_option="approve_fix")
    assert response.status_code == 409, response.text

    # The route's session rolled back on the HTTP error: only what it committed first may survive.
    async with await fresh(test_engine) as other:
        stored = await other.get(OrchestrationAuthorityDecision, decision_id)
        assert stored.status != "pending" and stored.selected_option != "approve_fix"
        stored_proposal = await other.get(OrchestrationAction, proposal_id)
        assert stored_proposal.status == "failed" and stored_proposal.error == "state_changed"
        applied = (await other.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "recovery_apply",
        ))).all()
        assert applied == []  # stale/drift 409 applies nothing


# ---- unsupported / capability gap ----------------------------------------------------------------

async def _gap(db, goal, run, agent):
    svc = gap_service_for(gap_payload(agent))
    from tests.test_orchestration_recovery_authority import trigger_for
    trigger = await svc.create_trigger(db, trigger_for(goal, run))
    action = await svc.diagnose(db, trigger.id)
    decision = (await db.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.related_action_id == action.id,
    ))).one()
    return svc, action, decision


@pytest.mark.asyncio
async def test_capability_gap_projects_unsupported_and_is_not_approvable(
    client, db_session, auth_headers, goal_run, test_agent, fail_on_effect,
):
    goal, run = goal_run
    _, _, decision = await _gap(db_session, goal, run, test_agent)

    item, _ = await get_decision(client, auth_headers, goal, decision)
    assert decision.decision_key.startswith("recovery_gap:")
    assert item["recovery"]["status"] == "unsupported"
    assert item["recovery"]["kind"] == "recovery_proposal_v1"
    assert [option["key"] for option in item["options"]] == ["capability_provided"]
    # F2: re-diagnosis follows the owner's "mark done", never an automatic capability detection.
    assert "mark it done" in item["recovery"]["after_approval"]
    assert "automatically" not in item["recovery"]["after_approval"].replace("Nothing happens automatically", "")

    for option in ("approve_fix", "reject"):
        refused = await answer(client, auth_headers, goal, decision, selected_option=option, reason="x")
        assert refused.status_code in (400, 409)
    await db_session.refresh(decision)
    assert decision.status == "pending"
    assert await actions_of(db_session, run, "recovery_apply") == []


@pytest.mark.asyncio
async def test_capability_provided_routes_to_answer_capability_gap(
    client, db_session, auth_headers, goal_run, test_agent, monkeypatch,
):
    goal, run = goal_run
    _, action, decision = await _gap(db_session, goal, run, test_agent)
    rediagnose = AsyncMock(return_value=action)
    monkeypatch.setattr(OrchestrationBlockerRecoveryService, "rediagnose", rediagnose)

    response = await answer(client, auth_headers, goal, decision, selected_option="capability_provided")

    assert response.status_code == 200, response.text
    assert response.json()["decision"]["status"] == "answered"
    assert response.json()["decision"]["selected_option"] == "capability_provided"
    rediagnose.assert_awaited_once()


@pytest.mark.asyncio
async def test_duplicate_capability_provided_is_idempotent(client, db_session, auth_headers, goal_run, test_agent):
    goal, run = goal_run
    _, _, decision = await _gap(db_session, goal, run, test_agent)
    first = await answer(client, auth_headers, goal, decision, selected_option="capability_provided")
    assert first.status_code == 200, first.text
    triggers = len(await actions_of(db_session, run, "recovery_trigger"))
    again = await answer(client, auth_headers, goal, decision, selected_option="capability_provided")
    assert again.status_code == 200, again.text
    assert len(await actions_of(db_session, run, "recovery_trigger")) == triggers


@pytest.mark.asyncio
async def test_goal_lock_is_released_after_approve(
    client, db_session, auth_headers, goal_run, test_agent, test_engine, fail_on_effect,
):
    import asyncio
    goal, run = goal_run
    _, _, _, decision = await diagnosed(db_session, goal, run, test_agent)
    assert (await answer(client, auth_headers, goal, decision, selected_option="approve_fix")).status_code == 200

    async def acquire():
        async with await fresh(test_engine) as other:
            async with OrchestrationService()._lock_goal_for_baseline_transition(other, goal.id):
                return True

    assert await asyncio.wait_for(asyncio.create_task(acquire()), timeout=5)
