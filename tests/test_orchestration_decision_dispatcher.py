import uuid
from types import SimpleNamespace

import pytest
from datetime import datetime, timezone
from unittest.mock import ANY, AsyncMock
from huddleroom.models.orchestration import OrchestrationGate
from huddleroom.models.task import Task
from huddleroom.services.orchestration_progress_view import OrchestrationProgressView
from tests.test_orchestration_progress_view import _authority_decision
from tests.test_orchestration_runtime_e2e import (
    _agent, _authorized_run, _run_actions, _seed_accepted_plan, _tasks_for_run,
)
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_service import OrchestrationService


@pytest.fixture(autouse=True)
def _fake_db_seams(request, monkeypatch):
    """Fake-db unit tests bypass the savepoint and attempt-key lookup; real-db tests (db_session) do not."""
    if "db_session" in request.fixturenames:
        return
    from contextlib import nullcontext

    async def same_key(self, _db, _run_id, key, *_a):
        return key, None

    monkeypatch.setattr(OrchestrationDecisionDispatcher, "_savepoint", staticmethod(lambda _db: nullcontext()))
    monkeypatch.setattr(OrchestrationDecisionDispatcher, "_attempt_key", same_key)


@pytest.mark.asyncio
async def test_dispatch_routes_request_plan_to_executor():
    service = OrchestrationService()
    service.execute_request_plan_action = AsyncMock()
    service._steering_action_fence = AsyncMock()
    dispatcher = OrchestrationDecisionDispatcher(service)
    run = type("R", (), {"id": uuid.uuid4()})()
    decision = type("D", (), {
        "id": uuid.uuid4(),
        "validator_status": "accepted",
        "parsed_decision": {"action_type": "request_plan", "agent_id": str(uuid.uuid4()),
                            "scope": "plan it", "work_function": "planning"},
    })()

    await dispatcher.dispatch(None, run, decision)

    service.execute_request_plan_action.assert_awaited_once()
    service._steering_action_fence.assert_not_awaited()
    _, kwargs = service.execute_request_plan_action.call_args
    assert kwargs["idempotency_key"].startswith(f"run:{run.id}:kind:request_plan:request:")
    assert kwargs["decision_id"] == decision.id


@pytest.mark.asyncio
async def test_dispatch_fences_versioned_steering_snapshot(monkeypatch):
    service = OrchestrationService()
    service.execute_noop_action = AsyncMock(return_value=SimpleNamespace(status="completed", id=uuid.uuid4()))
    service._steering_action_fence = AsyncMock(return_value=("fenced", None, None, None, None))
    steering = SimpleNamespace(assert_current_versions=AsyncMock(), link_result=AsyncMock())
    monkeypatch.setattr(
        "huddleroom.services.orchestration_decision_dispatcher.OrchestrationSteeringService", lambda: steering,
    )
    db = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace()))
    run = SimpleNamespace(id=uuid.uuid4(), goal_id=uuid.uuid4())
    decision = SimpleNamespace(
        id=uuid.uuid4(), validator_status="accepted", parsed_decision={"action_type": "noop"},
        input_snapshot={"steering": {"versions": {
            "inbox_version": 1, "direction_version": 1,
            "contract_version": "contract", "plan_version": "plan",
        }}},
    )

    await OrchestrationDecisionDispatcher(service).dispatch(db, run, decision)

    service._steering_action_fence.assert_awaited_once_with(db, run.id, ANY, decision.id)


@pytest.mark.asyncio
async def test_dispatch_routes_roadmap_replan_with_version_stable_key():
    service = OrchestrationService()
    service.execute_request_roadmap_replan_action = AsyncMock()
    service.roadmap_replan_action_key = AsyncMock()
    dispatcher = OrchestrationDecisionDispatcher(service)
    run = type("R", (), {"id": uuid.uuid4(), "plan_state": {"roadmap_version": 2}})()
    decision = type("D", (), {
        "id": uuid.uuid4(), "validator_status": "accepted",
        "parsed_decision": {"action_type": "request_roadmap_replan", "agent_id": str(uuid.uuid4()),
                            "scope": "Revise remaining work.", "reason": "New information."},
    })()
    service.roadmap_replan_action_key.return_value = f"run:{run.id}:kind:request_roadmap_replan:version:2"

    await dispatcher.dispatch(None, run, decision)

    _, kwargs = service.execute_request_roadmap_replan_action.call_args
    assert kwargs["idempotency_key"] == f"run:{run.id}:kind:request_roadmap_replan:version:2"


@pytest.mark.asyncio
async def test_dispatch_skips_rejected_but_persists_noop_wait():
    service = OrchestrationService()
    service.execute_request_plan_action = AsyncMock()
    service.execute_noop_action = AsyncMock()
    dispatcher = OrchestrationDecisionDispatcher(service)
    run = type("R", (), {"id": uuid.uuid4()})()
    rejected = type("D", (), {"id": uuid.uuid4(), "validator_status": "rejected",
                              "parsed_decision": {"action_type": "request_plan"}})()
    noop = type("D", (), {"id": uuid.uuid4(), "validator_status": "accepted",
                          "parsed_decision": {"action_type": "noop"}})()
    assert await dispatcher.dispatch(None, run, rejected) is None
    service.execute_request_plan_action.assert_not_awaited()
    await dispatcher.dispatch(None, run, noop)
    service.execute_noop_action.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_deduplicates_reason_only_changes_but_not_request_changes():
    service = OrchestrationService()
    service.execute_ask_human_action = AsyncMock()
    dispatcher = OrchestrationDecisionDispatcher(service)
    run = type("R", (), {"id": uuid.uuid4()})()

    def decision(question: str, reason: str):
        return type("D", (), {
            "id": uuid.uuid4(),
            "validator_status": "accepted",
            "parsed_decision": {
                "action_type": "ask_human", "question": question, "reason": reason,
            },
        })()

    await dispatcher.dispatch(None, run, decision("Choose a scope.", "First explanation."))
    await dispatcher.dispatch(None, run, decision("Choose a scope.", "Different explanation."))
    await dispatcher.dispatch(None, run, decision("Choose a budget.", "Different request."))

    keys = [call.kwargs["idempotency_key"] for call in service.execute_ask_human_action.await_args_list]
    assert keys[0] == keys[1]
    assert keys[2] != keys[0]


@pytest.mark.asyncio
async def test_dispatch_uses_canonical_key_for_equivalent_delegation_requests():
    """Whitespace-only regeneration must not create a second work request."""
    service = OrchestrationService()
    service.execute_create_delegation_task_action = AsyncMock()
    dispatcher = OrchestrationDecisionDispatcher(service)
    run = type("R", (), {"id": uuid.uuid4()})()

    def decision(scope: str, deliverable: str, reason: str):
        return type("D", (), {
            "id": uuid.uuid4(),
            "validator_status": "accepted",
            "parsed_decision": {
                "action_type": "create_delegation_task",
                "agent_id": str(uuid.uuid4()),
                "work_function": "implementation",
                "scope": scope,
                "deliverable": deliverable,
                "reason": reason,
            },
        })()

    agent_id = str(uuid.uuid4())
    first = decision("Implement the change.", "A tested patch.", "First explanation.")
    second = decision("  Implement the change.  ", " A tested patch. ", "Regenerated explanation.")
    first.parsed_decision["agent_id"] = agent_id
    second.parsed_decision["agent_id"] = agent_id
    first.parsed_decision["inputs"] = ["  prior context  "]
    second.parsed_decision["inputs"] = ["prior context"]

    await dispatcher.dispatch(None, run, first)
    await dispatcher.dispatch(None, run, second)

    keys = [
        call.kwargs["idempotency_key"]
        for call in service.execute_create_delegation_task_action.await_args_list
    ]
    assert keys == [keys[0], keys[0]]


@pytest.mark.asyncio
async def test_dispatch_warning_defaults_run_id_before_hashing():
    service = OrchestrationService()
    service.execute_record_warning_action = AsyncMock()
    dispatcher = OrchestrationDecisionDispatcher(service)
    run = type("R", (), {"id": uuid.uuid4()})()

    for extra in ({}, {"run_id": None}, {"run_id": ""}, {"run_id": run.id}):
        request = {
            "action_type": "record_warning",
            "warning_type": "risk",
            "severity": "warning",
            "message": "Check it.",
            **extra,
        }
        decision = type("D", (), {
            "id": uuid.uuid4(), "validator_status": "accepted", "parsed_decision": request,
        })()
        await dispatcher.dispatch(None, run, decision)

    calls = service.execute_record_warning_action.await_args_list
    assert [call.kwargs["idempotency_key"] for call in calls] == [calls[0].kwargs["idempotency_key"]] * 4
    assert [call.kwargs["request"]["run_id"] for call in calls] == [str(run.id)] * 4


@pytest.mark.asyncio
async def test_dispatch_keeps_nested_report_schema_whitespace_semantic():
    service = OrchestrationService()
    service.execute_create_delegation_task_action = AsyncMock()
    dispatcher = OrchestrationDecisionDispatcher(service)
    run = type("R", (), {"id": uuid.uuid4()})()
    request = {
        "action_type": "create_delegation_task",
        "agent_id": str(uuid.uuid4()),
        "work_function": "implementation",
        "scope": "Implement it.",
        "deliverable": "A patch.",
    }
    for report_schema in ({"label": " A "}, {"label": "A"}):
        decision = type("D", (), {
            "id": uuid.uuid4(),
            "validator_status": "accepted",
            "parsed_decision": {**request, "report_schema": report_schema},
        })()
        await dispatcher.dispatch(None, run, decision)

    keys = [call.kwargs["idempotency_key"] for call in service.execute_create_delegation_task_action.await_args_list]
    assert keys[0] != keys[1]


@pytest.mark.parametrize(
    ("action_type", "omitted", "explicit"),
    [
        (
            "create_delegation_task",
            {
                "agent_id": uuid.uuid4(), "work_function": "implementation",
                "scope": "Implement it.", "deliverable": "A patch.",
            },
            lambda request: {
                **request,
                "agent_id": str(request["agent_id"]),
                "inputs": [], "forbidden_work": [], "success_evidence": [],
                "budget": {}, "report_schema": {}, "parent_task_id": "",
                "source_session_id": "",
            },
        ),
        (
            "ask_human",
            {"question": "Choose."},
            lambda request: {
                **request, "work_function": None, "required_capabilities": [],
                "candidate_agent_ids": [], "gate_id": None,
            },
        ),
        (
            "record_warning",
            {"warning_type": "risk", "severity": "warning", "message": "Check it."},
            lambda request: {
                **request, "run_id": None, "source_process_run_id": None,
                "related_gate_id": None, "related_action_id": None, "related_agent_id": None,
            },
        ),
    ],
)
def test_canonical_decision_request_normalizes_replay_equivalents(action_type, omitted, explicit):
    service = OrchestrationService()
    omitted_request = {"action_type": action_type, **omitted}
    explicit_request = {"action_type": action_type, **explicit(omitted)}

    run_id = uuid.uuid4()
    canonical_omitted = service.canonical_decision_request(action_type, omitted_request, run_id=run_id)
    canonical_explicit = service.canonical_decision_request(action_type, explicit_request, run_id=run_id)

    assert canonical_omitted == canonical_explicit
    assert OrchestrationDecisionDispatcher.action_key(run_id, action_type, canonical_omitted) == (
        OrchestrationDecisionDispatcher.action_key(run_id, action_type, canonical_explicit)
    )


def test_canonical_decision_request_changes_key_when_semantics_change():
    service = OrchestrationService()
    run_id = uuid.uuid4()
    first = service.canonical_decision_request("ask_human", {
        "action_type": "ask_human", "question": "Choose scope.",
    })
    second = service.canonical_decision_request("ask_human", {
        "action_type": "ask_human", "question": "Choose budget.",
    })

    assert OrchestrationDecisionDispatcher.action_key(run_id, "ask_human", first) != (
        OrchestrationDecisionDispatcher.action_key(run_id, "ask_human", second)
    )


def test_canonical_warning_treats_uuid_objects_and_strings_as_one_request():
    """Changing UUID representation alone must not create another warning."""
    service = OrchestrationService()
    run_id = uuid.uuid4()
    related_gate_id = uuid.uuid4()
    base = {
        "action_type": "record_warning",
        "warning_type": "risk",
        "severity": "warning",
        "message": "Check it.",
    }

    object_request = service.canonical_decision_request(
        "record_warning", {**base, "related_gate_id": related_gate_id}, run_id=run_id
    )
    string_request = service.canonical_decision_request(
        "record_warning", {**base, "related_gate_id": str(related_gate_id)}, run_id=run_id
    )

    assert object_request == string_request
    assert OrchestrationDecisionDispatcher.action_key(run_id, "record_warning", object_request) == (
        OrchestrationDecisionDispatcher.action_key(run_id, "record_warning", string_request)
    )


def test_canonical_meeting_preserves_participant_order_as_semantic():
    """A different participant order remains a distinct coordination request."""
    service = OrchestrationService()
    run_id = uuid.uuid4()
    first_agent_id, second_agent_id = uuid.uuid4(), uuid.uuid4()
    base = {
        "action_type": "schedule_meeting",
        "topic": "Resolve the blocker.",
        "organizer_agent_id": str(first_agent_id),
    }

    first = service.canonical_decision_request(
        "schedule_meeting",
        {**base, "participant_agent_ids": [str(first_agent_id), str(second_agent_id)]},
    )
    second = service.canonical_decision_request(
        "schedule_meeting",
        {**base, "participant_agent_ids": [str(second_agent_id), str(first_agent_id)]},
    )

    assert OrchestrationDecisionDispatcher.action_key(run_id, "schedule_meeting", first) != (
        OrchestrationDecisionDispatcher.action_key(run_id, "schedule_meeting", second)
    )


def _plan_decision(applies_decision_id=None):
    parsed = {"action_type": "request_plan", "agent_id": str(uuid.uuid4()),
              "scope": "plan it", "work_function": "planning"}
    if applies_decision_id is not None:
        parsed["applies_decision_id"] = str(applies_decision_id)
    return SimpleNamespace(id=uuid.uuid4(), validator_status="accepted", parsed_decision=parsed,
                           input_snapshot={}, rejection_reason=None)


def _authority_db(decision_row):
    return SimpleNamespace(get=AsyncMock(return_value=decision_row))


def _answered(run_id, status="answered"):
    return SimpleNamespace(id=uuid.uuid4(), run_id=run_id, status=status)


def _service_returning(action_type, status, dispatch_contract=None):
    service = OrchestrationService()
    service._steering_action_fence = AsyncMock()
    action = SimpleNamespace(id=uuid.uuid4(), action_type=action_type, status=status,
                             dispatch_contract=dispatch_contract)
    service.execute_request_plan_action = AsyncMock(return_value=action)
    service.execute_noop_action = AsyncMock(return_value=action)
    return service, action


@pytest.mark.asyncio
async def test_completed_action_is_linked_to_applied_decision():
    run = SimpleNamespace(id=uuid.uuid4())
    answered = _answered(run.id)
    service, action = _service_returning("request_plan", "completed", {"owner": "planner"})
    decision = _plan_decision(answered.id)

    await OrchestrationDecisionDispatcher(service).dispatch(_authority_db(answered), run, decision)

    assert action.dispatch_contract == {"owner": "planner", "applies_decision_id": str(answered.id)}
    assert decision.validator_status == "accepted"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action_type", "status"),
    [("noop", "completed"), ("record_warning", "completed"), ("pause_run", "completed"),
     ("ask_human", "completed"), ("suggest_agent", "completed"), ("request_plan", "failed")],
)
async def test_noop_warning_and_failed_actions_are_not_linked(action_type, status):
    run = SimpleNamespace(id=uuid.uuid4())
    answered = _answered(run.id)
    service, action = _service_returning(action_type, status, {"owner": "x"})

    await OrchestrationDecisionDispatcher(service).dispatch(
        _authority_db(answered), run, _plan_decision(answered.id),
    )

    assert action.dispatch_contract == {"owner": "x"}


@pytest.mark.asyncio
async def test_decision_from_other_run_is_rejected_before_executor():
    run = SimpleNamespace(id=uuid.uuid4())
    other = _answered(uuid.uuid4())
    service, _ = _service_returning("request_plan", "completed")
    decision = _plan_decision(other.id)

    result = await OrchestrationDecisionDispatcher(service).dispatch(_authority_db(other), run, decision)

    assert result is None
    assert decision.validator_status == "rejected"
    assert decision.rejection_reason
    service.execute_request_plan_action.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["pending", "cancelled", "expired"])
async def test_non_answered_decision_is_rejected_before_executor(status):
    run = SimpleNamespace(id=uuid.uuid4())
    pending = _answered(run.id, status=status)
    service, _ = _service_returning("request_plan", "completed")
    decision = _plan_decision(pending.id)

    result = await OrchestrationDecisionDispatcher(service).dispatch(_authority_db(pending), run, decision)

    assert result is None
    assert decision.validator_status == "rejected"
    assert decision.rejection_reason
    service.execute_request_plan_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_decision_row_is_rejected_before_executor():
    run = SimpleNamespace(id=uuid.uuid4())
    service, _ = _service_returning("request_plan", "completed")
    decision = _plan_decision(uuid.uuid4())

    assert await OrchestrationDecisionDispatcher(service).dispatch(_authority_db(None), run, decision) is None
    assert decision.validator_status == "rejected"
    service.execute_request_plan_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_replayed_completed_action_is_still_linked():
    run = SimpleNamespace(id=uuid.uuid4())
    answered = _answered(run.id)
    service, action = _service_returning("request_plan", "completed", {"owner": "planner"})
    dispatcher = OrchestrationDecisionDispatcher(service)

    await dispatcher.dispatch(_authority_db(answered), run, _plan_decision(answered.id))
    action.dispatch_contract = {"owner": "planner"}  # executor overwrote it on replay
    await dispatcher.dispatch(_authority_db(answered), run, _plan_decision(answered.id))

    assert action.dispatch_contract["applies_decision_id"] == str(answered.id)


@pytest.mark.asyncio
async def test_replayed_action_keeps_original_applies_decision_id():
    run = SimpleNamespace(id=uuid.uuid4())
    first, second = _answered(run.id), _answered(run.id)
    service, action = _service_returning("request_plan", "completed", {"applies_decision_id": str(first.id)})

    await OrchestrationDecisionDispatcher(service).dispatch(
        _authority_db(second), run, _plan_decision(second.id),
    )

    assert action.dispatch_contract == {"applies_decision_id": str(first.id)}


@pytest.mark.asyncio
async def test_applies_decision_id_does_not_change_action_key():
    run = SimpleNamespace(id=uuid.uuid4())
    answered = _answered(run.id)
    service, _ = _service_returning("request_plan", "completed")
    dispatcher = OrchestrationDecisionDispatcher(service)
    plain = _plan_decision()
    linked = _plan_decision(answered.id)
    linked.parsed_decision = {**plain.parsed_decision, "applies_decision_id": str(answered.id)}

    await dispatcher.dispatch(None, run, plain)
    await dispatcher.dispatch(_authority_db(answered), run, linked)

    plain_key = service.execute_request_plan_action.call_args_list[0].kwargs["idempotency_key"]
    linked_key = service.execute_request_plan_action.call_args_list[1].kwargs["idempotency_key"]
    assert plain_key == linked_key


async def _verifiable_gate(db_session, test_project):
    producer, verifier = _agent("disp-producer", ["implementation"]), _agent("disp-verifier", ["validation"])
    producer.role, verifier.role = "developer", "validator"
    db_session.add_all([producer, verifier])
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    await _seed_accepted_plan(db_session, test_project, service, run, producer, plan_items=[{
        "id": "disp-item", "work_function": "implementation", "scope": "Deliver.",
        "deliverable": "Delivered.", "agent_id": str(producer.id),
    }])
    await service.tick(db_session, run.id)
    work_task = next(t for t in await _tasks_for_run(db_session, run.id)
                     if service._task_work_function(t) == "implementation")
    work_task.status = "done"
    gate = await db_session.get(OrchestrationGate, uuid.UUID(work_task.metadata_["orchestration"]["plan_item_gate_id"]))
    return service, goal, run, gate


def _verify_decision(gate, **extra):
    return SimpleNamespace(
        id=None, validator_status="accepted", input_snapshot={}, rejection_reason=None,
        parsed_decision={"action_type": "request_verification", "gate_id": str(gate.id),
                         "work_function": "validation", **extra},
    )


async def test_failed_verification_is_retryable_and_replays_are_idempotent(
    db_session, test_project, safe_effectiveness_review_continue,
):
    service, _goal, run, gate = await _verifiable_gate(db_session, test_project)
    dispatcher = OrchestrationDecisionDispatcher(service)

    first = await dispatcher.dispatch(db_session, run, _verify_decision(gate))
    assert first.status == "completed"
    assert ":attempt:" not in first.idempotency_key  # attempt 0 keeps the legacy key
    assert (await dispatcher.dispatch(db_session, run, _verify_decision(gate))).id == first.id

    first.status = "failed"
    await db_session.flush()
    second = await dispatcher.dispatch(db_session, run, _verify_decision(gate))
    assert second.id != first.id
    assert second.idempotency_key.endswith(":attempt:1")
    assert second.status == "completed"
    assert second.target_id != first.target_id
    assert await db_session.get(Task, second.target_id) is not None
    # replay of the successful retry: same action, no third verifier task
    assert (await dispatcher.dispatch(db_session, run, _verify_decision(gate))).id == second.id
    verifiers = [a for a in await _run_actions(db_session, run.id)
                 if a.action_type == "create_delegation_task" and ":verify_gate:" in a.idempotency_key]
    assert sorted(a.idempotency_key.rsplit(str(gate.id), 1)[1] for a in verifiers) == ["", ":attempt:1"]


async def test_progress_view_follow_up_cleared_only_by_substantive_continuation(
    db_session, test_project, safe_effectiveness_review_continue,
):
    service, goal, run, gate = await _verifiable_gate(db_session, test_project)
    related = (await _run_actions(db_session, run.id))[0]
    answered = await _authority_decision(
        db_session, goal, run, related, decided_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    async def follow_ups():
        view = await OrchestrationProgressView().build(db_session, goal, run)
        return [f["id"] for f in view.untracked_follow_ups if f["kind"] == "answered_decision"]

    assert await follow_ups() == [str(answered.id)]
    dispatcher = OrchestrationDecisionDispatcher(service)
    noop = SimpleNamespace(id=None, validator_status="accepted", input_snapshot={}, rejection_reason=None,
                           parsed_decision={"action_type": "noop", "applies_decision_id": str(answered.id)})
    await dispatcher.dispatch(db_session, run, noop)
    assert await follow_ups() == [str(answered.id)]

    action = await dispatcher.dispatch(
        db_session, run, _verify_decision(gate, applies_decision_id=str(answered.id)),
    )
    assert action.status == "completed"
    assert await follow_ups() == []


def test_advertised_action_schemas_match_dispatchable_executors():
    from huddleroom.services.orchestration_decision_validator import ALLOWED_ACTION_SCHEMAS

    assert set(ALLOWED_ACTION_SCHEMAS) == set(OrchestrationDecisionDispatcher._EXECUTORS)


def test_every_dispatcher_executor_method_exists_on_orchestration_service():
    for action_type, method_name in OrchestrationDecisionDispatcher._EXECUTORS.items():
        assert callable(getattr(OrchestrationService, method_name, None)), (action_type, method_name)
