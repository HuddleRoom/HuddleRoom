import uuid
from types import SimpleNamespace

import pytest
from unittest.mock import ANY, AsyncMock
from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher
from huddleroom.services.orchestration_service import OrchestrationService


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
