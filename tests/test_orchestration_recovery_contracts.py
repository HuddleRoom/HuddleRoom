"""Strict contracts for bounded, approval-gated orchestration recovery."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from huddleroom.schemas.orchestration_recovery import (
    BaselineRepairAdmissionV1,
    CapabilityGapAssessmentV1,
    ChangeV1,
    CorrectiveDisposition,
    EvidenceV1,
    ProposalDraftV1,
    RecoveryEvidenceQueryV1,
    RecoveryCallV1,
    RecoveryDispatchContext,
    RecoveryProposalV1,
    RecoveryProposalResponse,
    RecoveryTriggerV1,
    RetryTargetV1,
    TeamConfigurationApplyV1,
    ToolManifestV1,
    ToolReceiptV1,
    TransientAssessmentV1,
    VerificationCheckV1,
    VerificationResultV1,
)
from huddleroom.services.orchestration_recovery_analyzer import (
    MAX_RECOVERY_EVIDENCE_ROUNDS,
    RecoveryAnalyzer,
    parse_recovery_assessment,
)


UUID_A = UUID("11111111-1111-1111-1111-111111111111")
UUID_B = UUID("22222222-2222-2222-2222-222222222222")
SHA256 = "a" * 64


def _wake(*, recheck_after_seconds: int = 1) -> dict:
    return {
        "events": [{
            "event_type": "task.status_changed",
            "matcher": {"task_id": str(UUID_A)},
        }],
        "recheck_after_seconds": recheck_after_seconds,
        "expected_result": "The dependent task reports a terminal state.",
    }


def _evidence() -> dict:
    return {
        "label": "The planning gate is blocked",
        "value": {"status": "blocked"},
        "source_kind": "gate",
        "source_id": str(UUID_A),
    }


def _proposal_draft(**overrides: object) -> dict:
    payload: dict[str, object] = {
        "kind": "proposal",
        "diagnosis": "The active planning configuration cannot select an author.",
        "evidence": [_evidence()],
        "problem": "No eligible planning author is configured.",
        "general_updates": ["Assign an eligible planning author."],
        "expected_effect": "Planning can be retried once after the configuration is verified.",
        "calls": [{
            "tool": "team_configuration.apply",
            "arguments": {"agents": []},
            "rationale": "The accepted roster must include a planning author.",
        }],
    }
    payload.update(overrides)
    return payload


def _call(index: int) -> dict:
    return {
        "call_id": f"call-{index}",
        "tool": "team_configuration.apply",
        "tool_version": "1",
        "arguments": {"agents": []},
        "changes": [{
            "operation": "update",
            "scope": "global",
            "target_type": "agent",
            "target_id": str(UUID_A),
            "before": None,
            "after": {"role": "planner"},
        }],
    }


def _check(index: int) -> dict:
    return {
        "check_id": f"check-{index}",
        "verifier": "planning.ready",
        "verifier_version": "1",
        "arguments": {"goal_id": str(UUID_B)},
    }


def _proposal(**overrides: object) -> dict:
    payload: dict[str, object] = {
        "schema_version": 1,
        "proposal_id": str(UUID_A),
        "lineage_id": str(UUID_B),
        "attempt": 1,
        "source": {
            "kind": "planning_preflight",
            "id": str(UUID_A),
            "blocker_code": "plan_author_unavailable",
            "fingerprint": SHA256,
        },
        "resource_fingerprint": SHA256,
        "registry_version": "recovery-tools-v1",
        "problem": "No eligible planning author is configured.",
        "general_updates": ["Assign an eligible planning author."],
        "expected_effect": "Verify configuration, then retry planning once.",
        "evidence": [_evidence()],
        "calls": [_call(1)],
        "verification": [_check(1)],
        "retry": {"target": {"kind": "planning", "requested_agent_id": str(UUID_A)}, "max_attempts": 1},
        "authority_required": "human",
        "global_effect": True,
    }
    payload.update(overrides)
    return payload


def _error_types(error: ValidationError) -> set[str]:
    return {item["type"] for item in error.errors()}


def _query(payload: dict) -> object:
    """Keep tests compatible with the required discriminated query union."""
    return TypeAdapter(RecoveryEvidenceQueryV1).validate_python(payload)


def test_recovery_contract_rejects_unknown_and_broker_owned_fields():
    with pytest.raises(ValidationError) as caught:
        parse_recovery_assessment({**_proposal_draft(), "proposal_id": str(uuid4())})

    assert "extra_forbidden" in _error_types(caught.value)

    with pytest.raises(ValidationError) as caught:
        parse_recovery_assessment({
            **_proposal_draft(),
            "calls": [{
                **_proposal_draft()["calls"][0],
                "call_id": "model-must-not-choose-this",
                "tool_version": "999",
            }],
        })

    assert "extra_forbidden" in _error_types(caught.value)


@pytest.mark.parametrize("field", ["api_key", "authorization", "password", "token"])
def test_recovery_draft_rejects_secret_shaped_arguments(field: str):
    payload = _proposal_draft(calls=[{
        "tool": "team_configuration.apply",
        "arguments": {field: "do-not-store-this"},
        "rationale": "A repair should never transport credentials.",
    }])

    with pytest.raises(ValidationError, match="secret"):
        parse_recovery_assessment(payload)


@pytest.mark.parametrize("field", ["predicate", "where", "sql", "query", "expression"])
def test_recovery_draft_rejects_free_form_executable_predicates(field: str):
    payload = _proposal_draft(calls=[{
        "tool": "team_configuration.apply",
        "arguments": {field: "status == 'blocked'"},
        "rationale": "Registered tools receive canonical arguments only.",
    }])

    with pytest.raises(ValidationError, match="predicate|query|expression|SQL"):
        parse_recovery_assessment(payload)


@pytest.mark.parametrize(
    "field",
    ["python_service", "Python-Service", "service_name", "module", "function", "callable", "command", "code"],
)
def test_recovery_draft_rejects_executable_selector_key_families_recursively(field: str):
    payload = _proposal_draft(calls=[{
        "tool": "team_configuration.apply",
        "arguments": {"nested": {field: "untrusted implementation selector"}},
        "rationale": "The registry owns executable implementation selection.",
    }])

    with pytest.raises(ValidationError, match="executable|service|command|code"):
        parse_recovery_assessment(payload)


def test_assessments_require_nonblank_text_and_traceable_evidence():
    assert EvidenceV1.model_validate(_evidence()).source_id == UUID_A

    for field in ("label", "diagnosis", "problem", "expected_effect"):
        payload = _proposal_draft(**{field: "   "})
        with pytest.raises(ValidationError):
            parse_recovery_assessment(payload)

    with pytest.raises(ValidationError):
        EvidenceV1.model_validate({**_evidence(), "source_id": "not-a-uuid"})

    with pytest.raises(ValidationError):
        EvidenceV1.model_validate({**_evidence(), "source_kind": "database"})


def test_recovery_trigger_retry_targets_and_corrective_dispositions_are_closed_enums():
    target_adapter = TypeAdapter(RetryTargetV1)
    assert target_adapter.validate_python({"kind": "decision_loop"}).kind == "decision_loop"
    assert target_adapter.validate_python({"kind": "planning", "requested_agent_id": str(UUID_A)}).kind == "planning"
    assert target_adapter.validate_python({"kind": "action", "source_action_id": str(UUID_A)}).kind == "action"
    assert target_adapter.validate_python({"kind": "baseline_process", "process_type": "team_hierarchy"}).kind == "baseline_process"
    with pytest.raises(ValidationError):
        target_adapter.validate_python({"kind": "shell", "command": "repair"})

    trigger = RecoveryTriggerV1.model_validate({
        "schema_version": 1,
        "goal_id": str(UUID_A),
        "run_id": str(UUID_B),
        "source_kind": "planning_preflight",
        "source_id": str(UUID_A),
        "blocker_code": "plan_author_unavailable",
        "summary": "No eligible planning author.",
        "source_fingerprint": SHA256,
        "retry_target": {"kind": "decision_loop"},
    })
    assert trigger.retry_target.kind == "decision_loop"
    assert {member.value for member in CorrectiveDisposition} == {
        "ordinary", "same_attempt_reconciliation", "requires_recovery",
    }
    with pytest.raises(ValidationError):
        RecoveryTriggerV1.model_validate({**trigger.model_dump(mode="json"), "source_kind": "recovery_action"})


def test_evidence_queries_and_tool_manifest_have_only_documented_shapes():
    exact = (
        {"kind": "run_state", "run_id": str(UUID_A)},
        {"kind": "action", "action_id": str(UUID_A)},
        {"kind": "process", "process_type": "authority_interview"},
        {"kind": "team_configuration"},
        {"kind": "task", "task_id": str(UUID_A)},
        {"kind": "gate", "gate_id": str(UUID_A)},
        {"kind": "waits"},
        {"kind": "warning", "warning_id": str(UUID_A)},
        {"kind": "agent_execution", "agent_id": str(UUID_A)},
    )
    assert {request.kind for request in map(_query, exact)} == {
        "run_state", "action", "process", "team_configuration", "task", "gate", "waits", "warning", "agent_execution",
    }

    for payload in (
        {"kind": "run_state"},
        {"kind": "action", "action_id": str(UUID_A), "run_id": str(UUID_B)},
        {"kind": "process", "process_type": "not-a-process"},
        {"kind": "team_configuration", "action_id": str(UUID_A)},
        {"kind": "waits", "warning_id": str(UUID_A)},
    ):
        with pytest.raises(ValidationError):
            _query(payload)
    with pytest.raises(ValidationError):
        _query({"kind": "team_configuration", "source_id": str(UUID_A)})
    with pytest.raises(ValidationError):
        _query({"kind": "arbitrary_model", "source_id": str(UUID_A)})

    manifest = ToolManifestV1.model_validate({
        "name": "team_configuration.apply",
        "version": "1",
        "description": "Atomically updates accepted team configuration.",
        "arguments_schema": {"type": "object"},
    })
    assert manifest.name == "team_configuration.apply"
    with pytest.raises(ValidationError) as caught:
        ToolManifestV1.model_validate({**manifest.model_dump(), "python_service": "DangerousExecutor"})
    assert "extra_forbidden" in _error_types(caught.value)


def test_recovery_evidence_limits_are_public_and_match_the_contract_boundary():
    from huddleroom.services.orchestration_recovery_evidence import (
        RECOVERY_CONTEXT_MAX_BYTES,
        RECOVERY_EVIDENCE_MAX_BYTES,
    )

    assert RECOVERY_EVIDENCE_MAX_BYTES == 16_384
    assert RECOVERY_CONTEXT_MAX_BYTES == 65_536


def test_capability_gap_is_complete_and_cannot_smuggle_a_fix():
    valid = {
        "kind": "capability_gap",
        "diagnosis": "No registered repair tool can change this deployment boundary.",
        "evidence": [_evidence()],
        "missing_capability": "A deployed recovery adapter for the target system.",
        "required_human_action": "Deploy the registered adapter and retry recovery.",
        "wake_after": {"kind": "human_decision", "decision": "deploy_adapter"},
    }
    assert isinstance(parse_recovery_assessment(valid), CapabilityGapAssessmentV1)

    for invalid in (
        {key: value for key, value in valid.items() if key != "wake_after"},
        {**valid, "missing_capability": " "},
        {**valid, "calls": _proposal_draft()["calls"]},
    ):
        with pytest.raises(ValidationError):
            parse_recovery_assessment(invalid)


def test_transient_requires_a_live_wake_and_rejects_unknown_fields():
    valid = {
        "kind": "transient",
        "diagnosis": "The task producer is actively running.",
        "evidence": [_evidence()],
        "wake": _wake(),
    }
    assert isinstance(parse_recovery_assessment(valid), TransientAssessmentV1)

    with pytest.raises(ValidationError):
        parse_recovery_assessment({**valid, "wake": {"events": []}})
    with pytest.raises(ValidationError) as caught:
        parse_recovery_assessment({**valid, "unknown": True})
    assert "extra_forbidden" in _error_types(caught.value)


def test_canonical_proposal_limits_calls_checks_and_retry_to_one():
    assert isinstance(RecoveryProposalV1.model_validate(_proposal()), RecoveryProposalV1)

    for invalid in (
        _proposal(calls=[_call(index) for index in range(9)]),
        _proposal(verification=[_check(index) for index in range(9)]),
        _proposal(retry={"target": {"kind": "decision_loop"}, "max_attempts": 2}),
        _proposal(calls=[_call(1), _call(1)]),
        _proposal(verification=[_check(1), _check(1)]),
    ):
        with pytest.raises(ValidationError):
            RecoveryProposalV1.model_validate(invalid)

    with pytest.raises(ValidationError, match="secret"):
        RecoveryCallV1.model_validate({**_call(1), "arguments": {"token": "must-not-pass"}})
    with pytest.raises(ValidationError, match="predicate|query|expression|SQL"):
        VerificationCheckV1.model_validate({**_check(1), "arguments": {"predicate": "anything()"}})


def test_recovery_dispatch_context_is_server_only_and_discriminated():
    adapter = TypeAdapter(RecoveryDispatchContext)
    apply_context = adapter.validate_python({
        "purpose": "apply",
        "proposal_id": str(UUID_A),
        "call_id": "call-1",
        "authority_decision_id": str(UUID_B),
    })
    retry_context = adapter.validate_python({
        "purpose": "retry",
        "proposal_id": str(UUID_A),
        "approved_retry_ordinal": 1,
        "authority_decision_id": str(UUID_B),
    })
    assert apply_context.purpose == "apply"
    assert retry_context.purpose == "retry"

    with pytest.raises(ValidationError):
        adapter.validate_python({
            "purpose": "apply",
            "proposal_id": str(UUID_A),
            "call_id": "call-1",
            "approved_retry_ordinal": 1,
            "authority_decision_id": str(UUID_B),
        })

    with pytest.raises(ValidationError):
        parse_recovery_assessment({**_proposal_draft(), "dispatch_context": {"purpose": "retry"}})


def test_receipts_require_exact_safe_effect_identity():
    payload = {
        "proposal_id": str(UUID_A),
        "call_id": "call-1",
        "tool": "team_configuration.apply@1",
        "wrapper_action_id": str(UUID_A),
        "action_id": str(UUID_A),
        "idempotency_key": "recovery:proposal:apply:call-1",
        "canonical_request_hash": SHA256,
        "execution_identity": {
            "proposal_id": str(UUID_A),
            "call_id": "call-1",
            "authority_decision_id": str(UUID_B),
            "purpose": "apply",
        },
        "status": "applied",
        "target_refs": [{"kind": "agent", "id": str(UUID_A)}],
        "observed_post_state": {"role": "planner"},
        "evidence": [_evidence()],
    }
    assert isinstance(ToolReceiptV1.model_validate(payload), ToolReceiptV1)
    assert ToolReceiptV1.model_validate({
        **payload,
        "status": "failed",
        "error_code": "provider_failed",
        "error": "The registered tool reported a safe failure.",
    }).status == "failed"
    with pytest.raises(ValidationError):
        ToolReceiptV1.model_validate({**payload, "idempotency_key": ""})
    with pytest.raises(ValidationError):
        ToolReceiptV1.model_validate({**payload, "execution_identity": {**payload["execution_identity"], "purpose": "retry"}})


def test_remaining_shared_models_are_strict_and_preserve_wait_safety():
    assert BaselineRepairAdmissionV1.model_validate({
        "process_type": "team_hierarchy", "gate_id": str(UUID_A), "gate_fingerprint": SHA256,
    }).gate_id == UUID_A
    with pytest.raises(ValidationError):
        BaselineRepairAdmissionV1.model_validate({
            "process_type": "team_hierarchy", "gate_id": str(UUID_A), "gate_fingerprint": SHA256, "skip_all_gates": True,
        })

    configuration = TeamConfigurationApplyV1.model_validate({
        "agents": [], "eligibility": [], "assignments": [],
        "hierarchy": {
            "manager": {"kind": "none"}, "team_leads": [], "contributors": [], "reviewers": [],
            "validators": [], "specialists": [],
        },
        "documented_gaps": [],
    })
    assert configuration.agents == []
    with pytest.raises(ValidationError, match="secret"):
        TeamConfigurationApplyV1.model_validate({
            **configuration.model_dump(),
            "agents": [{
                "operation": "create", "ref": "new:planner",
                "definition": {
                    "name": "Planner", "role": "Planner", "provider": "configured", "model": "model",
                    "adapter_type": "api", "capabilities": [], "config": {"api_key": "must-not-pass"},
                },
            }],
        })

    pending = VerificationResultV1.model_validate({
        "check_id": "planning-ready", "status": "pending", "code": "waiting_for_task", "evidence": [_evidence()],
        "wake": _wake(),
    })
    assert pending.status == "pending"
    with pytest.raises(ValidationError):
        VerificationResultV1.model_validate({**pending.model_dump(mode="json"), "wake": None})

    response = RecoveryProposalResponse.model_validate({
        "kind": "recovery_proposal_v1", "proposal_id": str(UUID_A), "version": 1,
        "status": "pending_approval", "problem": "No planning author.",
        "general_updates": ["Assign an eligible planning author."],
        "after_approval": "Apply, verify, and retry once.", "global_effect": True,
    })
    assert response.status == "pending_approval"
    with pytest.raises(ValidationError):
        RecoveryProposalResponse.model_validate({**response.model_dump(mode="json"), "executable_request": {"sql": "DROP"}})


def test_team_configuration_is_a_strict_aggregate_with_valid_create_and_update():
    valid = {
        "agents": [
            {
                "operation": "create",
                "ref": "new:planner",
                "definition": {
                    "name": "Planner", "role": "Planner", "provider": "configured", "model": "model",
                    "adapter_type": "api", "capabilities": [], "config": {},
                },
            },
            {"operation": "update", "agent_id": str(UUID_A), "set": {"role": "Lead planner"}},
        ],
        "eligibility": [{"agent_ref": "new:planner", "work_functions": ["planning", "management"]}],
        "assignments": [{"work_function": "planning", "agent_ref": "new:planner"}],
        "hierarchy": {
            "manager": {"kind": "none"}, "team_leads": ["new:planner"], "contributors": [],
            "reviewers": [], "validators": [], "specialists": [],
        },
        "documented_gaps": [{"code": "human_approval", "rationale": "A human owns the final decision."}],
    }
    parsed = TeamConfigurationApplyV1.model_validate(valid)
    assert len(parsed.agents) == 2

    invalid = (
        {**valid, "agents": [{"operation": "delete", "agent_id": str(UUID_A)}]},
        {**valid, "agents": [{"operation": "update", "agent_id": str(UUID_A), "set": {"is_active": False}}]},
        {**valid, "agents": [{"operation": "create", "ref": "planner", "definition": valid["agents"][0]["definition"]}]},
        {**valid, "agents": [{"operation": "create", "ref": "new:planner", "definition": {**valid["agents"][0]["definition"], "unknown": True}}]},
        {**valid, "eligibility": [{"agent_ref": "new:planner", "work_functions": ["arbitrary_work"]}]},
        {**valid, "hierarchy": {**valid["hierarchy"], "manager": {"kind": "robot", "agent_ref": "new:planner"}}},
        {**valid, "hierarchy": {**valid["hierarchy"], "reviewers": ["invalid local reference"]}},
        {**valid, "documented_gaps": [{"code": "", "rationale": "gap"}]},
    )
    for payload in invalid:
        with pytest.raises(ValidationError):
            TeamConfigurationApplyV1.model_validate(payload)


def test_recovery_wakes_use_the_registered_normalized_contract_everywhere():
    transient = parse_recovery_assessment({
        "kind": "transient", "diagnosis": "A producer is running.", "evidence": [_evidence()], "wake": _wake(),
    })
    canonical = transient.wake.model_dump(mode="json") if hasattr(transient.wake, "model_dump") else transient.wake
    assert canonical["events"][0]["matcher"]["task_id"] == str(UUID_A)
    assert canonical["expected_result"] == "The dependent task reports a terminal state."
    assert canonical["recheck_after_seconds"] is not None

    for invalid_wake in (
        {**_wake(), "events": [{"event_type": "task.completed", "matcher": {"task_id": str(UUID_A)}}]},
        {**_wake(), "events": [{"event_type": "task.status_changed", "matcher": {"agent_id": str(UUID_A)}}]},
        {**_wake(), "events": [
            {"event_type": "task.status_changed", "matcher": {"task_id": str(UUID_A)}},
            {"event_type": "task.assigned", "matcher": {"task_id": str(UUID_A)}},
            {"event_type": "session.created", "matcher": {"session_id": str(UUID_A)}},
            {"event_type": "session.started", "matcher": {"session_id": str(UUID_A)}},
            {"event_type": "session.completed", "matcher": {"session_id": str(UUID_A)}},
            {"event_type": "session.failed", "matcher": {"session_id": str(UUID_A)}},
        ]},
        {key: value for key, value in _wake().items() if key != "expected_result"},
        {"events": [], "expected_result": "No event or timer"},
    ):
        with pytest.raises(ValidationError):
            parse_recovery_assessment({
                "kind": "transient", "diagnosis": "A producer is running.", "evidence": [_evidence()], "wake": invalid_wake,
            })

    waiting_receipt = {
        "proposal_id": str(UUID_A), "call_id": "call-1", "tool": "team_configuration.apply@1",
        "wrapper_action_id": str(UUID_A), "action_id": str(UUID_A), "idempotency_key": "recovery:proposal:apply:call-1",
        "canonical_request_hash": SHA256,
        "execution_identity": {"proposal_id": str(UUID_A), "call_id": "call-1", "authority_decision_id": str(UUID_B), "purpose": "apply"},
        "status": "waiting", "target_refs": [{"kind": "agent", "id": str(UUID_A)}],
        "observed_post_state": None, "evidence": [_evidence()], "wait": _wake(),
    }
    assert ToolReceiptV1.model_validate(waiting_receipt).status == "waiting"
    with pytest.raises(ValidationError):
        ToolReceiptV1.model_validate({key: value for key, value in waiting_receipt.items() if key != "wait"})
    with pytest.raises(ValidationError):
        ToolReceiptV1.model_validate({**waiting_receipt, "status": "applied"})

    assert VerificationResultV1.model_validate({
        "check_id": "wait-for-task", "status": "pending", "code": "pending", "evidence": [_evidence()], "wake": _wake(),
    }).status == "pending"
    assert isinstance(parse_recovery_assessment({
        "kind": "capability_gap", "diagnosis": "Adapter is absent.", "evidence": [_evidence()],
        "missing_capability": "A registered adapter.", "required_human_action": "Deploy the adapter.", "wake_after": _wake(),
    }), CapabilityGapAssessmentV1)


def test_model_visible_json_is_bounded_json_only_and_secret_free():
    nested_secret = {"level": {"access-token": "supersecret"}}
    too_deep: object = "leaf"
    for _ in range(9):
        too_deep = {"next": too_deep}
    over_bytes = "x" * 16_385
    invalid_evidence_values = (nested_secret, too_deep, over_bytes, ["item"] * 129, float("nan"), object())
    for value in invalid_evidence_values:
        with pytest.raises(ValidationError):
            EvidenceV1.model_validate({**_evidence(), "value": value})

    with pytest.raises(ValidationError):
        ChangeV1.model_validate({**_call(1)["changes"][0], "after": nested_secret})
    with pytest.raises(ValidationError):
        ToolManifestV1.model_validate({
            "name": "tool", "version": "1", "description": "safe", "arguments_schema": nested_secret,
        })
    with pytest.raises(ValidationError):
        ToolReceiptV1.model_validate({
            "proposal_id": str(UUID_A), "call_id": "call-1", "tool": "tool@1", "wrapper_action_id": str(UUID_A),
            "action_id": str(UUID_A), "idempotency_key": "key", "canonical_request_hash": SHA256,
            "execution_identity": {"proposal_id": str(UUID_A), "call_id": "call-1", "authority_decision_id": str(UUID_B), "purpose": "apply"},
            "status": "applied", "target_refs": [nested_secret], "observed_post_state": nested_secret, "evidence": [_evidence()],
        })
    with pytest.raises(ValidationError):
        RecoveryProposalResponse.model_validate({
            "kind": "recovery_proposal_v1", "proposal_id": str(UUID_A), "version": 1, "status": "failed",
            "problem": "A safe problem.", "general_updates": ["A safe update."], "after_approval": "A safe next step.",
            "global_effect": False, "exact_details": nested_secret,
        })
    with pytest.raises(ValidationError):
        RecoveryProposalResponse.model_validate({
            "kind": "recovery_proposal_v1", "proposal_id": str(UUID_A), "version": 1, "status": "failed",
            "problem": "A safe problem.", "general_updates": ["A safe update."], "after_approval": "A safe next step.",
            "global_effect": False, "evidence": [_evidence()] * 33,
        })


def test_general_safe_json_accepts_ordinary_code_data_but_arguments_do_not():
    assert EvidenceV1.model_validate({**_evidence(), "value": {"code": "plan_author_unavailable", "status": "blocked"}}).value[
        "code"
    ] == "plan_author_unavailable"
    assert ChangeV1.model_validate({
        **_call(1)["changes"][0], "before": {"code": "old"}, "after": {"code": "new", "status": "applied"},
    }).after["code"] == "new"
    assert ToolManifestV1.model_validate({
        "name": "tool", "version": "1", "description": "safe", "arguments_schema": {"properties": {"code": {"type": "string"}}},
    }).arguments_schema["properties"]["code"]["type"] == "string"

    for value in ({"code": "unsafe"}, {"command": "unsafe"}, {"function": "unsafe"}):
        with pytest.raises(ValidationError):
            RecoveryCallV1.model_validate({**_call(1), "arguments": value})
    with pytest.raises(ValidationError):
        TeamConfigurationApplyV1.model_validate({
            "agents": [{
                "operation": "create", "ref": "new:planner",
                "definition": {
                    "name": "Planner", "role": "Planner", "provider": "configured", "model": "model",
                    "adapter_type": "api", "capabilities": [], "config": {"command": "unsafe"},
                },
            }],
            "eligibility": [], "assignments": [],
            "hierarchy": {"manager": {"kind": "none"}, "team_leads": [], "contributors": [], "reviewers": [], "validators": [], "specialists": []},
            "documented_gaps": [],
        })


def test_wake_and_agent_text_are_bounded_secret_free_and_keep_omitted_updates_none():
    for expected_result in ("token=supersecret", "x" * 4_001):
        with pytest.raises(ValidationError):
            parse_recovery_assessment({
                "kind": "transient", "diagnosis": "A producer is running.", "evidence": [_evidence()],
                "wake": {**_wake(), "expected_result": expected_result},
            })

    human_wake = {
        "kind": "capability_gap", "diagnosis": "Adapter is absent.", "evidence": [_evidence()],
        "missing_capability": "A registered adapter.", "required_human_action": "Deploy the adapter.",
        "wake_after": {"kind": "human_decision", "decision": "deploy_adapter"},
    }
    assert isinstance(parse_recovery_assessment(human_wake), CapabilityGapAssessmentV1)
    for wake_after in (
        {"kind": "human_decision", "decision": "token=supersecret"},
        {"kind": "human_decision", "decision": " "},
        {"kind": "human_decision", "decision": "x" * 4_001},
        {"kind": "human_decision", "decision": "deploy", "extra": True},
    ):
        with pytest.raises(ValidationError):
            parse_recovery_assessment({**human_wake, "wake_after": wake_after})

    base = {
        "agents": [{
            "operation": "create", "ref": "new:planner",
            "definition": {
                "name": "Planner", "role": "Planner", "provider": "configured", "model": "model",
                "adapter_type": "api", "capabilities": ["planning"], "config": {},
            },
        }],
        "eligibility": [], "assignments": [],
        "hierarchy": {"manager": {"kind": "none"}, "team_leads": [], "contributors": [], "reviewers": [], "validators": [], "specialists": []},
        "documented_gaps": [],
    }
    nullable_create = TeamConfigurationApplyV1.model_validate({
        **base,
        "agents": [{
            **base["agents"][0],
            "definition": {
                **base["agents"][0]["definition"],
                "description": None,
                "system_prompt": None,
                "cli_runtime": None,
            },
        }],
    })
    assert nullable_create.agents[0].definition.description is None
    assert nullable_create.agents[0].definition.system_prompt is None
    assert nullable_create.agents[0].definition.cli_runtime is None

    nullable_update = TeamConfigurationApplyV1.model_validate({
        **base,
        "agents": [{
            "operation": "update", "agent_id": str(UUID_A),
            "set": {
                "name": None, "role": None, "description": None, "provider": None, "model": None,
                "system_prompt": None, "cli_runtime": None,
            },
        }],
    })
    for field in ("name", "role", "description", "provider", "model", "system_prompt", "cli_runtime"):
        assert getattr(nullable_update.agents[0].set, field) is None

    for field, maximum in (
        ("name", 255), ("role", 255), ("provider", 255), ("model", 255), ("description", 4_000),
        ("system_prompt", 4_000), ("cli_runtime", 255),
    ):
        for value in ("token=supersecret", "x" * (maximum + 1)):
            definition = {**base["agents"][0]["definition"], field: value}
            with pytest.raises(ValidationError):
                TeamConfigurationApplyV1.model_validate({**base, "agents": [{**base["agents"][0], "definition": definition}]})
    for value in ("token=supersecret", "x" * 256):
        with pytest.raises(ValidationError):
            TeamConfigurationApplyV1.model_validate({
                **base, "agents": [{**base["agents"][0], "definition": {**base["agents"][0]["definition"], "capabilities": [value]}}],
            })

    updated = TeamConfigurationApplyV1.model_validate({
        **base, "agents": [{"operation": "update", "agent_id": str(UUID_A), "set": {"role": "Planner"}}],
    })
    assert updated.agents[0].set.description is None
    for field in ("name", "role", "description", "provider", "model", "system_prompt", "cli_runtime"):
        with pytest.raises(ValidationError):
            TeamConfigurationApplyV1.model_validate({
                **base, "agents": [{"operation": "update", "agent_id": str(UUID_A), "set": {field: "token=supersecret"}}],
            })


def test_evidence_query_variants_have_exact_foreign_null_free_serialization():
    exact = (
        ({"kind": "run_state", "run_id": str(UUID_A)}, {"kind": "run_state", "run_id": UUID_A}),
        ({"kind": "action", "action_id": str(UUID_A)}, {"kind": "action", "action_id": UUID_A}),
        ({"kind": "process", "process_type": "authority_interview"}, {"kind": "process", "process_type": "authority_interview"}),
        ({"kind": "team_configuration"}, {"kind": "team_configuration"}),
        ({"kind": "task", "task_id": str(UUID_A)}, {"kind": "task", "task_id": UUID_A}),
        ({"kind": "gate", "gate_id": str(UUID_A)}, {"kind": "gate", "gate_id": UUID_A}),
        ({"kind": "waits"}, {"kind": "waits"}),
        ({"kind": "warning", "warning_id": str(UUID_A)}, {"kind": "warning", "warning_id": UUID_A}),
        ({"kind": "agent_execution", "agent_id": str(UUID_A)}, {"kind": "agent_execution", "agent_id": UUID_A}),
    )
    for payload, expected in exact:
        assert _query(payload).model_dump(exclude_none=False) == expected
    with pytest.raises(ValidationError):
        _query({"kind": "action", "action_id": str(UUID_A), "run_id": None})


@pytest.mark.asyncio
async def test_recovery_analyzer_callback_receives_the_concrete_query_variant():
    completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": json.dumps({
            "kind": "evidence_query", "query": {"kind": "action", "action_id": str(UUID_A)},
        })}}]},
        {"choices": [{"message": {"content": json.dumps(_proposal_draft())}}]},
    ])
    evidence_query = AsyncMock(return_value=EvidenceV1.model_validate(_evidence()))

    assert isinstance(
        await RecoveryAnalyzer(completion_fn=completion).analyze(
            {"goal": {"id": str(UUID_A)}, "run": {"id": str(UUID_B)}}, project_id=UUID_A, evidence_query=evidence_query,
        ),
        ProposalDraftV1,
    )
    assert evidence_query.await_args.args[0].model_dump(exclude_none=False) == {"kind": "action", "action_id": UUID_A}


@pytest.mark.asyncio
async def test_recovery_analyzer_stops_after_four_evidence_rounds_and_repairs_parser_errors():
    """The documented intermediate envelope is JSON-only: kind=evidence_query, query=<strict query>."""
    completion = AsyncMock(side_effect=[
        {"choices": [{"message": {"content": '{"kind":"evidence_query","query": {'}}]},
        {"choices": [{"message": {"content": json.dumps({
            "kind": "evidence_query",
            "query": {"kind": "run_state", "run_id": str(UUID_A)},
        })}}]},
        {"choices": [{"message": {"content": json.dumps({
            "kind": "evidence_query",
            "query": {"kind": "waits"},
        })}}]},
        {"choices": [{"message": {"content": json.dumps({
            "kind": "evidence_query",
            "query": {"kind": "warning", "warning_id": str(UUID_A)},
        })}}]},
        {"choices": [{"message": {"content": json.dumps({
            "kind": "evidence_query",
            "query": {"kind": "team_configuration"},
        })}}]},
        {"choices": [{"message": {"content": json.dumps(_proposal_draft())}}]},
    ])
    evidence_query = AsyncMock(return_value=EvidenceV1.model_validate(_evidence()))

    result = await RecoveryAnalyzer(completion_fn=completion).analyze(
        {"goal": {"id": str(UUID_A)}, "run": {"id": str(UUID_B)}},
        project_id=UUID_A,
        evidence_query=evidence_query,
    )

    assert isinstance(result, ProposalDraftV1)
    assert MAX_RECOVERY_EVIDENCE_ROUNDS == 4
    assert evidence_query.await_count == 4
    assert completion.await_count == 6


@pytest.mark.asyncio
async def test_recovery_analyzer_redacts_provider_failures_before_publication(monkeypatch):
    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    events = []

    async def publish(event):
        events.append(event)

    async def never_reset(_project_id):
        await asyncio.Event().wait()

    from huddleroom.services.agent_response_stream import AgentResponseInvocation as RealInvocation

    monkeypatch.setattr(
        "huddleroom.services.orchestration_recovery_analyzer.AgentResponseInvocation",
        lambda context: RealInvocation(context, publish=publish),
    )
    monkeypatch.setattr(
        "huddleroom.services.agent_response_relay.register_project_reset_monitor",
        AsyncMock(return_value=SimpleNamespace(generation=None)),
    )
    monkeypatch.setattr(
        "huddleroom.services.agent_response_relay.unregister_project_reset_monitor", AsyncMock(),
    )
    monkeypatch.setattr("huddleroom.services.agent_response_stream.wait_for_project_reset", never_reset)
    analyzer = RecoveryAnalyzer(completion_fn=AsyncMock(side_effect=RuntimeError(f"provider rejected {secret}")))

    with pytest.raises(RuntimeError) as caught:
        await analyzer.analyze(
            {"goal": {"id": str(UUID_A)}, "run": {"id": str(UUID_B)}},
            project_id=UUID_A,
            evidence_query=AsyncMock(),
        )

    assert "[REDACTED]" in str(caught.value)
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    terminal = next(event for event in events if event.event_type == "agent_response.terminal")
    assert secret not in str(terminal.payload)


@pytest.mark.asyncio
async def test_recovery_analyzer_redacts_evidence_callback_failures_and_drops_raw_cause():
    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    completion = AsyncMock(return_value={"choices": [{"message": {"content": json.dumps({
        "kind": "evidence_query", "query": {"kind": "run_state", "run_id": str(UUID_A)},
    })}}]})

    with pytest.raises(RuntimeError) as caught:
        await RecoveryAnalyzer(completion_fn=completion).analyze(
            {"goal": {"id": str(UUID_A)}, "run": {"id": str(UUID_B)}},
            project_id=UUID_A,
            evidence_query=AsyncMock(side_effect=RuntimeError(f"evidence callback leaked {secret}")),
        )

    assert "[REDACTED]" in str(caught.value)
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
