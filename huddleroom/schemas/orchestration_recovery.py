"""Strict, server-owned contracts for blocker recovery."""

from __future__ import annotations

from enum import StrEnum
import json
import math
import re
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationInfo, field_validator, model_validator

from huddleroom.models.orchestration_process import PROCESS_TYPES
from huddleroom.services.secret_redaction import redact_secrets as _redact_secrets  # not the adapter: avoids a schemas import cycle
from huddleroom.services.orchestration_wake_when import normalize_wake_when


_JSON_MAX_BYTES = 16_384
_JSON_MAX_DEPTH = 8
_JSON_MAX_ITEMS = 128


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _text(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must be nonblank")
    if _redact_secrets(value) != value or re.search(r"(?i)(api[_-]?key|access[_-]?token|token|password|authorization|bearer)\s*[:=]", value):
        raise ValueError("must not contain secrets")
    return value


def _nullable_text(value: str | None) -> str | None:
    return _text(value) if value is not None else None


def _capability_texts(values: list[str]) -> list[str]:
    if any(len(value) > 255 for value in values):
        raise ValueError("capability exceeds maximum length")
    return [_text(value) for value in values]


def _safe_json(value: Any) -> Any:
    def visit(item: Any, depth: int = 0) -> Any:
        if depth > _JSON_MAX_DEPTH:
            raise ValueError("JSON exceeds maximum depth")
        if item is None or isinstance(item, (str, int, bool)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("JSON floats must be finite")
            return item
        if isinstance(item, list):
            if len(item) > _JSON_MAX_ITEMS:
                raise ValueError("JSON container exceeds maximum items")
            return [visit(child, depth + 1) for child in item]
        if isinstance(item, dict):
            if len(item) > _JSON_MAX_ITEMS or not all(isinstance(key, str) for key in item):
                raise ValueError("JSON object has invalid keys or too many items")
            return {key: visit(child, depth + 1) for key, child in item.items()}
        raise ValueError("must be JSON-compatible")

    normalized = visit(value)
    _reject_secret_keys(normalized)
    encoded = json.dumps(normalized, allow_nan=False, separators=(",", ":"), ensure_ascii=False)
    if len(encoded.encode()) > _JSON_MAX_BYTES:
        raise ValueError("JSON exceeds maximum bytes")
    if _redact_secrets(encoded) != encoded:
        raise ValueError("JSON must not contain secrets")
    return normalized


def _reject_secret_keys(value: Any) -> Any:
    if isinstance(value, list):
        return [_reject_secret_keys(item) for item in value]
    if not isinstance(value, dict):
        return value
    blocked = {"apikey", "accesstoken", "authorization", "password", "token", "secret", "bearer"}
    for key, item in value.items():
        normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
        if normalized in blocked:
            raise ValueError("secret keys are not allowed")
        _reject_secret_keys(item)
    return value


def _reject_unsafe_keys(value: Any, *, allow_documented_gap_code: bool = False, path: tuple[str, ...] = ()) -> Any:
    _reject_secret_keys(value)
    if isinstance(value, list):
        return [_reject_unsafe_keys(item, allow_documented_gap_code=allow_documented_gap_code, path=(*path, "*")) for item in value]
    if not isinstance(value, dict):
        return value
    blocked = {"predicate", "where", "sql", "query", "expression", "pythonservice", "servicename", "module", "function", "callable", "command", "code"}
    for key, item in value.items():
        normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
        documented_gap_code = allow_documented_gap_code and path == ("documented_gaps", "*") and normalized == "code"
        if normalized in blocked and not documented_gap_code:
            raise ValueError("predicate/query or executable service/command/code keys are not allowed")
        _reject_unsafe_keys(item, allow_documented_gap_code=allow_documented_gap_code, path=(*path, key))
    return value


def _safe_call_arguments(value: dict[str, Any], *, team_configuration: bool) -> dict[str, Any]:
    return _reject_unsafe_keys(_safe_json(value), allow_documented_gap_code=team_configuration)


class CorrectiveDisposition(StrEnum):
    ORDINARY = "ordinary"
    SAME_ATTEMPT_RECONCILIATION = "same_attempt_reconciliation"
    REQUIRES_RECOVERY = "requires_recovery"


class _DecisionLoopRetryTarget(_StrictModel):
    kind: Literal["decision_loop"]


class _PlanningRetryTarget(_StrictModel):
    kind: Literal["planning"]
    requested_agent_id: UUID | None = None


class _ActionRetryTarget(_StrictModel):
    kind: Literal["action"]
    source_action_id: UUID


class _BaselineProcessRetryTarget(_StrictModel):
    kind: Literal["baseline_process"]
    process_type: str

    @field_validator("process_type")
    @classmethod
    def _process_type(cls, value: str) -> str:
        if value not in PROCESS_TYPES:
            raise ValueError("unsupported process type")
        return value


RetryTargetV1 = Annotated[
    _DecisionLoopRetryTarget | _PlanningRetryTarget | _ActionRetryTarget | _BaselineProcessRetryTarget,
    Field(discriminator="kind"),
]


class RecoveryTriggerV1(_StrictModel):
    schema_version: Literal[1]
    goal_id: UUID
    run_id: UUID
    source_kind: Literal["reported_blocker", "failed_action", "planning_preflight", "eventless_noop"]
    source_id: UUID
    blocker_code: str = Field(min_length=1, max_length=255)
    summary: str = Field(min_length=1, max_length=4_000)
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    retry_target: RetryTargetV1

    _nonblank = field_validator("blocker_code", "summary")(_text)


class EvidenceV1(_StrictModel):
    label: str = Field(min_length=1, max_length=1_000)
    value: Any
    source_kind: Literal["action", "task", "gate", "process", "decision", "warning", "run", "agent", "artifact"]
    source_id: UUID

    _nonblank = field_validator("label")(_text)
    _json = field_validator("value")(_safe_json)


class _RunStateQueryV1(_StrictModel):
    kind: Literal["run_state"]
    run_id: UUID


class _ActionQueryV1(_StrictModel):
    kind: Literal["action"]
    action_id: UUID


class _ProcessQueryV1(_StrictModel):
    kind: Literal["process"]
    process_type: str

    @field_validator("process_type")
    @classmethod
    def _process_type(cls, value: str) -> str:
        if value not in PROCESS_TYPES:
            raise ValueError("unsupported process type")
        return value


class _TeamConfigurationQueryV1(_StrictModel):
    kind: Literal["team_configuration"]


class _TaskQueryV1(_StrictModel):
    kind: Literal["task"]
    task_id: UUID


class _GateQueryV1(_StrictModel):
    kind: Literal["gate"]
    gate_id: UUID


class _WaitsQueryV1(_StrictModel):
    kind: Literal["waits"]


class _WarningQueryV1(_StrictModel):
    kind: Literal["warning"]
    warning_id: UUID


class _AgentExecutionQueryV1(_StrictModel):
    kind: Literal["agent_execution"]
    agent_id: UUID


RecoveryEvidenceQueryV1 = Annotated[
    _RunStateQueryV1 | _ActionQueryV1 | _ProcessQueryV1 | _TeamConfigurationQueryV1 |
    _TaskQueryV1 | _GateQueryV1 | _WaitsQueryV1 | _WarningQueryV1 | _AgentExecutionQueryV1,
    Field(discriminator="kind"),
]
RECOVERY_EVIDENCE_QUERY_ADAPTER = TypeAdapter(RecoveryEvidenceQueryV1)


class ToolManifestV1(_StrictModel):
    name: str = Field(min_length=1, max_length=255)
    version: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=2_000)
    arguments_schema: dict[str, Any]

    _nonblank = field_validator("name", "version", "description")(_text)
    _schema_json = field_validator("arguments_schema")(_safe_json)


class _WakeV1(_StrictModel):
    events: list[dict[str, Any]] = Field(default_factory=list)
    recheck_after_seconds: int | None = None
    expected_result: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        return normalize_wake_when(value)

    _expected_result = field_validator("expected_result")(_text)


class _HumanDecisionWakeV1(_StrictModel):
    kind: Literal["human_decision"]
    decision: str = Field(min_length=1, max_length=4_000)
    _decision = field_validator("decision")(_text)


class _AssessmentBase(_StrictModel):
    diagnosis: str = Field(min_length=1, max_length=4_000)
    evidence: list[EvidenceV1] = Field(min_length=1, max_length=32)

    _nonblank = field_validator("diagnosis")(_text)


class TransientAssessmentV1(_AssessmentBase):
    kind: Literal["transient"]
    wake: _WakeV1


class RecoveryCallDraftV1(_StrictModel):
    tool: str = Field(min_length=1, max_length=255)
    arguments: dict[str, Any]
    rationale: str = Field(min_length=1, max_length=4_000)

    _nonblank = field_validator("tool", "rationale")(_text)

    @field_validator("arguments")
    @classmethod
    def _safe_arguments(cls, value: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        return _safe_call_arguments(value, team_configuration=info.data.get("tool") == "team_configuration.apply@1")


class ProposalDraftV1(_AssessmentBase):
    kind: Literal["proposal"]
    problem: str = Field(min_length=1, max_length=4_000)
    general_updates: list[str] = Field(min_length=1, max_length=16)
    expected_effect: str = Field(min_length=1, max_length=4_000)
    calls: list[RecoveryCallDraftV1] = Field(min_length=1, max_length=8)

    _nonblank = field_validator("diagnosis", "problem", "expected_effect")(_text)

    @field_validator("general_updates")
    @classmethod
    def _updates(cls, value: list[str]) -> list[str]:
        return [_text(item) for item in value]


class CapabilityGapAssessmentV1(_AssessmentBase):
    kind: Literal["capability_gap"]
    missing_capability: str = Field(min_length=1, max_length=4_000)
    required_human_action: str = Field(min_length=1, max_length=4_000)
    wake_after: _WakeV1 | _HumanDecisionWakeV1

    _nonblank = field_validator("diagnosis", "missing_capability", "required_human_action")(_text)

RecoveryAssessmentV1 = Annotated[
    TransientAssessmentV1 | ProposalDraftV1 | CapabilityGapAssessmentV1,
    Field(discriminator="kind"),
]


class ChangeV1(_StrictModel):
    operation: Literal["create", "update", "assign", "replace", "delegate", "request"]
    scope: Literal["global", "project", "goal", "run"]
    target_type: str = Field(min_length=1, max_length=255)
    target_id: UUID | str
    before: Any | None = None
    after: Any

    _target = field_validator("target_type")(_text)
    _before = field_validator("before")(_safe_json)
    _after = field_validator("after")(_safe_json)


class RecoveryCallV1(_StrictModel):
    call_id: str = Field(min_length=1, max_length=100)
    tool: str = Field(min_length=1, max_length=255)
    tool_version: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any]
    changes: list[ChangeV1] = Field(min_length=1, max_length=32)

    _nonblank = field_validator("call_id", "tool", "tool_version")(_text)
    @field_validator("arguments")
    @classmethod
    def _safe_arguments(cls, value: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        return _safe_call_arguments(value, team_configuration=(
            info.data.get("tool") == "team_configuration.apply" and info.data.get("tool_version") == "1"
        ))


class VerificationCheckV1(_StrictModel):
    check_id: str = Field(min_length=1, max_length=100)
    verifier: str = Field(min_length=1, max_length=255)
    verifier_version: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any]

    _nonblank = field_validator("check_id", "verifier", "verifier_version")(_text)
    @field_validator("arguments")
    @classmethod
    def _safe_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_unsafe_keys(_safe_json(value))


class _ProposalSourceV1(_StrictModel):
    kind: Literal["reported_blocker", "failed_action", "planning_preflight", "eventless_noop"]
    id: UUID
    blocker_code: str = Field(min_length=1, max_length=255)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    _code = field_validator("blocker_code")(_text)


class _SealedRetryV1(_StrictModel):
    target: RetryTargetV1
    max_attempts: Literal[1]


class RecoveryProposalV1(_StrictModel):
    schema_version: Literal[1]
    proposal_id: UUID
    lineage_id: UUID
    attempt: int = Field(ge=1)
    source: _ProposalSourceV1
    resource_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_version: str = Field(min_length=1, max_length=100)
    problem: str = Field(min_length=1, max_length=4_000)
    general_updates: list[str] = Field(min_length=1, max_length=16)
    expected_effect: str = Field(min_length=1, max_length=4_000)
    evidence: list[EvidenceV1] = Field(min_length=1, max_length=32)
    calls: list[RecoveryCallV1] = Field(min_length=1, max_length=8)
    verification: list[VerificationCheckV1] = Field(min_length=1, max_length=8)
    retry: _SealedRetryV1
    authority_required: Literal["human"]
    global_effect: bool

    _nonblank = field_validator("registry_version", "problem", "expected_effect")(_text)

    @field_validator("general_updates")
    @classmethod
    def _proposal_updates(cls, value: list[str]) -> list[str]:
        return [_text(item) for item in value]

    @model_validator(mode="after")
    def _unique_broker_ids(self) -> "RecoveryProposalV1":
        if len({call.call_id for call in self.calls}) != len(self.calls) or len({check.check_id for check in self.verification}) != len(self.verification):
            raise ValueError("call and verification IDs must be unique")
        return self


class BaselineRepairAdmissionV1(_StrictModel):
    process_type: str
    gate_id: UUID
    gate_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("process_type")
    @classmethod
    def _process_type(cls, value: str) -> str:
        if value not in PROCESS_TYPES:
            raise ValueError("unsupported process type")
        return value


class RecoveryApplyDispatchContext(_StrictModel):
    purpose: Literal["apply"]
    proposal_id: UUID
    call_id: str = Field(min_length=1, max_length=100)
    authority_decision_id: UUID
    baseline_repair_admission: BaselineRepairAdmissionV1 | None = None

    _call = field_validator("call_id")(_text)


class RecoveryRetryDispatchContext(_StrictModel):
    purpose: Literal["retry"]
    proposal_id: UUID
    approved_retry_ordinal: int = Field(ge=1)
    authority_decision_id: UUID


RecoveryDispatchContext = Annotated[
    RecoveryApplyDispatchContext | RecoveryRetryDispatchContext,
    Field(discriminator="purpose"),
]


_WORK_FUNCTIONS = frozenset({"planning", "investigation", "implementation", "review", "validation", "summarization", "management"})


def _agent_ref(value: str) -> str:
    if value.startswith("new:"):
        name = value[4:]
        if name and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name):
            return value
        raise ValueError("new agent refs must be new:<name>")
    try:
        return str(UUID(value))
    except ValueError:
        raise ValueError("agent ref must be a UUID or new:<name>") from None


class _AgentDefinitionV1(_StrictModel):
    name: str = Field(min_length=1, max_length=255)
    role: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=4_000)
    provider: str = Field(min_length=1, max_length=255)
    model: str = Field(min_length=1, max_length=255)
    system_prompt: str | None = Field(default=None, max_length=4_000)
    adapter_type: Literal["api", "cli", "routine"] = "api"
    cli_runtime: str | None = Field(default=None, max_length=255)
    capabilities: list[str] = Field(default_factory=list, max_length=128)
    config: dict[str, Any] = Field(default_factory=dict)

    _text_fields = field_validator("name", "role", "provider", "model")(_text)
    _nullable_text_fields = field_validator("description", "system_prompt", "cli_runtime")(_nullable_text)
    @field_validator("capabilities")
    @classmethod
    def _capabilities(cls, value: list[str]) -> list[str]:
        return _capability_texts(value)

    @field_validator("config")
    @classmethod
    def _config(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_unsafe_keys(_safe_json(value))


class _AgentUpdateV1(_StrictModel):
    name: str | None = Field(default=None, max_length=255)
    role: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=4_000)
    provider: str | None = Field(default=None, max_length=255)
    model: str | None = Field(default=None, max_length=255)
    system_prompt: str | None = Field(default=None, max_length=4_000)
    adapter_type: Literal["api", "cli", "routine"] | None = None
    cli_runtime: str | None = Field(default=None, max_length=255)
    capabilities: list[str] | None = Field(default=None, max_length=128)
    config: dict[str, Any] | None = None
    is_active: Literal[True] | None = None

    _text_fields = field_validator("name", "role", "description", "provider", "model", "system_prompt", "cli_runtime")(_nullable_text)
    @field_validator("capabilities")
    @classmethod
    def _capabilities(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return _capability_texts(value)

    @field_validator("config")
    @classmethod
    def _config(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return _reject_unsafe_keys(_safe_json(value)) if value is not None else None

    @model_validator(mode="after")
    def _changed(self) -> "_AgentUpdateV1":
        if not self.model_fields_set:
            raise ValueError("update set must not be empty")
        return self


class _CreateAgentV1(_StrictModel):
    operation: Literal["create"]
    ref: str
    definition: _AgentDefinitionV1

    _ref = field_validator("ref")(_agent_ref)

    @model_validator(mode="after")
    def _new_ref(self) -> "_CreateAgentV1":
        if not self.ref.startswith("new:"):
            raise ValueError("created agents require a new:<name> ref")
        return self


class _UpdateAgentV1(_StrictModel):
    operation: Literal["update"]
    agent_id: UUID
    set: _AgentUpdateV1


class _EligibilityV1(_StrictModel):
    agent_ref: str
    work_functions: list[str] = Field(min_length=1, max_length=7)

    _ref = field_validator("agent_ref")(_agent_ref)

    @field_validator("work_functions")
    @classmethod
    def _functions(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value) or any(item not in _WORK_FUNCTIONS for item in value):
            raise ValueError("unsupported work function")
        return value


class _AssignmentV1(_StrictModel):
    work_function: str
    agent_ref: str

    _ref = field_validator("agent_ref")(_agent_ref)

    @field_validator("work_function")
    @classmethod
    def _function(cls, value: str) -> str:
        if value not in _WORK_FUNCTIONS:
            raise ValueError("unsupported work function")
        return value


class _AgentManagerV1(_StrictModel):
    kind: Literal["agent"]
    agent_ref: str
    _ref = field_validator("agent_ref")(_agent_ref)


class _HumanManagerV1(_StrictModel):
    kind: Literal["human"]
    user_id: UUID


class _NoManagerV1(_StrictModel):
    kind: Literal["none"]


class _HierarchyV1(_StrictModel):
    manager: Annotated[_AgentManagerV1 | _HumanManagerV1 | _NoManagerV1, Field(discriminator="kind")]
    team_leads: list[str] = Field(max_length=128)
    contributors: list[str] = Field(max_length=128)
    reviewers: list[str] = Field(max_length=128)
    validators: list[str] = Field(max_length=128)
    specialists: list[str] = Field(max_length=128)

    @field_validator("team_leads", "contributors", "reviewers", "validators", "specialists")
    @classmethod
    def _members(cls, value: list[str]) -> list[str]:
        return [_agent_ref(member) for member in value]


class _DocumentedGapV1(_StrictModel):
    code: str = Field(min_length=1, max_length=255)
    rationale: str = Field(min_length=1, max_length=4_000)
    _text = field_validator("code", "rationale")(_text)


class TeamConfigurationApplyV1(_StrictModel):
    agents: list[Annotated[_CreateAgentV1 | _UpdateAgentV1, Field(discriminator="operation")]] = Field(max_length=128)
    eligibility: list[_EligibilityV1] = Field(max_length=128)
    assignments: list[_AssignmentV1] = Field(max_length=128)
    hierarchy: _HierarchyV1
    documented_gaps: list[_DocumentedGapV1] = Field(max_length=128)

    @model_validator(mode="before")
    @classmethod
    def _safe_aggregate(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        agents = value.get("agents")
        if isinstance(agents, list):
            for agent in agents:
                if not isinstance(agent, dict):
                    continue
                definition = agent.get("definition")
                update = agent.get("set")
                if isinstance(definition, dict) and "config" in definition:
                    _safe_json(definition["config"])
                if isinstance(update, dict) and "config" in update:
                    _safe_json(update["config"])
                if "config" in agent:
                    _safe_json(agent["config"])
        return value


class _ApplyExecutionIdentityV1(_StrictModel):
    proposal_id: UUID
    call_id: str = Field(min_length=1, max_length=100)
    authority_decision_id: UUID
    purpose: Literal["apply"]

    _call = field_validator("call_id")(_text)


class ToolReceiptV1(_StrictModel):
    proposal_id: UUID
    call_id: str = Field(min_length=1, max_length=100)
    tool: str = Field(min_length=1, max_length=255)
    wrapper_action_id: UUID
    action_id: UUID
    idempotency_key: str = Field(min_length=1, max_length=512)
    canonical_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_identity: _ApplyExecutionIdentityV1
    status: Literal["applied", "waiting", "failed", "effect_unknown"]
    target_refs: list[dict[str, Any]] = Field(min_length=1, max_length=64)
    observed_post_state: Any
    evidence: list[EvidenceV1] = Field(min_length=1, max_length=32)
    error_code: str | None = Field(default=None, max_length=255)
    error: str | None = Field(default=None, max_length=4_000)
    wait: _WakeV1 | None = None

    _nonblank = field_validator("call_id", "tool", "idempotency_key")(_text)
    _targets = field_validator("target_refs")(_safe_json)
    _post_state = field_validator("observed_post_state")(_safe_json)

    @field_validator("error_code", "error")
    @classmethod
    def _safe_error(cls, value: str | None) -> str | None:
        return _text(value) if value is not None else value

    @model_validator(mode="after")
    def _wait_state(self) -> "ToolReceiptV1":
        if (self.status == "waiting") != (self.wait is not None):
            raise ValueError("waiting receipts require a wait and terminal receipts forbid one")
        return self


class VerificationResultV1(_StrictModel):
    check_id: str = Field(min_length=1, max_length=100)
    status: Literal["passed", "pending", "failed"]
    code: str = Field(min_length=1, max_length=255)
    evidence: list[EvidenceV1] = Field(min_length=1, max_length=32)
    wake: _WakeV1 | None = None

    _nonblank = field_validator("check_id", "code")(_text)

    @model_validator(mode="after")
    def _pending_wakes(self) -> "VerificationResultV1":
        if self.status == "pending" and self.wake is None:
            raise ValueError("pending verification requires a wake")
        if self.status != "pending" and self.wake is not None:
            raise ValueError("only pending verification may include a wake")
        return self


class RecoveryProposalResponse(_StrictModel):
    kind: Literal["recovery_proposal_v1"]
    proposal_id: UUID
    version: int = Field(ge=1)
    status: Literal["pending_approval", "applying", "verifying", "retrying", "resolved", "rejected", "failed", "unsupported"]
    problem: str = Field(min_length=1, max_length=4_000)
    general_updates: list[str] = Field(min_length=1, max_length=16)
    after_approval: str = Field(min_length=1, max_length=4_000)
    global_effect: bool
    exact_details: dict[str, Any] | None = None
    evidence: list[EvidenceV1] | None = Field(default=None, max_length=32)
    error: str | None = Field(default=None, max_length=4_000)
    source_blocker_id: UUID | None = None
    source_action_id: UUID | None = None
    source_decision_id: UUID | None = None

    _nonblank = field_validator("problem", "after_approval")(_text)

    @field_validator("general_updates")
    @classmethod
    def _response_updates(cls, value: list[str]) -> list[str]:
        return [_text(item) for item in value]

    @field_validator("error")
    @classmethod
    def _safe_error(cls, value: str | None) -> str | None:
        return _text(value) if value is not None else value

    _details = field_validator("exact_details")(_safe_json)
