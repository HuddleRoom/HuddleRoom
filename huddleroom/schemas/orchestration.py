import json
import uuid
from decimal import Decimal, DecimalException
from datetime import datetime, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter
from pydantic import (
    BaseModel, ConfigDict, Field, StrictInt, ValidationInfo, computed_field,
    field_validator, model_validator,
)
from pydantic_core import PydanticCustomError

from huddleroom.config import settings
from huddleroom.schemas.agent import AgentCreate


ROADMAP_BUDGET_DIMENSIONS = frozenset({"max_tokens", "max_turns", "max_hours"})
MAX_BUDGET_EXPONENT = 1000


def _require_nonblank_reason(value: str) -> str:
    """Strip whitespace and ensure reason is not blank."""
    text_value = value.strip()
    if not text_value:
        raise ValueError("reason is required")
    return text_value


def public_process_outputs(outputs: dict | None, process_type: str | None = None) -> dict:
    """Return process outputs without persisted LM retry inputs or diagnostics."""
    public = dict(outputs or {})
    checkpoint = public.pop("_lm_retry", None)
    for private_key in ("error", "retryable", "semantic_error"):
        public.pop(private_key, None)
    available = valid_lm_retry_checkpoint(checkpoint, process_type)
    public["lm_retry"] = {
        "available": available,
        "kind": checkpoint.get("kind") if isinstance(checkpoint, dict) else None,
        "warning_id": checkpoint.get("warning_id") if isinstance(checkpoint, dict) else None,
        "model": settings.orchestration_model if available else None,
        "hint": (
            f"The current orchestration model ({settings.orchestration_model}) could not "
            "produce a valid structured response; self-repair exhausted after 3 attempts. "
            "Consider upgrading settings.orchestration_model to a stronger model."
        ) if available else None,
    }
    return public


def _is_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _valid_semantic_payload(value: Any) -> bool:
    if not (
        isinstance(value, dict)
        and value.get("status") in {"approved", "improvement_proposed"}
        and _is_string_list(value.get("problems"))
        and isinstance(value.get("reason"), str)
        and _is_string_list(value.get("approved_work_functions"))
    ):
        return False
    if value["status"] == "improvement_proposed":
        return isinstance(value.get("proposed_description"), str) and isinstance(value.get("proposed_persona"), str)
    return value.get("proposed_description") is None and value.get("proposed_persona") is None


def _valid_deterministic_assessment(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if not isinstance(value.get("fit_summary"), str) or not all(
        _is_string_list(value.get(key))
        for key in (
            "proposed_work_functions", "strengths", "risks", "recommended_changes",
            "approved_for_work_functions",
        )
    ):
        return False
    warnings = value.get("warnings")
    return isinstance(warnings, list) and all(
        isinstance(warning, dict)
        and all(isinstance(warning.get(key), str) for key in ("warning_type", "severity", "message"))
        for warning in warnings
    )


def _valid_agent_snapshot(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    required_strings = ("name", "role", "provider", "model", "adapter_type")
    optional_strings = ("description", "system_prompt", "cli_runtime")
    return (
        all(key in value and isinstance(value[key], str) for key in required_strings)
        and all(key in value and (value[key] is None or isinstance(value[key], str)) for key in optional_strings)
        and isinstance(value.get("capabilities"), list)
        and isinstance(value.get("config"), dict)
        and isinstance(value.get("is_active"), bool)
    )


def _valid_agent_target(value: Any) -> bool:
    if not isinstance(value, dict) or not _is_uuid(value.get("agent_id")):
        return False
    if not (
        _valid_agent_snapshot(value.get("agent_snapshot"))
        and isinstance(value.get("goal_snapshot"), dict)
        and _valid_deterministic_assessment(value.get("deterministic_assessment"))
        and _is_string_list(value.get("candidate_work_functions"))
    ):
        return False
    load = value.get("load_snapshot")
    return isinstance(load, dict) and all(
        isinstance(load.get(key), int) and not isinstance(load.get(key), bool) and load[key] >= 0
        for key in ("active_sessions", "active_tasks", "outcome_hints")
    )


def valid_lm_retry_checkpoint(checkpoint: Any, process_type: str | None = None) -> bool:
    """Validate the private retry envelopes we can safely execute."""
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
        return False
    kind = checkpoint.get("kind")
    owners = {"goal_definition": "goal_analysis", "manager_selection": "manager_selection",
              "agent_definition_review": "agent_definition_review", "team_hierarchy": "team_hierarchy",
              "effectiveness_review": "effectiveness_review"}
    if process_type is not None and (process_type not in owners or kind != owners[process_type]):
        return False
    request = checkpoint.get("request")
    if not isinstance(request, dict) or not isinstance(request.get("model"), str):
        return False
    if not isinstance(request.get("messages"), list) or not isinstance(request.get("response_format"), dict):
        return False
    if kind == "goal_analysis":
        continuation = checkpoint.get("continuation")
        working_goal = continuation.get("working_goal") if isinstance(continuation, dict) else None
        return (
            isinstance(continuation, dict)
            and isinstance(continuation.get("clarification_round"), int)
            and isinstance(continuation.get("context"), dict)
            and isinstance(working_goal, dict)
            and isinstance(working_goal.get("objective"), str)
            and isinstance(working_goal.get("success_criteria"), list)
        )
    if kind == "agent_definition_review":
        targets, completed, cursor = checkpoint.get("targets"), checkpoint.get("completed"), checkpoint.get("cursor")
        if not (
            isinstance(checkpoint.get("fingerprint"), str)
            and isinstance(checkpoint.get("coverage_fingerprint"), str)
            and isinstance(checkpoint.get("model"), str)
            and isinstance(targets, list) and all(_valid_agent_target(target) for target in targets)
            and isinstance(completed, dict) and isinstance(cursor, int) and 0 <= cursor < len(targets)
        ):
            return False
        target_ids = [target["agent_id"] for target in targets]
        return set(completed) == set(target_ids[:cursor]) and all(
            _valid_semantic_payload(payload) for payload in completed.values()
        )
    if kind == "manager_selection":
        messages = request["messages"]
        if len(messages) < 2 or any(
            not isinstance(message, dict)
            or message.get("role") != role
            or not isinstance(message.get("content"), str)
            or not message["content"]
            for message, role in zip(messages[:2], ("system", "user"))
        ):
            return False
        try:
            payload = json.loads(messages[1]["content"])
        except (TypeError, ValueError):
            return False
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        recommendation = payload.get("deterministic_recommendation") if isinstance(payload, dict) else None
        return (
            set(checkpoint) == {"kind", "version", "request"}
            and isinstance(candidates, list)
            and all(isinstance(candidate, dict) and isinstance(candidate.get("key"), str)
                    for candidate in candidates)
            and isinstance(recommendation, str)
            and recommendation in {candidate["key"] for candidate in candidates}
        )
    if kind == "team_hierarchy":
        messages = request["messages"]
        if len(messages) < 2 or any(
            not isinstance(message, dict)
            or message.get("role") != role
            or not isinstance(message.get("content"), str)
            or not message["content"]
            for message, role in zip(messages[:2], ("system", "user"))
        ):
            return False
        try:
            payload = json.loads(messages[1]["content"])
        except (TypeError, ValueError):
            return False
        return (
            set(checkpoint) == {"kind", "version", "request"}
            and isinstance(payload, dict)
            and set(payload) in ({
                "schema_version", "goal", "selected_manager",
                "required_work_functions", "agents",
            }, {
                "schema_version", "goal", "selected_manager",
                "required_work_functions", "agents", "resolved_proposal_exclusions",
            })
            and payload.get("schema_version") == 1
            and isinstance(payload.get("goal"), dict)
            and isinstance(payload.get("selected_manager"), dict)
            and _is_string_list(payload.get("required_work_functions"))
            and isinstance(payload.get("agents"), list)
            and isinstance(payload.get("resolved_proposal_exclusions", []), list)
        )
    if kind == "effectiveness_review":
        messages = request["messages"]
        if len(messages) < 2 or any(
            not isinstance(message, dict) or message.get("role") != role
            or not isinstance(message.get("content"), str) or not message["content"]
            for message, role in zip(messages[:2], ("system", "user"))
        ):
            return False
        try:
            payload = json.loads(messages[1]["content"])
        except (TypeError, ValueError):
            return False
        return (
            set(checkpoint) == {"kind", "version", "request"}
            and isinstance(payload, dict)
            and payload.get("schema_version") == 1
            and isinstance(payload.get("goal"), dict)
            and isinstance(payload.get("triggers"), list)
            and isinstance(payload.get("checks"), list)
        )
    return False


class OrchestrationGoalCreate(BaseModel):
    objective: str = Field(min_length=1)
    original_request: str | None = Field(default=None, min_length=1)
    success_criteria: list[dict] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)
    budget: dict = Field(default_factory=dict)
    explicit_multi_work_function: bool = False

    @field_validator("original_request")
    @classmethod
    def _validate_original_request(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text_value = value.strip()
        if not text_value:
            raise ValueError("original request is required")
        return text_value


def _budget_amounts(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("budget dimensions are required")
    normalized = {}
    for key, raw in value.items():
        if key not in ROADMAP_BUDGET_DIMENSIONS or isinstance(raw, bool):
            raise ValueError(f"invalid budget dimension '{key}'")
        try:
            amount = Decimal(str(raw))
            if not amount.is_finite() or amount < 0 or abs(amount.adjusted()) > MAX_BUDGET_EXPONENT:
                raise ValueError
            normalized[key] = format(amount.normalize(), "f")
        except (DecimalException, ValueError) as exc:
            raise ValueError(f"invalid amount for '{key}'") from exc
    return normalized


class OrchestrationContinuousActivation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cron: str = Field(min_length=1)
    timezone: str = Field(min_length=1)
    missed_slots: Literal["coalesce"] = "coalesce"

    @model_validator(mode="after")
    def validate_schedule(self):
        if not croniter.is_valid(self.cron):
            raise ValueError("invalid cron expression")
        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("unknown timezone") from exc
        return self


class OrchestrationContinuousFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter_type: Literal["schedule"]
    enabled: bool = True


def _trimmed(value: str, maximum: int, field: str) -> str:
    value = value.strip()
    if not value or len(value) > maximum:
        raise ValueError(f"{field} is invalid")
    return value


class DiscoveryCandidateOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin_key: str
    objective: str
    source_refs: list[str] = Field(min_length=1, max_length=16)

    _key = field_validator("origin_key")(lambda value: _trimmed(value, 255, "origin_key"))
    _objective = field_validator("objective")(lambda value: _trimmed(value, 4000, "objective"))

    @field_validator("source_refs")
    @classmethod
    def validate_refs(cls, values):
        return [_trimmed(value, 2000, "source_ref") for value in values]


class DiscoveryBatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    candidates: list[DiscoveryCandidateOutput]

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("schema_version must be integer 1")
        return value


def parse_discovery_batch(raw: str, candidate_limit: int) -> DiscoveryBatchOutput:
    batch = DiscoveryBatchOutput.model_validate_json(raw)
    origins = [candidate.origin_key for candidate in batch.candidates]
    if len(batch.candidates) > candidate_limit or len(set(origins)) != len(origins):
        raise ValueError("invalid discovery batch")
    return batch


class OrchestrationContinuousChildTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1)
    success_criteria: list[dict] = Field(min_length=1)
    constraints: dict = Field(default_factory=dict)


class OrchestrationContinuousRollingBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_seconds: StrictInt = Field(gt=0)
    limits: dict[str, str]

    @field_validator("limits", mode="before")
    @classmethod
    def validate_limits(cls, value):
        return _budget_amounts(value)


class OrchestrationDiscoveryBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_cycle: dict[str, str]
    rolling: OrchestrationContinuousRollingBudget

    @field_validator("per_cycle", mode="before")
    @classmethod
    def validate_per_cycle(cls, value):
        return _budget_amounts(value)

    @model_validator(mode="after")
    def validate_budget(self):
        if set(self.per_cycle) != set(self.rolling.limits):
            raise ValueError("discovery budget dimensions differ")
        if any(
            Decimal(self.per_cycle[key]) > Decimal(self.rolling.limits[key])
            for key in self.per_cycle
        ):
            raise ValueError("invalid discovery budget")
        return self


class OrchestrationDiscoverySource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: uuid.UUID
    work_function: str
    instructions: str
    source_filter: str
    origin_key_rule: str
    access_requirements: list[str] = Field(default_factory=list, max_length=16)
    max_candidates: StrictInt = Field(ge=1, le=100)
    timeout_seconds: StrictInt = Field(ge=1, le=86400)
    budget: OrchestrationDiscoveryBudget

    _function = field_validator("work_function")(lambda value: _trimmed(value, 100, "work_function"))
    _instructions = field_validator("instructions")(lambda value: _trimmed(value, 8000, "instructions"))
    _filter = field_validator("source_filter")(lambda value: _trimmed(value, 4000, "source_filter"))
    _rule = field_validator("origin_key_rule")(lambda value: _trimmed(value, 1000, "origin_key_rule"))
    _access = field_validator("access_requirements")(
        lambda values: [_trimmed(value, 200, "access_requirement") for value in values]
    )


class OrchestrationContinuousStopCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["manual", "deadline", "max_cycles"]
    deadline: datetime | None = None
    max_cycles: StrictInt | None = Field(default=None, gt=0)

    @field_validator("deadline")
    @classmethod
    def require_aware_deadline(cls, value):
        if value is not None and value.tzinfo is None:
            raise ValueError("deadline must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_condition(self):
        if self.mode == "deadline" and self.deadline is None:
            raise ValueError("deadline stop requires deadline")
        if self.mode == "max_cycles" and self.max_cycles is None:
            raise ValueError("max_cycles stop requires max_cycles")
        if self.mode != "deadline" and self.deadline is not None:
            raise ValueError("deadline is valid only for deadline stop")
        if self.mode != "max_cycles" and self.max_cycles is not None:
            raise ValueError("max_cycles is valid only for max_cycles stop")
        return self


class OrchestrationContinuousPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activation: OrchestrationContinuousActivation
    adapter_filter: OrchestrationContinuousFilter | None = None
    discovery_source: OrchestrationDiscoverySource | None = None
    cycle_mode: Literal["direct", "discovery"]
    child_template: OrchestrationContinuousChildTemplate
    per_case_budget: dict[str, str]
    response_target_seconds: StrictInt = Field(gt=0)
    max_active_cases: StrictInt = Field(gt=0)
    max_backlog: StrictInt = Field(gt=0)
    rolling_budget: OrchestrationContinuousRollingBudget
    stop_condition: OrchestrationContinuousStopCondition

    @field_validator("per_case_budget", mode="before")
    @classmethod
    def validate_per_case_budget(cls, value):
        return _budget_amounts(value)

    @model_validator(mode="after")
    def validate_dimensions(self):
        if set(self.per_case_budget) != set(self.rolling_budget.limits):
            raise ValueError("per-case and rolling budgets must use identical dimensions")
        if any(
            Decimal(self.per_case_budget[key]) > Decimal(self.rolling_budget.limits[key])
            for key in self.per_case_budget
        ):
            raise ValueError("per-case budget cannot exceed the rolling limit")
        if self.cycle_mode == "direct":
            if self.adapter_filter is None or self.discovery_source is not None:
                raise ValueError("direct mode requires only adapter_filter")
        elif self.discovery_source is None or self.adapter_filter is not None:
            raise ValueError("discovery mode requires only discovery_source")
        if (
            self.discovery_source is not None
            and set(self.discovery_source.budget.per_cycle) != set(self.per_case_budget)
        ):
            raise ValueError("discovery and case budgets must use identical dimensions")
        return self


class OrchestrationContinuousPolicyUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: OrchestrationContinuousPolicy


class OrchestrationContinuousStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _require_nonblank_reason(value)


class OrchestrationSupersedeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_type: Literal["outcome", "roadmap", "continuous"] = "outcome"


class OrchestrationGateOverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gate_id: uuid.UUID
    decision: Literal["accept", "reject"]
    reason: str = Field(min_length=1)
    evidence_metadata: dict = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _require_nonblank_reason(value)


class OrchestrationGoalWeightOverrideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight: Literal["trivial", "standard", "substantial"]
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _require_nonblank_reason(value)


class OrchestrationDelegationContract(BaseModel):
    goal_id: uuid.UUID
    run_id: uuid.UUID
    action_id: uuid.UUID
    agent_id: uuid.UUID
    work_function: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    deliverable: str = Field(min_length=1)
    forbidden_work: list[str] = Field(default_factory=list)
    success_evidence: list[str] = Field(default_factory=list)
    budget: dict = Field(default_factory=dict)
    report_schema: dict = Field(default_factory=dict)
    parent_task_id: uuid.UUID | None = None
    orchestrator_context: dict = Field(default_factory=dict)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Keep the persisted contract canonical when no context is present."""
        payload = super().model_dump(**kwargs)
        if not self.orchestrator_context:
            payload.pop("orchestrator_context", None)
        return payload


class OrchestrationPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    title: str | None = None
    work_function: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    deliverable: str = Field(min_length=1)
    agent_id: uuid.UUID | None = None
    inputs: list[str] = Field(default_factory=list)
    forbidden_work: list[str] = Field(default_factory=list)
    success_evidence: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    required_evidence: dict = Field(
        default_factory=lambda: {"required_source_types": ["task"], "min_count": 1}
    )
    success_criterion_keys: list[str] = Field(default_factory=list)
    # Predecessor plan-item ids this item depends on. Accepts either the bare
    # plan item id or its gate key ("plan_item:<id>") -- see
    # OrchestrationService._item_dependencies_accepted.
    depends_on: list[str] = Field(default_factory=list)

    @field_validator("id", "title", "work_function", "scope", "deliverable", "agent_id", mode="before")
    @classmethod
    def _strip_text(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            text_value = value.strip()
            return text_value or None
        return value

    @field_validator(
        "inputs", "forbidden_work", "success_evidence", "required_capabilities", "depends_on",
        "success_criterion_keys", mode="before"
    )
    @classmethod
    def _clean_text_list(cls, value):
        if value is None:
            return []
        if not isinstance(value, (list, tuple, set)):
            return value
        return list(dict.fromkeys(item_text for item in value if (item_text := str(item).strip())))


class OrchestrationRoadmapPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_key: str = Field(min_length=1, max_length=80)
    unit_type: Literal["task", "goal"]
    title: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    mutates_shared_state: bool = True
    staging_boundary: dict | None = None
    allocation: dict[str, Decimal] = Field(default_factory=dict)
    work_function: str | None = None
    scope: str | None = None
    deliverable: str | None = None
    agent_id: uuid.UUID | None = None
    inputs: list[str] = Field(default_factory=list)
    forbidden_work: list[str] = Field(default_factory=list)
    success_evidence: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    objective: str | None = None
    success_criteria: list[dict] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)

    @field_validator("item_key", "title", "work_function", "scope", "deliverable", "objective", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator("depends_on", "inputs", "forbidden_work", "success_evidence", "required_capabilities", mode="before")
    @classmethod
    def clean_list(cls, value):
        if value is None:
            return []
        if not isinstance(value, (list, tuple, set)):
            return value
        return list(dict.fromkeys(text for item in value if (text := str(item).strip())))

    @field_validator("allocation", mode="before")
    @classmethod
    def reject_boolean_allocations(cls, value):
        if isinstance(value, dict) and any(isinstance(amount, bool) for amount in value.values()):
            raise ValueError("allocation values must be numbers, not booleans")
        return value

    @model_validator(mode="after")
    def validate_unit(self):
        if self.mutates_shared_state and self.staging_boundary is not None:
            boundary = self.staging_boundary
            if boundary.get("type") not in {"git_branch", "git_worktree", "transaction", "draft"}:
                raise ValueError("mutable work requires a supported staging boundary")
            if boundary.get("reversible") is not True or not str(boundary.get("identifier", "")).strip():
                raise ValueError("mutable work requires a named reversible staging boundary")
        if self.unit_type == "task":
            if not all((self.work_function, self.scope, self.deliverable)):
                raise ValueError("task item requires work_function, scope, and deliverable")
            if self.objective is not None or self.success_criteria or self.constraints or self.allocation:
                raise ValueError("task item cannot define child goal fields or allocation")
        else:
            if not self.objective or not self.success_criteria or not self.allocation:
                raise ValueError("goal item requires objective, success_criteria, and allocation")
            if any((self.work_function, self.scope, self.deliverable, self.agent_id)):
                raise ValueError("goal item cannot define task-only fields")
        return self


class OrchestrationRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    goal_id: uuid.UUID
    status: str
    phase: str
    baseline_authorized: bool = True
    cycle_key: str | None = None
    # ponytail: default lets the response validate before `condition` (a
    # derived, non-column value) is injected by the router; always
    # overwritten before reaching a client.
    condition: str = ""
    event_cursor: int | None
    plan_state: dict
    active_blockers: list
    budget_state: dict
    retry_state: dict
    started_at: datetime
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class OrchestrationDecisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    decision_type: str
    input_snapshot: dict
    llm_output: Any
    parsed_decision: dict
    validator_status: str
    rejection_reason: str | None
    reason: str | None
    created_at: datetime
    updated_at: datetime


class OrchestrationActionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    decision_id: uuid.UUID | None
    idempotency_key: str
    action_type: str
    request: dict
    target_type: str | None
    target_id: uuid.UUID | None
    status: str
    error: str | None
    created_at: datetime
    updated_at: datetime


class OrchestrationAgentSuggestionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    missing_work_function: str
    reason: str
    suggested_role: str | None
    suggested_capabilities: list
    suggested_adapter_type: str | None
    suggested_model: str | None
    suggested_system_prompt_outline: str | None
    status: str
    created_at: datetime
    updated_at: datetime


class OrchestrationGateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    success_criterion_key: str
    gate_type: str
    required_evidence: dict
    status: str
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime
    accepted_at: datetime | None
    failed_at: datetime | None


class OrchestrationEvidenceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    run_id: uuid.UUID
    gate_id: uuid.UUID
    source_type: str
    source_id: uuid.UUID | None
    observed_event_id: uuid.UUID | None
    producer_agent_id: uuid.UUID | None
    verdict: str
    evidence_metadata: dict
    created_at: datetime
    updated_at: datetime


class OrchestrationGoalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    project_id: uuid.UUID
    objective: str
    original_request: str
    success_criteria: list
    constraints: dict
    budget: dict
    orchestrator_context: dict = Field(default_factory=dict)
    status: str
    weight: str
    weight_overridden_by: str | None = None
    manager_agent_id: uuid.UUID | None = None
    manager_user_id: uuid.UUID | None = None
    authority_model: str | None = None
    goal_type: str
    supersedes_goal_id: uuid.UUID | None = None
    parent_goal_id: uuid.UUID | None = None
    roadmap_version_id: uuid.UUID | None = None
    roadmap_item_key: str | None = None
    continuous_policy: dict | None = None
    continuous_state: dict | None = None
    continuous_origin_key: str | None = None
    created_by_user_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime
    needs_you_count: int = 0


class OrchestrationRoadmapVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: int
    plan_artifact_id: uuid.UUID
    fingerprint: str
    created_at: datetime


class OrchestrationRoadmapItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    item_key: str
    unit_type: str
    task_id: uuid.UUID | None
    child_goal_id: uuid.UUID | None
    gate_id: uuid.UUID
    released_at: datetime
    completed_at: datetime | None


class OrchestrationRoadmapBudgetResponse(BaseModel):
    caps: dict[str, str]
    direct_spend: dict[str, str]
    settled_child_spend: dict[str, str]
    active_reservations: dict[str, str]
    active_commitments: dict[str, str]
    remaining: dict[str, str]


class OrchestrationSupervisionResponse(BaseModel):
    """Read-only, durable execution facts for the goal detail surface."""

    condition: Literal[
        "working", "waiting", "needs_you", "needs_attention", "paused",
        "stopped", "cancelled", "completed",
    ]
    operation: str
    next_action: str
    rationale: str
    criterion: dict | None = None
    verified_progress: list[dict] = Field(default_factory=list)
    useful_learning: list[dict] = Field(default_factory=list)
    accepted_evidence: list[OrchestrationEvidenceResponse] = Field(default_factory=list)
    workers: list[dict] = Field(default_factory=list)
    waits: list[dict] = Field(default_factory=list)
    recovery_history: list[dict] = Field(default_factory=list)
    pending_direction: "OrchestrationAuthorityDecisionResponse | None" = None
    budget: dict = Field(default_factory=dict)
    transition: dict | None = None


class OrchestrationGoalDetailResponse(BaseModel):
    goal: OrchestrationGoalResponse
    run: OrchestrationRunResponse | None = None
    decisions_count: int = 0
    decisions: list[OrchestrationDecisionResponse] = Field(default_factory=list)
    actions_count: int = 0
    actions: list[OrchestrationActionResponse] = Field(default_factory=list)
    gates_count: int = 0
    gates: list[OrchestrationGateResponse] = Field(default_factory=list)
    evidence_count: int = 0
    evidence: list[OrchestrationEvidenceResponse] = Field(default_factory=list)
    agent_suggestions_count: int = 0
    agent_suggestions: list[OrchestrationAgentSuggestionResponse] = Field(default_factory=list)
    # The dashboard merges typed decisions/actions by created_at.
    timeline: list = Field(default_factory=list)
    roadmap_version: OrchestrationRoadmapVersionResponse | None = None
    roadmap_items: list[OrchestrationRoadmapItemResponse] | None = None
    children: list[uuid.UUID] | None = None
    budget_summary: OrchestrationRoadmapBudgetResponse | None = None
    supervision: OrchestrationSupervisionResponse | None = None


class OrchestrationProcessRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    goal_id: uuid.UUID
    run_id: uuid.UUID | None
    process_type: str
    process_version: int
    status: str
    trigger_reason: str
    input_snapshot: dict
    outputs: dict
    skipped_by: str | None
    override_reason: str | None
    superseded_by_id: uuid.UUID | None
    started_at: datetime
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @field_validator("outputs", mode="before")
    @classmethod
    def _public_outputs(cls, value: dict | None, info: ValidationInfo) -> dict:
        return public_process_outputs(value, info.data.get("process_type"))

    @field_validator("started_at", "completed_at", "created_at", "updated_at", mode="after")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class OrchestrationAgentReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    goal_id: uuid.UUID
    run_id: uuid.UUID | None
    agent_id: uuid.UUID | None
    source_process_run_id: uuid.UUID | None
    review_context: str | None
    proposed_work_functions: list
    definition_snapshot: dict
    fit_summary: str
    strengths: list
    risks: list
    recommended_changes: list
    approved_for_work_functions: list
    created_at: datetime
    updated_at: datetime


class OrchestrationWarningResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    goal_id: uuid.UUID
    run_id: uuid.UUID | None
    warning_type: str
    severity: str
    message: str
    source_process_run_id: uuid.UUID | None
    related_gate_id: uuid.UUID | None
    related_action_id: uuid.UUID | None
    related_agent_id: uuid.UUID | None
    source_agent_review_id: uuid.UUID | None
    related_authority_decision_id: uuid.UUID | None
    acknowledged_by: str | None
    acknowledged_at: datetime | None
    active: bool
    resolved_by: str | None
    resolved_reason: str | None
    resolved_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @computed_field  # type: ignore[misc]
    @property
    def blocks_completion(self) -> bool:
        if not self.active:
            return False
        return self.severity in ("blocker", "hard_stop") or self.acknowledged_at is None


class OrchestrationWarningReasonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _require_nonblank_reason(value)


class OrchestrationAuthorityDecisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    goal_id: uuid.UUID
    run_id: uuid.UUID | None
    decision_key: str
    title: str
    status: str
    authority: str
    runtime_identity: str | None = None
    contract_version: str | None = None
    continuation: dict | None = None
    continuation_action_id: uuid.UUID | None = None
    continuation_applied_at: datetime | None = None
    authority_agent_id: uuid.UUID | None
    source_process_run_id: uuid.UUID | None
    question: str
    context: str | None
    options: list
    recommendation: str | None
    consequences: str | None
    selected_option: str | None
    reason: str | None
    decided_by_user_id: uuid.UUID | None
    decided_by_agent_id: uuid.UUID | None
    overrides_recommendation: bool
    created_warning_id: uuid.UUID | None
    related_gate_id: uuid.UUID | None
    related_action_id: uuid.UUID | None
    asked_at: datetime
    decided_at: datetime | None
    created_at: datetime
    updated_at: datetime


class OrchestrationDecisionAnswerResponse(BaseModel):
    decision: OrchestrationAuthorityDecisionResponse
    process: dict | None = None
    continuation_applied: bool = False


class OrchestrationCheckpointResponse(BaseModel):
    goal_id: uuid.UUID
    items: list[OrchestrationAuthorityDecisionResponse]
    deferred_count: int
    max_questions: int


class OrchestrationDecisionAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_option: str = Field(min_length=1)
    contract_version: str | None = Field(default=None, min_length=1)
    reason: str | None = None
    edited_agent: AgentCreate | None = None
    edited_description: str | None = None
    edited_persona: str | None = None


class OrchestrationAgentDefinitionReviewBatchAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: uuid.UUID
    selected_option: str = Field(min_length=1)
    reason: str | None = None
    edited_description: str | None = None
    edited_persona: str | None = None


class OrchestrationAgentDefinitionReviewBatchAnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[OrchestrationAgentDefinitionReviewBatchAnswer] = Field(min_length=1)


class OrchestrationAgentDefinitionReviewBatchAnswerResponse(BaseModel):
    decisions: list[OrchestrationAuthorityDecisionResponse]
    process: dict


class OrchestrationDecisionCancelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _require_nonblank_reason(value)


class OrchestrationProcessStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _require_nonblank_reason(value)


class OrchestrationProcessSkipRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        return _require_nonblank_reason(value)


class OrchestrationGoalDefinitionRecoverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["proceed", "another_round"]


class OrchestrationDebugStepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_type: Literal[
        "goal_definition",
        "manager_selection",
        "agent_definition_review",
        "team_hierarchy",
        "effectiveness_review",
        "goal_closeout",
    ]


class OrchestrationDebugRerunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_type: Literal[
        "goal_definition",
        "manager_selection",
        "agent_definition_review",
        "team_hierarchy",
        "effectiveness_review",
        "goal_closeout",
    ] | None = None


class OrchestrationConversationSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: uuid.UUID
    content: str

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        return value.strip()


class OrchestrationConversationErrorResponse(BaseModel):
    code: str


class OrchestrationConversationManifestSourceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    status: str
    freshness_at: str | None = None
    available: int
    included: int
    omitted: int
    truncated: bool
    references: list[str]


class OrchestrationConversationManifestResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str | None = None
    excluded_categories: list[str] = Field(default_factory=list)
    sources: list[OrchestrationConversationManifestSourceResponse] = Field(default_factory=list)
    truncated: bool = False


class OrchestrationConversationInvestigationSourceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reference: str
    operation: Literal["list", "read", "search"]
    status: Literal["included", "restricted", "unsafe", "binary", "too_large", "changed", "omitted_by_limit"]
    freshness_at: str | None = None
    truncated: bool = False


class OrchestrationConversationInvestigationReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: str
    uncertainty: str
    sources: list[str]


class OrchestrationConversationInvestigationResponse(BaseModel):
    investigation_id: uuid.UUID
    status: Literal[
        "pending", "running", "completed", "limited", "failed",
        "cancelled", "unavailable", "interrupted_unknown",
    ]
    objective: str
    attempt_count: int
    repair_count: int
    retry_count: int
    sources: list[OrchestrationConversationInvestigationSourceResponse] = Field(default_factory=list)
    report: OrchestrationConversationInvestigationReportResponse | None = None
    error: OrchestrationConversationErrorResponse | None = None
    started_at: datetime | None
    deadline_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


class OrchestrationConversationTurnResponse(BaseModel):
    message_id: uuid.UUID
    response_id: uuid.UUID
    client_request_id: uuid.UUID
    sequence: int
    actor_id: uuid.UUID
    content: str
    message_created_at: datetime
    status: str
    run_id: uuid.UUID | None
    answer: str | None
    error: OrchestrationConversationErrorResponse | None
    started_at: datetime | None
    deadline_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    context_version: str
    context_manifest: OrchestrationConversationManifestResponse
    investigation: OrchestrationConversationInvestigationResponse | None = None
    proposed_steering: "OrchestrationSteeringProposalResponse | None" = None
    feedback: "OrchestrationConversationFeedbackResponse | None" = None
    feedback_eligible: bool = False


class OrchestrationConversationFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: Literal["helpful", "not_helpful"]
    reason: Literal[
        "unanswered", "incorrect", "missing_context", "stale_context",
        "unclear", "too_limited", "other",
    ] | None = None

    @model_validator(mode="after")
    def _validate_rating_reason(self):
        if (self.rating == "helpful" and self.reason is None) or (
            self.rating == "not_helpful" and self.reason is not None
        ):
            return self
        raise PydanticCustomError(
            "feedback_reason_inconsistent", "Feedback rating and reason are inconsistent"
        )


class OrchestrationConversationFeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_id: uuid.UUID
    rating: Literal["helpful", "not_helpful"]
    reason: Literal[
        "unanswered", "incorrect", "missing_context", "stale_context",
        "unclear", "too_limited", "other",
    ] | None
    created_at: datetime


class _StrictLearningResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrchestrationConversationLearningLatencyResponse(_StrictLearningResponse):
    count: int
    invalid: int
    median: float | None
    p95: float | None


class OrchestrationConversationLearningWindowResponse(_StrictLearningResponse):
    start_at: datetime
    end_at: datetime
    generated_at: datetime
    goal_filtered: bool


class OrchestrationConversationLearningSampleResponse(_StrictLearningResponse):
    goals: int
    operators: int
    questions: int


class OrchestrationConversationStatusCountsResponse(_StrictLearningResponse):
    pending: int
    running: int
    completed: int
    failed: int
    interrupted_unknown: int


class OrchestrationInvestigationStatusCountsResponse(_StrictLearningResponse):
    pending: int
    running: int
    completed: int
    limited: int
    failed: int
    cancelled: int
    unavailable: int
    interrupted_unknown: int


class OrchestrationDossierSourceCountsResponse(_StrictLearningResponse):
    goal: int
    run: int
    accepted_plan: int
    decisions: int
    actions: int
    gates: int
    evidence: int
    processes: int
    warnings: int
    memory: int
    artifact: int
    agents: int
    prior_turns: int


class OrchestrationInvestigationOmissionCountsResponse(_StrictLearningResponse):
    restricted: int
    unsafe: int
    binary: int
    too_large: int
    changed: int
    omitted_by_limit: int


class OrchestrationProposalStatusCountsResponse(_StrictLearningResponse):
    proposed: int
    dismissed: int
    promoted: int


class OrchestrationSteeringStatusCountsResponse(_StrictLearningResponse):
    pending: int
    being_considered: int
    applied: int
    deferred: int
    rejected: int
    superseded: int
    needs_clarification: int
    withdrawn: int


class OrchestrationSteeringReasonCountsResponse(_StrictLearningResponse):
    submitted: int
    considering: int
    run_changed: int
    steering_ineligible: int
    target_already_started: int
    supersedes_required: int
    invalid_supersedes_request: int
    superseded: int
    applied: int
    withdrawn: int


class OrchestrationFeedbackReasonCountsResponse(_StrictLearningResponse):
    unanswered: int
    incorrect: int
    missing_context: int
    stale_context: int
    unclear: int
    too_limited: int
    other: int


class OrchestrationConversationLearningChatResponse(_StrictLearningResponse):
    status_counts: OrchestrationConversationStatusCountsResponse
    answered: int
    terminal_without_answer: int
    latency_seconds: OrchestrationConversationLearningLatencyResponse


class OrchestrationConversationLearningInvestigationResponse(_StrictLearningResponse):
    triggered: int
    accounted_terminal: int
    status_counts: OrchestrationInvestigationStatusCountsResponse
    attempts: int
    repairs: int
    retries: int
    accumulated_tokens: int
    latency_seconds: OrchestrationConversationLearningLatencyResponse


class OrchestrationConversationLearningContextResponse(_StrictLearningResponse):
    turns_truncated: int
    turns_with_omissions: int
    omitted_records: int
    truncated_source_counts: OrchestrationDossierSourceCountsResponse
    investigation_omission_status_counts: OrchestrationInvestigationOmissionCountsResponse


class OrchestrationConversationLearningSteeringResponse(_StrictLearningResponse):
    proposal_status_counts: OrchestrationProposalStatusCountsResponse
    request_status_counts: OrchestrationSteeringStatusCountsResponse
    request_reason_counts: OrchestrationSteeringReasonCountsResponse
    direct_requests: int
    proposal_derived_requests: int
    requests_with_result_actions: int
    result_links: int
    submit_to_considered_seconds: OrchestrationConversationLearningLatencyResponse
    submit_to_finished_seconds: OrchestrationConversationLearningLatencyResponse


class OrchestrationConversationLearningFeedbackResponse(_StrictLearningResponse):
    rated: int
    unrated_answered: int
    helpful: int
    not_helpful: int
    not_helpful_reason_counts: OrchestrationFeedbackReasonCountsResponse


class OrchestrationConversationLearningReportResponse(_StrictLearningResponse):
    window: OrchestrationConversationLearningWindowResponse
    sample: OrchestrationConversationLearningSampleResponse
    chat: OrchestrationConversationLearningChatResponse
    investigations: OrchestrationConversationLearningInvestigationResponse
    context_limits: OrchestrationConversationLearningContextResponse
    steering: OrchestrationConversationLearningSteeringResponse
    operator_feedback: OrchestrationConversationLearningFeedbackResponse


class OrchestrationSteeringSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_request_id: uuid.UUID
    directive: str = Field(min_length=1, max_length=4_000)
    target_type: Literal["goal", "plan_item", "task"]
    target_id: str = Field(min_length=1, max_length=255)
    scope: Literal["item", "run", "goal"] = "run"
    lifetime: Literal["selected_item", "remaining_current_run", "future_runs"] = "remaining_current_run"
    impact_summary: str = Field(min_length=1, max_length=1_000)
    source_proposal_id: uuid.UUID | None = None
    supersedes_request_id: uuid.UUID | None = None

    @field_validator("directive", "target_id", "impact_summary", mode="before")
    @classmethod
    def _strip_required(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        value = value.strip()
        if not value:
            raise ValueError("value is required")
        return value


class OrchestrationSteeringTransitionResponse(BaseModel):
    status: str
    reason_code: str
    actor: str
    created_at: datetime


class OrchestrationSteeringProposalResponse(BaseModel):
    proposal_id: uuid.UUID
    response_id: uuid.UUID
    status: Literal["proposed", "dismissed", "promoted"]
    directive: str
    target_type: Literal["goal", "plan_item", "task"]
    target_id: str
    scope: Literal["item", "run", "goal"]
    lifetime: Literal["selected_item", "remaining_current_run", "future_runs"]
    impact_summary: str
    dismissed_at: datetime | None
    promoted_request_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class OrchestrationSteeringRequestResponse(BaseModel):
    request_id: uuid.UUID
    client_request_id: uuid.UUID
    sequence: int
    directive: str
    target_type: Literal["goal", "plan_item", "task"]
    target_id: str
    scope: Literal["item", "run", "goal"]
    lifetime: Literal["selected_item", "remaining_current_run", "future_runs"]
    impact_summary: str
    source_proposal_id: uuid.UUID | None
    supersedes_request_id: uuid.UUID | None
    status: Literal["pending", "being_considered", "applied", "deferred", "rejected", "superseded", "needs_clarification", "withdrawn"]
    reason_code: str
    submitted_at: datetime
    considered_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime
    transitions: list[OrchestrationSteeringTransitionResponse] = Field(default_factory=list)
    result_action_ids: list[uuid.UUID] = Field(default_factory=list)


class OrchestrationSteeringLedgerResponse(BaseModel):
    enabled: bool
    eligibility: Literal["active", "paused", "unstarted", "terminal", "forbidden"]
    eligibility_reason: str | None
    inbox_version: int
    direction_version: int
    requests: list[OrchestrationSteeringRequestResponse] = Field(default_factory=list)
    proposals: list[OrchestrationSteeringProposalResponse] = Field(default_factory=list)


class OrchestrationConversationAllowanceResponse(BaseModel):
    enabled: bool
    limit: int
    used: int
    remaining: int


class OrchestrationConversationHistoryResponse(BaseModel):
    items: list[OrchestrationConversationTurnResponse]
    total: int
    omitted: int
    allowance: OrchestrationConversationAllowanceResponse
    steering: OrchestrationSteeringLedgerResponse


class OrchestrationDebugActionResponse(BaseModel):
    action: Literal["step", "retry", "rerun_last", "recover_goal_definition"]
    goal_id: uuid.UUID
    run_id: uuid.UUID
    process_type: str
    process: dict


class OrchestrationTickResponse(BaseModel):
    run_id: uuid.UUID
    status: str
    processed_events: int
    evidence_created: int = 0
    gates_validated: int = 0
    recoveries_created: int = 0
    tick_emitted: bool
    event_cursor: int | None
    baseline_process: dict | None = None
    manager_selection_process: dict | None = None
    agent_definition_review_process: dict | None = None
    team_hierarchy_process: dict | None = None
    effectiveness_review_process: dict | None = None
    goal_closeout_process: dict | None = None
    authority_interview: dict | None = None
    authorized_execution: dict | None = None
