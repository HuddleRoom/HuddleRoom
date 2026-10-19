"""Bounded interpretation and normalized conversation history."""
from datetime import datetime
from typing import Annotated, Literal
import json
import uuid
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticCustomError
from huddleroom.schemas.orchestration import OrchestrationConversationTurnResponse, OrchestrationConversationAllowanceResponse, OrchestrationSteeringLedgerResponse
from huddleroom.schemas.agent import AgentCreate
from huddleroom.schemas.orchestration import OrchestrationAgentDefinitionReviewBatchAnswer
from huddleroom.schemas.orchestration_advisor import ProjectAdvisorAllowance, ProjectAdvisorCitation

ChatOperation = Literal["submit_steering", "promote_steering", "withdraw_steering", "dismiss_proposal", "start_baseline", "start_execution", "restart_execution", "pause", "resume", "cancel", "answer_authority", "answer_runtime_question", "approve_recovery", "reject_recovery", "answer_agent_definition_batch", "answer_hierarchy"]

class ChatSubmitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(min_length=1, max_length=12000)
    client_request_id: uuid.UUID
    context_goal_id: uuid.UUID | None = None
    action_id: uuid.UUID | None = None
    start_scope: Literal["planning_only", "execution"] | None = None

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise PydanticCustomError("empty_content", "content must not be empty")
        return value

ChatText = Annotated[str, Field(max_length=12000)]
VersionKey = Annotated[str, Field(max_length=128)]


class ChatAgentEdit(AgentCreate):
    model_config = ConfigDict(extra="forbid")
    name: ChatText
    role: ChatText
    description: ChatText | None = None
    provider: ChatText
    model: ChatText
    system_prompt: ChatText | None = None
    cli_runtime: ChatText | None = None
    capabilities: list[ChatText] = Field(default_factory=list, max_length=100)
    config: dict = Field(default_factory=dict)

    @field_validator("config", mode="before")
    @classmethod
    def bounded_config(cls, value):
        entries = 0

        def visit(node, depth=0):
            nonlocal entries
            if depth > 8:
                raise ValueError("agent config exceeds nesting limit")
            if isinstance(node, dict):
                entries += len(node)
                for key, child in node.items():
                    if not isinstance(key, str) or len(key) > 12000:
                        raise ValueError("agent config key exceeds limit")
                    visit(child, depth + 1)
            elif isinstance(node, list):
                entries += len(node)
                for child in node:
                    visit(child, depth + 1)
            elif isinstance(node, str):
                if len(node) > 12000:
                    raise ValueError("agent config string exceeds limit")
            elif node is not None and not isinstance(node, (bool, int, float)):
                raise ValueError("agent config must contain JSON values")
            if entries > 100:
                raise ValueError("agent config exceeds entry limit")

        visit(value)
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("agent config must contain bounded JSON values") from exc
        if len(encoded) > 24000:
            raise ValueError("agent config exceeds serialized limit")
        return value


class ChatBatchAnswer(OrchestrationAgentDefinitionReviewBatchAnswer):
    model_config = ConfigDict(extra="forbid")
    selected_option: str = Field(min_length=1, max_length=4000)
    reason: ChatText | None = None
    edited_description: ChatText | None = None
    edited_persona: ChatText | None = None


class ChatActionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directive: str | None = Field(default=None, max_length=12000)
    target_type: Literal["goal", "plan_item", "task"] | None = None
    target_id: uuid.UUID | None = None
    scope: Literal["item", "run", "goal"] | None = None
    lifetime: Literal["selected_item", "remaining_current_run", "future_runs"] | None = None
    impact_summary: str | None = Field(default=None, max_length=4000)
    supersedes_request_id: uuid.UUID | None = None
    selected_option: str | None = Field(default=None, max_length=4000)
    answer: str | None = Field(default=None, max_length=12000)
    reason: str | None = Field(default=None, max_length=12000)
    contract_version: str | None = Field(default=None, max_length=128)
    edited_agent: ChatAgentEdit | None = None
    edited_description: str | None = Field(default=None, max_length=12000)
    edited_persona: str | None = Field(default=None, max_length=12000)
    answers: list[ChatBatchAnswer] = Field(default_factory=list, max_length=100)
    start_scope: Literal["planning_only", "execution"] | None = None

class ChatActionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: uuid.UUID
    operation: ChatOperation
    project_id: uuid.UUID
    goal_id: uuid.UUID
    run_id: uuid.UUID | None = None
    target_id: uuid.UUID | None = None
    frozen_versions: dict[VersionKey, ChatText | int | None] = Field(default_factory=dict, max_length=30)
    payload: ChatActionPayload = Field(default_factory=ChatActionPayload)
    summary: str = Field(max_length=12000)
    status: Literal["suggested", "awaiting_confirmation", "prepared", "stale", "completed", "failed"] = "prepared"
    requires_confirmation: bool = False

class ChatActionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: uuid.UUID
    operation: ChatOperation
    project_id: uuid.UUID
    goal_id: uuid.UUID
    run_id: uuid.UUID | None = None
    status: Literal["accepted", "pending", "applied", "completed", "recorded", "failed", "interrupted_unknown"]
    source_kind: str | None = None
    source_record_id: uuid.UUID | None = None
    message: str = Field(default="", max_length=12000)
    error_code: str | None = None

class ChatInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal["question", "instruction", "acceptance", "clarification", "unsupported"]
    answer: str | None = Field(default=None, max_length=24000)
    goal_id: uuid.UUID | None = None
    named_goal: str | None = Field(default=None, max_length=500)
    operation: ChatOperation | None = None
    payload: ChatActionPayload = Field(default_factory=ChatActionPayload)
    clarification: str | None = Field(default=None, max_length=4000)

class ChatSourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["project_advisor", "goal_conversation", "steering_proposal", "steering_request", "authority_decision", "orchestration_action"]
    id: uuid.UUID
    goal_id: uuid.UUID | None = None

class ChatTurnResponse(BaseModel):
    id: str
    client_request_id: uuid.UUID | None = None
    actor_id: uuid.UUID
    content: str
    answer: str | None = None
    status: str
    created_at: datetime
    context_goal_id: uuid.UUID | None = None
    run_id: uuid.UUID | None = None
    historical: bool = False
    source_references: list[ChatSourceReference] = Field(default_factory=list)
    citations: list[ProjectAdvisorCitation] = Field(default_factory=list)
    source_goal_ids: list[uuid.UUID] = Field(default_factory=list)
    off_topic: bool = False
    error: dict | None = None
    goal_turn: OrchestrationConversationTurnResponse | None = None
    actions: list[ChatActionDraft] = Field(default_factory=list)
    receipts: list[ChatActionReceipt] = Field(default_factory=list)
    accounting_snapshot: dict = Field(default_factory=dict)

class ChatHistoryResponse(BaseModel):
    items: list[ChatTurnResponse]
    total: int
    omitted: int = 0
    allowance: ProjectAdvisorAllowance
    goal_allowances: dict[str, OrchestrationConversationAllowanceResponse] = Field(default_factory=dict)
    steering: dict[str, OrchestrationSteeringLedgerResponse] = Field(default_factory=dict)
