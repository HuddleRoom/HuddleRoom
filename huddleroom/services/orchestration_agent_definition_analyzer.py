from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Awaitable, Callable, Mapping
from uuid import UUID

import litellm

from huddleroom.config import settings
from huddleroom.services.llm_structured_repair import complete_with_repair, _default_fix_prompt
from huddleroom.services.orchestration_llm_decision_adapter import (
    _full_completion_error,
    _redact_secrets,
    _safe_completion_error,
    orchestrator_preamble,
)


@dataclass(frozen=True)
class SemanticAgentAssessment:
    status: str
    problems: tuple[str, ...]
    reason: str
    approved_work_functions: tuple[str, ...]
    proposed_description: str | None = None
    proposed_persona: str | None = None


class SemanticAgentAnalysisError(RuntimeError):
    def __init__(self, category: str, error: str, request: Any, raw_response: str | None = None):
        super().__init__(error)
        self.category = category
        self.full_error = error
        self.request = request
        self.raw_response = raw_response


def redact_semantic_payload(value: Any) -> Any:
    """Return a JSON-safe copy with nested credentials redacted."""
    return json.loads(_redact_secrets(json.dumps(value, sort_keys=True, default=str)))


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty")
    return value.strip()


def _text_list(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{key} must contain non-empty strings")
    return tuple(item.strip() for item in value)


def _unfence_json(raw: str) -> str:
    stripped = raw.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        label, newline, payload = stripped[3:-3].partition("\n")
        if newline and label.strip() in {"", "json"}:
            return payload
    return raw


def _repair_instruction(error_text: str) -> str:
    """Tailored repair guidance for agent-definition review failures.

    Detects specific validation errors and provides targeted corrective prompts.
    Falls back to generic fix prompt for unrecognized errors.
    """
    if "must not reference a goal or project" in error_text:
        return (
            "Your previous proposal was rejected because proposed_description or "
            "proposed_persona referenced a goal or project (they contained the word "
            "'goal' or 'project', or named a specific goal/project). Rewrite BOTH "
            "proposed_description and proposed_persona to describe ONLY the agent's "
            "generic role and responsibilities, with no mention of any goal, project, "
            "or specific task. Return only the corrected JSON object, no commentary."
        )
    if "must contain exactly the required fields" in error_text:
        return (
            "Your previous response had wrong keys. Return a JSON object with EXACTLY "
            "these keys and no others: status, problems, reason, approved_work_functions, "
            "proposed_description, proposed_persona. Return only the JSON object."
        )
    if "approved assessment proposals must be null" in error_text:
        return (
            "status is 'approved', so proposed_description and proposed_persona MUST both "
            "be null. Set them to null and return only the corrected JSON object."
        )
    return _default_fix_prompt(error_text)


def parse_semantic_agent_assessment(
    payload: Any,
    candidate_work_functions: list[str],
    agent_snapshot: Mapping[str, Any] | None = None,
) -> SemanticAgentAssessment:
    if not isinstance(payload, Mapping):
        raise ValueError("semantic assessment must be an object")
    required_keys = {
        "status", "problems", "reason", "approved_work_functions",
        "proposed_description", "proposed_persona",
    }
    if set(payload) != required_keys:
        raise ValueError("semantic assessment must contain exactly the required fields")
    status = payload.get("status")
    if status not in {"approved", "improvement_proposed"}:
        raise ValueError("status must be approved or improvement_proposed")
    problems = _text_list(payload, "problems")
    reason = _required_text(payload, "reason")
    approved_raw = _text_list(payload, "approved_work_functions")
    candidate_by_casefold = {name.casefold(): name for name in candidate_work_functions}
    if status == "improvement_proposed":
        if not problems:
            raise ValueError("improvement_proposed requires problems")
        proposed_description = _required_text(payload, "proposed_description")
        proposed_persona = _required_text(payload, "proposed_persona")
        if re.search(r"\b(?:goal|project)\b", f"{proposed_description}\n{proposed_persona}", re.IGNORECASE):
            raise ValueError("proposals must not reference a goal or project")
        # ponytail: at temperature 0 a model that echoes an unchanged field
        # deterministically re-emits it, so hard-raising here permanently stalls
        # the pipeline. An improvement needs BOTH fields to genuinely differ;
        # anything less is not a deliverable improvement -> downgrade to approved
        # rather than store a non-improvement or dead-end the run.
        if agent_snapshot is not None and (
            proposed_description == str(agent_snapshot.get("description") or "").strip()
            or proposed_persona == str(agent_snapshot.get("system_prompt") or "").strip()
        ):
            status, problems = "approved", ()
            proposed_description = proposed_persona = None
    else:
        if payload.get("proposed_description") is not None or payload.get("proposed_persona") is not None:
            raise ValueError("approved assessment proposals must be null")
        proposed_description = proposed_persona = None
    approved = tuple(
        candidate_by_casefold[name.casefold()]
        for name in approved_raw
        if name.casefold() in candidate_by_casefold
    )
    return SemanticAgentAssessment(status, problems, reason, approved, proposed_description, proposed_persona)


class AgentDefinitionSemanticAnalyzer:
    def __init__(self, completion_fn: Callable[..., Awaitable[Any]] | None = None) -> None:
        self._completion_fn = completion_fn or litellm.acompletion

    async def review(
        self,
        agent_snapshot: Mapping[str, Any],
        goal_snapshot: Mapping[str, Any],
        candidate_work_functions: list[str],
        project=None,
        *,
        project_id: UUID | None = None,
    ) -> SemanticAgentAssessment:
        request_payload = self.build_request(
            agent_snapshot, goal_snapshot, candidate_work_functions, project=project
        )
        return await self.review_request(request_payload, candidate_work_functions, project_id=project_id)

    @staticmethod
    def build_request(
        agent_snapshot: Mapping[str, Any],
        goal_snapshot: Mapping[str, Any],
        candidate_work_functions: list[str],
        project=None,
    ) -> dict[str, Any]:
        payload = redact_semantic_payload({
            "agent": agent_snapshot,
            "goal": goal_snapshot,
            "candidate_work_functions": candidate_work_functions,
        })
        system_message_content = (
            "Review this agent definition for clarity, role ownership, behavioral guidance, boundaries, and fitness "
            "for the supplied goal. Judge description and persona/system_prompt explicitly. Identify concrete "
            "weaknesses. For improvement_proposed, provide constrained proposed_description and proposed_persona: both "
            "must be non-empty, role-specific, generic, differ from the candidate fields, and contain no goal or project "
            "information. Preserve valid constraints and do not assign capabilities or work functions outside "
            "candidate_work_functions. approved_work_functions must contain only exact labels copied from "
            "candidate_work_functions; never invent, rephrase, or infer labels. If candidate_work_functions is empty, "
            "approved_work_functions must be []. Agents description, persona and system_prompt should not be project specific, only "
            "role specific."
            "Return JSON only with exactly: status (approved or improvement_proposed), problems (string array), reason "
            "(string), approved_work_functions (string array), proposed_description (string or null), proposed_persona "
            "(string or null)."
        )
        messages = [
            {"role": "system", "content": orchestrator_preamble(project, goal=goal_snapshot) + "\n\n" + system_message_content},
            {"role": "user", "content": json.dumps(payload, sort_keys=True, default=str)},
        ]
        return {"model": settings.orchestration_model, "messages": messages,
                "response_format": {"type": "json_object"}, "temperature": 0, "max_tokens": 32768}

    async def review_request(
        self, request: Mapping[str, Any], candidate_work_functions: list[str], *, project_id: UUID | None = None
    ) -> SemanticAgentAssessment:
        request_payload = redact_semantic_payload(dict(request))
        model = request_payload.get("model", settings.orchestration_model)
        if settings.debug:
            print(_redact_secrets(f"LLM request model={model} messages={request_payload.get('messages')}"), flush=True)

        # Precompute agent_snapshot from request before completion (needed by parser)
        agent_snapshot_content = request_payload.get("messages", [{}, {}])[1].get("content")
        agent_snapshot = json.loads(agent_snapshot_content).get("agent") if isinstance(agent_snapshot_content, str) else None

        try:
            def _parse(raw):
                if settings.debug:
                    print(_redact_secrets(f"LLM response model={model} content={raw}"), flush=True)
                return parse_semantic_agent_assessment(
                    json.loads(_unfence_json(raw)),
                    candidate_work_functions,
                    agent_snapshot,
                )
            args = (self._completion_fn, request_payload, _parse)
            if project_id is None:
                result = await complete_with_repair(*args, fix_prompt_fn=_repair_instruction)
            else:
                from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext
                result = await complete_with_repair(*args, fix_prompt_fn=_repair_instruction, invocation=AgentResponseInvocation(InvocationContext(
                    project_id, "system", "orchestrator", "Orchestrator", "api", "agent_definition_review",
                    model, "Review this agent definition for the proposed work functions.",
                )))
            return result
        except Exception as exc:
            full_error = _full_completion_error(exc)
            if settings.debug:
                is_live_e2e = (
                    os.environ.get("HUDDLEROOM_ORCHESTRATION_BASELINE_E2E") == "true"
                    or os.environ.get("RALLY_ORCHESTRATION_BASELINE_E2E") == "true"
                )
                formatter = _full_completion_error if is_live_e2e else _safe_completion_error
                print(f"LLM failed model={model} error={formatter(exc)}", flush=True)
            # ponytail: collapsed provider_error/invalid_response distinction; complete_with_repair exhaustion doesn't expose raw
            raise SemanticAgentAnalysisError(
                "invalid_response",
                full_error,
                request_payload,
                None,
            ) from exc
