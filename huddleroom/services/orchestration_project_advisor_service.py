"""Read-only "Ask the orchestrator" project advisor service (Decisions 2,4,5,7 / Phase 3).

Lean, request-scoped service modeled on OrchestrationDecisionAdapter.decide: one
synchronous complete_with_repair call, one persisted ProjectAdvisorTurn row. No
steering, no tool-calls, no reservation/claim state machine, no Celery. The
ONLY database write is the single ProjectAdvisorTurn row per ask().
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Awaitable, Callable, Mapping

import litellm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn, advisor_allowance_used
from huddleroom.models.project import Project
from huddleroom.services.llm_structured_repair import complete_with_repair
from huddleroom.services.orchestration_conversation_service import ConversationDomainError
from huddleroom.services.orchestration_llm_decision_adapter import orchestrator_preamble
from huddleroom.services.orchestration_project_advisor_context import build_advisor_context
from huddleroom.services.secret_redaction import redact_secrets

CompletionFn = Callable[..., Awaitable[Any]]

_CITATION_TYPES = {"goal", "decision", "meeting"}
_HISTORY_LIMIT = 20
_MAX_ERROR_LEN = 500
_finite_advisor_locks: dict[tuple[uuid.UUID, uuid.UUID], asyncio.Lock] = {}

# History ordering: newest-first (created_at desc). The frontend history list
# renders most-recent turn at the top of the Ask tab.
_RESPONSE_INSTRUCTIONS = (
    "You are the read-only project advisor: you explain project state, you never act. "
    "You must never create, start, steer, or otherwise change any goal, decision, task, "
    "or meeting -- if asked to do something, explain what exists instead of doing it. "
    "Answer only questions about this project's goals, decisions, and recent activity, "
    "grounded strictly in the JSON context object in the user message; do not invent facts "
    "not present in that context. "
    "If the question is off-topic (not about this project's state), set off_topic to true "
    "and give a short deflection answer instead of trying to answer it. "
    "Cite every goal, decision, or meeting you relied on to answer in citations "
    "(omit citations entirely for off-topic answers). "
    'Return exactly one JSON object with this shape: {"answer": "string", '
    '"citations": [{"type": "goal|decision|meeting", "id": "string", '
    '"goal_id": "string for decisions", "label": "string"}], '
    '"off_topic": true|false}. No markdown, no commentary, JSON only.'
)


def _parse_advisor_response(raw_content: str, context: Mapping[str, Any]) -> dict:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output was not valid JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("LLM output JSON root must be an object")

    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("LLM output must contain a non-empty 'answer' string")

    off_topic = payload.get("off_topic")
    if not isinstance(off_topic, bool):
        raise ValueError("LLM output 'off_topic' must be a boolean")

    citations = payload.get("citations")
    if not isinstance(citations, list):
        raise ValueError("LLM output 'citations' must be a list")

    decision_goals = {
        (decision.get("id"), decision.get("goal_id"))
        for decision in context.get("recent_decisions", [])
        if isinstance(decision, Mapping)
    }
    clean_citations = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError("each citation must be an object")
        ctype, cid, label = citation.get("type"), citation.get("id"), citation.get("label")
        if (
            ctype not in _CITATION_TYPES
            or not isinstance(cid, str) or not cid.strip()
            or not isinstance(label, str) or not label.strip()
        ):
            raise ValueError(
                "each citation needs type in goal|decision|meeting, a non-empty id, and a non-empty label"
            )
        if ctype == "decision":
            goal_id = citation.get("goal_id")
            if not isinstance(goal_id, str) or not goal_id.strip() or (cid, goal_id) not in decision_goals:
                raise ValueError("each decision citation needs a goal_id matching recent_decisions context")
            clean_citations.append({"type": ctype, "id": cid, "goal_id": goal_id, "label": label})
        else:
            clean_citations.append({"type": ctype, "id": cid, "label": label})

    return {"answer": answer.strip(), "citations": clean_citations, "off_topic": off_topic}


def _extract_usage(response: Any) -> int | None:
    def get(value: Any, key: str) -> Any:
        return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)

    usage = get(response, "usage")
    prompt, completion = get(usage, "prompt_tokens"), get(usage, "completion_tokens")
    if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (prompt, completion)):
        return prompt + completion
    return None


class OrchestrationProjectAdvisorService:
    def __init__(self, model: str | None = None, completion_fn: CompletionFn | None = None) -> None:
        self.model = model or settings.orchestration_model
        self._completion_fn = completion_fn or litellm.acompletion

    async def ask(
        self, db: AsyncSession, project_id: uuid.UUID, actor_id: uuid.UUID, question: str
    ) -> ProjectAdvisorTurn:
        question = question.strip() if isinstance(question, str) else ""
        if not question:
            raise ConversationDomainError("invalid_content", 422, "Question is required")

        limit = settings.orchestration_advisor_allowance_tokens
        if limit == 0:
            raise ConversationDomainError("advisor_disabled", 409, "Project advisor is disabled")

        if limit > 0:
            lock = _finite_advisor_locks.setdefault((project_id, actor_id), asyncio.Lock())
            async with lock:
                return await self._ask(db, project_id, actor_id, question, limit)
        return await self._ask(db, project_id, actor_id, question, limit)

    async def _ask(
        self, db: AsyncSession, project_id: uuid.UUID, actor_id: uuid.UUID, question: str, limit: int
    ) -> ProjectAdvisorTurn:
        if limit > 0:
            used = await advisor_allowance_used(db, project_id, actor_id)
            if used >= limit:
                raise ConversationDomainError(
                    "advisor_allowance_exhausted", 429, "Project advisor allowance is exhausted"
                )

        context = await build_advisor_context(db, project_id)
        project = await db.get(Project, project_id)
        project_dict = (
            {"id": str(project_id), "name": project.name, "description": project.description}
            if project is not None else None
        )

        system_text = orchestrator_preamble(project=project_dict) + "\n\n" + _RESPONSE_INSTRUCTIONS
        messages = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "context": context}, sort_keys=True, default=str
                ),
            },
        ]
        request = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

        usage_holder = {"tokens": 0}

        def _observe(response: Any) -> None:
            usage = _extract_usage(response)
            if usage is not None:
                usage_holder["tokens"] += usage

        try:
            parsed = await complete_with_repair(
                self._completion_fn,
                request,
                lambda content: _parse_advisor_response(content, context),
                response_observer=_observe,
            )
        except Exception as exc:
            error_text = redact_secrets(" ".join(str(exc).split()))
            if len(error_text) > _MAX_ERROR_LEN:
                error_text = error_text[: _MAX_ERROR_LEN - 3] + "..."
            turn = ProjectAdvisorTurn(
                project_id=project_id, actor_id=actor_id, question=question,
                answer=None, citations=[], off_topic=False,
                tokens_used=usage_holder["tokens"], status="failed", error=error_text,
            )
            db.add(turn)
            # Commit explicitly (mirrors _settle/_fail_known in
            # orchestration_conversation_service.py): the failed turn must
            # be durably persisted even though we re-raise right after —
            # relying on the router's request-scoped session to commit
            # would lose this row, since raising rolls that session back.
            await db.commit()
            raise

        turn = ProjectAdvisorTurn(
            project_id=project_id, actor_id=actor_id, question=question,
            answer=parsed["answer"], citations=parsed["citations"], off_topic=parsed["off_topic"],
            tokens_used=usage_holder["tokens"], status="completed",
        )
        db.add(turn)
        await db.commit()
        await db.refresh(turn)
        return turn

    async def history(
        self, db: AsyncSession, project_id: uuid.UUID, actor_id: uuid.UUID
    ) -> list[ProjectAdvisorTurn]:
        """Last 20 turns for this project+actor, newest-first (created_at desc)."""
        result = await db.execute(
            select(ProjectAdvisorTurn)
            .where(ProjectAdvisorTurn.project_id == project_id, ProjectAdvisorTurn.actor_id == actor_id)
            .order_by(ProjectAdvisorTurn.created_at.desc(), ProjectAdvisorTurn.id.desc())
            .limit(_HISTORY_LIMIT)
        )
        return list(result.scalars().all())
