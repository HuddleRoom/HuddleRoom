from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import ceil
from re import findall
from statistics import median
from typing import Literal, Mapping

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.orchestration_conversation import (
    FEEDBACK_REASON_CHECK,
    ConversationFeedback,
    ConversationMessage,
    ConversationResponse,
    ConversationInvestigation,
    conversation_feedback_id,
)
from huddleroom.models.orchestration_steering import (
    OrchestrationSteeringProposal,
    OrchestrationSteeringRequest,
    OrchestrationSteeringResultLink,
    OrchestrationSteeringTransition,
)


FEEDBACK_REASONS = frozenset(findall(r"'([^']+)'", FEEDBACK_REASON_CHECK)) - {
    "helpful",
    "not_helpful",
}


FeedbackRating = Literal["helpful", "not_helpful"]
FeedbackReason = Literal[
    "unanswered",
    "incorrect",
    "missing_context",
    "stale_context",
    "unclear",
    "too_limited",
    "other",
]


@dataclass(frozen=True)
class ConversationFeedbackInput:
    rating: FeedbackRating
    reason: FeedbackReason | None


@dataclass(frozen=True)
class ConversationLearningWindow:
    start_at: datetime
    end_at: datetime
    goal_id: uuid.UUID | None = None


@dataclass(frozen=True)
class ConversationLearningReport:
    window: Mapping[str, object]
    sample: Mapping[str, int]
    chat: Mapping[str, object]
    investigations: Mapping[str, object]
    context_limits: Mapping[str, object]
    steering: Mapping[str, object]
    operator_feedback: Mapping[str, object]


CHAT_STATUSES = ("pending", "running", "completed", "failed", "interrupted_unknown")
INVESTIGATION_STATUSES = (
    "pending", "running", "completed", "limited", "failed", "cancelled", "unavailable",
    "interrupted_unknown",
)
REQUEST_STATUSES = (
    "pending", "being_considered", "applied", "deferred", "rejected", "superseded",
    "needs_clarification", "withdrawn",
)
REQUEST_REASONS = (
    "submitted", "considering", "run_changed", "steering_ineligible", "target_already_started",
    "supersedes_required", "invalid_supersedes_request", "superseded", "applied", "withdrawn",
)
TRUNCATED_SOURCES = (
    "goal", "run", "accepted_plan", "decisions", "actions", "gates", "evidence", "processes",
    "warnings", "memory", "artifact", "agents", "prior_turns",
)
OMISSION_STATUSES = ("restricted", "unsafe", "binary", "too_large", "changed", "omitted_by_limit")


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _latency(pairs: list[tuple[datetime | None, datetime | None]]) -> dict[str, object]:
    values: list[float] = []
    invalid = 0
    for start, end in pairs:
        start, end = _utc(start), _utc(end)
        if start is None or end is None:
            continue
        seconds = (end - start).total_seconds()
        if seconds < 0:
            invalid += 1
        else:
            values.append(round(seconds, 3))
    ordered = sorted(values)
    return {
        "count": len(values), "invalid": invalid,
        "median": median(values) if values else None,
        "p95": ordered[ceil(.95 * len(ordered)) - 1] if ordered else None,
    }


def _fixed(keys: tuple[str, ...]) -> dict[str, int]:
    return dict.fromkeys(keys, 0)


class ConversationLearningError(Exception):
    code: str
    status_code: int
    message: str

    def __init__(self, code: str, status_code: int, message: str):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message


class OrchestrationConversationLearningService:
    """Persist immutable, actor-owned answer feedback without runtime effects."""

    @staticmethod
    def _ineligible() -> ConversationLearningError:
        return ConversationLearningError(
            "conversation_feedback_ineligible", 409, "Conversation feedback is ineligible"
        )

    @staticmethod
    def _replay_or_conflict(
        existing: ConversationFeedback | None, feedback: ConversationFeedbackInput
    ) -> ConversationFeedback:
        if existing is not None and (existing.rating, existing.reason) == (
            feedback.rating,
            feedback.reason,
        ):
            return existing
        raise ConversationLearningError(
            "conversation_feedback_already_recorded", 409, "Conversation feedback is already recorded"
        )

    @classmethod
    def _validate_feedback(cls, feedback: ConversationFeedbackInput) -> None:
        if feedback.rating == "helpful" and feedback.reason is None:
            return
        if (
            feedback.rating == "not_helpful"
            and isinstance(feedback.reason, str)
            and feedback.reason in FEEDBACK_REASONS
        ):
            return
        raise cls._ineligible()

    @staticmethod
    def _window(window: ConversationLearningWindow) -> tuple[datetime, datetime]:
        start_at, end_at = _utc(window.start_at), _utc(window.end_at)
        if (
            window.start_at.tzinfo is None or window.start_at.utcoffset() is None
            or window.end_at.tzinfo is None or window.end_at.utcoffset() is None
            or start_at >= end_at or end_at - start_at > timedelta(days=90)
        ):
            raise ConversationLearningError(
                "conversation_learning_invalid_window", 422, "Conversation learning window is invalid"
            )
        return start_at, end_at

    async def summarize(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        window: ConversationLearningWindow,
    ) -> ConversationLearningReport:
        """Derive one content-free bounded report from durable source rows."""
        start_at, end_at = self._window(window)
        if window.goal_id is not None and not await db.scalar(
            select(OrchestrationGoal.id).where(
                OrchestrationGoal.id == window.goal_id, OrchestrationGoal.project_id == project_id
            )
        ):
            raise ConversationLearningError(
                "conversation_learning_goal_not_found", 404, "Conversation learning goal not found"
            )

        goal_where = [OrchestrationGoal.project_id == project_id]
        if window.goal_id is not None:
            goal_where.append(OrchestrationGoal.id == window.goal_id)
        # ponytail: bounded 90-day projection; materialize only if measured volume makes this slow.
        messages = (await db.execute(
            select(
                ConversationMessage.id, ConversationMessage.goal_id, ConversationMessage.actor_id,
                ConversationResponse.id, ConversationResponse.status,
                func.length(func.trim(ConversationResponse.answer, " \n\t\r")) > 0,
                ConversationResponse.started_at, ConversationResponse.finished_at,
                ConversationResponse.context_manifest,
            ).join(ConversationResponse, ConversationResponse.message_id == ConversationMessage.id)
            .join(OrchestrationGoal, OrchestrationGoal.id == ConversationMessage.goal_id)
            .where(*goal_where, ConversationMessage.created_at >= start_at, ConversationMessage.created_at < end_at)
        )).all()
        chat_counts = _fixed(CHAT_STATUSES)
        answered = terminal_without_answer = 0
        response_ids: set[uuid.UUID] = set()
        message_goals: set[uuid.UUID] = set()
        message_actors: set[uuid.UUID] = set()
        chat_latency: list[tuple[datetime | None, datetime | None]] = []
        context = {
            "turns_truncated": 0, "turns_with_omissions": 0, "omitted_records": 0,
            "truncated_source_counts": _fixed(TRUNCATED_SOURCES),
            "investigation_omission_status_counts": _fixed(OMISSION_STATUSES),
        }
        for message_id, goal_id, actor_id, response_id, status, has_answer, started, finished, manifest in messages:
            response_ids.add(response_id)
            message_goals.add(goal_id)
            message_actors.add(actor_id)
            terminal = _utc(finished) is not None and _utc(finished) < end_at
            as_of = status if terminal else ("running" if _utc(started) is not None and _utc(started) < end_at else "pending")
            chat_counts[as_of] += 1
            if as_of == "completed" and has_answer:
                answered += 1
            elif as_of in {"completed", "failed", "interrupted_unknown"} and not has_answer:
                terminal_without_answer += 1
            if terminal:
                chat_latency.append((started, finished))
            if isinstance(manifest, dict):
                sources = manifest.get("sources")
                valid_sources = [item for item in sources if isinstance(item, dict) and item.get("source") in TRUNCATED_SOURCES] if isinstance(sources, list) else []
                if manifest.get("truncated") is True:
                    context["turns_truncated"] += 1
                omissions = 0
                for item in valid_sources:
                    source = item["source"]
                    if item.get("truncated") is True:
                        context["truncated_source_counts"][source] += 1
                    omitted = item.get("omitted")
                    if isinstance(omitted, int) and not isinstance(omitted, bool) and omitted > 0:
                        omissions += omitted
                if omissions:
                    context["turns_with_omissions"] += 1
                    context["omitted_records"] += omissions

        investigations = (await db.execute(
            select(
                ConversationInvestigation.status, ConversationInvestigation.attempt_count,
                ConversationInvestigation.repair_count, ConversationInvestigation.retry_count,
                ConversationInvestigation.accumulated_tokens, ConversationInvestigation.started_at,
                ConversationInvestigation.finished_at, ConversationInvestigation.input_manifest,
            ).join(ConversationResponse, ConversationResponse.id == ConversationInvestigation.response_id)
            .join(ConversationMessage, ConversationMessage.id == ConversationResponse.message_id)
            .join(OrchestrationGoal, OrchestrationGoal.id == ConversationMessage.goal_id)
            .where(*goal_where, ConversationInvestigation.created_at >= start_at, ConversationInvestigation.created_at < end_at)
        )).all()
        investigation_counts = _fixed(INVESTIGATION_STATUSES)
        investigation_latency: list[tuple[datetime | None, datetime | None]] = []
        accounted = attempts = repairs = retries = tokens = 0
        for status, attempt, repair, retry, total_tokens, started, finished, manifest in investigations:
            terminal = _utc(finished) is not None and _utc(finished) < end_at
            as_of = status if terminal else ("running" if _utc(started) is not None and _utc(started) < end_at else "pending")
            investigation_counts[as_of] += 1
            if terminal:
                accounted += 1
                attempts += attempt
                repairs += repair
                retries += retry
                tokens += total_tokens
                investigation_latency.append((started, finished))
            if isinstance(manifest, dict) and isinstance(manifest.get("omissions"), list):
                for omission in manifest["omissions"]:
                    if isinstance(omission, dict) and omission.get("status") in OMISSION_STATUSES:
                        context["investigation_omission_status_counts"][omission["status"]] += 1

        proposals = (await db.execute(
            select(OrchestrationSteeringProposal.id, OrchestrationSteeringProposal.dismissed_at,
                   OrchestrationSteeringProposal.goal_id, OrchestrationSteeringProposal.actor_id)
            .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationSteeringProposal.goal_id)
            .where(*goal_where, OrchestrationSteeringProposal.created_at >= start_at,
                   OrchestrationSteeringProposal.created_at < end_at)
        )).all()
        requests = (await db.execute(
            select(OrchestrationSteeringRequest.id, OrchestrationSteeringRequest.goal_id,
                   OrchestrationSteeringRequest.actor_id, OrchestrationSteeringRequest.source_proposal_id,
                   OrchestrationSteeringRequest.submitted_at)
            .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationSteeringRequest.goal_id)
            .where(*goal_where, OrchestrationSteeringRequest.submitted_at >= start_at,
                   OrchestrationSteeringRequest.submitted_at < end_at)
        )).all()
        request_ids = {row[0] for row in requests}
        promoted_ids = set((await db.scalars(
            select(OrchestrationSteeringRequest.source_proposal_id).where(
                OrchestrationSteeringRequest.source_proposal_id.is_not(None),
                OrchestrationSteeringRequest.submitted_at < end_at,
            )
        )).all())
        proposal_counts = _fixed(("proposed", "dismissed", "promoted"))
        for proposal_id, dismissed_at, goal_id, actor_id in proposals:
            message_goals.add(goal_id)
            message_actors.add(actor_id)
            proposal_counts["dismissed" if _utc(dismissed_at) is not None and _utc(dismissed_at) < end_at else "promoted" if proposal_id in promoted_ids else "proposed"] += 1
        for _, goal_id, actor_id, _, _ in requests:
            message_goals.add(goal_id)
            message_actors.add(actor_id)

        transitions = (await db.execute(
            select(OrchestrationSteeringTransition.request_id, OrchestrationSteeringTransition.to_status,
                   OrchestrationSteeringTransition.reason_code, OrchestrationSteeringTransition.created_at,
                   OrchestrationSteeringTransition.sequence)
            .where(OrchestrationSteeringTransition.request_id.in_(request_ids),
                   OrchestrationSteeringTransition.created_at < end_at)
            .order_by(OrchestrationSteeringTransition.request_id, OrchestrationSteeringTransition.created_at,
                      OrchestrationSteeringTransition.sequence)
        )).all() if request_ids else []
        transitions_by_request: dict[uuid.UUID, list[tuple[str, str, datetime]]] = {}
        for request_id, status, reason, created_at, sequence in transitions:
            transitions_by_request.setdefault(request_id, []).append((status, reason, created_at))
        request_counts, reason_counts = _fixed(REQUEST_STATUSES), _fixed(REQUEST_REASONS)
        considered_pairs: list[tuple[datetime | None, datetime | None]] = []
        finished_pairs: list[tuple[datetime | None, datetime | None]] = []
        terminal_statuses = set(REQUEST_STATUSES) - {"pending", "being_considered"}
        for request_id, _, _, source_proposal_id, submitted_at in requests:
            history = transitions_by_request.get(request_id, [])
            status, reason = history[-1][:2] if history else ("pending", "submitted")
            request_counts[status] += 1
            if reason in reason_counts:
                reason_counts[reason] += 1
            considered = next((created for target, _, created in history if target == "being_considered"), None)
            finished = next((created for target, _, created in history if target in terminal_statuses), None)
            if considered is not None:
                considered_pairs.append((submitted_at, considered))
            if finished is not None:
                finished_pairs.append((submitted_at, finished))
        links = (await db.execute(
            select(OrchestrationSteeringResultLink.request_id).where(
                OrchestrationSteeringResultLink.request_id.in_(request_ids),
                OrchestrationSteeringResultLink.created_at < end_at,
            )
        )).all() if request_ids else []

        feedback_rows = (await db.execute(
            select(ConversationFeedback.response_id, ConversationFeedback.rating, ConversationFeedback.reason)
            .where(ConversationFeedback.response_id.in_(response_ids), ConversationFeedback.created_at < end_at)
        )).all() if response_ids else []
        feedback = {"rated": 0, "unrated_answered": 0, "helpful": 0, "not_helpful": 0,
                    "not_helpful_reason_counts": _fixed(tuple(FEEDBACK_REASONS))}
        answered_ids = {row[3] for row in messages if (_utc(row[7]) is not None and _utc(row[7]) < end_at and row[4] == "completed" and row[5])}
        for response_id, rating, reason in feedback_rows:
            if response_id in answered_ids:
                feedback["rated"] += 1
                feedback[rating] += 1
                if rating == "not_helpful" and reason in FEEDBACK_REASONS:
                    feedback["not_helpful_reason_counts"][reason] += 1
        feedback["unrated_answered"] = answered - feedback["rated"]
        return ConversationLearningReport(
            window={"start_at": start_at, "end_at": end_at, "generated_at": datetime.now(timezone.utc), "goal_filtered": window.goal_id is not None},
            sample={"goals": len(message_goals), "operators": len(message_actors), "questions": len(messages)},
            chat={"status_counts": chat_counts, "answered": answered, "terminal_without_answer": terminal_without_answer, "latency_seconds": _latency(chat_latency)},
            investigations={"triggered": len(investigations), "accounted_terminal": accounted, "status_counts": investigation_counts, "attempts": attempts, "repairs": repairs, "retries": retries, "accumulated_tokens": tokens, "latency_seconds": _latency(investigation_latency)},
            context_limits=context,
            steering={"proposal_status_counts": proposal_counts, "request_status_counts": request_counts, "request_reason_counts": reason_counts, "direct_requests": sum(source is None for _, _, _, source, _ in requests), "proposal_derived_requests": sum(source is not None for _, _, _, source, _ in requests), "requests_with_result_actions": len({row[0] for row in links}), "result_links": len(links), "submit_to_considered_seconds": _latency(considered_pairs), "submit_to_finished_seconds": _latency(finished_pairs)},
            operator_feedback=feedback,
        )

    async def record_feedback(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        actor_id: uuid.UUID,
        response_id: uuid.UUID,
        feedback: ConversationFeedbackInput,
    ) -> ConversationFeedback:
        """Create or replay one immutable feedback row for the owning actor."""
        self._validate_feedback(feedback)
        lineage = await db.execute(
            select(
                ConversationResponse.status,
                ConversationResponse.answer,
                ConversationMessage.actor_id,
            )
            .join(ConversationMessage, ConversationResponse.message_id == ConversationMessage.id)
            .join(OrchestrationGoal, ConversationMessage.goal_id == OrchestrationGoal.id)
            .where(
                ConversationResponse.id == response_id,
                OrchestrationGoal.id == goal_id,
                OrchestrationGoal.project_id == project_id,
            )
        )
        response = lineage.one_or_none()
        if response is None:
            raise ConversationLearningError(
                "conversation_feedback_not_found", 404, "Conversation response not found"
            )
        status, answer, owner_id = response
        if owner_id != actor_id:
            raise ConversationLearningError(
                "conversation_feedback_forbidden", 403, "Conversation feedback is forbidden"
            )
        if status != "completed" or not isinstance(answer, str) or not answer.strip():
            raise self._ineligible()

        feedback_id = conversation_feedback_id(response_id, actor_id)
        existing = await db.get(ConversationFeedback, feedback_id)
        if existing is not None:
            return self._replay_or_conflict(existing, feedback)

        recorded = ConversationFeedback(
            id=feedback_id,
            response_id=response_id,
            actor_id=actor_id,
            rating=feedback.rating,
            reason=feedback.reason,
        )
        try:
            async with db.begin_nested():
                db.add(recorded)
                await db.flush()
        except IntegrityError:
            return self._replay_or_conflict(await db.get(ConversationFeedback, feedback_id), feedback)
        return recorded
