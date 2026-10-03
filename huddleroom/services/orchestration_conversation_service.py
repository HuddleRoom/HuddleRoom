"""Durable, non-streaming goal conversation dispatch."""

# Reuses the existing service's lock contract on the same fresh session.
# pylint: disable=protected-access

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import timedelta
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Literal

import litellm
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.config import settings
from huddleroom.database import AsyncSessionLocal
from huddleroom.models.orchestration_conversation import (
    ConversationFeedback,
    ConversationMessage,
    ConversationInvestigation,
    ConversationReservation,
    ConversationResponse,
    conversation_message_id,
    conversation_provider_request_id,
    conversation_reservation_id,
    conversation_response_id,
)
from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.orchestration_steering import OrchestrationSteeringProposal, steering_proposal_id
from huddleroom.services.orchestration_conversation_dossier import ConversationDossierBuilder
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_steering import (
    OrchestrationSteeringService,
    SteeringDomainError,
    SteeringDraft,
    normalize_draft,
)

if TYPE_CHECKING:
    from huddleroom.services.orchestration_conversation_investigation import InvestigationRequest


@dataclass(frozen=True)
class ConversationTurn:
    message: ConversationMessage
    response: ConversationResponse
    investigation: ConversationInvestigation | None = None
    provider_messages: list[dict[str, str]] | None = None
    feedback: ConversationFeedback | None = None
    feedback_eligible: bool = False


@dataclass(frozen=True)
class ConversationProviderResult:
    kind: Literal["answer", "investigation", "proposal", "invalid", "unknown"]
    answer: str | None
    request: InvestigationRequest | None
    usage: int | None
    proposal: SteeringDraft | None = None


@dataclass(frozen=True)
class _ChatAuthority:
    response_id: uuid.UUID
    context_version: str
    provider_request_id: str
    dossier: str


PROPOSE_STEERING_TOOL = {
    "type": "function",
    "function": {
        "name": "respond_with_proposed_steering",
        "description": "Create a review-only steering draft for explicit human review; it never applies steering.",
        "strict": True,
        "parameters": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "answer": {"type": "string", "minLength": 1, "maxLength": 8_000},
                "proposal": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "directive": {"type": "string", "minLength": 1, "maxLength": 4_000},
                        "target_type": {"type": "string", "enum": ["goal", "plan_item", "task"]},
                        "target_id": {"type": "string", "minLength": 1, "maxLength": 255},
                        "scope": {"type": "string", "enum": ["item", "run", "goal"]},
                        "lifetime": {"type": "string", "enum": ["selected_item", "remaining_current_run", "future_runs"]},
                        "impact_summary": {"type": "string", "minLength": 1, "maxLength": 1_000},
                    },
                    "required": ["directive", "target_type", "target_id", "scope", "lifetime", "impact_summary"],
                },
            },
            "required": ["answer", "proposal"],
        },
    },
}


class ConversationDomainError(Exception):
    def __init__(self, code: str, status_code: int, message: str):
        super().__init__(code)
        self.code, self.status_code, self.message = code, status_code, message


class OrchestrationConversationService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
        completion_fn: Callable[..., Any] = litellm.acompletion,
        lookup_fn: Callable[[str], Any] | None = None,
        orchestration_service: OrchestrationService | None = None,
        investigation_service: Any | None = None,
    ):
        self._session_factory = session_factory
        self._completion_fn = completion_fn
        self._lookup_fn = lookup_fn
        self._orchestration = orchestration_service or OrchestrationService()
        if investigation_service is None:
            # The investigation module imports ConversationDomainError from here.
            from huddleroom.services.orchestration_conversation_investigation import ConversationInvestigationService
            investigation_service = ConversationInvestigationService(
                session_factory, completion_fn, lookup_fn, self._orchestration
            )
        self._investigations = investigation_service

    async def submit(
        self,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        actor_id: uuid.UUID,
        client_request_id: uuid.UUID,
        content: str,
    ) -> ConversationTurn:
        content = content.strip() if isinstance(content, str) else ""
        if not content:
            raise ConversationDomainError(
                "conversation_invalid_content", 422, "Conversation content is required"
            )
        prepared, created = await self._prepare(
            project_id, goal_id, actor_id, client_request_id, content
        )
        if not created:
            return prepared
        claimed = await self._claim(goal_id, prepared.response.id)
        if claimed is None:
            return await self._turn(prepared.response.id)
        provider_messages = prepared.provider_messages
        assert provider_messages is not None
        investigation_allowed = self._investigation_allowed(claimed)
        steering_allowed = self._steering_allowed(claimed)
        authority = self._authority(claimed) if investigation_allowed or steering_allowed else None
        try:
            completion_kwargs: dict[str, Any] = {
                "model": settings.orchestration_model,
                "messages": provider_messages,
                "stream": False,
                "temperature": 0,
                "max_tokens": self._completion_tokens(steering_allowed),
                "litellm_call_id": claimed.provider_request_id,
            }
            tools = []
            if investigation_allowed:
                from huddleroom.services.orchestration_conversation_investigation import REQUEST_INVESTIGATION_TOOL
                tools.append(REQUEST_INVESTIGATION_TOOL)
            if steering_allowed:
                tools.append(PROPOSE_STEERING_TOOL)
            if tools:
                completion_kwargs.update(tools=tools, tool_choice="auto")
            raw = await asyncio.wait_for(self._completion_fn(**completion_kwargs), timeout=120)
            result = self._normalize(raw, investigation_allowed, steering_allowed)
        except Exception:  # provider outcome is uncertain; never redispatch it.
            result = ConversationProviderResult("unknown", None, None, None)
            if self._lookup_fn is not None:
                try:
                    if investigation_allowed:
                        from huddleroom.services.orchestration_conversation_investigation import _lookup_provider
                        raw = await _lookup_provider(self._lookup_fn, claimed.provider_request_id)
                    else:
                        raw = await self._lookup_fn(claimed.provider_request_id)
                    if raw is not None:
                        result = self._normalize(raw, investigation_allowed, steering_allowed)
                except Exception:
                    pass
        return await self._continue_result(project_id, goal_id, claimed.id, actor_id, result, authority)

    async def history(
        self, project_id: uuid.UUID, goal_id: uuid.UUID, actor_id: uuid.UUID
    ) -> list[ConversationTurn]:
        async with self._session_factory() as db:
            if await self._orchestration.get_goal(db, project_id, goal_id) is None:
                raise ConversationDomainError(
                    "goal_not_found", 404, "Orchestration goal not found"
                )
            rows = await db.execute(
                select(
                    ConversationMessage,
                    ConversationResponse,
                    ConversationInvestigation,
                    ConversationFeedback,
                )
                .select_from(ConversationMessage)
                .join(ConversationResponse, ConversationResponse.message_id == ConversationMessage.id)
                .outerjoin(ConversationInvestigation, ConversationInvestigation.response_id == ConversationResponse.id)
                .outerjoin(
                    ConversationFeedback,
                    (ConversationFeedback.response_id == ConversationResponse.id)
                    & (ConversationFeedback.actor_id == actor_id),
                )
                .where(
                    ConversationMessage.goal_id == goal_id,
                )
                .order_by(ConversationMessage.sequence)
            )
            return [
                ConversationTurn(
                    message=message,
                    response=response,
                    investigation=investigation,
                    feedback=feedback,
                    feedback_eligible=(
                        message.actor_id == actor_id
                        and response.status == "completed"
                        and isinstance(response.answer, str)
                        and bool(response.answer.strip())
                        and feedback is None
                    ),
                )
                for message, response, investigation, feedback in rows.all()
            ]

    async def recover_all(self) -> None:
        async with self._session_factory() as db:
            goal_ids = (
                await db.scalars(
                    select(ConversationMessage.goal_id)
                    .join(ConversationResponse)
                    .join(ConversationReservation)
                    .where(
                        (ConversationResponse.status == "pending")
                        | (ConversationResponse.status == "running")
                    )
                    .distinct()
                )
            ).all()
        for goal_id in goal_ids:
            await self.recover_goal(goal_id)
        await self._investigations.recover_all()

    async def recover_goal(self, goal_id: uuid.UUID) -> None:
        async with self._session_factory() as db:
            attempts = (
                await db.execute(
                    select(
                        ConversationResponse.id,
                        ConversationResponse.provider_request_id,
                        ConversationResponse.status,
                        ConversationResponse.dossier,
                        ConversationResponse.context_version,
                        ConversationMessage.actor_id,
                        OrchestrationGoal.project_id,
                    )
                    .select_from(ConversationResponse)
                    .join(
                        ConversationMessage,
                        ConversationMessage.id == ConversationResponse.message_id,
                    )
                    .join(
                        ConversationReservation,
                        ConversationReservation.response_id == ConversationResponse.id,
                    )
                    .join(
                        OrchestrationGoal,
                        OrchestrationGoal.id == ConversationMessage.goal_id,
                    )
                    .where(
                        ConversationMessage.goal_id == goal_id,
                        (
                            (ConversationResponse.status == "pending")
                            & (ConversationReservation.status == "reserved")
                            & (
                                ConversationResponse.created_at
                                < _utcnow() - timedelta(seconds=120)
                            )
                        )
                        | (
                            (ConversationResponse.status == "running")
                            & (ConversationReservation.status == "committed")
                            & (ConversationResponse.deadline_at < _utcnow())
                        ),
                    )
                )
            ).all()
        for (
            response_id,
            provider_request_id,
            status,
            dossier,
            context_version,
            actor_id,
            project_id,
        ) in attempts:
            if status == "pending":
                await self._release_interrupted(goal_id, response_id)
                continue
            result = ConversationProviderResult("unknown", None, None, None)
            allowed = isinstance(dossier, dict) and isinstance(dossier.get("_conversation_runtime"), dict) and dossier["_conversation_runtime"].get("investigation_enabled") is True
            steering_allowed = isinstance(dossier, dict) and isinstance(dossier.get("_conversation_runtime"), dict) and dossier["_conversation_runtime"].get("steering_enabled") is True
            authority = self._authority_values(response_id, context_version, provider_request_id, dossier) if allowed or steering_allowed else None
            if self._lookup_fn is not None:
                try:
                    if allowed:
                        from huddleroom.services.orchestration_conversation_investigation import _lookup_provider
                        raw = await _lookup_provider(self._lookup_fn, provider_request_id)
                    else:
                        raw = await self._lookup_fn(provider_request_id)
                    if raw is not None:
                        result = self._normalize(
                            raw,
                            allowed, steering_allowed,
                        )
                except Exception:
                    pass
            try:
                await self._continue_result(project_id, goal_id, response_id, actor_id, result, authority)
            except ConversationDomainError as exc:
                if (
                    result.kind != "investigation"
                    or exc.code != "conversation_investigation_ineligible"
                    or not await self._has_terminal_recovery_winner(
                        project_id,
                        goal_id,
                        response_id,
                        actor_id,
                        provider_request_id,
                        context_version,
                        dossier,
                    )
                ):
                    raise
        await self._investigations.recover_goal(goal_id)

    async def _has_terminal_recovery_winner(
        self,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        response_id: uuid.UUID,
        actor_id: uuid.UUID,
        provider_request_id: str,
        context_version: str,
        dossier: dict[str, Any],
    ) -> bool:
        """Recognize only a sibling's durable terminal transition."""
        async with self._session_factory() as db:
            response = await db.get(ConversationResponse, response_id)
            reservation = await db.get(
                ConversationReservation, conversation_reservation_id(response_id)
            )
            goal = await db.get(OrchestrationGoal, goal_id)
            message = (
                await db.get(ConversationMessage, response.message_id)
                if response is not None
                else None
            )
        return (
            response is not None
            and response.provider_request_id == provider_request_id
            and response.context_version == context_version
            and response.dossier == dossier
            and response.status in {"completed", "failed", "interrupted_unknown"}
            and message is not None
            and message.goal_id == goal_id
            and message.actor_id == actor_id
            and goal is not None
            and goal.project_id == project_id
            and reservation is not None
            and reservation.response_id == response_id
            and reservation.goal_id == goal_id
            and reservation.actor_id == actor_id
            and reservation.status in {"settled", "held_unknown"}
        )

    async def _release_interrupted(
        self, goal_id: uuid.UUID, response_id: uuid.UUID
    ) -> bool:
        async with self._session_factory() as db:
            async with db.begin():
                async with self._orchestration._lock_goal_for_baseline_transition(
                    db, goal_id
                ):
                    response = await db.get(ConversationResponse, response_id)
                    reservation = await db.get(
                        ConversationReservation,
                        conversation_reservation_id(response_id),
                    )
                    if (
                        response is None
                        or reservation is None
                        or response.status != "pending"
                        or reservation.status != "reserved"
                    ):
                        return False
                    response.status, response.error, response.finished_at = (
                        "failed",
                        {"code": "interrupted_before_dispatch"},
                        _utcnow(),
                    )
                    (
                        reservation.status,
                        reservation.released_tokens,
                        reservation.released_at,
                    ) = "released", reservation.reserved_tokens, _utcnow()
                    return True

    async def _prepare(
        self,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        actor_id: uuid.UUID,
        request_id: uuid.UUID,
        content: str,
    ) -> tuple[ConversationTurn, bool]:
        async with self._session_factory() as db:
            async with db.begin():
                async with self._orchestration._lock_goal_for_baseline_transition(
                    db, goal_id
                ):
                    goal = await self._orchestration.get_goal(db, project_id, goal_id)
                    if goal is None:
                        raise ConversationDomainError(
                            "goal_not_found", 404, "Orchestration goal not found"
                        )
                    message_id = conversation_message_id(goal_id, actor_id, request_id)
                    message = await db.get(ConversationMessage, message_id)
                    if message is not None:
                        response = await db.get(
                            ConversationResponse, conversation_response_id(message.id)
                        )
                        if message.content != content:
                            raise ConversationDomainError(
                                "idempotency_conflict",
                                409,
                                "Request ID has different content",
                            )
                        investigation = await db.scalar(
                            select(ConversationInvestigation).where(
                                ConversationInvestigation.response_id == response.id
                            )
                        )
                        return ConversationTurn(
                            message=message, response=response, investigation=investigation
                        ), False
                    if settings.orchestration_conversation_allowance_tokens <= 0:
                        raise ConversationDomainError(
                            "conversation_disabled", 409, "Conversation is disabled"
                        )
                    pairs = (
                        await db.execute(
                            select(ConversationMessage, ConversationResponse)
                            .join(ConversationResponse)
                            .where(
                                ConversationMessage.goal_id == goal_id,
                                ConversationResponse.status == "completed",
                            )
                            .order_by(ConversationMessage.sequence)
                        )
                    ).all()
                    run = await self._orchestration.get_run_for_goal(
                        db, project_id, goal_id
                    )
                    build = await ConversationDossierBuilder(db).build(
                        goal, run, content, pairs
                    )
                    steering_allowed = (
                        settings.orchestration_conversation_steering_enabled
                        and run is not None
                        and OrchestrationSteeringService._eligibility(goal, run, actor_id)[0] in {"active", "paused"}
                    )
                    demand = (
                        len(
                            json.dumps(
                                build.provider_messages,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ).encode("utf-8")
                        )
                        + 64
                        + self._completion_tokens(steering_allowed)
                    )
                    from huddleroom.services.orchestration_conversation_investigation import conversation_allowance_used
                    used = await conversation_allowance_used(db, goal_id, actor_id)
                    if (
                        int(used) + demand
                        > settings.orchestration_conversation_allowance_tokens
                    ):
                        raise ConversationDomainError(
                            "conversation_exhausted",
                            429,
                            "Conversation allowance exhausted",
                        )
                    sequence = (
                        int(
                            await db.scalar(
                                select(
                                    func.coalesce(
                                        func.max(ConversationMessage.sequence), 0
                                    )
                                ).where(ConversationMessage.goal_id == goal_id)
                            )
                        )
                        + 1
                    )
                    message = ConversationMessage(
                        id=message_id,
                        goal_id=goal_id,
                        actor_id=actor_id,
                        client_request_id=request_id,
                        sequence=sequence,
                        content=content,
                    )
                    runtime = {}
                    if settings.orchestration_conversation_investigation_enabled:
                        runtime["investigation_enabled"] = True
                    if steering_allowed:
                        runtime["steering_enabled"] = True
                    response = ConversationResponse(
                        id=conversation_response_id(message_id),
                        message_id=message_id,
                        run_id=build.run_id,
                        dossier={**build.dossier, "_conversation_runtime": runtime} if runtime else build.dossier,
                        context_manifest=build.manifest,
                        context_version=build.context_version,
                        provider_request_id=conversation_provider_request_id(
                            conversation_response_id(message_id)
                        ),
                    )
                    reservation = ConversationReservation(
                        id=conversation_reservation_id(response.id),
                        response_id=response.id,
                        goal_id=goal_id,
                        actor_id=actor_id,
                        ceiling_snapshot=settings.orchestration_conversation_allowance_tokens,
                        reserved_tokens=demand,
                    )
                    db.add_all((message, response))
                    await db.flush()
                    db.add(reservation)
                    await db.flush()
                    return ConversationTurn(
                        message=message, response=response, provider_messages=build.provider_messages
                    ), True

    async def _claim(
        self, goal_id: uuid.UUID, response_id: uuid.UUID
    ) -> ConversationResponse | None:
        async with self._session_factory() as db:
            async with db.begin():
                async with self._orchestration._lock_goal_for_baseline_transition(
                    db, goal_id
                ):
                    response = await db.get(ConversationResponse, response_id)
                    reservation = await db.get(
                        ConversationReservation,
                        conversation_reservation_id(response_id),
                    )
                    if (
                        response is None
                        or reservation is None
                        or response.status != "pending"
                        or reservation.status != "reserved"
                    ):
                        return None
                    now = _utcnow()
                    response.status, response.started_at, response.deadline_at = (
                        "running",
                        now,
                        now + timedelta(seconds=120),
                    )
                    reservation.status, reservation.committed_at = "committed", now
                    await db.flush()
                    return response

    async def _settle(
        self, goal_id: uuid.UUID, response_id: uuid.UUID, answer: str, usage: int | None,
        authority=None, proposal: SteeringDraft | None = None,
    ) -> ConversationTurn:
        async with self._session_factory() as db:
            async with db.begin():
                async with self._orchestration._lock_goal_for_baseline_transition(
                    db, goal_id
                ):
                    response = await db.get(ConversationResponse, response_id)
                    reservation = await db.get(
                        ConversationReservation,
                        conversation_reservation_id(response_id),
                    )
                    if (
                        not self._authority_matches(response, authority)
                    ):
                        raise ConversationDomainError("conversation_investigation_ineligible", 409, "Conversation investigation is ineligible")
                    if (
                        response.status != "running"
                        or reservation.status != "committed"
                    ):
                        return await self._turn(response_id)
                    now = _utcnow()
                    response.status, response.answer, response.finished_at = (
                        "completed",
                        answer,
                        now,
                    )
                    if proposal is not None and self._steering_allowed(response):
                        message = await db.get(ConversationMessage, response.message_id)
                        await self._persist_proposal(db, message, response, proposal)
                    if usage is None or usage > reservation.reserved_tokens:
                        reservation.status = "held_unknown"
                    else:
                        (
                            reservation.status,
                            reservation.settled_tokens,
                            reservation.released_tokens,
                        ) = "settled", usage, reservation.reserved_tokens - usage
                        reservation.settled_at = reservation.released_at = now
                    return ConversationTurn(
                        message=await db.get(ConversationMessage, response.message_id),
                        response=response,
                    )

    async def _fail_known(
        self, goal_id: uuid.UUID, response_id: uuid.UUID, code: str, usage: int | None, authority=None
    ) -> ConversationTurn:
        async with self._session_factory() as db:
            async with db.begin():
                async with self._orchestration._lock_goal_for_baseline_transition(db, goal_id):
                    response = await db.get(ConversationResponse, response_id)
                    reservation = await db.get(
                        ConversationReservation, conversation_reservation_id(response_id)
                    )
                    if (
                        response is None or reservation is None
                    ):
                        return await self._turn(response_id)
                    if not self._authority_matches(response, authority):
                        raise ConversationDomainError("conversation_investigation_ineligible", 409, "Conversation investigation is ineligible")
                    if (
                        response.status != "running"
                        or reservation.status != "committed"
                    ):
                        return await self._turn(response_id)
                    now = _utcnow()
                    response.status, response.error, response.finished_at = "failed", {"code": code}, now
                    if usage is None or usage > reservation.reserved_tokens:
                        reservation.status = "held_unknown"
                    else:
                        reservation.status, reservation.settled_tokens, reservation.released_tokens = (
                            "settled", usage, reservation.reserved_tokens - usage
                        )
                        reservation.settled_at = reservation.released_at = now
                    return ConversationTurn(
                        message=await db.get(ConversationMessage, response.message_id), response=response
                    )

    async def _persist_proposal(
        self, db: AsyncSession, message: ConversationMessage, response: ConversationResponse,
        draft: SteeringDraft,
    ) -> None:
        """Persist a proposal under the already-held goal lock, without rereading settings."""
        goal = await db.get(OrchestrationGoal, message.goal_id)
        run = await self._orchestration.get_run_for_goal(db, goal.project_id, goal.id) if goal else None
        if (
            goal is None or run is None or response.run_id != run.id
            or OrchestrationSteeringService._eligibility(goal, run, message.actor_id)[0] not in {"active", "paused"}
        ):
            return
        try:
            await OrchestrationSteeringService(self._orchestration)._validate_target(db, goal, run, draft)
        except SteeringDomainError:
            return
        proposal_id = steering_proposal_id(response.id)
        if await db.get(OrchestrationSteeringProposal, proposal_id) is None:
            db.add(OrchestrationSteeringProposal(
                id=proposal_id, response_id=response.id, goal_id=goal.id, actor_id=message.actor_id,
                draft={
                    "directive": draft.directive, "target_type": draft.target_type,
                    "target_id": draft.target_id, "scope": draft.scope,
                    "lifetime": draft.lifetime, "impact_summary": draft.impact_summary,
                },
            ))

    async def _continue_result(
        self,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        response_id: uuid.UUID,
        actor_id: uuid.UUID,
        result: ConversationProviderResult,
        authority: _ChatAuthority | None = None,
    ) -> ConversationTurn:
        if result.kind == "answer":
            assert result.answer is not None
            return await self._settle(goal_id, response_id, result.answer, result.usage, authority)
        if result.kind == "proposal":
            assert result.answer is not None and result.proposal is not None
            return await self._settle(
                goal_id, response_id, result.answer, result.usage, authority, result.proposal
            )
        if result.kind == "invalid":
            return await self._fail_known(
                goal_id, response_id, "invalid_investigation_request", result.usage, authority
            )
        if result.kind == "unknown":
            return await self._hold_unknown(goal_id, response_id, "provider_outcome_unknown", authority)
        assert result.request is not None
        investigation = await self._investigations.execute(
            project_id, goal_id, actor_id, response_id, authority.context_version if authority else (await self._turn(response_id)).response.context_version,
            result.request, result.usage, authority,
        )
        return await self._turn(response_id, investigation)

    async def _hold_unknown(
        self, goal_id: uuid.UUID, response_id: uuid.UUID, code: str, authority=None
    ) -> ConversationTurn:
        async with self._session_factory() as db:
            async with db.begin():
                async with self._orchestration._lock_goal_for_baseline_transition(
                    db, goal_id
                ):
                    response = await db.get(ConversationResponse, response_id)
                    reservation = await db.get(
                        ConversationReservation,
                        conversation_reservation_id(response_id),
                    )
                    if (
                        not self._authority_matches(response, authority)
                    ):
                        raise ConversationDomainError("conversation_investigation_ineligible", 409, "Conversation investigation is ineligible")
                    if (
                        response.status != "running"
                        or reservation.status != "committed"
                    ):
                        return await self._turn(response_id)
                    return await self._hold(db, response, reservation, code)

    async def _hold(
        self,
        db: AsyncSession,
        response: ConversationResponse,
        reservation: ConversationReservation,
        code: str,
    ) -> ConversationTurn:
        now = _utcnow()
        response.status, response.error, response.finished_at = (
            "interrupted_unknown",
            {"code": code},
            now,
        )
        reservation.status = "held_unknown"
        return ConversationTurn(
            message=await db.get(ConversationMessage, response.message_id), response=response
        )

    async def _turn(
        self, response_id: uuid.UUID, investigation: ConversationInvestigation | None = None
    ) -> ConversationTurn:
        async with self._session_factory() as db:
            response = await db.get(ConversationResponse, response_id)
            if investigation is None:
                investigation = await db.scalar(
                    select(ConversationInvestigation).where(
                        ConversationInvestigation.response_id == response_id
                    )
                )
            return ConversationTurn(
                message=await db.get(ConversationMessage, response.message_id),
                response=response,
                investigation=investigation,
            )

    @staticmethod
    def _normalize(
        raw: Any, investigation_allowed: bool = False, steering_allowed: bool = False,
    ) -> ConversationProviderResult:
        def get(value: Any, key: str, default: Any = None) -> Any:
            return (
                value.get(key, default)
                if isinstance(value, Mapping)
                else getattr(value, key, default)
            )

        choices = get(raw, "choices", [])
        if not investigation_allowed and not steering_allowed:
            message = get(choices[0], "message") if choices else None
            content = get(message, "content")
            return ConversationProviderResult(
                "answer" if isinstance(content, str) and content.strip() else "unknown",
                content.strip() if isinstance(content, str) and content.strip() else None,
                None,
                OrchestrationConversationService._usage(raw, get),
            )
        return OrchestrationConversationService._normalize_enabled(raw, get, investigation_allowed, steering_allowed)

    @staticmethod
    def _normalize_enabled(
        raw: Any, get: Callable[[Any, str, Any], Any], investigation_allowed: bool, steering_allowed: bool,
    ) -> ConversationProviderResult:
        choices = get(raw, "choices", [])
        if not isinstance(choices, list) or len(choices) != 1:
            return ConversationProviderResult("invalid", None, None, OrchestrationConversationService._usage(raw, get))
        total = OrchestrationConversationService._usage(raw, get)
        message = get(choices[0], "message")
        raw_content = get(message, "content")
        if raw_content is not None and not isinstance(raw_content, str):
            return ConversationProviderResult("invalid", None, None, total)
        content = raw_content.strip() if raw_content and raw_content.strip() else None
        tool_calls = get(message, "tool_calls")
        if content and not tool_calls:
            return ConversationProviderResult("answer", content, None, total)
        if not content and isinstance(tool_calls, list) and len(tool_calls) == 1:
            function = get(tool_calls[0], "function")
            if investigation_allowed and get(function, "name") == "request_investigation":
                return OrchestrationConversationService._investigation_request(
                    get(function, "arguments"), total
                )
            if steering_allowed and get(function, "name") == "respond_with_proposed_steering":
                return OrchestrationConversationService._steering_proposal(get(function, "arguments"), total)
        if content or tool_calls:
            return ConversationProviderResult("invalid", None, None, total)
        return ConversationProviderResult("unknown", None, None, total)

    @staticmethod
    def _investigation_request(arguments: Any, usage: int | None) -> ConversationProviderResult:
        from huddleroom.services.orchestration_conversation_investigation import parse_investigation_request
        try:
            request = parse_investigation_request(arguments)
        except ValueError:
            return ConversationProviderResult("invalid", None, None, usage)
        return ConversationProviderResult("investigation", None, request, usage)

    @staticmethod
    def _steering_proposal(arguments: Any, usage: int | None) -> ConversationProviderResult:
        try:
            payload = json.loads(arguments) if isinstance(arguments, str) else None
            if not isinstance(payload, dict) or set(payload) != {"answer", "proposal"}:
                raise ValueError
            answer, proposal = payload["answer"], payload["proposal"]
            keys = {"directive", "target_type", "target_id", "scope", "lifetime", "impact_summary"}
            if (
                not isinstance(answer, str) or not 1 <= len(answer.strip()) <= 8_000
                or not isinstance(proposal, dict) or set(proposal) != keys
            ):
                raise ValueError
            if any(not isinstance(proposal[key], str) for key in keys):
                raise ValueError
            draft = normalize_draft(SteeringDraft(**proposal))
            if draft.target_type not in {"goal", "plan_item", "task"}:
                raise ValueError
            if (draft.scope == "item") != (draft.target_type in {"plan_item", "task"}):
                raise ValueError
        except (SteeringDomainError, TypeError, ValueError, json.JSONDecodeError):
            return ConversationProviderResult("invalid", None, None, usage)
        return ConversationProviderResult("proposal", answer.strip(), None, usage, draft)

    @staticmethod
    def _usage(raw: Any, get: Callable[[Any, str, Any], Any]) -> int | None:
        usage = get(raw, "usage")
        prompt, completion = get(usage, "prompt_tokens"), get(usage, "completion_tokens")
        if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (prompt, completion)):
            return prompt + completion
        return None

    @staticmethod
    def _investigation_allowed(response: ConversationResponse) -> bool:
        runtime = response.dossier.get("_conversation_runtime") if isinstance(response.dossier, dict) else None
        return isinstance(runtime, dict) and runtime.get("investigation_enabled") is True

    @staticmethod
    def _steering_allowed(response: ConversationResponse) -> bool:
        runtime = response.dossier.get("_conversation_runtime") if isinstance(response.dossier, dict) else None
        return isinstance(runtime, dict) and runtime.get("steering_enabled") is True

    @staticmethod
    def _completion_tokens(steering_allowed: bool) -> int:
        return 1_600 if steering_allowed and settings.orchestration_model == "openrouter/minimax/minimax-m3" else 800

    @staticmethod
    def _authority(response: ConversationResponse) -> _ChatAuthority:
        return OrchestrationConversationService._authority_values(
            response.id, response.context_version, response.provider_request_id, response.dossier
        )

    @staticmethod
    def _authority_values(response_id, context_version, provider_request_id, dossier) -> _ChatAuthority:
        return _ChatAuthority(response_id, context_version, provider_request_id, json.dumps(
            dossier, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ))

    @staticmethod
    def _authority_matches(response: ConversationResponse, authority: _ChatAuthority | None) -> bool:
        return authority is None or (
            response.id == authority.response_id
            and response.context_version == authority.context_version
            and response.provider_request_id == authority.provider_request_id
            and json.dumps(response.dossier, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == authority.dossier
        )
