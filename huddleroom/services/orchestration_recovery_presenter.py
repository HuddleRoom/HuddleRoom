"""Read-only typed projection of recovery decisions for the decision routes."""
import json
import re
import uuid
from typing import Any

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationAction
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.schemas.orchestration import OrchestrationAuthorityDecisionResponse
from huddleroom.schemas.orchestration_recovery import RecoveryProposalResponse
from huddleroom.services.orchestration_blocker_recovery import OrchestrationBlockerRecoveryService
from huddleroom.services.orchestration_recovery_configuration import RecoveryConfigurationService
from huddleroom.services.secret_redaction import redact_secrets as _redact_secrets

PROPOSAL_PREFIX = "recovery:"
GAP_PREFIX = "recovery_gap:"
_STATUSES = ("pending_approval", "applying", "verifying", "retrying", "resolved", "rejected", "failed", "unsupported")
_GAP_AFTER = ("Nothing happens automatically. After you provide the capability and mark it done, "
              "HuddleRoom re-diagnoses the blocker.")


def is_recovery_decision(decision: OrchestrationAuthorityDecision) -> bool:
    return decision.decision_key.startswith((PROPOSAL_PREFIX, GAP_PREFIX))


def _clean(value: Any) -> Any:
    """Semantic + textual redaction of arbitrary stored JSON before it is re-validated."""
    if isinstance(value, str):
        return _redact_secrets(value)
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return value


def _json(raw: str | None) -> dict:
    try:
        value = json.loads(raw or "")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _source_links(source: dict) -> dict:
    try:
        source_id = uuid.UUID(str(source.get("id")))
    except ValueError:
        return {}
    key = {"failed_action": "source_action_id", "reported_blocker": "source_decision_id"}.get(
        source.get("kind"), "source_blocker_id")
    return {key: source_id}


class RecoveryProposalPresenter:
    def __init__(self) -> None:
        self._service = OrchestrationBlockerRecoveryService()

    async def present(
        self, db: AsyncSession, decision: OrchestrationAuthorityDecision,
    ) -> RecoveryProposalResponse | None:
        if not is_recovery_decision(decision):
            return None
        action = (await db.get(OrchestrationAction, decision.related_action_id)
                  if decision.related_action_id is not None else None)
        if decision.decision_key.startswith(GAP_PREFIX):
            return self._gap(decision, action)
        return await self._proposal(db, decision, action)

    async def present_many(
        self, db: AsyncSession, decisions: list[OrchestrationAuthorityDecision],
    ) -> list[RecoveryProposalResponse | None]:
        return [await self.present(db, decision) for decision in decisions]

    @staticmethod
    def _safe_error(raw: str | None) -> str | None:
        if not raw:
            return None
        line = _redact_secrets(raw.strip().splitlines()[0] if raw.strip() else "")[:200]
        return line if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", line) else "The recovery attempt failed."

    @staticmethod
    def _minimal(decision, action, status: str) -> RecoveryProposalResponse:
        """Fail-closed fallback when stored context is unusable: status and plain question only."""
        try:
            pid = uuid.UUID(str(((action.request if action is not None else None) or {}).get("proposal_id")))
        except ValueError:
            pid = decision.id
        question = _redact_secrets(decision.question or "")[:4_000].strip()
        for text in (question, "Recovery proposal."):
            try:
                return RecoveryProposalResponse(
                    kind="recovery_proposal_v1", proposal_id=pid, version=1, status=status, problem=text,
                    general_updates=[text], after_approval=_GAP_AFTER, global_effect=False)
            except ValidationError:
                continue
        raise AssertionError("unreachable")

    async def _proposal(self, db, decision, action) -> RecoveryProposalResponse:
        data = _clean(_json(decision.context))
        if action is not None and action.action_type == "recovery_proposal":
            data["status"] = (await self._service.project_status(db, action)).value
            if action.status == "failed" and action.error and action.error != "rejected_by_human":
                data["error"] = action.error
        else:
            # No linked proposal: never leave the stored pending_approval (fail closed, not approvable).
            data["status"] = "rejected" if decision.selected_option == "reject" else "failed"
        status = data["status"] if data.get("status") in _STATUSES else "failed"
        if data.get("exact_details") is not None:
            data["exact_details"] = RecoveryConfigurationService._disclosure_value(data["exact_details"])
        if data.get("error") is not None:
            data["error"] = self._safe_error(str(data["error"]))
        # On validation failure null only the offending optional fields; otherwise a minimal projection.
        for _ in range(2):
            try:
                return RecoveryProposalResponse.model_validate(data)
            except ValidationError as exc:
                bad = {err["loc"][0] for err in exc.errors() if err["loc"]}
                if not bad or not bad <= {"error", "evidence", "exact_details"}:
                    break
                for name in bad:
                    data[name] = None
        return self._minimal(decision, action, status)

    def _gap(self, decision, action) -> RecoveryProposalResponse | None:
        request = (action.request if action is not None else None) or {}
        data = _clean(_json(decision.context))
        try:
            gap_id = uuid.UUID(str(request.get("gap_id") or decision.decision_key[len(GAP_PREFIX):]))
            return RecoveryProposalResponse(
                kind="recovery_proposal_v1", proposal_id=gap_id,
                version=((action.dispatch_contract or {}).get("attempt") or 1) if action is not None else 1,
                status="unsupported", problem=data.get("missing_capability") or decision.question,
                general_updates=[data.get("required_human_action") or decision.question],
                after_approval=_GAP_AFTER, global_effect=False, **_source_links(request.get("source") or {}),
            )
        except (ValidationError, ValueError):
            return self._minimal(decision, action, "unsupported")


_presenter: RecoveryProposalPresenter | None = None


async def present_decision_responses(
    db: AsyncSession, decisions: list[OrchestrationAuthorityDecision],
) -> list[OrchestrationAuthorityDecisionResponse]:
    """Response models with the typed recovery projection; raw context never leaves for recovery decisions."""
    global _presenter
    _presenter = _presenter or RecoveryProposalPresenter()
    projected = await _presenter.present_many(db, decisions)
    return [
        OrchestrationAuthorityDecisionResponse.model_validate(decision).model_copy(
            update={"context": None, "recovery": recovery} if is_recovery_decision(decision) else {})
        for decision, recovery in zip(decisions, projected)
    ]


async def present_decision_response(
    db: AsyncSession, decision: OrchestrationAuthorityDecision,
) -> OrchestrationAuthorityDecisionResponse:
    return (await present_decision_responses(db, [decision]))[0]
