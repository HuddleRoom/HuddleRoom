"""Bounded structured diagnosis for recovery blockers."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable
from uuid import UUID

from pydantic import TypeAdapter

from huddleroom.config import settings
from huddleroom.schemas.orchestration_recovery import (
    EvidenceV1,
    RECOVERY_EVIDENCE_QUERY_ADAPTER,
    RecoveryAssessmentV1,
    RecoveryEvidenceQueryV1,
)
from huddleroom.services.llm_structured_repair import complete_with_repair
from huddleroom.services.orchestration_completion import get_orchestration_completion, orchestration_runtime_metadata
from huddleroom.services.orchestration_llm_decision_adapter import _redact_secrets
from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext


MAX_RECOVERY_EVIDENCE_ROUNDS = 4
RAW_OUTPUT_MAX_CHARS = 16_384
_ASSESSMENT = TypeAdapter(RecoveryAssessmentV1)


def parse_recovery_assessment(payload: object) -> RecoveryAssessmentV1:
    return _ASSESSMENT.validate_python(payload)


class RecoveryAnalyzer:
    def __init__(self, completion_fn: Callable[..., Awaitable[Any]] | None = None) -> None:
        self._completion_fn = get_orchestration_completion(completion_fn)

    @staticmethod
    def _request(snapshot: dict, evidence: list[EvidenceV1]) -> dict[str, Any]:
        prompt = (
            "Diagnose the recovery blocker. Return one JSON object only. You may request evidence with "
            '{"kind":"evidence_query","query":{"kind":"run_state",...}}. Otherwise return one terminal '
            "assessment: transient, proposal, or capability_gap. Never author IDs, versions, retry targets, "
            "authority, hashes, registry data, secrets, predicates, queries, or executable code."
        )
        return {
            "model": settings.orchestration_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": _redact_secrets(json.dumps({"snapshot": snapshot, "evidence": [item.model_dump(mode="json") for item in evidence]}, sort_keys=True, default=str))},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

    async def analyze(
        self,
        snapshot: dict,
        *,
        project_id: UUID,
        evidence_query: Callable[[RecoveryEvidenceQueryV1], Awaitable[EvidenceV1]],
        on_raw: Callable[[str], None] | None = None,
    ) -> RecoveryAssessmentV1:
        """`on_raw` receives each redacted, bounded raw model output (for audit storage)."""
        evidence: list[EvidenceV1] = []
        for _ in range(MAX_RECOVERY_EVIDENCE_ROUNDS + 1):
            request = self._request(snapshot, evidence)

            def parse(raw: str) -> Any:
                if on_raw is not None:
                    on_raw(_redact_secrets(raw)[:RAW_OUTPUT_MAX_CHARS])
                payload = json.loads(raw)
                if isinstance(payload, dict) and set(payload) == {"kind", "query"} and payload.get("kind") == "evidence_query":
                    return RECOVERY_EVIDENCE_QUERY_ADAPTER.validate_python(payload["query"])
                return parse_recovery_assessment(payload)

            try:
                invocation_kind, runtime_model = orchestration_runtime_metadata(self._completion_fn, request["model"])

                async def safe_completion(**values: Any) -> Any:
                    try:
                        return await self._completion_fn(**values)
                    except Exception as exc:
                        raise RuntimeError(_redact_secrets(str(exc))) from None

                result = await complete_with_repair(
                    safe_completion, request, parse,
                    invocation=AgentResponseInvocation(InvocationContext(
                        project_id, "system", "orchestrator", "Orchestrator", invocation_kind,
                        "recovery_analysis", runtime_model, "Diagnose a blocker using bounded evidence.",
                    )),
                )
            except Exception as exc:
                raise RuntimeError(_redact_secrets(f"Recovery analysis failed: {exc}")) from None
            if getattr(result, "kind", None) not in {"run_state", "action", "process", "team_configuration", "task", "gate", "waits", "warning", "agent_execution"}:
                return result
            if len(evidence) == MAX_RECOVERY_EVIDENCE_ROUNDS:
                raise ValueError("recovery evidence round limit reached")
            try:
                evidence.append(await evidence_query(result))
            except Exception as exc:
                raise RuntimeError(_redact_secrets(f"Recovery evidence failed: {exc}")) from None
        raise RuntimeError("recovery analysis exhausted")
