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
from huddleroom.services.agent_response_stream import (
    AgentResponseCall, AgentResponseInvocation, CompletionObserver, InvocationContext, supports_observer,
)


MAX_RECOVERY_EVIDENCE_ROUNDS = 4
RAW_OUTPUT_MAX_CHARS = 16_384
_ASSESSMENT = TypeAdapter(RecoveryAssessmentV1)


class RecoveryProviderUnavailable(RuntimeError):
    """Completion invocation failed before a usable response was returned."""


class RecoveryContractRejected(ValueError):
    """The bounded repair attempts exhausted the response contract."""


class RecoveryEvidenceUnavailable(RuntimeError):
    """An authoritative evidence lookup failed."""


class RecoveryEvidenceLimitReached(ValueError):
    """The analyzer requested evidence beyond its bounded allowance."""


def _recovery_examples() -> dict[str, dict]:
    """Shape examples only: replace example IDs with authoritative context IDs."""
    source_id = "00000000-0000-0000-0000-000000000001"
    evidence = [{"label": "Observed run state", "value": {"status": "blocked"},
                 "source_kind": "run", "source_id": source_id}]
    return {
        "transient": {"kind": "transient", "diagnosis": "A live producer is pending", "evidence": evidence,
                      "wake": {"events": [], "recheck_after_seconds": 30, "expected_result": "Producer finishes"}},
        "proposal": {"kind": "proposal", "diagnosis": "Configuration prevents progress", "evidence": evidence,
                     "problem": "Required capability is absent", "general_updates": ["Assign a capable agent"],
                     "expected_effect": "Blocked work can resume",
                     "calls": [{"tool": "team_configuration.apply@1", "arguments": {},
                                "rationale": "Use the registry arguments schema to provide the required assignment"}]},
        "capability_gap": {"kind": "capability_gap", "diagnosis": "Human authorization is required",
                           "evidence": evidence, "missing_capability": "Approved execution authority",
                           "required_human_action": "Review the required authorization",
                           "wake_after": {"kind": "human_decision", "decision": "Authorize the required work"}},
        "evidence_query": {"kind": "evidence_query", "query": {"kind": "run_state", "run_id": source_id}},
    }


def parse_recovery_assessment(payload: object) -> RecoveryAssessmentV1:
    return _ASSESSMENT.validate_python(payload)


class RecoveryAnalyzer:
    def __init__(self, completion_fn: Callable[..., Awaitable[Any]] | None = None) -> None:
        self._completion_fn = get_orchestration_completion(completion_fn)

    @staticmethod
    def _request(snapshot: dict, evidence: list[EvidenceV1]) -> dict[str, Any]:
        prompt = (
            "Diagnose the recovery blocker. Return one JSON object only. You may request evidence with "
            'an evidence_query object. Otherwise return one terminal '
            "assessment: transient, proposal, or capability_gap. Never author IDs, versions, retry targets, "
            "authority, hashes, registry data, secrets, predicates, queries, or executable code. "
            "Reuse authoritative IDs from the snapshot/evidence for evidence citations and queries. "
            "The schemas below are authoritative; include every required field. "
            + json.dumps({
                "assessment_schema": _ASSESSMENT.json_schema(),
                "evidence_query_schema": RECOVERY_EVIDENCE_QUERY_ADAPTER.json_schema(),
                "examples": _recovery_examples(),
            }, sort_keys=True)
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

            invocation_kind, runtime_model = orchestration_runtime_metadata(self._completion_fn, request["model"])

            async def classified_completion(values: dict[str, Any], *, observing: bool = False) -> Any:
                observation_seen = False
                observer = values.get("observer")
                if observer is not None:
                    async def observed(stream, text):
                        nonlocal observation_seen
                        observation_seen = True
                        await observer(stream, text)
                    values = {**values, "observer": observed}
                try:
                    response = await self._completion_fn(**values)
                except Exception as exc:
                    if not observation_seen and (observing or values.get("stream") is True) and AgentResponseCall._streaming_unsupported(exc):
                        raise
                    raise RecoveryProviderUnavailable(_redact_secrets(str(exc))) from None
                if hasattr(response, "__aiter__"):
                    async def classified_stream():
                        seen = False
                        try:
                            async for chunk in response:
                                seen = True
                                yield chunk
                        except Exception as exc:
                            if not seen and not observation_seen and values.get("stream") is True and AgentResponseCall._streaming_unsupported(exc):
                                raise
                            raise RecoveryProviderUnavailable(_redact_secrets(str(exc))) from None
                    return classified_stream()
                return response

            async def plain_completion(**values: Any) -> Any:
                return await classified_completion(values)

            async def observed_completion(*, observer: CompletionObserver | None = None, **values: Any) -> Any:
                return await classified_completion({**values, "observer": observer}, observing=observer is not None)

            safe_completion = observed_completion if supports_observer(self._completion_fn) else plain_completion

            try:
                result = await complete_with_repair(
                    safe_completion, request, parse,
                    invocation=AgentResponseInvocation(InvocationContext(
                        project_id, "system", "orchestrator", "Orchestrator", invocation_kind,
                        "recovery_analysis", runtime_model, "Diagnose a blocker using bounded evidence.",
                    )),
                )
            except ValueError as exc:
                raise RecoveryContractRejected(_redact_secrets(str(exc))) from None
            if getattr(result, "kind", None) not in {"run_state", "action", "process", "team_configuration", "task", "gate", "waits", "warning", "agent_execution"}:
                return result
            if len(evidence) == MAX_RECOVERY_EVIDENCE_ROUNDS:
                raise RecoveryEvidenceLimitReached("recovery evidence round limit reached")
            try:
                evidence.append(await evidence_query(result))
            except Exception as exc:
                raise RecoveryEvidenceUnavailable(_redact_secrets(f"Recovery evidence failed: {exc}")) from None
        raise RuntimeError("recovery analysis exhausted")
