"""Pure builder for the verifier/reviewer brief: shows the actual deliverable
and asks for a judgement against its own conditions only."""
from __future__ import annotations

from typing import Any, Iterable

MAX_SCOPE_CHARS = 1500
MAX_DELIVERABLE_CHARS = 1000
MAX_EVIDENCE_ITEMS = 10
MAX_EVIDENCE_ITEM_CHARS = 500
MAX_CRITERIA = 20
MAX_CRITERION_CHARS = 800
MAX_REPORT_EXCERPT_CHARS = 4000
MAX_ARTIFACTS = 20

JUDGE_INSTRUCTION = "Judge this deliverable against these conditions only, not the overall goal."
FRESH_EVIDENCE_REASON = "Verify the latest work for the stale orchestration gate. Report fresh evidence only."


def _clip(text: Any, limit: int, label: str) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n[{label} truncated: showing {limit} of {len(text)} characters]"


def _limit(items: Iterable[Any], limit: int, label: str) -> tuple[list[Any], str | None]:
    items = list(items or [])
    if len(items) <= limit:
        return items, None
    return items[:limit], f"[{label} truncated: showing {limit} of {len(items)}]"


def _field(artifact: Any, name: str) -> Any:
    return artifact.get(name) if isinstance(artifact, dict) else getattr(artifact, name, None)


def render_artifact(artifact: Any) -> str:
    """Artifact is a dict or object with `id` and `version` and/or `content_hash` (or `hash`)."""
    version = _field(artifact, "version")
    digest = _field(artifact, "content_hash") or _field(artifact, "hash")
    parts = [f"Artifact {_field(artifact, 'id')}"]
    parts.append(f"version {version}" if version is not None else "version unknown")
    parts.append(f"hash {digest}" if digest else "hash unknown")
    return ", ".join(parts)


def build_review_inputs(
    scope: str,
    deliverable: str,
    success_evidence: Iterable[str],
    criterion_lines: Iterable[str],
    report_excerpt: str,
    marker_ref: str | None,
    artifacts: Iterable[Any],
    *,
    stale_evidence: bool = False,
) -> dict[str, Any]:
    """Return {"scope": str, "deliverable": str, "inputs": list[str]} with every section bounded.

    `stale_evidence` is True only when the gate failed because its evidence went
    stale; only then does the brief mention the stale orchestration gate.
    """
    scope_text = _clip(scope, MAX_SCOPE_CHARS, "Scope")
    full_scope = f"{FRESH_EVIDENCE_REASON} {JUDGE_INSTRUCTION}" if stale_evidence else JUDGE_INSTRUCTION
    full_scope = f"{full_scope}\n\n{scope_text}" if scope_text else full_scope

    evidence, evidence_note = _limit(success_evidence, MAX_EVIDENCE_ITEMS, "Success evidence")
    criteria, criteria_note = _limit(criterion_lines, MAX_CRITERIA, "Criteria")
    arts, arts_note = _limit(artifacts, MAX_ARTIFACTS, "Artifacts")

    inputs = ["Success evidence:"] + [f"- {_clip(e, MAX_EVIDENCE_ITEM_CHARS, 'Item')}" for e in evidence]
    if evidence_note:
        inputs.append(evidence_note)
    inputs.append("Conditions to judge against:")
    inputs += [f"- {_clip(c, MAX_CRITERION_CHARS, 'Criterion')}" for c in criteria]
    if criteria_note:
        inputs.append(criteria_note)
    excerpt = _clip(report_excerpt, MAX_REPORT_EXCERPT_CHARS, "Producer report")
    ref = f" (marker {marker_ref})" if marker_ref else ""
    inputs.append(f"Producer report{ref}:\n{excerpt or '[no report content]'}")
    inputs.append("Artifacts:")
    inputs += [f"- {render_artifact(a)}" for a in arts] or ["- none"]
    if arts_note:
        inputs.append(arts_note)

    return {
        "scope": full_scope,
        "deliverable": _clip(deliverable, MAX_DELIVERABLE_CHARS, "Deliverable"),
        "inputs": inputs,
    }
