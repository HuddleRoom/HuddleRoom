"""Pure helpers: which result kind a delegated task must return, lossless output normalization, and
bounded result correction (failure fingerprint, no-progress rule, correction brief)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from huddleroom.services.llm_structured_repair import _unfence_json

PLANNING_ENVELOPE = "planning_envelope"
VERIFICATION_VERDICT = "verification_verdict"
FINAL_SUMMARY = "final_summary"
DECISION_REPORT = "decision_report"
CONTINUOUS_CANDIDATES = "continuous_candidates"
WORK_REPORT = "work_report"

RESULT_KINDS = frozenset({
    PLANNING_ENVELOPE, VERIFICATION_VERDICT, FINAL_SUMMARY, DECISION_REPORT, CONTINUOUS_CANDIDATES, WORK_REPORT,
})
_LEGACY_KINDS = {"canonical_work_report": WORK_REPORT}
# Kinds whose validators return exact errors; decision reports and continuous candidates keep their own handling.
CORRECTABLE_KINDS = frozenset({WORK_REPORT, PLANNING_ENVELOPE, VERIFICATION_VERDICT, FINAL_SUMMARY})
CORRECTION_INSTRUCTION = (
    "Correct and re-emit this result; keep valid content and stated uncertainty; do not redo the work, "
    "repeat tool or external effects, invent evidence or widen scope."
)


def expected_result_kind(contract: dict | None, has_gate: bool) -> str:
    """Stored valid kind wins (legacy value mapped); otherwise derive from the report schema keys."""
    contract = contract or {}
    stored = contract.get("expected_result")
    stored = _LEGACY_KINDS.get(stored, stored) if isinstance(stored, str) else None
    if stored in RESULT_KINDS:
        return stored
    schema = contract.get("report_schema")
    keys = schema.keys() if isinstance(schema, dict) else ()
    if "plan_kind" in keys and not has_gate:
        return PLANNING_ENVELOPE
    for key, kind in (
        ("verdict", VERIFICATION_VERDICT),
        ("criteria", FINAL_SUMMARY),
        ("selected_option", DECISION_REPORT),
        ("candidates", CONTINUOUS_CANDIDATES),
    ):
        if key in keys:
            return kind
    return WORK_REPORT


def _no_duplicate_keys(pairs: list) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r} in JSON object")
        result[key] = value
    return result


def _decode_object(text: str, start: int) -> int:
    """Return end index of the JSON object starting at ``start``; ValueError if not a valid object."""
    try:
        obj, end = json.JSONDecoder(object_pairs_hook=_no_duplicate_keys).raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise ValueError(f"result is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("result is not a JSON object")
    return end


def normalize_result_output(raw: str) -> tuple[str, str | None]:
    """Return (exact JSON object substring, "fence" | "framing" | None); ValueError if not exactly one object."""
    if not isinstance(raw, str):
        raise ValueError("result output is not text")
    unfenced = _unfence_json(raw)
    how = "fence" if unfenced is not raw else None
    text = unfenced.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("result contains no JSON object")
    end = _decode_object(text, start)
    # ponytail: any further brace after the object counts as a second object; relax only if real outputs need it
    if "{" in text[end:]:
        raise ValueError("result contains more than one JSON object")
    if how is None and (start or end != len(text)):
        how = "framing"
    elif how == "fence" and (start or end != len(text)):
        raise ValueError("fenced result contains text besides one JSON object")
    return text[start:end], how


def stable_hash(value: Mapping[str, Any]) -> str:
    """Short deterministic digest of a JSON-compatible mapping (also backs OrchestrationService._stable_hash)."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def failure_fingerprint(kind: str, normalized_output: str, errors: Sequence[str]) -> str:
    return stable_hash({"kind": kind, "output": normalized_output, "errors": sorted(str(e) for e in errors)})


def next_step(history: Sequence[Mapping[str, Any]], fingerprint: str, error_count: int) -> str:
    """"correct" | "stop" for the newest failure, given the earlier failures of this cycle (oldest first).

    The first failure gets one correction; a second only when the corrected result failed differently and with
    fewer errors; any repeated fingerprint stops at once."""
    if any(item.get("fingerprint") == fingerprint for item in history):
        return "stop"
    if not history:
        return "correct"
    if len(history) == 1 and error_count < len(history[-1].get("errors") or []):
        return "correct"
    return "stop"


def build_correction_brief(
    task_brief: str, failed_output: str, schema: Mapping[str, Any], errors: Sequence[str], version: Mapping[str, Any],
) -> str:
    """Output-only correction brief; the failed output is carried whole and last, never truncated."""
    return "\n\n".join([
        "Result correction: your previous result for this task failed validation.",
        CORRECTION_INSTRUCTION,
        "Original task brief:\n" + task_brief,
        "Expected result schema:\n" + json.dumps(schema, indent=2, sort_keys=True),
        "Validation errors:\n" + "\n".join(f"- {error}" for error in errors),
        "Current version: " + json.dumps(version, sort_keys=True, default=str),
        "Complete failed output (verbatim, between the markers):\n<<<FAILED_OUTPUT\n" + failed_output
        + "\nFAILED_OUTPUT>>>",
    ])
