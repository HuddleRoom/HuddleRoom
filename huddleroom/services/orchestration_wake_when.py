"""Normalizer for the orchestrator's `wake_when` wait condition.

Kept free of service imports so it stays cycle-free.
"""

import json
import uuid
from collections.abc import Mapping

from huddleroom.config import settings

ORCHESTRATOR_WAIT_OWNER_TYPE = "orchestrator_decision"
WAKE_RECHECK_EVENT_TYPE = "orchestration.wake_recheck"
# ponytail: fixed backstop avoids hot-looping on a persistent provider error; make it a setting if operators need to tune it.
BACKSTOP_RECHECK_SECONDS = 300
MAX_WAKE_EVENTS = 5

EVENT_MATCHER_KEYS: dict[str, frozenset[str]] = {
    "task.status_changed": frozenset({"task_id"}),
    "task.assigned": frozenset({"task_id"}),
    "session.created": frozenset({"session_id", "task_id", "graph_run_id"}),
    "session.started": frozenset({"session_id", "task_id"}),
    "session.completed": frozenset({"session_id", "task_id"}),
    "session.failed": frozenset({"session_id", "task_id"}),
    "session.cancelled": frozenset({"session_id", "task_id"}),
    "session.resumed": frozenset({"session_id", "task_id"}),
    "authority.decision_resolved": frozenset({"decision_id"}),
    "artifact.created": frozenset({"artifact_id"}),
    "artifact.content_changed": frozenset({"artifact_id"}),
    "artifact.breaking_change": frozenset({"artifact_id"}),
    "meeting.scheduled": frozenset({"meeting_id"}),
    "meeting.concluded": frozenset({"meeting_id"}),
    "graph.run_completed": frozenset({"graph_run_id"}),
    "graph.run_failed": frozenset({"graph_run_id"}),
    "graph.run_advanced": frozenset({"graph_run_id"}),
}

_ALLOWED_TOP_KEYS = frozenset({"events", "recheck_after_seconds", "expected_result"})
_EVENT_KEYS = frozenset({"event_type", "matcher"})


def clamp_recheck_seconds(seconds: int) -> int:
    lo = settings.orchestration_reconcile_interval_seconds
    hi = max(lo, settings.orchestration_wake_max_seconds)
    return max(lo, min(seconds, hi))


def _normalize_event(item) -> dict:
    if not isinstance(item, Mapping):
        raise ValueError("each event must be an object with event_type and matcher")
    if set(item) != _EVENT_KEYS:
        raise ValueError("each event must have exactly the keys event_type and matcher")
    event_type = item["event_type"]
    if not isinstance(event_type, str) or event_type not in EVENT_MATCHER_KEYS:
        raise ValueError(
            f"unsupported event_type {event_type!r}; allowed: {sorted(EVENT_MATCHER_KEYS)}"
        )
    matcher = item["matcher"]
    if not isinstance(matcher, Mapping) or not matcher:
        raise ValueError(f"matcher for {event_type} must be a non-empty object")
    if not all(isinstance(key, str) for key in matcher):
        raise ValueError(f"matcher keys for {event_type} must be strings")
    allowed = EVENT_MATCHER_KEYS[event_type]
    extra = set(matcher) - allowed
    if extra:
        raise ValueError(
            f"matcher keys {sorted(extra)} not valid for {event_type}; allowed: {sorted(allowed)}"
        )
    canonical = {}
    for key, value in matcher.items():
        if not isinstance(value, str):
            raise ValueError(f"matcher value for {key} must be a UUID string")
        try:
            canonical[key] = str(uuid.UUID(value))
        except ValueError:
            raise ValueError(f"matcher value for {key} must be a UUID string") from None
    return {"event_type": event_type, "matcher": canonical}


def normalize_wake_when(raw) -> dict:
    if not isinstance(raw, Mapping):
        raise ValueError("wake_when must be an object")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError("wake_when keys must be strings")
    unknown = set(raw) - _ALLOWED_TOP_KEYS
    if unknown:
        raise ValueError(f"unknown wake_when keys {sorted(unknown)}; allowed: {sorted(_ALLOWED_TOP_KEYS)}")

    raw_events = raw.get("events")
    if raw_events is None:
        raw_events = []
    if not isinstance(raw_events, list):
        raise ValueError("events must be a list")
    # Dedupe on canonical form (order-preserving) before the cap, so repeats don't count against it.
    unique: dict[str, dict] = {}
    for item in raw_events:
        event = _normalize_event(item)
        unique.setdefault(json.dumps(event, sort_keys=True), event)
    events = list(unique.values())
    if len(events) > MAX_WAKE_EVENTS:
        raise ValueError(f"at most {MAX_WAKE_EVENTS} events allowed")

    recheck = raw.get("recheck_after_seconds")
    if recheck is not None:
        if isinstance(recheck, bool) or not isinstance(recheck, int) or recheck <= 0:
            raise ValueError("recheck_after_seconds must be a positive integer")
        recheck = clamp_recheck_seconds(recheck)

    if not events and recheck is None:
        raise ValueError("wake_when needs at least one event or recheck_after_seconds")

    expected = raw.get("expected_result")
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError("expected_result must be a non-blank string")

    return {"events": events, "recheck_after_seconds": recheck, "expected_result": expected.strip()}


def wake_when_prompt_table() -> str:
    return json.dumps({event: sorted(keys) for event, keys in EVENT_MATCHER_KEYS.items()}, sort_keys=True)
