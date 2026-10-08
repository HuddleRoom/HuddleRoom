import json
import uuid

import pytest

from huddleroom.config import settings
from huddleroom.services.orchestration_supervision_scheduler import SUPPORTED_EVENTS
from huddleroom.services.orchestration_wake_when import (
    EVENT_MATCHER_KEYS,
    normalize_wake_when,
    wake_when_prompt_table,
)

U1 = str(uuid.uuid4())
U2 = str(uuid.uuid4())


def _events(*items):
    return {"events": list(items), "expected_result": "something happens"}


def _ev(event_type="task.status_changed", matcher=None):
    return {"event_type": event_type, "matcher": matcher if matcher is not None else {"task_id": U1}}


def test_accepts_events_only():
    out = normalize_wake_when(_events(_ev()))
    assert out["events"] == [{"event_type": "task.status_changed", "matcher": {"task_id": U1}}]
    assert out["recheck_after_seconds"] is None
    assert out["expected_result"] == "something happens"


def test_accepts_recheck_only(monkeypatch):
    monkeypatch.setattr(settings, "orchestration_reconcile_interval_seconds", 30)
    monkeypatch.setattr(settings, "orchestration_wake_max_seconds", 3600)
    out = normalize_wake_when({"recheck_after_seconds": 600, "expected_result": "ok"})
    assert out["events"] == []
    assert out["recheck_after_seconds"] == 600


def test_accepts_events_and_recheck(monkeypatch):
    monkeypatch.setattr(settings, "orchestration_reconcile_interval_seconds", 30)
    monkeypatch.setattr(settings, "orchestration_wake_max_seconds", 3600)
    out = normalize_wake_when({**_events(_ev()), "recheck_after_seconds": 600})
    assert len(out["events"]) == 1
    assert out["recheck_after_seconds"] == 600


def test_rejects_no_events_and_no_recheck():
    with pytest.raises(ValueError, match="wake_when needs at least one event or recheck_after_seconds"):
        normalize_wake_when({"expected_result": "ok"})


def test_rejects_empty_events_list_without_recheck():
    with pytest.raises(ValueError, match="wake_when needs at least one event or recheck_after_seconds"):
        normalize_wake_when(_events())


def test_rejects_unsupported_event_type():
    with pytest.raises(ValueError, match=r"unsupported event_type 'bogus\.event'"):
        normalize_wake_when(_events(_ev("bogus.event")))


def test_rejects_list_event_type():
    with pytest.raises(ValueError, match="unsupported event_type"):
        normalize_wake_when(_events(_ev(["task.status_changed"])))


def test_rejects_dict_event_type():
    with pytest.raises(ValueError, match="unsupported event_type"):
        normalize_wake_when(_events(_ev({"name": "task.status_changed"})))


@pytest.mark.parametrize("events", [{"event_type": "task.status_changed"}, "task.status_changed"])
def test_rejects_events_not_a_list(events):
    with pytest.raises(ValueError, match="events must be a list"):
        normalize_wake_when({"events": events, "expected_result": "ok"})


def test_rejects_event_item_with_extra_key():
    item = {**_ev(), "note": "extra"}
    with pytest.raises(ValueError, match="each event must have exactly the keys event_type and matcher"):
        normalize_wake_when(_events(item))


def test_rejects_non_string_expected_result():
    with pytest.raises(ValueError, match="expected_result must be a non-blank string"):
        normalize_wake_when({"recheck_after_seconds": 60, "expected_result": 42})


def test_rejects_matcher_key_not_in_event_payload():
    with pytest.raises(ValueError, match=r"matcher keys \['decision_id'\] not valid for task\.status_changed"):
        normalize_wake_when(_events(_ev("task.status_changed", {"decision_id": U1})))


def test_rejects_non_string_matcher_key():
    with pytest.raises(ValueError, match="matcher keys for task.status_changed must be strings"):
        normalize_wake_when(_events(_ev("task.status_changed", {1: U1})))


def test_rejects_empty_matcher():
    with pytest.raises(ValueError, match="matcher for task.status_changed must be a non-empty object"):
        normalize_wake_when(_events(_ev("task.status_changed", {})))


def test_rejects_non_uuid_matcher_value():
    with pytest.raises(ValueError, match="matcher value for task_id must be a UUID string"):
        normalize_wake_when(_events(_ev("task.status_changed", {"task_id": "not-a-uuid"})))


def test_canonicalizes_uppercase_uuid():
    upper = U1.upper()
    out = normalize_wake_when(_events(_ev("task.status_changed", {"task_id": upper})))
    assert out["events"][0]["matcher"] == {"task_id": U1}


def test_clamps_recheck_low_and_high(monkeypatch):
    monkeypatch.setattr(settings, "orchestration_reconcile_interval_seconds", 30)
    monkeypatch.setattr(settings, "orchestration_wake_max_seconds", 3600)
    assert normalize_wake_when({"recheck_after_seconds": 1, "expected_result": "x"})["recheck_after_seconds"] == 30
    assert normalize_wake_when({"recheck_after_seconds": 10**6, "expected_result": "x"})["recheck_after_seconds"] == 3600
    assert normalize_wake_when({"recheck_after_seconds": 120, "expected_result": "x"})["recheck_after_seconds"] == 120


@pytest.mark.parametrize("value", [True, False, 0, -1])
def test_rejects_bool_zero_negative_recheck(value):
    with pytest.raises(ValueError, match="recheck_after_seconds must be a positive integer"):
        normalize_wake_when({"recheck_after_seconds": value, "expected_result": "x"})


@pytest.mark.parametrize("text", ["", "   "])
def test_rejects_blank_expected_result(text):
    with pytest.raises(ValueError, match="expected_result must be a non-blank string"):
        normalize_wake_when({"recheck_after_seconds": 60, "expected_result": text})


def test_rejects_more_than_five_events():
    items = [_ev("task.status_changed", {"task_id": str(uuid.uuid4())}) for _ in range(6)]
    with pytest.raises(ValueError, match="at most 5 events allowed"):
        normalize_wake_when(_events(*items))


def test_rejects_unknown_top_level_key():
    with pytest.raises(ValueError, match=r"unknown wake_when keys \['bogus'\]"):
        normalize_wake_when({"recheck_after_seconds": 60, "expected_result": "x", "bogus": 1})


def test_rejects_non_string_top_level_key():
    with pytest.raises(ValueError, match="wake_when keys must be strings"):
        normalize_wake_when({1: "x", "recheck_after_seconds": 60, "expected_result": "x"})


def test_duplicate_events_are_deduplicated():
    dup = _ev("task.status_changed", {"task_id": U1.upper()})
    other = _ev("artifact.created", {"artifact_id": U2})
    out = normalize_wake_when(_events(_ev(), other, dup, _ev()))
    assert out["events"] == [
        {"event_type": "task.status_changed", "matcher": {"task_id": U1}},
        {"event_type": "artifact.created", "matcher": {"artifact_id": U2}},
    ]


def test_cap_applies_after_deduplication():
    items = [_ev() for _ in range(6)]
    out = normalize_wake_when(_events(*items))
    assert len(out["events"]) == 1


def test_normalize_is_idempotent(monkeypatch):
    monkeypatch.setattr(settings, "orchestration_reconcile_interval_seconds", 30)
    monkeypatch.setattr(settings, "orchestration_wake_max_seconds", 3600)
    once = normalize_wake_when({
        "events": [_ev("session.created", {"graph_run_id": U2.upper(), "task_id": U1})],
        "recheck_after_seconds": 5,
        "expected_result": "  done  ",
    })
    assert normalize_wake_when(once) == once


def test_matcher_table_equals_supported_events_minus_orchestration_events():
    assert set(EVENT_MATCHER_KEYS) == SUPPORTED_EVENTS - {"orchestration.run_completed", "orchestration.steering_changed"}


def test_prompt_table_is_sorted_json_of_matcher_keys():
    table = json.loads(wake_when_prompt_table())
    assert table["session.created"] == sorted({"session_id", "task_id", "graph_run_id"})
    assert set(table) == set(EVENT_MATCHER_KEYS)
