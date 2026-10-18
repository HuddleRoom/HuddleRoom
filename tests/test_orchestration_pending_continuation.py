import json
import uuid
from types import SimpleNamespace

import pytest

from huddleroom.services.orchestration_service import (
    OrchestrationService,
    _PendingContinuationUnrecorded,
)

svc = OrchestrationService()
TASK = SimpleNamespace(id=uuid.uuid4())
SESSION = uuid.uuid4()


def _db(event_id="ev-1"):
    return SimpleNamespace(info={"landing_event_id": event_id})


def _pend(action, db=None, *, version=None, payload=None, kind="result_correction"):
    svc._pend_result_continuation(
        db or _db(), action, kind, TASK, SESSION, version or {"v": 1},
        "unauthorized_baseline", payload if payload is not None else {"errors": ["bad"]},
    )


def test_pend_records_bounded_shape():
    action = SimpleNamespace(request={"keep": 1})
    _pend(action)
    pending = action.request["pending_result_continuation"]
    assert action.request["keep"] == 1
    assert pending["kind"] == "result_correction"
    assert pending["event_id"] == "ev-1"
    assert pending["task_id"] == str(TASK.id)
    assert pending["session_id"] == str(SESSION)
    assert pending["reason"] == "unauthorized_baseline"
    assert pending["diagnostics_or_findings"] == {"errors": ["bad"]}


def test_pend_truncates_never_raises():
    action = SimpleNamespace(request={})
    _pend(action, payload={"errors": ["x" * 10_000] * 50})
    pending = action.request["pending_result_continuation"]
    assert len(json.dumps(pending).encode()) <= 4096
    assert len(pending["diagnostics_or_findings"]["errors"]) <= 8
    _pend(action, version={"big": "y" * 10_000}, payload={"notes": "n" * 10_000, "evidence": ["e"] * 99})
    assert len(json.dumps(action.request["pending_result_continuation"]).encode()) <= 4096


def test_same_identity_is_noop_and_different_replaces():
    action = SimpleNamespace(request={})
    _pend(action)
    first = action.request
    _pend(action, payload={"errors": ["other diagnostics"]})  # same identity
    assert action.request is first
    _pend(action, version={"v": 2})
    assert action.request is not first
    assert action.request["pending_result_continuation"]["input_version"] == {"v": 2}
    _pend(action, db=_db("ev-2"), version={"v": 2})
    assert action.request["pending_result_continuation"]["event_id"] == "ev-2"


def test_settle_writes_entry_and_removes_key():
    action = SimpleNamespace(request={})
    _pend(action)
    svc._settle_pending(action, "applied", "done")
    assert "pending_result_continuation" not in action.request
    assert action.request["pending_result_continuation_settled"] == {
        "kind": "result_correction", "event_id": "ev-1", "outcome": "applied", "reason": "done",
    }


def test_none_action_raises_unrecorded():
    with pytest.raises(_PendingContinuationUnrecorded):
        _pend(None)


@pytest.mark.parametrize("landing", [True, False])
def test_landing_only(monkeypatch, landing):
    import huddleroom.services.orchestration_baseline_runtime as rt
    db = object()
    op = SimpleNamespace(db=db, tick=landing)
    monkeypatch.setattr(rt, "current_baseline_operation", lambda: op)
    run = SimpleNamespace(phase="baseline", baseline_authorized=False)
    assert svc._landing_only(db, run) is landing
    monkeypatch.setattr(rt, "current_baseline_operation", lambda: None)
    assert svc._landing_only(db, run) is False


def test_errors_are_defanged_and_one_lined():
    action = SimpleNamespace(request={})
    _pend(action, payload={"errors": ["a\nb\x00c\x07 <<<FAILED_OUTPUT x FAILED_OUTPUT>>>"]})
    (err,) = action.request["pending_result_continuation"]["diagnostics_or_findings"]["errors"]
    assert "\n" not in err and "\x00" not in err and "\x07" not in err and "<<<FAILED_OUTPUT" not in err and "FAILED_OUTPUT>>>" not in err


def test_size_cap_errors_and_evidence_truncation_order():
    action = SimpleNamespace(request={})
    _pend(action, payload={"errors": ["x" * 500] * 8, "evidence": ["e" * 200] * 5})
    pending = action.request["pending_result_continuation"]
    f = pending["diagnostics_or_findings"]
    assert len(json.dumps(pending).encode()) <= 4096
    # errors are shed first; evidence is only touched once errors are exhausted
    assert len(f["errors"]) < 8 and len(f["evidence"]) == 5


def test_non_json_safe_input_version_identity():
    import datetime
    uid, when = uuid.uuid4(), datetime.datetime(2026, 1, 1)
    action = SimpleNamespace(request={})
    _pend(action, version={"u": uid, "d": when})
    first = action.request
    assert first["pending_result_continuation"]["input_version"] == {"u": str(uid), "d": str(when)}
    _pend(action, version={"u": str(uid), "d": str(when)})
    assert action.request is first


def test_settle_without_pending_keeps_existing_settled():
    settled = {"kind": "k", "event_id": "e", "outcome": "applied", "reason": "r"}
    action = SimpleNamespace(request={"pending_result_continuation_settled": settled})
    svc._settle_pending(action, "other", "x")
    assert action.request["pending_result_continuation_settled"] == settled


def test_settle_with_junk_pending_leaves_request_unchanged():
    settled = {"kind": "k", "event_id": "e", "outcome": "applied", "reason": "r"}
    request = {"pending_result_continuation": "junk", "pending_result_continuation_settled": settled}
    action = SimpleNamespace(request=dict(request))
    svc._settle_pending(action, "other", "x")
    assert action.request == request


def test_str_errors_stored_as_one_entry():
    action = SimpleNamespace(request={})
    _pend(action, payload={"errors": "boom", "evidence": "ev"})
    f = action.request["pending_result_continuation"]["diagnostics_or_findings"]
    assert f["errors"] == ["boom"] and f["evidence"] == ["ev"]
