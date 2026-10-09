import json
import uuid
from types import SimpleNamespace

import pytest

from huddleroom.services.orchestration_plan_ingestion import (
    PlanIngestionError,
    check_disposition,
    parse_envelope,
    plan_artifact_id,
    plan_output_hash,
)

GOAL = SimpleNamespace(success_criteria=[{"key": "c1"}], budget={"caps": {"max_tokens": "1000"}})


def _item(**kw):
    base = {"id": "a", "work_function": "implementation", "scope": "s", "deliverable": "d", "why_runnable": "ready"}
    return {**base, **kw}


def _env(kind="implementation", items=None, **kw):
    return {"schema_version": 1, "plan_kind": kind, "summary": "sum", "plan_items": items or [_item()], **kw}


def _parse(data, kind="implementation", goal=GOAL):
    return parse_envelope(data if isinstance(data, str) else json.dumps(data), kind, goal)


def _rejects(data, text, kind="implementation"):
    with pytest.raises(PlanIngestionError) as exc:
        _parse(data, kind)
    assert text in exc.value.reason
    assert not exc.value.stale


@pytest.mark.parametrize("kind", ["implementation", "delivery"])
def test_valid_outcome_envelopes(kind):
    out = _parse(_env(kind), kind)
    assert out["plan_kind"] == kind and out["plan_items"][0]["id"] == "a"


def test_valid_roadmap_envelope():
    item = {
        "item_key": "k", "unit_type": "task", "title": "t", "work_function": "implementation",
        "scope": "s", "deliverable": "d", "why_runnable": "ready", "allocation": {},
    }
    assert _parse(_env("roadmap", [item]), "roadmap")["plan_items"][0]["item_key"] == "k"


def test_rejections():
    _rejects("here is my plan", "JSON object")
    _rejects("```json\n" + json.dumps(_env()) + "\n```", "JSON object")
    _rejects(_env(extra=1), "extra: Extra inputs")
    _rejects(_env(schema_version=2), "schema_version")
    _rejects(_env("delivery"), "plan_kind 'delivery' does not match expected 'implementation'")
    _rejects(_env(items=[_item(why_runnable="  ")]), "why_runnable")
    _rejects('{"schema_version":1,"schema_version":1}', "duplicate key")
    _rejects(_env(items=[_item(depends_on=["zzz"])]), "unknown plan item 'zzz'")
    _rejects(_env(items=[_item(id="a", depends_on=["b"]), _item(id="b", depends_on=["a"])]), "dependency cycle")


def test_disposition():
    tid = uuid.uuid4()
    norm = _parse(_env())
    with pytest.raises(PlanIngestionError, match=str(tid)):
        check_disposition(norm, [tid])
    check_disposition({**norm, "summary": f"skipping {tid}: obsolete"}, [tid])
    adopted = _parse(_env(items=[_item(adopt_task_id=str(tid))]))
    check_disposition(adopted, [tid])


def test_ids_and_hash():
    t = uuid.uuid4()
    assert plan_artifact_id(t, 1) == plan_artifact_id(t, 1) != plan_artifact_id(t, 2)
    assert plan_output_hash({"a": 1, "b": 2}) == plan_output_hash({"b": 2, "a": 1})
