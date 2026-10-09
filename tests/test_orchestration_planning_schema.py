import uuid

import pytest
from pydantic import ValidationError

from huddleroom.schemas.orchestration import (
    PLAN_ITEM_PROMPT_FIELDS,
    PLANNING_ENVELOPE_ADAPTER,
    ROADMAP_ITEM_PROMPT_FIELDS,
    AuthorizationScope,
    OrchestrationDelegationContract,
    OrchestrationPlanItem,
    OrchestrationRoadmapPlanItem,
    planning_envelope_report_schema,
)
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_work_report import CANONICAL_WORK_REPORT_SCHEMA, validate_work_report

AGENT = "11111111-1111-1111-1111-111111111111"
LEGACY_IMPL = dict(id="a", work_function="dev", scope="s", deliverable="d", agent_id=AGENT, depends_on=["x"])
LEGACY_TASK = dict(item_key="t", unit_type="task", title="T", work_function="dev", scope="s", deliverable="d")
LEGACY_GOAL = dict(item_key="g", unit_type="goal", title="G", objective="o",
                   success_criteria=[{"key": "k"}], allocation={"cost": "5"})
# Pinned before the schema change: legacy dumps must keep identical fingerprints.
PINNED = [
    (OrchestrationPlanItem, LEGACY_IMPL, "53774f9fb1fa937c343aa375d23978d9c357c5a65f2ed48fa5326b73a610eb0b"),
    (OrchestrationRoadmapPlanItem, LEGACY_TASK, "99647257fa220c578380b7b1baba24e9826911020b21b9d46bdf555aad7dc4ba"),
    (OrchestrationRoadmapPlanItem, LEGACY_GOAL, "030e3542ecd55e2592e43ba66bd925f626a42c6cb0216b13077a7f28f22d878f"),
]
NEW_ITEM_KEYS = {"adopt_task_id", "artifact_ids", "priority", "why_runnable"}


@pytest.mark.parametrize("cls,data,expected", PINNED)
def test_legacy_item_fingerprint_unchanged(cls, data, expected):
    dumped = cls(**data).model_dump(mode="json")
    assert not (NEW_ITEM_KEYS | {"goal_type"}) & set(dumped)
    assert OrchestrationService._accepted_plan_fingerprint([dumped]) == expected


def test_non_default_new_item_values_are_dumped():
    tid = uuid.uuid4()
    item = OrchestrationPlanItem(**LEGACY_IMPL, adopt_task_id=tid, priority=7, why_runnable="ready")
    dumped = item.model_dump(mode="json")
    assert dumped["adopt_task_id"] == str(tid) and dumped["priority"] == 7 and dumped["why_runnable"] == "ready"
    assert "artifact_ids" not in dumped
    goal = OrchestrationRoadmapPlanItem(**LEGACY_GOAL, goal_type="roadmap")
    assert goal.model_dump(mode="json")["goal_type"] == "roadmap"


def test_contract_minimal_dump_has_only_legacy_keys():
    ids = [uuid.uuid4() for _ in range(4)]
    c = OrchestrationDelegationContract(
        goal_id=ids[0], run_id=ids[1], action_id=ids[2], agent_id=ids[3],
        work_function="dev", scope="s", deliverable="d",
    )
    assert set(c.model_dump(mode="json")) == {
        "goal_id", "run_id", "action_id", "agent_id", "work_function", "scope", "inputs", "deliverable",
        "forbidden_work", "success_evidence", "budget", "report_schema", "parent_task_id",
    }
    full = c.model_copy(update={"title": "T", "priority": 3, "authorization_scope": AuthorizationScope()})
    dumped = full.model_dump(mode="json")
    assert dumped["title"] == "T" and dumped["priority"] == 3
    assert dumped["authorization_scope"]["workspace_mutation"] == "read_only"


def test_authorization_scope_fails_closed():
    scope = AuthorizationScope()
    assert not scope.model_spend.allowed and not scope.external_writes.allowed and not scope.live_sends.allowed
    with pytest.raises(ValidationError):
        AuthorizationScope(live_sends={"allowed": "yes"})
    with pytest.raises(ValidationError):
        AuthorizationScope(surprise=1)


def test_item_validation():
    with pytest.raises(ValidationError, match="task item cannot define goal_type"):
        OrchestrationRoadmapPlanItem(**LEGACY_TASK, goal_type="auto")
    for bad in (101, True):
        with pytest.raises(ValidationError):
            OrchestrationPlanItem(**LEGACY_IMPL, priority=bad)


def _item(**extra):
    return {**LEGACY_IMPL, "why_runnable": "no deps pending", **extra}


def _envelope(kind="implementation", items=None, **extra):
    return {"schema_version": 1, "plan_kind": kind, "summary": "plan",
            "plan_items": items if items is not None else [_item()], **extra}


@pytest.mark.parametrize("kind", ["delivery", "implementation"])
def test_envelope_accepts_outcome_kinds(kind):
    assert PLANNING_ENVELOPE_ADAPTER.validate_python(_envelope(kind)).plan_kind == kind


def test_envelope_accepts_roadmap():
    items = [{**LEGACY_TASK, "why_runnable": "ok"}, {**LEGACY_GOAL, "why_runnable": "ok", "goal_type": "auto"}]
    assert PLANNING_ENVELOPE_ADAPTER.validate_python(_envelope("roadmap", items)).plan_kind == "roadmap"


@pytest.mark.parametrize("payload,match", [
    (_envelope(extra=1), None),
    ({**_envelope(), "schema_version": 2}, None),
    (_envelope(items=[]), None),
    (_envelope(items=[{**LEGACY_TASK, "why_runnable": "ok"}]), None),
    (_envelope(items=[_item(), _item(id="b", why_runnable="  ")]), r"plan_items\[1\]\.why_runnable"),
    ({**_envelope(), "summary": "  "}, None),
])
def test_envelope_rejects(payload, match):
    with pytest.raises(ValidationError, match=match):
        PLANNING_ENVELOPE_ADAPTER.validate_python(payload)


def test_prompt_fields_match_models():
    assert set(PLAN_ITEM_PROMPT_FIELDS) == set(OrchestrationPlanItem.model_fields)
    assert set(ROADMAP_ITEM_PROMPT_FIELDS) == set(OrchestrationRoadmapPlanItem.model_fields)
    schema = planning_envelope_report_schema("roadmap")
    assert schema["plan_kind"] == "roadmap" and schema["plan_items"] == [ROADMAP_ITEM_PROMPT_FIELDS]
    assert planning_envelope_report_schema("delivery")["plan_items"] == [PLAN_ITEM_PROMPT_FIELDS]


def test_canonical_report_schema_matches_validator():
    sample = {
        "status": "done", "changes": [], "evidence": [], "criterion_progress": {}, "decisions": [],
        "risks": [], "open_questions": [], "next_step": "x", "collaboration_need": None,
    }
    assert set(sample) == set(CANONICAL_WORK_REPORT_SCHEMA)
    import json
    assert validate_work_report(json.dumps(sample)).report is not None


@pytest.mark.parametrize("cls,data,expected", PINNED)
def test_legacy_item_json_serialization_matches_dump(cls, data, expected):
    import json

    item = cls(**data)
    assert json.loads(item.model_dump_json()) == item.model_dump(mode="json")
    assert not (NEW_ITEM_KEYS | {"goal_type"}) & set(json.loads(item.model_dump_json()))
