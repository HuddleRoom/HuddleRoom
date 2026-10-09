import uuid
from types import SimpleNamespace

from huddleroom.schemas.orchestration import OrchestrationDelegationContract
from huddleroom.services.orchestration_service import OrchestrationService


def test_delegation_description_renders_criterion_inputs_with_descriptions():
    goal = SimpleNamespace(
        objective="Ship it",
        orchestrator_context={},
        success_criteria=[{"key": "criterion_1", "id": "c1", "description": "Login works"}, "bad-entry"],
    )
    contract = OrchestrationDelegationContract(
        goal_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        action_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
        work_function="build",
        scope="s",
        deliverable="d",
        inputs=["criterion:criterion_1", "criterion:unknown", "plain input"],
    )

    text = OrchestrationService._delegation_task_description(goal, contract)

    assert "- criterion:criterion_1 — Login works" in text
    assert "- criterion:unknown\n" in text + "\n"
    assert "- plain input" in text
    assert contract.inputs == ["criterion:criterion_1", "criterion:unknown", "plain input"]


def _contract(**kw):
    base = dict(
        goal_id=uuid.uuid4(), run_id=uuid.uuid4(), action_id=uuid.uuid4(), agent_id=uuid.uuid4(),
        work_function="build", scope="the scope", deliverable="the deliverable",
    )
    return OrchestrationDelegationContract(**{**base, **kw})


def _goal(**kw):
    return SimpleNamespace(
        objective="Ship it", orchestrator_context={}, success_criteria=[],
        goal_type="outcome", planning_mode=None, **kw,
    )


def test_description_sections_render_in_order():
    dep, art = uuid.uuid4(), uuid.uuid4()
    text = OrchestrationService._delegation_task_description(
        _goal(),
        _contract(
            why_runnable="deps done", decision_use="pick option", inputs=["in1"], depends_on=[dep],
            artifact_ids=[art], forbidden_work=["no x"], success_evidence=["proof"],
        ),
    )
    markers = [
        "Objective:", "Scope:", "Deliverable:", "Why runnable: deps done", "Decision use: pick option",
        "Inputs:", f"Dependencies (accepted predecessor task IDs; use their reports as inputs):\n- {dep}",
        f"Artifacts (existing Artifact IDs to read):\n- {art}", "Forbidden work:", "Success evidence:",
        "Authorization scope:", "Required output:",
    ]
    positions = [text.index(m) for m in markers]
    assert positions == sorted(positions)


def test_description_omits_empty_sections_and_renders_canonical_schema():
    from huddleroom.services.orchestration_work_report import CANONICAL_WORK_REPORT_SCHEMA
    import json

    text = OrchestrationService._delegation_task_description(
        _goal(), _contract(report_schema={"status": "done|blocked|failed", "notes": "str"}),
    )
    for absent in ("Why runnable", "Decision use", "Dependencies", "Artifacts (", "Forbidden work"):
        assert absent not in text
    assert "not separately declared" in text
    assert json.dumps(CANONICAL_WORK_REPORT_SCHEMA, indent=2) in text
    assert '"notes"' not in text
    assert "do not register Artifacts" in text


def test_description_verification_schema_verbatim():
    import json

    schema = {"verdict": "pass|fail", "evidence": "list[str]"}
    text = OrchestrationService._delegation_task_description(_goal(), _contract(report_schema=schema))
    assert json.dumps(schema, indent=2) in text
    assert "do not register Artifacts" not in text


def test_description_declared_authorization_scope():
    from huddleroom.schemas.orchestration import AuthorizationScope

    scope = AuthorizationScope(
        model_spend={"allowed": True, "remaining_by_dimension": {"usd": "5"}},
        external_writes={"allowed": True, "targets": ["a", "b"]},
        tool_constraints=["no shell"],
    )
    text = OrchestrationService._delegation_task_description(_goal(), _contract(authorization_scope=scope))
    assert "- Model spend: allowed (remaining: usd=5)" in text
    assert "- External writes: allowed to a, b" in text
    assert "- Live sends: not allowed" in text
    assert "- Workspace mutation: read_only" in text
    assert "- Tool constraints: no shell" in text


def test_planning_contract_renders_envelope_for_goal_kind():
    for goal_type, kind in (("outcome", "implementation"), ("roadmap", "roadmap")):
        goal = SimpleNamespace(**{**_goal().__dict__, "goal_type": goal_type})
        text = OrchestrationService._delegation_task_description(
            goal, _contract(work_function="planning", report_schema={"artifact_id": "uuid"}),
        )
        assert f'"plan_kind": "{kind}"' in text
        assert '"artifact_id"' not in text
        assert "do not register Artifacts" in text


def test_delegation_title_explicit_truncated_and_fallback():
    title = OrchestrationService._delegation_task_title
    assert title("build", "deliv", " Short title ") == "Short title"
    assert len(title("build", "deliv", "word " * 40)) <= 80
    assert title("build", "deliv", "  ") == title("build", "deliv") == "Build: deliv"


def test_legacy_create_request_has_no_title_and_stable_action_key():
    from huddleroom.services.orchestration_decision_dispatcher import OrchestrationDecisionDispatcher

    request = OrchestrationService()._canonical_delegation_task_request(
        {"agent_id": str(uuid.UUID(int=1)), "work_function": "build", "scope": "s", "deliverable": "d"}
    )
    assert "title" not in request
    with_title = OrchestrationService()._canonical_delegation_task_request(
        {"agent_id": str(uuid.UUID(int=1)), "work_function": "build", "scope": "s", "deliverable": "d", "title": " T "}
    )
    assert with_title["title"] == "T"
    run_id = uuid.UUID(int=2)
    assert OrchestrationDecisionDispatcher.action_key(run_id, "create_delegation_task", request) != \
        OrchestrationDecisionDispatcher.action_key(run_id, "create_delegation_task", with_title)
