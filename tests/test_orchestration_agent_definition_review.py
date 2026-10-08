import copy
import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")

from huddleroom.models.agent import Agent
from huddleroom.models.memory_item import MemoryItem
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.orchestration_process import (
    OrchestrationAgentReview,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.models.task import Task
from huddleroom.models.session import Session
from huddleroom.services.orchestration_agent_definition_analyzer import (
    AgentDefinitionSemanticAnalyzer,
    SemanticAgentAssessment,
    parse_semantic_agent_assessment,
)
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

_semantic_review = AgentDefinitionSemanticAnalyzer.review
_semantic_review_request = AgentDefinitionSemanticAnalyzer.review_request


@pytest.fixture(autouse=True)
def approve_semantic_definitions_by_default(monkeypatch):
    async def approve(self, request, candidate_work_functions, *, project_id=None):
        return SemanticAgentAssessment(
            "approved", (), "Definition is actionable.", tuple(candidate_work_functions)
        )

    monkeypatch.setattr(
        "huddleroom.services.orchestration_agent_definition_analyzer.AgentDefinitionSemanticAnalyzer.review_request",
        approve,
    )


class SequencedSemanticAnalyzer:
    def __init__(self):
        self.calls = 0

    async def review(self, agent_snapshot, goal_snapshot, candidate_work_functions, *, project_id=None):
        self.calls += 1
        if self.calls == 1:
            return SemanticAgentAssessment(
                "improvement_proposed",
                ("Persona does not assign implementation ownership.",),
                "Makes the definition actionable.",
                ("implementation",),
                "Developer role responsible for implementation.",
                "Implement scoped changes and report verification evidence.",
            )
        return SemanticAgentAssessment(
            "approved", (), "Definition is actionable.", ("implementation",)
        )

    @staticmethod
    def build_request(agent_snapshot, goal_snapshot, candidate_work_functions, project=None):
        return {"agent": agent_snapshot, "goal": goal_snapshot,
                "candidate_work_functions": candidate_work_functions}

    async def review_request(self, request, candidate_work_functions, *, project_id=None):
        return await self.review(request["agent"], request["goal"], candidate_work_functions, project_id=project_id)


class ThreeProposalAnalyzer:
    def __init__(self):
        self.calls = 0

    async def review(self, agent_snapshot, _goal_snapshot, candidate_work_functions, *, project_id=None):
        self.calls += 1
        number = int(agent_snapshot["name"].rsplit("-", 1)[1]) + 1
        return SemanticAgentAssessment(
            "improvement_proposed",
            ("The definition needs a clearer ownership statement.",),
            "Makes the definition actionable.",
            tuple(candidate_work_functions),
            f"Proposed description {number}.",
            f"Proposed persona {number}.",
        )

    @staticmethod
    def build_request(agent_snapshot, goal_snapshot, candidate_work_functions, project=None):
        return {"agent": agent_snapshot, "goal": goal_snapshot,
                "candidate_work_functions": candidate_work_functions}

    async def review_request(self, request, candidate_work_functions, *, project_id=None):
        return await self.review(request["agent"], request["goal"], candidate_work_functions, project_id=project_id)


def test_semantic_assessment_requires_problems():
    with pytest.raises(ValueError, match="requires problems"):
        parse_semantic_agent_assessment({
            "status": "improvement_proposed",
            "problems": [],
            "reason": "Clarifies ownership.",
            "approved_work_functions": ["implementation"],
            "proposed_description": "Developer role responsible for implementation.",
            "proposed_persona": "Implement scoped changes and report verification evidence.",
        }, ["implementation"])



def test_semantic_assessment_downgrades_non_improving_proposal_to_approved():
    snapshot = {
        "description": "Helps with code.",
        "system_prompt": "Be useful.",
    }
    assessment = parse_semantic_agent_assessment({
        "status": "improvement_proposed",
        "problems": ["Persona is vague."],
        "reason": "Could be clearer.",
        "approved_work_functions": ["implementation"],
        "proposed_description": "Developer role responsible for implementation.",
        "proposed_persona": "Be useful.",  # echoes candidate system_prompt unchanged
    }, ["implementation"], snapshot)

    assert assessment.status == "approved"
    assert assessment.problems == ()
    assert assessment.proposed_description is None
    assert assessment.proposed_persona is None
    assert assessment.approved_work_functions == ("implementation",)


def test_semantic_assessment_filters_unknown_approved_work_functions():
    assessment = parse_semantic_agent_assessment({
        "status": "approved",
        "problems": [],
        "reason": "Definition is actionable.",
        "approved_work_functions": ["implementation", "invented"],
        "proposed_description": None,
        "proposed_persona": None,
    }, ["implementation"])

    assert assessment.approved_work_functions == ("implementation",)


@pytest.mark.asyncio
async def test_semantic_analyzer_uses_action_prompt_and_parses_complete_rewrite(monkeypatch):
    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review_request", _semantic_review_request)
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "status": "improvement_proposed",
            "problems": ["Persona is vague about implementation ownership."],
            "reason": "The replacement turns a generic persona into executable instructions.",
            "approved_work_functions": ["implementation"],
            "proposed_description": "Developer role responsible for implementation.",
            "proposed_persona": "Implement scoped changes and report verification evidence.",
        })) )])

    result = await AgentDefinitionSemanticAnalyzer(completion_fn=completion).review(
        {"name": "python-developer", "description": "Helps with code.", "system_prompt": "Be useful."},
        {"objective": "Implement the API", "success_criteria": [], "constraints": {}},
        ["implementation"],
    )

    assert result.status == "improvement_proposed"
    assert result.approved_work_functions == ("implementation",)
    assert calls[0]["max_tokens"] == 32768
    assert calls[0]["response_format"] == {"type": "json_object"}
    system_prompt = calls[0]["messages"][0]["content"]
    assert "provide constrained proposed_description and proposed_persona" in system_prompt
    assert "Return JSON only" in system_prompt


@pytest.mark.asyncio
async def test_semantic_analyzer_unwraps_unlabelled_fence_without_relaxing_schema(monkeypatch):
    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review_request", _semantic_review_request)

    async def completion(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="```\n" + json.dumps({
            "status": "approved", "problems": [], "reason": "Actionable.",
            "approved_work_functions": ["implementation"],
            "proposed_description": None, "proposed_persona": None,
        }) + "\n```"))])

    result = await AgentDefinitionSemanticAnalyzer(completion_fn=completion).review(
        {"name": "python-developer", "description": "Helps with code.", "system_prompt": "Be useful."},
        {"objective": "Implement the API", "success_criteria": [], "constraints": {}}, ["implementation"],
    )

    assert result.status == "approved"


@pytest.mark.asyncio
async def test_semantic_analyzer_keeps_invalid_schema_strict_after_unwrapping(monkeypatch):
    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review_request", _semantic_review_request)

    async def completion(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="```json\n" + json.dumps({
            "status": "approved", "problems": [], "reason": "Actionable.",
            "approved_work_functions": ["implementation"],
            "proposed_description": None, "proposed_persona": None, "extra": True,
        }) + "\n```"))])

    with pytest.raises(Exception):
        await AgentDefinitionSemanticAnalyzer(completion_fn=completion).review(
            {"name": "python-developer", "description": "Helps with code.", "system_prompt": "Be useful."},
            {"objective": "Implement the API", "success_criteria": [], "constraints": {}}, ["implementation"],
        )


def _agent(**overrides) -> Agent:
    values = {
        "id": uuid.uuid4(),
        "name": "implementation-agent",
        "role": "developer",
        "description": "Implements scoped product changes.",
        "system_prompt": "Implement the requested change and verify it.",
        "provider": "example",
        "model": "unranked-model",
        "adapter_type": "cli",
        "cli_runtime": "codex",
        "capabilities": ["implementation"],
        "config": {
            "temperature": 0.2,
            "reasoning_effort": "medium",
            "tools": ["editor"],
        },
        "is_active": True,
    }
    values.update(overrides)
    return Agent(**values)


def _warning_types(assessment: dict) -> set[str]:
    assert all(warning["severity"] == "warning" for warning in assessment["warnings"])
    return {warning["warning_type"] for warning in assessment["warnings"]}


@pytest.mark.parametrize(
    ("agent", "proposed", "inactive_assignee", "warning_type", "eligible"),
    [
        (
            _agent(name="reviewer", role="developer"),
            ["implementation", "review", "summarization"],
            False,
            "agent_review_role_name_mismatch",
            ["summarization"],
        ),
        (
            _agent(name="implementation", role="reviewer"),
            ["implementation", "review", "planning"],
            False,
            "agent_review_role_name_mismatch",
            ["planning"],
        ),
        (
            _agent(description=" ", system_prompt=None),
            ["implementation", "planning"],
            False,
            "agent_review_vague_definition",
            [],
        ),
        (
                _agent(adapter_type="api", cli_runtime=None, config={"temperature": 0.31, "reasoning_effort": "high"}),
                ["review", "validation", "implementation"],
                False,
                "agent_review_weak_validation_config",
                [],
        ),
        (
            _agent(adapter_type="api", cli_runtime=None, config={"temperature": 0.1, "reasoning_effort": "low"}),
            ["validation", "planning"],
            False,
            "agent_review_weak_validation_config",
            ["planning"],
        ),
        (
            _agent(config={"allow_global_scope": True}),
            ["implementation"],
            False,
            "agent_review_unsafe_permissions",
            [],
        ),
        (
            _agent(adapter_type="cli", cli_runtime="claude_code", config={}),
            ["implementation"],
            False,
            "agent_review_unsafe_permissions",
            [],
        ),
        *[
            (
                _agent(adapter_type="cli", cli_runtime=runtime, config={}),
                ["implementation"],
                False,
                "agent_review_unsafe_permissions",
                [],
            )
            for runtime in ("copilot", "opencode", "pi")
        ],
        (
            _agent(is_active=False),
            ["implementation", "review"],
            True,
            "agent_review_inactive_assignee",
            [],
        ),
        (
            _agent(
                name="manager",
                role="manager",
                description="Own delivery and coordinate decisions.",
                system_prompt="Manage the team and edit files yourself.",
            ),
            ["management", "planning"],
            False,
            "agent_review_manager_overreach",
            ["planning"],
        ),
    ],
)
def test_assessment_applies_each_warning_to_only_relevant_functions(
    agent, proposed, inactive_assignee, warning_type, eligible
):
    from huddleroom.services.orchestration_agent_definition_review import assess_agent_definition

    assessment = assess_agent_definition(
        agent,
        proposed,
        inactive_assignee=inactive_assignee,
    )

    assert warning_type in _warning_types(assessment)
    assert assessment["approved_for_work_functions"] == eligible

    # Verify that vague_definition warning includes the agent's name
    if warning_type == "agent_review_vague_definition":
        vague_warning = next((w for w in assessment["warnings"] if w["warning_type"] == "agent_review_vague_definition"), None)
        assert vague_warning is not None
        assert agent.name in vague_warning["message"]


def test_safe_validation_definition_retains_eligibility():
    from huddleroom.services.orchestration_agent_definition_review import assess_agent_definition

    assessment = assess_agent_definition(
        _agent(name="validator", role="tester", capabilities=["validation"]),
        ["validation"],
    )

    assert assessment["approved_for_work_functions"] == ["validation"]
    assert not assessment["warnings"]


def test_missing_optional_review_settings_are_risks_not_warnings():
    from huddleroom.services.orchestration_agent_definition_review import assess_agent_definition

    assessment = assess_agent_definition(
        _agent(name="validator", role="tester", capabilities=["validation"], config={}),
        ["validation"],
    )

    assert assessment["approved_for_work_functions"] == ["validation"]
    assert not assessment["warnings"]
    assert any("config.temperature" in risk for risk in assessment["risks"])
    assert any("config.reasoning_effort" in risk for risk in assessment["risks"])
    assert any("config.tools" in risk for risk in assessment["risks"])


@pytest.mark.parametrize("name", ["agent-7", "blue-team-helper", "Ada"])
def test_arbitrary_names_do_not_create_role_mismatch(name):
    from huddleroom.services.orchestration_agent_definition_review import assess_agent_definition

    assessment = assess_agent_definition(_agent(name=name, role="reviewer"), ["review"])

    assert "agent_review_role_name_mismatch" not in _warning_types(assessment)
    assert assessment["approved_for_work_functions"] == ["review"]


def test_declared_tools_and_model_are_recorded_but_not_ranked_or_warned():
    from huddleroom.services.orchestration_agent_definition_review import assess_agent_definition

    assessment = assess_agent_definition(
        _agent(
            provider="unknown-provider",
            model="unknown-model",
            adapter_type="cli",
            cli_runtime="codex",
            config={"temperature": 0.2, "reasoning_effort": "high", "tools": ["shell", "editor"]},
        ),
        ["implementation"],
    )

    assert not assessment["warnings"]
    assert assessment["approved_for_work_functions"] == ["implementation"]
    assert "unknown-provider/unknown-model" in assessment["fit_summary"]
    assert any("shell" in strength and "editor" in strength for strength in assessment["strengths"])


@pytest.mark.parametrize(
    ("agent", "proposed", "eligible", "requires_cli"),
    [
        (_agent(adapter_type="api"), ["implementation"], [], True),
        (_agent(adapter_type="api"), ["planning", "implementation"], ["planning"], True),
        (_agent(adapter_type="cli", cli_runtime="codex"), ["implementation"], ["implementation"], False),
        (
            _agent(
                adapter_type="api",
                description="Writing documents to files.",
                system_prompt="Create the requested document yourself.",
            ),
            ["summarization"],
            [],
            True,
        ),
        (
            _agent(adapter_type="api", description="Producing documents."),
            ["summarization"],
            [],
            True,
        ),
        (
            _agent(adapter_type="api", description="Builds documents."),
            ["summarization"],
            [],
            True,
        ),
        (
            _agent(adapter_type="api", description="Edits Markdown files."),
            ["summarization"],
            [],
            True,
        ),
        (
            _agent(adapter_type="api", description="Writes documentation."),
            ["summarization"],
            [],
            True,
        ),
        (
            _agent(adapter_type="api", description="Writes summaries to files."),
            ["summarization"],
            [],
            True,
        ),
        (
            _agent(adapter_type="api", description="Summarizes meetings.", system_prompt="Write a plain summary."),
            ["planning", "review", "meeting", "summarization"],
            ["planning", "review", "meeting", "summarization"],
            False,
        ),
    ],
)
def test_api_workspace_review_requires_cli_only_for_workspace_work(agent, proposed, eligible, requires_cli):
    from huddleroom.services.orchestration_agent_definition_review import assess_agent_definition

    assessment = assess_agent_definition(agent, proposed)

    assert ("agent_review_api_workspace_required" in _warning_types(assessment)) is requires_cli
    assert assessment["approved_for_work_functions"] == eligible
    if requires_cli:
        assert "CLI adapter" in assessment["recommended_changes"][-1]


def test_negated_manager_artifact_work_is_not_overreach():
    from huddleroom.services.orchestration_agent_definition_review import assess_agent_definition

    assessment = assess_agent_definition(
        _agent(
            name="manager",
            role="manager",
            description="Coordinates delivery.",
            system_prompt="Coordinate the team. Do not implement changes or edit files.",
        ),
        ["management"],
    )

    assert "agent_review_manager_overreach" not in _warning_types(assessment)
    assert assessment["approved_for_work_functions"] == ["management"]


@pytest.mark.parametrize(
    ("system_prompt", "overreach"),
    [
        ("Coordinates engineers implementing scoped changes", False),
        ("Do not implement changes, but edit files yourself.", True),
        ("Have engineers implement the changes, not yourself.", False),
        ("Your role is to implement changes.", True),
    ],
)
def test_manager_overreach_requires_self_or_imperative_artifact_work(
    system_prompt, overreach
):
    from huddleroom.services.orchestration_agent_definition_review import assess_agent_definition

    assessment = assess_agent_definition(
        _agent(
            name="manager",
            role="manager",
            description="Coordinates delivery.",
            system_prompt=system_prompt,
        ),
        ["management"],
    )

    assert ("agent_review_manager_overreach" in _warning_types(assessment)) is overreach
    assert assessment["approved_for_work_functions"] == ([] if overreach else ["management"])


@pytest.mark.parametrize(
    "system_prompt",
    [
        "You cannot implement code yourself",
        "You can't edit files yourself.",
        "You don't implement changes.",
        "The manager doesn't write code.",
        "You mustn't implement changes.",
        "You shouldn't edit files.",
        "You won't produce deliverables.",
    ],
)
def test_manager_overreach_recognizes_clause_local_negations(system_prompt):
    from huddleroom.services.orchestration_agent_definition_review import assess_agent_definition

    assessment = assess_agent_definition(
        _agent(
            name="manager",
            role="manager",
            description="Coordinates delivery.",
            system_prompt=system_prompt,
        ),
        ["management"],
    )

    assert "agent_review_manager_overreach" not in _warning_types(assessment)
    assert assessment["approved_for_work_functions"] == ["management"]


def test_definition_only_score_reuses_profiles_without_load_signals():
    from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper

    fit = OrchestrationRosterMapper().score_agent_definition(
        _agent(adapter_type="cli", cli_runtime="codex", config={}),
        "implementation",
    )

    assert fit.work_function == "implementation"
    assert {"role:developer", "capability:implementation"}.issubset(fit.matched_signals)
    assert "adapter:api" not in fit.matched_signals
    assert not any(signal.startswith("config:") for signal in fit.matched_signals)
    assert fit.load.penalty == 0
    assert not any(
        signal.startswith(("load_penalty:", "outcome_memory:"))
        for signal in fit.matched_signals
    )


@pytest_asyncio.fixture
async def review_goal(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the reviewed change",
        success_criteria=[{"key": "done", "description": "change works"}],
        constraints={"scope": "project"},
        weight="standard",
        authority_model="no_manager",
    )
    db_session.add(goal)
    await db_session.flush()
    return goal


@pytest_asyncio.fixture
async def review_run(db_session, review_goal):
    run = OrchestrationRun(goal_id=review_goal.id)
    db_session.add(run)
    await db_session.flush()
    return run


async def _persist_agent(db_session, **overrides):
    agent = _agent(**overrides)
    db_session.add(agent)
    await db_session.flush()
    return agent


async def _assign(db_session, project_id, run_id, agent, work_function, *, title="work"):
    task = Task(
        project_id=project_id,
        title=title,
        assigned_to=agent.id,
        metadata_={
            "orchestration": {
                "run_id": str(run_id),
                "work_function": work_function,
            }
        },
    )
    db_session.add(task)
    await db_session.flush()
    return task


@pytest.mark.asyncio
async def test_current_run_assignments_keeps_unmarked_decision_work(db_session, review_goal, review_run):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    agent = await _persist_agent(db_session)
    decision_work = await _assign(db_session, review_goal.project_id, review_run.id, agent, "decision")
    delivery = await _assign(db_session, review_goal.project_id, review_run.id, agent, "decision")
    delivery.metadata_ = {
        **delivery.metadata_,
        "orchestration": {
            **delivery.metadata_["orchestration"],
            "authority_decision_id": str(uuid.uuid4()),
        },
    }
    await db_session.flush()

    assignments = await AgentDefinitionReviewProcess()._current_run_assignments(
        db_session, review_goal.project_id, review_run.id
    )

    assert [(task.id, work_function) for task, _agent, work_function in assignments] == [
        (decision_work.id, "decision")
    ]


@pytest.mark.asyncio
async def test_current_run_assignments_excludes_linked_legacy_delivery_without_writing_marker(
    db_session, review_goal, review_run
):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    agent = await _persist_agent(db_session)
    task = await _assign(db_session, review_goal.project_id, review_run.id, agent, "decision")
    action = OrchestrationAction(
        run_id=review_run.id,
        idempotency_key=f"test:legacy-decision:{uuid.uuid4()}",
        action_type="create_delegation_task",
        target_type="task",
        target_id=task.id,
        status="completed",
    )
    db_session.add(action)
    await db_session.flush()
    task.metadata_ = {
        **task.metadata_, "orchestration": {**task.metadata_["orchestration"], "action_id": str(action.id)},
    }
    decision = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, review_goal.id, decision_key=f"test:legacy-decision:{uuid.uuid4()}",
        title="Approve", question="Approve?", authority="manager", authority_agent_id=agent.id,
        options=["approve"], run_id=review_run.id,
    )
    await OrchestrationAuthorityDecisionService().link_delegation_action(db_session, decision, action_id=action.id)

    assert await AgentDefinitionReviewProcess()._current_run_assignments(
        db_session, review_goal.project_id, review_run.id
    ) == []
    assert "authority_decision_id" not in task.metadata_["orchestration"]


async def _advance(db_session, goal, run):
    from huddleroom.services.orchestration_agent_definition_review import (
        AgentDefinitionReviewProcess,
    )

    return await AgentDefinitionReviewProcess().advance(db_session, goal, run)


async def _current_process(db_session, goal):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    return await OrchestrationProcessService().get_current(
        db_session, goal.id, "agent_definition_review"
    )


def _batch_answer_url(project_id, goal_id):
    return (
        f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}"
        "/decisions/agent-definition-review/batch-answer"
    )


async def _three_pending_proposals(db_session, review_goal, review_run):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    agents = [await _persist_agent(
        db_session,
        id=uuid.UUID(f"00000000-0000-0000-0000-0000000001{index:02d}"),
        name=f"agent-{index}",
        description="Original description.",
        system_prompt="Original persona.",
    ) for index in range(3)]
    for agent in agents:
        await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    process_service = OrchestrationProcessService()
    manager_selection = await process_service.start_process(
        db_session, review_goal.id, process_type="manager_selection",
        trigger_reason="test prerequisite", run_id=review_run.id,
    )
    await process_service.complete_process(db_session, manager_selection)
    analyzer = ThreeProposalAnalyzer()
    process = AgentDefinitionReviewProcess(analyzer=analyzer)
    assert (await process.advance(db_session, review_goal, review_run))["status"] == "waiting_decision"
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, review_goal.id, status="pending"
    )
    return agents, analyzer, process, await _current_process(db_session, review_goal), decisions


@pytest.mark.asyncio
async def test_agent_definition_review_batch_applies_mixed_answers_once_without_rerun(
    client, db_session, test_project, review_goal, review_run
):
    agents, analyzer, process, current, decisions = await _three_pending_proposals(
        db_session, review_goal, review_run
    )

    response = await client.post(_batch_answer_url(test_project.id, review_goal.id), json={"answers": [
        {"decision_id": str(decisions[0].id), "selected_option": "approve"},
        {"decision_id": str(decisions[1].id), "selected_option": "edit",
         "edited_description": " Edited description. ", "edited_persona": " Edited persona. "},
        {"decision_id": str(decisions[2].id), "selected_option": "reject", "reason": "Keep it."},
    ]})

    assert response.status_code == 200, response.text
    assert analyzer.calls == 3
    await db_session.refresh(current)
    assert current.status == "completed" and current.superseded_by_id is None
    assert current.outputs["fingerprint"] == await process.current_fingerprint(
        db_session, review_goal, review_run
    )
    assert current.outputs["coverage_fingerprint"] == await process.current_coverage_fingerprint(
        db_session, review_goal, review_run
    )
    assert [agent.description for agent in agents] == [
        "Proposed description 1.", "Edited description.", "Original description.",
    ]


@pytest.mark.asyncio
async def test_agent_definition_review_batch_rejects_duplicate_ids_without_changes(
    client, db_session, test_project, review_goal, review_run
):
    agents, _analyzer, _process, current, decisions = await _three_pending_proposals(
        db_session, review_goal, review_run
    )

    response = await client.post(_batch_answer_url(test_project.id, review_goal.id), json={"answers": [
        {"decision_id": str(decisions[0].id), "selected_option": "approve"},
        {"decision_id": str(decisions[0].id), "selected_option": "approve"},
    ]})

    assert response.status_code == 400
    assert [decision.status for decision in decisions] == ["pending"] * 3
    assert [agent.description for agent in agents] == ["Original description."] * 3
    assert current.status == "waiting_decision"


@pytest.mark.asyncio
async def test_agent_definition_review_batch_rejects_blank_edit_without_changes(
    client, db_session, test_project, review_goal, review_run
):
    agents, _analyzer, _process, current, decisions = await _three_pending_proposals(
        db_session, review_goal, review_run
    )

    response = await client.post(_batch_answer_url(test_project.id, review_goal.id), json={"answers": [
        {"decision_id": str(decisions[0].id), "selected_option": "edit",
         "edited_description": " ", "edited_persona": "Edited persona."},
        {"decision_id": str(decisions[1].id), "selected_option": "approve"},
        {"decision_id": str(decisions[2].id), "selected_option": "reject"},
    ]})

    assert response.status_code == 400
    assert [decision.status for decision in decisions] == ["pending"] * 3
    assert [agent.description for agent in agents] == ["Original description."] * 3
    assert current.status == "waiting_decision"


@pytest.mark.asyncio
async def test_agent_definition_review_batch_rejects_stale_pending_set(
    client, db_session, test_project, review_goal, review_run, test_user
):
    agents, _analyzer, _process, current, decisions = await _three_pending_proposals(
        db_session, review_goal, review_run
    )
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decisions[0], selected_option="approve", reason=None,
        decided_by_user_id=test_user.id,
    )

    response = await client.post(_batch_answer_url(test_project.id, review_goal.id), json={"answers": [
        {"decision_id": str(decision.id), "selected_option": "approve"} for decision in decisions
    ]})

    assert response.status_code == 409
    assert [decision.status for decision in decisions] == ["answered", "pending", "pending"]
    assert [agent.description for agent in agents] == ["Original description."] * 3
    assert current.status == "waiting_decision"


@pytest.mark.asyncio
async def test_agent_definition_review_batch_rejects_schema_invalid_body_as_bad_request(
    client, test_project, review_goal
):
    response = await client.post(
        _batch_answer_url(test_project.id, review_goal.id), json={"answers": []}
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_retry_failed_agent_review_reuses_completed_frozen_requests(
    db_session, review_goal, review_run, monkeypatch
):
    """A retry must resume at the failed semantic request, not restart the loop."""
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    class RetryAnalyzer:
        def __init__(self, failed_agent_name, agent_ids):
            self.failed_agent_name = failed_agent_name
            self.agent_ids = agent_ids
            self.calls = []
            self.requests = []
            self.build_count = 0
            self.failure_count = 0

        def build_request(self, agent_snapshot, goal_snapshot, candidate_work_functions, project=None):
            self.build_count += 1
            return {
                "agent": copy.deepcopy(agent_snapshot),
                "goal": copy.deepcopy(goal_snapshot),
                "candidate_work_functions": list(candidate_work_functions),
                "build_count": self.build_count,
            }

        async def review_request(self, request, candidate_work_functions, *, project_id=None):
            self.requests.append(copy.deepcopy(request))
            agent_id = self.agent_ids[request["agent"]["name"]]
            self.calls.append(agent_id)
            if request["agent"]["name"] == self.failed_agent_name and self.failure_count < 2:
                self.failure_count += 1
                error = RuntimeError(f"provider unavailable {self.failure_count}")
                error.request = request
                error.raw_response = '{"partial": true}'
                error.category = "provider_error"
                error.full_error = f"provider unavailable {self.failure_count} with diagnostic evidence"
                raise error
            if request["agent"]["name"] == self.failed_agent_name:
                return SemanticAgentAssessment(
                    "improvement_proposed",
                    ("Definition needs a clearer scope.",),
                    "Propose a focused replacement definition.",
                    tuple(candidate_work_functions),
                )
            return SemanticAgentAssessment(
                "approved", (), "Definition is actionable.", tuple(candidate_work_functions)
            )

    agent_a = await _persist_agent(db_session, id=uuid.UUID("00000000-0000-0000-0000-000000000101"), name="agent-a")
    agent_b = await _persist_agent(db_session, id=uuid.UUID("00000000-0000-0000-0000-000000000102"), name="agent-b")
    agent_c = await _persist_agent(db_session, id=uuid.UUID("00000000-0000-0000-0000-000000000103"), name="agent-c")
    for agent in (agent_a, agent_b, agent_c):
        await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    analyzer = RetryAnalyzer(
        agent_b.name,
        {agent.name: agent.id for agent in (agent_a, agent_b, agent_c)},
    )
    process = AgentDefinitionReviewProcess(analyzer=analyzer)

    first = await process.advance(db_session, review_goal, review_run)
    current = await _current_process(db_session, review_goal)
    checkpoint = copy.deepcopy(current.outputs["_lm_retry"])

    assert first["retryable"] is True
    assert analyzer.calls == [agent_a.id, agent_b.id]
    assert checkpoint["cursor"] == 1
    assert set(checkpoint["completed"]) == {str(agent_a.id)}
    assert current.outputs["semantic_error"]["raw_response"] == '{"partial": true}'
    frozen_model = checkpoint["model"]
    unrelated = await process.warning_service.create_warning(
        db_session,
        review_goal.id,
        warning_type="agent_definition_review_analyzer_error",
        severity="warning",
        message="unrelated analyzer warning",
        run_id=review_run.id,
        source_process_run_id=current.id,
        related_agent_id=agent_c.id,
    )

    monkeypatch.setattr(
        "huddleroom.services.orchestration_agent_definition_review.settings.orchestration_model",
        "changed-after-failure",
    )
    second_failure = await process.retry_failed(db_session, review_goal, review_run, current)

    assert second_failure["retry_failed"] is True
    retry_warning = await db_session.get(OrchestrationWarning, uuid.UUID(current.outputs["_lm_retry"]["warning_id"]))
    assert "provider unavailable 2" in retry_warning.message
    assert "provider unavailable 1" not in retry_warning.message

    review_run.active_blockers.append({
        "kind": "agent_definition_review_analyzer_error",
        "warning_id": str(unrelated.id),
        "reason": "unrelated analyzer warning",
    })
    result = await process.retry_failed(db_session, review_goal, review_run, current)

    assert analyzer.calls == [agent_a.id, agent_b.id, agent_b.id, agent_b.id, agent_c.id]
    assert analyzer.requests[2] == checkpoint["request"]
    assert analyzer.requests[-1]["model"] == frozen_model
    assert analyzer.build_count == 3
    assert result["status"] == "waiting_decision"
    reviews = list((await db_session.execute(select(OrchestrationAgentReview))).scalars())
    assert len(reviews) == 3
    assert current.id == (await _current_process(db_session, review_goal)).id
    warnings = list((await db_session.execute(select(OrchestrationWarning))).scalars())
    assert all(not warning.active for warning in warnings if warning.related_agent_id == agent_b.id)
    assert unrelated.active is True
    assert review_goal.status == "blocked"
    assert review_run.status == "blocked"
    assert "_lm_retry" not in current.outputs
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, review_goal.id, status="pending"
    )
    assert len(decisions) == 1
    assert review_run.active_blockers == [{
        "kind": "agent_definition_review_analyzer_error",
        "warning_id": str(unrelated.id),
        "reason": "unrelated analyzer warning",
    }]
    assert current.outputs["coverage_fingerprint"] == await process.current_coverage_fingerprint(
        db_session, review_goal, review_run, covered_target_ids=set(current.outputs["target_ids"]),
    )


@pytest.mark.asyncio
async def test_semantic_rewrite_waits_for_human_then_updates_without_rerun(
    db_session, review_goal, review_run, test_user
):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    agent = await _persist_agent(
        db_session, description="Helps with code.", system_prompt="Be useful."
    )
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    analyzer = SequencedSemanticAnalyzer()
    process = AgentDefinitionReviewProcess(analyzer=analyzer)

    first = await process.advance(db_session, review_goal, review_run)
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, review_goal.id, status="pending"
    ))[0]
    assert first["status"] == "waiting_decision"
    assert agent.description == "Helps with code."
    assert agent.system_prompt == "Be useful."
    context = json.loads(decision.context)
    assert context["original_description"] == "Helps with code."
    assert context["proposed_persona"].startswith("Implement scoped changes")

    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session,
        decision,
        selected_option="approve",
        reason="Clearer ownership and boundaries.",
        decided_by_user_id=test_user.id,
    )
    second = await process.advance(db_session, review_goal, review_run)

    assert second["status"] == "completed"
    assert agent.description == "Developer role responsible for implementation."
    assert agent.system_prompt.startswith("Implement scoped changes")
    assert analyzer.calls == 1
    current = await _current_process(db_session, review_goal)
    assert current.outputs["eligibility"] == {str(agent.id): ["implementation"]}
    assert str(decision.id) in current.outputs["decision_ids"]


def test_semantic_approved_functions_normalize_candidate_casing():
    assessment = parse_semantic_agent_assessment({
        "status": "approved", "problems": [], "reason": "Actionable.",
        "approved_work_functions": ["IMPLEMENTATION"],
        "proposed_description": None, "proposed_persona": None,
    }, ["implementation"])

    assert assessment.approved_work_functions == ("implementation",)


@pytest.mark.asyncio
async def test_rejected_semantic_rewrite_preserves_definition(
    db_session, review_goal, review_run, test_user
):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    agent = await _persist_agent(db_session, description="Original.", system_prompt="Original persona.")
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    process = AgentDefinitionReviewProcess(analyzer=SequencedSemanticAnalyzer())
    await process.advance(db_session, review_goal, review_run)
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, review_goal.id, status="pending"
    ))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session,
        decision,
        selected_option="reject",
        reason="Keep current wording.",
        decided_by_user_id=test_user.id,
    )

    result = await process.advance(db_session, review_goal, review_run)

    assert result["status"] == "completed"
    assert (agent.description, agent.system_prompt) == ("Original.", "Original persona.")


@pytest.mark.asyncio
async def test_legacy_valid_edit_applies_agent_definition(
    db_session, review_goal, review_run, test_user
):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    agent = await _persist_agent(db_session, description="Original.", system_prompt="Original persona.")
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    process = AgentDefinitionReviewProcess(analyzer=SequencedSemanticAnalyzer())
    await process.advance(db_session, review_goal, review_run)
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, review_goal.id, status="pending"
    ))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="edit",
        reason=json.dumps({"description": " Edited description. ", "persona": " Edited persona. "}),
        decided_by_user_id=test_user.id,
    )

    await process.advance(db_session, review_goal, review_run)

    assert (agent.description, agent.system_prompt) == ("Edited description.", "Edited persona.")


@pytest.mark.asyncio
async def test_legacy_malformed_edit_requeues_and_parks_for_a_new_answer(
    db_session, review_goal, review_run, test_user
):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    agent = await _persist_agent(db_session, description="Original.", system_prompt="Original persona.")
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    process = AgentDefinitionReviewProcess(analyzer=SequencedSemanticAnalyzer())
    await process.advance(db_session, review_goal, review_run)
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, review_goal.id, status="pending"
    ))[0]
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="edit", reason="not JSON", decided_by_user_id=test_user.id,
    )

    result = await process.advance(db_session, review_goal, review_run)
    assert result["status"] == "waiting_decision"
    assert (agent.description, agent.system_prompt) == ("Original.", "Original persona.")
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, review_goal.id)
    assert [item.status for item in decisions] == ["answered", "pending"]
    assert decisions[1].decision_key == decision.decision_key
    assert decisions[1].source_process_run_id == decision.source_process_run_id


@pytest.mark.asyncio
async def test_semantic_request_and_persisted_snapshot_redact_nested_credentials(
    db_session, review_goal, review_run, monkeypatch
):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review_request", _semantic_review_request)
    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "status": "approved", "problems": [], "proposed_description": None,
            "proposed_persona": None, "reason": "Actionable.",
            "approved_work_functions": ["implementation"],
        })))])

    agent = await _persist_agent(
        db_session,
        config={"tools": ["editor"], "nested": {"api_key": "secret-value", "token": "token-value"}},
    )
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    process = AgentDefinitionReviewProcess(
        analyzer=AgentDefinitionSemanticAnalyzer(completion_fn=completion)
    )
    await process.advance(db_session, review_goal, review_run)
    review = (await db_session.execute(select(OrchestrationAgentReview))).scalar_one()

    assert "secret-value" not in captured["messages"][1]["content"]
    assert "token-value" not in captured["messages"][1]["content"]
    assert review.definition_snapshot["config"]["nested"] == {
        "api_key": "[REDACTED]", "token": "[REDACTED]"
    }


@pytest.mark.asyncio
async def test_exact_logged_ceo_review_uses_validated_model_proposal(
    db_session, review_goal, review_run, test_user, monkeypatch
):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review_request", _semantic_review_request)
    async def completion(**_kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "status": "improvement_proposed",
            "problems": ["Description is circular and vague."],
            "reason": "The proposal makes the management role actionable.",
            "approved_work_functions": ["management"],
            "proposed_description": (
                "Management role responsible for product direction, prioritization, "
                "acceptance criteria, and delegation."
            ),
            "proposed_persona": (
                "Set product direction, prioritize work, define acceptance criteria, and "
                "delegate within configured capabilities. Do not expand permissions. "
                "Report verification evidence."
            ),
        })))])

    agent = await _persist_agent(
        db_session,
        name="CEO",
        role="Manager",
        description="Manager responsible for management on the supplied goal.",
        system_prompt=(
            "Own only assigned work for management. Use only configured tools: none. "
            "Use only configured capabilities: product direction, prioritization, acceptance criteria, delegation. "
            "Do not expand permissions. Report verification evidence."
        ),
        config={},
        capabilities=["product direction", "prioritization", "acceptance criteria", "delegation"],
    )
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "management")
    await AgentDefinitionReviewProcess(
        analyzer=AgentDefinitionSemanticAnalyzer(completion_fn=completion)
    ).advance(db_session, review_goal, review_run)
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, review_goal.id, status="pending"
    ))[0]
    context = json.loads(decision.context)
    assert context["proposed_description"] == (
        "Management role responsible for product direction, prioritization, acceptance criteria, and delegation."
    )
    proposed_persona = (
        "Set product direction, prioritize work, define acceptance criteria, and delegate within configured "
        "capabilities. Do not expand permissions. Report verification evidence."
    )
    assert context["proposed_persona"] == proposed_persona
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", reason="Use the suggestion.",
        decided_by_user_id=test_user.id,
    )
    await AgentDefinitionReviewProcess(
        analyzer=AgentDefinitionSemanticAnalyzer(completion_fn=completion)
    ).advance(db_session, review_goal, review_run)
    assert agent.system_prompt == proposed_persona


@pytest.mark.asyncio
async def test_goal_drift_leaves_pending_semantic_proposal_parked_with_suggestion(
    db_session, review_goal, review_run
):
    """Item 3, orchestrator override: goal drift while a decision is pending
    must never cancel it -- it stays parked, and a one-time suggestion is
    raised instead (was: auto-cancel + restart)."""
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    agent = await _persist_agent(db_session)
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    process = AgentDefinitionReviewProcess(analyzer=SequencedSemanticAnalyzer())
    await process.advance(db_session, review_goal, review_run)
    pending = (await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, review_goal.id, status="pending"
    ))[0]
    review_goal.objective = "A changed semantic objective"
    await db_session.flush()

    result = await process.advance(db_session, review_goal, review_run)

    assert result["status"] == "waiting_decision"
    await db_session.refresh(pending)
    assert pending.status == "pending"

    warnings = await OrchestrationWarningService().list_warnings(db_session, review_goal.id, active_only=True)
    assert any(w.warning_type == "agent_definition_review_stale_inputs" for w in warnings)


@pytest.mark.asyncio
async def test_trivial_review_process_completes_compressed_without_reviews_or_warnings(
    db_session, review_goal, review_run
):
    review_goal.weight = "trivial"

    summary = await _advance(db_session, review_goal, review_run)
    current = (
        await db_session.execute(select(OrchestrationProcessRun))
    ).scalar_one()
    memory = (
        await db_session.execute(select(OrchestrationMemorySection))
    ).scalar_one()

    assert summary == {
        "process_type": "agent_definition_review",
        "status": "completed",
        "questions_created": 0,
    }
    assert current.outputs["compressed"] is True
    assert current.outputs["review_ids"] == []
    assert current.outputs["target_ids"] == []
    assert current.outputs["eligibility"] == {}
    assert current.outputs["warning_count"] == 0
    assert current.outputs["gates"] == {"agent_definitions_reviewed": True}
    assert memory.section_key == "agent_definition_review"
    assert memory.created_by == "orchestrator:agent_definition_review"
    assert not (await db_session.execute(select(OrchestrationAgentReview))).scalars().all()
    assert not (await db_session.execute(select(OrchestrationWarning))).scalars().all()


@pytest.mark.asyncio
async def test_standard_reviews_manager_and_current_run_assignees_once_with_exact_warning_links(
    db_session, review_goal, review_run
):
    manager = await _persist_agent(
        db_session,
        name="manager",
        role="manager",
        description="Coordinates delivery.",
        system_prompt="Manage the team and edit files yourself.",
    )
    builder = await _persist_agent(db_session)
    other_run_agent = await _persist_agent(db_session, name="other-run-agent")
    review_goal.authority_model = "agent_manager"
    review_goal.manager_agent_id = manager.id
    await _assign(db_session, review_goal.project_id, review_run.id, builder, "implementation")
    await _assign(
        db_session,
        review_goal.project_id,
        uuid.uuid4(),
        other_run_agent,
        "review",
        title="other run",
    )

    await _advance(db_session, review_goal, review_run)
    current = (
        await db_session.execute(select(OrchestrationProcessRun))
    ).scalar_one()
    reviews = list(
        (
            await db_session.execute(
                select(OrchestrationAgentReview).order_by(OrchestrationAgentReview.agent_id)
            )
        ).scalars()
    )
    warnings = list((await db_session.execute(select(OrchestrationWarning))).scalars())
    by_agent = {review.agent_id: review for review in reviews}

    assert set(by_agent) == {manager.id, builder.id}
    assert by_agent[manager.id].proposed_work_functions == ["management"]
    assert by_agent[manager.id].approved_for_work_functions == []
    assert by_agent[builder.id].proposed_work_functions == ["implementation"]
    assert by_agent[builder.id].approved_for_work_functions == ["implementation"]
    assert {warning.source_agent_review_id for warning in warnings} == {by_agent[manager.id].id}
    assert all(warning.related_agent_id == manager.id for warning in warnings)
    assert current.outputs["eligibility"] == {
        str(builder.id): ["implementation"],
        str(manager.id): [],
    }
    assert current.outputs["warning_ids"] == [str(warning.id) for warning in warnings]
    assert current.outputs["warning_count"] == len(warnings)

    first_fingerprint = current.outputs["fingerprint"]
    await _advance(db_session, review_goal, review_run)
    assert len((await db_session.execute(select(OrchestrationProcessRun))).scalars().all()) == 1
    assert len((await db_session.execute(select(OrchestrationAgentReview))).scalars().all()) == 2
    assert len((await db_session.execute(select(OrchestrationWarning))).scalars().all()) == len(warnings)
    assert current.outputs["fingerprint"] == first_fingerprint


@pytest.mark.asyncio
async def test_analyzer_failure_parks_without_persisting_reviews_and_creates_warning(
    db_session, review_goal, review_run, monkeypatch
):
    """SPR #85: an analyzer-level provider error must PARK the review (blocker +
    warning) instead of persisting a partial/incorrect review for the failed agent."""
    agent = await _persist_agent(db_session)
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")

    async def boom(self, request, candidate_work_functions, *, project_id=None):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "huddleroom.services.orchestration_agent_definition_analyzer.AgentDefinitionSemanticAnalyzer.review_request",
        boom,
    )

    result = await _advance(db_session, review_goal, review_run)

    assert result["retryable"] is True
    assert review_goal.status == "blocked"
    assert review_run.status == "blocked"
    assert any(
        b["kind"] == "agent_definition_review_analyzer_error" for b in review_run.active_blockers
    )
    assert (await db_session.execute(select(OrchestrationAgentReview))).scalars().all() == []

    warnings = list((await db_session.execute(select(OrchestrationWarning))).scalars())
    assert any(
        w.warning_type == "agent_definition_review_analyzer_error" and w.active
        for w in warnings
    )


@pytest.mark.asyncio
async def test_analyzer_failure_recovery_resolves_warning_and_unblocks(
    db_session, review_goal, review_run, test_project, monkeypatch
):
    """Recovering from a parked analyzer failure via the operator-facing
    baseline/step endpoint must resolve the analyzer-error warning, clear the
    blocker, and restore goal/run status -- mirrors the goal_definition
    recovery contract (test_goal_definition_debug_recovery_restores_status_...)."""
    from huddleroom.services import orchestration_debug_service
    from huddleroom.services.orchestration_agent_definition_analyzer import SemanticAgentAssessment
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService
    from tests.conftest import complete_baseline_processes

    # Complete every baseline process once (with a safe analyzer) so
    # goal_definition + manager_selection are terminal, matching the real
    # gate agent_definition_review advances behind.
    agent = await _persist_agent(db_session)
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    await complete_baseline_processes(db_session, review_goal, review_run)

    # Force a new review round: add a second, materially different assignment.
    # Completed + stale inputs is now a one-time suggestion, not an
    # auto-rerun (item 3), so the fresh review round is started via the
    # human-facing rerun endpoint (the "Approve" action), exactly as the
    # real recovery flow would.
    other_agent = await _persist_agent(db_session, name="second-reviewer")
    await _assign(
        db_session, review_goal.project_id, review_run.id, other_agent, "review", title="second"
    )

    async def boom(self, request, candidate_work_functions):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "huddleroom.services.orchestration_agent_definition_analyzer.AgentDefinitionSemanticAnalyzer.review_request",
        boom,
    )
    process = AgentDefinitionReviewProcess()
    monkeypatch.setattr(orchestration_debug_service, "AgentDefinitionReviewProcess", lambda: process)
    await OrchestrationDebugService().rerun_last(
        db_session, test_project.id, review_goal.id, "agent_definition_review"
    )

    async def approve(self, request, candidate_work_functions, *, project_id=None):
        return SemanticAgentAssessment("approved", (), "ok", tuple(candidate_work_functions))

    monkeypatch.setattr(
        "huddleroom.services.orchestration_agent_definition_analyzer.AgentDefinitionSemanticAnalyzer.review_request",
        approve,
    )
    monkeypatch.setattr(orchestration_debug_service, "AgentDefinitionReviewProcess", lambda: process)

    with pytest.raises(HTTPException, match="baseline/retry"):
        await OrchestrationDebugService().step(
            db_session, test_project.id, review_goal.id, "agent_definition_review"
        )
    result = await OrchestrationDebugService().retry_failed(
        db_session, test_project.id, review_goal.id, "agent_definition_review"
    )
    assert result["process"]["status"] == "completed"

    warnings = list((await db_session.execute(select(OrchestrationWarning))).scalars())
    target = next(w for w in warnings if w.warning_type == "agent_definition_review_analyzer_error")
    assert target.active is False
    assert target.resolved_by == "orchestrator:agent_definition_review"
    assert not any(
        b["kind"] == "agent_definition_review_analyzer_error" for b in review_run.active_blockers
    )
    assert review_goal.status == "active"
    assert review_run.status == "running"


@pytest.mark.asyncio
async def test_tick_short_circuits_when_agent_definition_review_blocked(
    db_session, review_goal, review_run, monkeypatch
):
    """CRITICAL fix: once agent_definition_review is parked on an analyzer error,
    OrchestrationService.tick() must short-circuit before re-invoking the
    analyzer (no auto-retry against litellm on every tick)."""
    from huddleroom.services import orchestration_service
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.conftest import complete_baseline_processes

    agent = await _persist_agent(db_session)
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    await complete_baseline_processes(db_session, review_goal, review_run)

    review_run.active_blockers = [
        {"kind": "agent_definition_review_analyzer_error", "reason": "provider unavailable"}
    ]
    review_goal.status = "blocked"
    review_run.status = "blocked"
    await db_session.flush()

    class NoAutoAdvance:
        async def advance(self, *args, **kwargs):
            raise AssertionError("tick re-invoked agent_definition_review while blocked")

    monkeypatch.setattr(orchestration_service, "AgentDefinitionReviewProcess", NoAutoAdvance)

    # Must not raise -- tick short-circuits before touching AgentDefinitionReviewProcess.
    await OrchestrationService().tick(db_session, review_run.id)


@pytest.mark.asyncio
async def test_review_context_records_load_observations_without_affecting_decisions(
    db_session, review_goal, review_run, monkeypatch
):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper

    agent = await _persist_agent(db_session)
    assigned_task = await _assign(
        db_session, review_goal.project_id, review_run.id, agent, "implementation"
    )
    db_session.add_all(
        [
            Session(
                task_id=assigned_task.id,
                agent_id=agent.id,
                project_id=review_goal.project_id,
                adapter_type="api",
                status="running",
            ),
            MemoryItem(
                agent_id=agent.id,
                project_id=review_goal.project_id,
                scope="project",
                content="Successfully completed the assigned work.",
            ),
        ]
    )
    await db_session.flush()

    load_queries = 0
    original_loads = OrchestrationRosterMapper.loads_by_agent

    async def counted_loads(self, db, project_id):
        nonlocal load_queries
        load_queries += 1
        return await original_loads(self, db, project_id)

    monkeypatch.setattr(OrchestrationRosterMapper, "loads_by_agent", counted_loads)

    await _advance(db_session, review_goal, review_run)
    first_process = await _current_process(db_session, review_goal)
    first_review = (
        await db_session.execute(select(OrchestrationAgentReview))
    ).scalar_one()
    first_context = json.loads(first_review.review_context)
    assert first_context["active_session_count"] == 1
    assert first_context["active_task_count"] == 1
    assert first_context["available_outcome_hint_count"] == 1
    assert first_context["semantic_assessment"]["status"] == "approved"
    assert load_queries == 1

    extra_task = Task(
        project_id=review_goal.project_id,
        title="non-orchestration load",
        assigned_to=agent.id,
    )
    db_session.add(extra_task)
    await db_session.flush()
    db_session.add_all(
        [
            Session(
                task_id=extra_task.id,
                agent_id=agent.id,
                project_id=review_goal.project_id,
                adapter_type="api",
                status="pending",
            ),
            MemoryItem(
                agent_id=agent.id,
                project_id=review_goal.project_id,
                scope="project",
                content="Review passed and was accepted.",
            ),
        ]
    )
    await db_session.flush()
    await OrchestrationProcessService().start_process(
        db_session,
        review_goal.id,
        process_type="agent_definition_review",
        trigger_reason="test: observe changed load",
        run_id=review_run.id,
    )

    await _advance(db_session, review_goal, review_run)
    second_process = await _current_process(db_session, review_goal)
    reviews = list(
        (
            await db_session.execute(
                select(OrchestrationAgentReview).order_by(OrchestrationAgentReview.created_at)
            )
        ).scalars()
    )
    second_context = json.loads(reviews[-1].review_context)
    assert second_context["active_session_count"] == 2
    assert second_context["active_task_count"] == 2
    assert second_context["available_outcome_hint_count"] == 2
    assert second_context["semantic_assessment"]["status"] == "approved"
    assert load_queries == 2
    assert second_process.outputs["fingerprint"] == first_process.outputs["fingerprint"]
    assert second_process.outputs["eligibility"] == first_process.outputs["eligibility"]
    assert second_process.outputs["warning_count"] == first_process.outputs["warning_count"]


@pytest.mark.asyncio
async def test_substantial_reviews_all_active_agents_and_inactive_current_run_assignees(
    db_session, review_goal, review_run
):
    review_goal.weight = "substantial"
    active_builder = await _persist_agent(db_session)
    active_reviewer = await _persist_agent(
        db_session,
        name="reviewer",
        role="reviewer",
        capabilities=["review"],
    )
    inactive_assignee = await _persist_agent(
        db_session,
        name="retired-builder",
        is_active=False,
    )
    inactive_unassigned = await _persist_agent(
        db_session,
        name="retired-reviewer",
        role="reviewer",
        is_active=False,
    )
    await _assign(
        db_session, review_goal.project_id, review_run.id, inactive_assignee, "implementation"
    )

    await _advance(db_session, review_goal, review_run)
    reviews = list((await db_session.execute(select(OrchestrationAgentReview))).scalars())
    by_agent = {review.agent_id: review for review in reviews}
    warnings = list((await db_session.execute(select(OrchestrationWarning))).scalars())

    assert active_builder.id in by_agent
    assert active_reviewer.id in by_agent
    assert inactive_assignee.id in by_agent
    assert inactive_unassigned.id not in by_agent
    assert "implementation" in by_agent[active_builder.id].proposed_work_functions
    assert "review" in by_agent[active_reviewer.id].proposed_work_functions
    inactive_warning = next(
        warning for warning in warnings if warning.warning_type == "agent_review_inactive_assignee"
    )
    assert inactive_warning.related_agent_id == inactive_assignee.id
    assert inactive_warning.source_agent_review_id == by_agent[inactive_assignee.id].id


@pytest.mark.asyncio
async def test_substantial_active_agent_drops_unmatched_assigned_function(
    db_session, review_goal, review_run
):
    review_goal.weight = "substantial"
    builder = await _persist_agent(db_session)
    await _assign(
        db_session,
        review_goal.project_id,
        review_run.id,
        builder,
        "quantum_tuning",
    )

    await _advance(db_session, review_goal, review_run)
    review = (
        await db_session.execute(
            select(OrchestrationAgentReview).where(
                OrchestrationAgentReview.agent_id == builder.id
            )
        )
    ).scalar_one()

    assert review.proposed_work_functions == ["implementation"]
    assert review.approved_for_work_functions == ["implementation"]


@pytest.mark.asyncio
async def test_running_process_with_changed_fingerprint_restarts_without_mixing_reviews(
    db_session, review_goal, review_run
):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    agent = await _persist_agent(
        db_session,
        config={"allow_global_scope": True},
    )
    await _assign(
        db_session, review_goal.project_id, review_run.id, agent, "implementation"
    )
    process_service = OrchestrationProcessService()
    partial = await process_service.start_process(
        db_session,
        review_goal.id,
        process_type="agent_definition_review",
        trigger_reason="partially persisted attempt",
        run_id=review_run.id,
        input_snapshot={"fingerprint": "stale-fingerprint"},
    )
    old_review = await OrchestrationAgentReviewService().create_review(
        db_session,
        review_goal.id,
        agent_id=agent.id,
        fit_summary="Partial unsafe review.",
        proposed_work_functions=["implementation"],
        approved_for_work_functions=[],
        run_id=review_run.id,
        source_process_run_id=partial.id,
    )
    old_warning = await OrchestrationWarningService().create_warning(
        db_session,
        review_goal.id,
        warning_type="agent_review_unsafe_permissions",
        severity="warning",
        message="Partial unsafe finding.",
        run_id=review_run.id,
        source_process_run_id=partial.id,
        related_agent_id=agent.id,
        source_agent_review_id=old_review.id,
    )
    agent.config = {
        "temperature": 0.2,
        "reasoning_effort": "medium",
        "tools": ["editor"],
    }
    await db_session.flush()

    await _advance(db_session, review_goal, review_run)
    current = await _current_process(db_session, review_goal)
    reviews = list(
        (
            await db_session.execute(
                select(OrchestrationAgentReview).order_by(OrchestrationAgentReview.created_at)
            )
        ).scalars()
    )

    assert current.id != partial.id
    assert partial.superseded_by_id == current.id
    assert len(reviews) == 2
    new_review = next(review for review in reviews if review.id != old_review.id)
    assert current.outputs["review_ids"] == [str(new_review.id)]
    assert new_review.source_process_run_id == current.id
    assert new_review.definition_snapshot["config"] == agent.config
    assert old_warning.active is False

    await _advance(db_session, review_goal, review_run)
    assert (await _current_process(db_session, review_goal)).id == current.id
    assert len((await db_session.execute(select(OrchestrationProcessRun))).scalars().all()) == 2
    assert len((await db_session.execute(select(OrchestrationAgentReview))).scalars().all()) == 2
    assert len((await db_session.execute(select(OrchestrationWarning))).scalars().all()) == 1


@pytest.mark.asyncio
async def test_empty_standard_roster_suggests_rerun_when_assignment_appears(
    db_session, review_goal, review_run
):
    """Item 3, orchestrator override: an assignment appearing after a
    completed empty-roster review is a one-time suggestion, not an
    auto-rerun -- the completed row and its fingerprint stand until a human
    approves a rerun (mirrors the debug service's rerun endpoint: a fresh
    process row, superseding the old one)."""
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    await _advance(db_session, review_goal, review_run)
    first = (
        await db_session.execute(select(OrchestrationProcessRun))
    ).scalar_one()
    assert first.outputs["target_ids"] == []
    first_fingerprint = first.outputs["fingerprint"]

    unsafe = await _persist_agent(
        db_session,
        config={"allow_global_scope": True},
    )
    task = await _assign(
        db_session, review_goal.project_id, review_run.id, unsafe, "implementation"
    )
    await _advance(db_session, review_goal, review_run)
    second = await _current_process(db_session, review_goal)
    assert second.id == first.id
    assert second.outputs["fingerprint"] == first_fingerprint

    first_warning = (
        await db_session.execute(
            select(OrchestrationWarning).where(OrchestrationWarning.active.is_(True))
        )
    ).scalar_one()
    assert first_warning.warning_type == "agent_definition_review_stale_inputs"
    assert first_warning.source_process_run_id == first.id

    # Repeated ticks against the same fingerprint change don't pile up a
    # second warning for the same process run.
    task.metadata_ = {
        "orchestration": {
            "run_id": str(review_run.id),
            "work_function": "planning",
        }
    }
    await db_session.flush()
    await _advance(db_session, review_goal, review_run)
    assert (await _current_process(db_session, review_goal)).id == first.id
    active_warnings = (
        await db_session.execute(
            select(OrchestrationWarning).where(OrchestrationWarning.active.is_(True))
        )
    ).scalars().all()
    assert len(active_warnings) == 1

    # A human approving the suggestion (the rerun endpoint's own "human
    # requested" path: start a fresh row, supersede the old one) produces a
    # new process run with the recomputed fingerprint.
    fresh = await OrchestrationProcessService().start_process(
        db_session,
        review_goal.id,
        process_type="agent_definition_review",
        trigger_reason="human requested: rerun of agent_definition_review",
        run_id=review_run.id,
        input_snapshot=first.input_snapshot,
        process_version=first.process_version,
    )
    first.superseded_by_id = fresh.id
    await db_session.flush()
    await _advance(db_session, review_goal, review_run)
    third = await _current_process(db_session, review_goal)
    assert third.id != first.id
    assert third.outputs["fingerprint"] != first_fingerprint


@pytest.mark.asyncio
async def test_fingerprint_sensitive_to_assignment_identity_manager_and_snapshot_changes(
    db_session, review_goal, review_run
):
    """Fingerprint sensitivity is now observed via the side-effect-free
    current_fingerprint() rather than repeated auto-reruns (completed +
    stale inputs is a one-time suggestion, not an auto-rerun, per item 3)."""
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    process = AgentDefinitionReviewProcess()
    first_agent = await _persist_agent(db_session)
    task = await _assign(
        db_session, review_goal.project_id, review_run.id, first_agent, "implementation"
    )
    fingerprints = [await process.current_fingerprint(db_session, review_goal, review_run)]

    db_session.add(
        Task(
            project_id=review_goal.project_id,
            title="replacement identity",
            assigned_to=first_agent.id,
            metadata_=task.metadata_,
        )
    )
    await db_session.delete(task)
    await db_session.flush()
    fingerprints.append(await process.current_fingerprint(db_session, review_goal, review_run))

    manager = await _persist_agent(db_session, name="manager", role="manager")
    review_goal.authority_model = "agent_manager"
    review_goal.manager_agent_id = manager.id
    await db_session.flush()
    fingerprints.append(await process.current_fingerprint(db_session, review_goal, review_run))

    first_agent.description = "Definition changed after review."
    await db_session.flush()
    fingerprints.append(await process.current_fingerprint(db_session, review_goal, review_run))

    assert len(set(fingerprints)) == 4


@pytest.mark.asyncio
async def test_direct_gate_allows_same_agent_function_assignment_count_drift(
    db_session, review_goal, review_run
):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.conftest import complete_baseline_processes

    agent = await _persist_agent(db_session)
    first_task = await _assign(
        db_session, review_goal.project_id, review_run.id, agent, "implementation"
    )
    await complete_baseline_processes(db_session, review_goal, review_run)
    reviewed = await _current_process(db_session, review_goal)

    db_session.add(
        Task(
            project_id=review_goal.project_id,
            title="second covered assignment",
            assigned_to=agent.id,
            status="done",
            metadata_=first_task.metadata_,
        )
    )
    await db_session.flush()
    process = AgentDefinitionReviewProcess()

    assert await process.current_fingerprint(db_session, review_goal, review_run) != reviewed.outputs["fingerprint"]
    assert (
        await process.current_coverage_fingerprint(db_session, review_goal, review_run)
        == reviewed.outputs["coverage_fingerprint"]
    )
    await OrchestrationService()._ensure_baseline_processes_ready(
        db_session, review_goal.id
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("material_change", ["agent", "function"])
async def test_direct_gate_blocks_material_assignment_coverage_drift(
    db_session, review_goal, review_run, material_change
):
    from fastapi import HTTPException

    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.conftest import complete_baseline_processes

    agent = await _persist_agent(db_session)
    task = await _assign(
        db_session, review_goal.project_id, review_run.id, agent, "implementation"
    )
    await complete_baseline_processes(db_session, review_goal, review_run)
    if material_change == "agent":
        task.assigned_to = (await _persist_agent(db_session, name="replacement-agent")).id
    else:
        task.metadata_ = {
            "orchestration": {
                "run_id": str(review_run.id),
                "work_function": "review",
            }
        }
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationService()._ensure_baseline_processes_ready(
            db_session, review_goal.id
        )

    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_direct_gate_allows_plan_action_assignment_after_baseline_coverage(
    db_session, review_goal, review_run
):
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.conftest import complete_baseline_processes

    agent = await _persist_agent(db_session)
    baseline_task = await _assign(
        db_session, review_goal.project_id, review_run.id, agent, "implementation"
    )
    await complete_baseline_processes(db_session, review_goal, review_run)
    reviewed = await _current_process(db_session, review_goal)
    stored_coverage_fingerprint = reviewed.outputs["coverage_fingerprint"]

    # Scenario 1: Agent has both baseline and action-tagged assignments.
    # The action-tagged work function (planning) must not contaminate coverage.
    db_session.add(
        Task(
            project_id=review_goal.project_id,
            title="plan action task",
            assigned_to=agent.id,
            status="done",
            metadata_={
                "orchestration": {
                    "run_id": str(review_run.id),
                    "action_id": str(uuid.uuid4()),
                    "work_function": "planning",
                }
            },
        )
    )
    await db_session.flush()
    process = AgentDefinitionReviewProcess()

    # Coverage should not change: action-tagged work_function is sanitized out
    assert (
        await process.current_coverage_fingerprint(
            db_session, review_goal, review_run, covered_target_ids={str(agent.id)}
        )
        == stored_coverage_fingerprint
    )
    await OrchestrationService()._ensure_baseline_processes_ready(
        db_session, review_goal.id
    )

    # Scenario 2: Replace the baseline assignment to simulate task reassignment while
    # action-tagged task exists. Baseline removed so agent only has action-tagged.
    # The agent must still be represented in coverage (even with sanitized work_function="")
    # to prevent false readiness 409 when the agent otherwise "drops out of the payload".
    new_baseline_task = Task(
        project_id=review_goal.project_id,
        title="replaced baseline",
        assigned_to=agent.id,
        metadata_=baseline_task.metadata_,
    )
    db_session.add(new_baseline_task)
    await db_session.delete(baseline_task)
    await db_session.flush()

    # Even with task identity changed, coverage should remain stable because the agent
    # and its work_function haven't changed. Readiness must not raise.
    await OrchestrationService()._ensure_baseline_processes_ready(
        db_session, review_goal.id
    )


@pytest.mark.asyncio
async def test_tick_suggests_rerun_when_assignment_identity_changes_but_coverage_does_not(
    db_session, review_goal, review_run, monkeypatch
):
    """Completed + stale inputs (assignment identity changed) is a one-time
    suggestion via tick(), not an auto-rerun (item 3, orchestrator
    override): the completed row and its stored fingerprint stand."""
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
    from huddleroom.services.orchestration_manager_analyzer import (
        ManagerAssessment,
        ManagerSelectionAnalyzer,
    )
    from huddleroom.services.orchestration_service import OrchestrationService
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
    from tests.conftest import complete_baseline_processes

    async def review(_self, _payload, project=None, *, project_id=None):
        return ManagerAssessment("override", "human_as_manager", "test fixture")

    monkeypatch.setattr(ManagerSelectionAnalyzer, "review", review)

    agent = await _persist_agent(db_session)
    task = await _assign(
        db_session, review_goal.project_id, review_run.id, agent, "implementation"
    )
    await complete_baseline_processes(db_session, review_goal, review_run)
    first = await _current_process(db_session, review_goal)
    db_session.add(
        Task(
            project_id=review_goal.project_id,
            title="replacement identity",
            assigned_to=agent.id,
            metadata_=task.metadata_,
        )
    )
    await db_session.delete(task)
    await db_session.flush()

    await OrchestrationService().tick(db_session, review_run.id)
    second = await _current_process(db_session, review_goal)

    assert second.id == first.id
    assert second.outputs["fingerprint"] == first.outputs["fingerprint"]

    # The live-input fingerprint has actually changed (identity changed,
    # coverage did not) -- verified side-effect-free, confirming this is a
    # real staleness condition, not a no-op.
    live_fingerprint = await AgentDefinitionReviewProcess().current_fingerprint(
        db_session, review_goal, review_run
    )
    assert live_fingerprint != first.outputs["fingerprint"]

    warnings = await OrchestrationWarningService().list_warnings(db_session, review_goal.id, active_only=True)
    stale_warnings = [w for w in warnings if w.warning_type == "agent_definition_review_stale_inputs"]
    assert len(stale_warnings) == 1
    assert stale_warnings[0].source_process_run_id == first.id


@pytest.mark.asyncio
async def test_post_baseline_task_for_covered_function_resyncs_fingerprint_without_rerun(
    db_session, review_goal, review_run
):
    """Post-baseline, a new execution task for an already-reviewed agent with an
    already-covered work function moves the assignment fingerprint but not
    coverage: the completed review stands, no rerun row is created, and the
    stored fingerprint is silently re-stamped."""
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    agent = await _persist_agent(db_session)
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    await _advance(db_session, review_goal, review_run)
    first = await _current_process(db_session, review_goal)
    first_id = first.id
    first_fingerprint = first.outputs["fingerprint"]
    first_coverage = first.outputs["coverage_fingerprint"]

    review_run.phase = "authorized"
    await _assign(
        db_session,
        review_goal.project_id,
        review_run.id,
        agent,
        "implementation",
        title="execution task",
    )
    await db_session.flush()

    result = await _advance(db_session, review_goal, review_run)
    second = await _current_process(db_session, review_goal)

    assert result["status"] == "completed"
    assert second.id == first_id
    assert second.status == "completed"
    assert second.outputs["coverage_fingerprint"] == first_coverage
    assert second.outputs["fingerprint"] != first_fingerprint
    live_fingerprint = await AgentDefinitionReviewProcess().current_fingerprint(
        db_session, review_goal, review_run
    )
    assert second.outputs["fingerprint"] == live_fingerprint


@pytest.mark.asyncio
async def test_tick_backfills_legacy_completed_audit_missing_coverage_fingerprint(
    db_session, review_goal, review_run
):
    """A pre-coverage-fingerprint legacy row (field didn't exist yet) is
    silently backfilled in place -- same no-warning/no-rerun treatment as a
    PROCESS_VERSION bump, not a user-facing stale-inputs suggestion, since
    nothing about the reviewed inputs actually changed."""
    agent = await _persist_agent(db_session)
    await _assign(
        db_session, review_goal.project_id, review_run.id, agent, "implementation"
    )
    await _advance(db_session, review_goal, review_run)
    first = await _current_process(db_session, review_goal)
    first.outputs = {
        key: value
        for key, value in first.outputs.items()
        if key != "coverage_fingerprint"
    }
    await db_session.flush()

    await _advance(db_session, review_goal, review_run)
    second = await _current_process(db_session, review_goal)

    assert second.id == first.id
    assert second.outputs["coverage_fingerprint"]


@pytest.mark.asyncio
async def test_process_version_bump_alone_is_silently_absorbed(db_session, review_goal, review_run):
    """A pure PROCESS_VERSION bump (fingerprint *formula* changed, e.g.
    dropping provider/model from the hash) with no actual drift in the
    underlying agent/goal state is silently re-stamped -- no stale-inputs
    warning."""
    from huddleroom.services.orchestration_agent_definition_review import PROCESS_VERSION

    agent = await _persist_agent(db_session)
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    await _advance(db_session, review_goal, review_run)
    first = await _current_process(db_session, review_goal)
    first.process_version = PROCESS_VERSION - 1
    await db_session.flush()

    await _advance(db_session, review_goal, review_run)
    second = await _current_process(db_session, review_goal)

    assert second.id == first.id
    assert second.process_version == PROCESS_VERSION
    warnings = (
        await db_session.execute(
            select(OrchestrationWarning).where(OrchestrationWarning.active.is_(True))
        )
    ).scalars().all()
    assert warnings == []


@pytest.mark.asyncio
async def test_process_version_bump_does_not_absorb_real_drift(db_session, review_goal, review_run):
    """A PROCESS_VERSION bump co-occurring with a genuine input change (a
    covered agent's system_prompt changed) must NOT be silently absorbed --
    it falls through to the normal stale-inputs path (one suggestion,
    baseline phase)."""
    from huddleroom.services.orchestration_agent_definition_review import PROCESS_VERSION

    agent = await _persist_agent(db_session)
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")
    await _advance(db_session, review_goal, review_run)
    first = await _current_process(db_session, review_goal)
    first.process_version = PROCESS_VERSION - 1
    await db_session.flush()

    agent.system_prompt = (agent.system_prompt or "") + " Updated instructions."
    await db_session.flush()

    await _advance(db_session, review_goal, review_run)
    second = await _current_process(db_session, review_goal)

    assert second.id == first.id
    assert second.process_version == PROCESS_VERSION
    warnings = (
        await db_session.execute(
            select(OrchestrationWarning).where(OrchestrationWarning.active.is_(True))
        )
    ).scalars().all()
    assert len(warnings) == 1
    assert warnings[0].warning_type == "agent_definition_review_stale_inputs"
    assert warnings[0].source_process_run_id == first.id


@pytest.mark.asyncio
async def test_human_skipped_review_is_terminal_until_force_started(
    db_session, review_goal, review_run
):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    service = OrchestrationProcessService()
    skipped = await service.skip_process(
        db_session,
        review_goal.id,
        process_type="agent_definition_review",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="accept the risk",
        run_id=review_run.id,
    )
    agent = await _persist_agent(db_session)
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")

    summary = await _advance(db_session, review_goal, review_run)
    current = await service.get_current(db_session, review_goal.id, "agent_definition_review")
    warnings = list((await db_session.execute(select(OrchestrationWarning))).scalars())
    memory = (
        await db_session.execute(
            select(OrchestrationMemorySection).where(
                OrchestrationMemorySection.section_key == "agent_definition_review"
            )
        )
    ).scalar_one()

    assert summary["status"] == "skipped"
    assert current.id == skipped.id
    assert {warning.warning_type for warning in warnings} == {
        "agent_definition_review_skipped",
        "agent_definition_review_not_performed",
    }
    assert "Some agents may be misconfigured" in memory.body
    assert memory.created_by == "orchestrator:agent_definition_review"
    assert not (await db_session.execute(select(OrchestrationAgentReview))).scalars().all()

    forced = await service.start_process(
        db_session,
        review_goal.id,
        process_type="agent_definition_review",
        trigger_reason="human force-start",
        run_id=review_run.id,
    )
    assert forced.id != skipped.id
    assert skipped.superseded_by_id == forced.id

    assert (await _advance(db_session, review_goal, review_run))["status"] == "completed"
    assert all(
        not warning.active
        for warning in (await db_session.execute(select(OrchestrationWarning))).scalars()
    )
    assert len((await db_session.execute(select(OrchestrationAgentReview))).scalars().all()) == 1


@pytest.mark.asyncio
async def test_tick_runs_agent_review_after_goal_definition_and_manager_selection(
    db_session, test_project
):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="fix typo in README",
            success_criteria=[
                {"key": "fixed", "description": "typo gone", "evidence": "diff"}
            ],
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True

    result = await service.tick(db_session, run.id)

    assert result["baseline_process"]["status"] == "completed"
    assert result["manager_selection_process"]["status"] == "completed"
    assert result["agent_definition_review_process"]["status"] == "completed"
    assert result["team_hierarchy_process"]["status"] == "completed"
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "agent_definition_review"
    )
    assert current.status == "completed"


@pytest.mark.asyncio
async def test_tick_does_not_start_agent_review_before_manager_selection_terminal(
    db_session, review_goal, review_run, monkeypatch
):
    from huddleroom.services.orchestration_manager_analyzer import (
        ManagerAssessment,
        ManagerSelectionAnalyzer,
    )
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_service import OrchestrationService

    async def review(_self, payload, project=None, *, project_id=None):
        return ManagerAssessment(
            "override", "human_as_manager", "test fixture"
        )

    monkeypatch.setattr(ManagerSelectionAnalyzer, "review", review)

    process_service = OrchestrationProcessService()
    await process_service.skip_process(
        db_session,
        review_goal.id,
        process_type="goal_definition",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="test: skip goal definition",
        run_id=review_run.id,
    )

    result = await OrchestrationService().tick(db_session, review_run.id)

    assert result["manager_selection_process"]["status"] == "waiting_decision"
    assert result["agent_definition_review_process"] is None
    assert result["team_hierarchy_process"] is None
    assert await process_service.get_current(
        db_session, review_goal.id, "agent_definition_review"
    ) is None


@pytest.mark.asyncio
async def test_forward_progress_gate_requires_agent_definition_review(
    db_session, review_goal, review_run
):
    from fastapi import HTTPException

    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_service import OrchestrationService

    process_service = OrchestrationProcessService()
    for process_type in ("goal_definition", "manager_selection"):
        process = await process_service.start_process(
            db_session,
            review_goal.id,
            process_type=process_type,
            trigger_reason="test setup",
            run_id=review_run.id,
        )
        await process_service.complete_process(db_session, process)

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationService()._ensure_baseline_processes_ready(
            db_session, review_goal.id
        )
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "Goal definition, manager selection, agent definition review, and "
        "team hierarchy must complete before planning or delegation"
    )

    await _advance(db_session, review_goal, review_run)
    await process_service.skip_process(
        db_session,
        review_goal.id,
        process_type="team_hierarchy",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="test: accept hierarchy risk",
        run_id=review_run.id,
    )
    await OrchestrationService()._ensure_baseline_processes_ready(
        db_session, review_goal.id
    )


@pytest.mark.asyncio
async def test_direct_planning_and_delegation_block_stale_completed_review_without_writes(
    db_session, review_goal, review_run
):
    from fastapi import HTTPException

    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.conftest import complete_baseline_processes

    agent = await _persist_agent(db_session)
    await _assign(
        db_session, review_goal.project_id, review_run.id, agent, "implementation"
    )
    await complete_baseline_processes(db_session, review_goal, review_run)
    agent.description = "Definition changed after the terminal review."
    await db_session.flush()

    before = (
        len((await db_session.execute(select(OrchestrationProcessRun))).scalars().all()),
        len((await db_session.execute(select(OrchestrationAgentReview))).scalars().all()),
        len((await db_session.execute(select(OrchestrationWarning))).scalars().all()),
    )
    service = OrchestrationService()
    calls = (
        (
            service.execute_request_plan_action,
            {
                "agent_id": str(agent.id),
                "work_function": "planning",
                "scope": "Plan the reviewed change",
            },
            "stale-review-plan",
        ),
        (
            service.execute_create_delegation_task_action,
            {
                "agent_id": str(agent.id),
                "work_function": "implementation",
                "scope": "Implement the reviewed change",
                "deliverable": "Reviewed implementation",
            },
            "stale-review-delegation",
        ),
    )
    for action, request, idempotency_key in calls:
        with pytest.raises(HTTPException) as exc_info:
            await action(
                db_session,
                run_id=review_run.id,
                request=request,
                idempotency_key=idempotency_key,
            )
        assert exc_info.value.status_code == 409

    after = (
        len((await db_session.execute(select(OrchestrationProcessRun))).scalars().all()),
        len((await db_session.execute(select(OrchestrationAgentReview))).scalars().all()),
        len((await db_session.execute(select(OrchestrationWarning))).scalars().all()),
    )
    assert after == before


@pytest.mark.asyncio
async def test_direct_gate_uses_active_run_and_keeps_skipped_review_terminal(
    db_session, review_goal, review_run
):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    from huddleroom.services.orchestration_service import OrchestrationService

    process_service = OrchestrationProcessService()
    for process_type in ("goal_definition", "manager_selection"):
        process = await process_service.start_process(
            db_session,
            review_goal.id,
            process_type=process_type,
            trigger_reason="test setup",
            run_id=review_run.id,
        )
        await process_service.complete_process(db_session, process)
    skipped = await process_service.skip_process(
        db_session,
        review_goal.id,
        process_type="agent_definition_review",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="accepted review risk",
        run_id=review_run.id,
    )
    review_run.status = "completed"
    active_run = OrchestrationRun(goal_id=review_goal.id)
    db_session.add(active_run)
    await db_session.flush()
    agent = await _persist_agent(db_session)
    await _assign(
        db_session, review_goal.project_id, active_run.id, agent, "implementation"
    )
    await process_service.skip_process(
        db_session,
        review_goal.id,
        process_type="team_hierarchy",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="accepted hierarchy risk",
        run_id=active_run.id,
    )

    await OrchestrationService()._ensure_baseline_processes_ready(
        db_session, review_goal.id, allow_heal=False
    )

    current = await _current_process(db_session, review_goal)
    assert current.id == skipped.id
    assert current.status == "skipped"
    assert not (await db_session.execute(select(OrchestrationAgentReview))).scalars().all()


@pytest.mark.asyncio
async def test_direct_gate_fingerprints_active_run_not_review_process_run(
    db_session, review_goal, review_run
):
    from fastapi import HTTPException

    from huddleroom.services.orchestration_service import OrchestrationService
    from tests.conftest import complete_baseline_processes

    agent = await _persist_agent(db_session)
    await _assign(
        db_session, review_goal.project_id, review_run.id, agent, "implementation"
    )
    await complete_baseline_processes(db_session, review_goal, review_run)
    review_run.status = "completed"
    active_run = OrchestrationRun(goal_id=review_goal.id)
    db_session.add(active_run)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await OrchestrationService().execute_request_plan_action(
            db_session,
            run_id=active_run.id,
            request={
                "agent_id": str(agent.id),
                "work_function": "planning",
                "scope": "Plan the reviewed change",
            },
            idempotency_key="active-run-fingerprint",
        )

    assert exc_info.value.status_code == 409


def _agent_reviews_url(project_id, goal_id):
    return (
        f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}"
        "/agent-reviews"
    )


@pytest.mark.asyncio
async def test_agent_review_api_scopes_goal_to_project_and_returns_empty_history(
    client, auth_headers, db_session, test_project, review_goal
):
    from huddleroom.models.project import Project
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    other_goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Other scoped goal",
        success_criteria=[],
    )
    db_session.add(other_goal)
    await db_session.flush()
    other_agent = await _persist_agent(db_session, name="other-goal-agent")
    await OrchestrationAgentReviewService().create_review(
        db_session,
        other_goal.id,
        agent_id=other_agent.id,
        fit_summary="Review from another goal.",
    )

    empty = await client.get(
        _agent_reviews_url(test_project.id, review_goal.id), headers=auth_headers
    )
    assert empty.status_code == 200
    assert empty.json() == []

    other_project = Project(name="Other project", description="", config={})
    db_session.add(other_project)
    await db_session.flush()
    wrong_project = await client.get(
        _agent_reviews_url(other_project.id, review_goal.id), headers=auth_headers
    )
    assert wrong_project.status_code == 404


@pytest.mark.asyncio
async def test_agent_review_api_filters_by_agent_and_orders_created_at_then_id(
    client, auth_headers, db_session, test_project, review_goal
):
    first_agent = await _persist_agent(db_session, name="first-agent")
    second_agent = await _persist_agent(db_session, name="second-agent")
    timestamp = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    later_id = uuid.UUID("00000000-0000-0000-0000-000000000002")
    earlier_id = uuid.UUID("00000000-0000-0000-0000-000000000001")

    def review(review_id, agent, fit_summary):
        return OrchestrationAgentReview(
            id=review_id,
            goal_id=review_goal.id,
            agent_id=agent.id,
            review_context="agent definition review",
            proposed_work_functions=["implementation"],
            definition_snapshot={"name": agent.name},
            fit_summary=fit_summary,
            strengths=["scoped"],
            risks=[],
            recommended_changes=[],
            approved_for_work_functions=["implementation"],
            created_at=timestamp,
            updated_at=timestamp,
        )

    db_session.add_all(
        [
            review(later_id, first_agent, "later id"),
            review(earlier_id, second_agent, "earlier id"),
        ]
    )
    await db_session.flush()

    response = await client.get(
        _agent_reviews_url(test_project.id, review_goal.id), headers=auth_headers
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [str(earlier_id), str(later_id)]
    assert set(response.json()[0]) == {
        "id",
        "goal_id",
        "run_id",
        "agent_id",
        "source_process_run_id",
        "review_context",
        "proposed_work_functions",
        "definition_snapshot",
        "fit_summary",
        "strengths",
        "risks",
        "recommended_changes",
        "approved_for_work_functions",
        "created_at",
        "updated_at",
    }

    filtered = await client.get(
        _agent_reviews_url(test_project.id, review_goal.id),
        params={"agent_id": str(first_agent.id)},
        headers=auth_headers,
    )
    assert filtered.status_code == 200
    assert [row["id"] for row in filtered.json()] == [str(later_id)]


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_agent_review_api_requires_authentication(
    client, test_project, review_goal
):
    from huddleroom.config import Settings

    with patch("huddleroom.dependencies.settings", Settings(auth_enabled=True)):
        response = await client.get(
            _agent_reviews_url(test_project.id, review_goal.id)
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_force_start_agent_review_waits_for_terminal_manager_selection_and_reverses_skip(
    client, auth_headers, db_session, test_project, review_goal, review_run
):
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    process_service = OrchestrationProcessService()
    manager_selection = await process_service.start_process(
        db_session,
        review_goal.id,
        process_type="manager_selection",
        trigger_reason="manager resolution in progress",
        run_id=review_run.id,
    )
    url = (
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{review_goal.id}"
        "/processes/agent_definition_review/start"
    )

    blocked = await client.post(
        url, json={"reason": "review now"}, headers=auth_headers
    )
    assert blocked.status_code == 409
    assert "manager selection" in blocked.json()["detail"].lower()
    assert await _current_process(db_session, review_goal) is None

    await process_service.complete_process(db_session, manager_selection)
    skipped = await process_service.skip_process(
        db_session,
        review_goal.id,
        process_type="agent_definition_review",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="accepted review risk",
        run_id=review_run.id,
    )
    started = await client.post(
        url, json={"reason": "review after all"}, headers=auth_headers
    )
    assert started.status_code == 200
    assert started.json()["status"] == "running"
    assert started.json()["id"] != str(skipped.id)
    await db_session.refresh(skipped)
    assert skipped.superseded_by_id == uuid.UUID(started.json()["id"])


@pytest.mark.asyncio
async def test_blank_definition_agent_vague_warning_resolves_on_approve(
    db_session, review_goal, review_run, test_user
):
    """When a human approves a proposal for a blank-definition agent,
    the agent_review_vague_definition warning should be resolved."""
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess

    # Create an agent with blank description and system_prompt
    agent = await _persist_agent(db_session, description=None, system_prompt=None)
    await _assign(db_session, review_goal.project_id, review_run.id, agent, "implementation")

    # Run the review process - this creates the vague_definition warning
    process = AgentDefinitionReviewProcess(analyzer=SequencedSemanticAnalyzer())
    result = await process.advance(db_session, review_goal, review_run)

    assert result["status"] == "waiting_decision"

    # Get the proposal
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, review_goal.id, status="pending"
    ))[0]

    # Verify warning is active before approval
    active_warnings = list(
        (await db_session.execute(
            select(OrchestrationWarning).where(OrchestrationWarning.active.is_(True))
        )).scalars()
    )
    vague_warning = next(
        (w for w in active_warnings
         if w.warning_type == "agent_review_vague_definition" and w.related_agent_id == agent.id),
        None
    )
    assert vague_warning is not None

    # Approve the proposal
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session,
        decision,
        selected_option="approve",
        reason="Use the suggestion.",
        decided_by_user_id=test_user.id,
    )

    # Advance to apply the change
    result = await process.advance(db_session, review_goal, review_run)
    assert result["status"] == "completed"

    # Verify the warning is now resolved
    vague_warnings = list(
        (await db_session.execute(
            select(OrchestrationWarning).where(
                OrchestrationWarning.warning_type == "agent_review_vague_definition",
                OrchestrationWarning.related_agent_id == agent.id,
            )
        )).scalars()
    )
    assert len(vague_warnings) == 1
    assert vague_warnings[0].active is False
    assert vague_warnings[0].resolved_by == "orchestrator:agent_definition_review"
    assert vague_warnings[0].resolved_reason == "definition provided via approved proposal"


def test_agent_review_route_is_read_only_and_not_agent_facing():
    from huddleroom.main import create_app

    paths = create_app().openapi()["paths"]
    review_paths = [path for path in paths if "agent-reviews" in path]
    assert len(review_paths) == 1
    assert set(paths[review_paths[0]]) == {"get"}
    assert review_paths[0].startswith("/api/v1/projects/")
    assert not review_paths[0].startswith("/api/v1/agent")
