import asyncio
import json
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.schemas.agent import AgentCreate
from huddleroom.services.orchestration_agent_definition_analyzer import (
    AgentDefinitionSemanticAnalyzer,
    SemanticAgentAssessment,
)
from huddleroom.services.orchestration_agent_review_service import OrchestrationAgentReviewService
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_team_hierarchy import (
    TeamHierarchyProcess,
    infer_required_work_functions,
)
from huddleroom.services.orchestration_team_hierarchy_analyzer import (
    TeamHierarchyAnalysis,
    TeamHierarchyAnalyzer,
    canonical_agent_create,
    parse_team_hierarchy_analysis,
)
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService


TEAM_HIERARCHY_SKIP_MESSAGE = (
    "Team hierarchy review was skipped. Work may be delegated without clear "
    "ownership or escalation paths."
)


def test_canonical_agent_create_accepts_omitted_defaults_and_rejects_invalid_identity_fields():
    definition = _proposed_agent()["definition"]

    minimal_definition = {
        key: definition[key]
        for key in ("name", "role", "provider", "model")
    }
    assert canonical_agent_create(minimal_definition, exact_keys=True).model_dump(mode="json") == {
        **minimal_definition,
        "description": None,
        "system_prompt": None,
        "adapter_type": "api",
        "cli_runtime": None,
        "capabilities": [],
        "config": {},
    }
    for field in ("name", "role", "provider", "model"):
        with pytest.raises(ValueError, match="complete non-empty"):
            canonical_agent_create({**minimal_definition, field: "  "}, exact_keys=True)
    with pytest.raises(ValueError, match="complete non-empty"):
        canonical_agent_create({**minimal_definition, "system_prompt": "  "}, exact_keys=True)
    with pytest.raises(ValueError, match="complete AgentCreate"):
        canonical_agent_create({**minimal_definition, "extra": True}, exact_keys=True)

def test_hierarchy_parser_rejects_disconnected_graph_and_inexact_gaps():
    request = {
        "agents": [{"id": "worker"}, {"id": "lead"}],
        "required_work_functions": ["implementation", "review"],
    }
    base = {
        "proposed_agents": [],
        "assignments": [{"work_function": "implementation", "agent_ref": "worker"}],
        "reporting_lines": [{"agent_ref": "worker", "reports_to": "lead"}],
        "documented_gaps": ["review"],
        "rationale": "One gap remains.",
        "self_review": "Checked graph and gaps.",
    }
    with pytest.raises(ValueError, match="assigned agents"):
        parse_team_hierarchy_analysis(base, request)
    with pytest.raises(ValueError, match="exactly match"):
        parse_team_hierarchy_analysis({
            **base,
            "reporting_lines": [{"agent_ref": "worker", "reports_to": "manager"}],
            "documented_gaps": ["review", "review"],
        }, request)


def test_hierarchy_parser_normalizes_proposal_id_that_echoes_the_prefix():
    request = {
        "agents": [{"id": "worker"}],
        "required_work_functions": ["implementation", "review"],
    }
    proposal = _proposed_agent(proposal_id="proposal:reviewer", name="reviewer")
    payload = {
        "proposed_agents": [proposal],
        "assignments": [
            {"work_function": "implementation", "agent_ref": "worker"},
            {"work_function": "review", "agent_ref": "proposal:reviewer"},
        ],
        "reporting_lines": [
            {"agent_ref": "worker", "reports_to": "manager"},
            {"agent_ref": "proposal:reviewer", "reports_to": "manager"},
        ],
        "documented_gaps": [],
        "rationale": "Reviewer covers independent review.",
        "self_review": "Checked references and coverage.",
    }

    parsed = parse_team_hierarchy_analysis(payload, request)

    assert parsed.proposed_agents[0]["proposal_id"] == "reviewer"
    assert {a["work_function"]: a["agent_ref"] for a in parsed.assignments}["review"] == "proposal:reviewer"


def _analysis(*, reporting_lines, documented_gaps=("review",)):
    return {
        "proposed_agents": [],
        "assignments": [{"work_function": "implementation", "agent_ref": "worker"}],
        "reporting_lines": reporting_lines,
        "documented_gaps": list(documented_gaps),
        "rationale": "Covers the assigned work.",
        "self_review": "Checked graph and gaps.",
    }


@pytest.mark.parametrize(
    "reporting_lines",
    [
        [{"agent_ref": "worker", "reports_to": "worker"}],
        [
            {"agent_ref": "worker", "reports_to": "lead"},
            {"agent_ref": "lead", "reports_to": "worker"},
        ],
        [{"agent_ref": "worker", "reports_to": "lead"}],
    ],
)
def test_hierarchy_parser_rejects_cycles_and_unassigned_parents(reporting_lines):
    request = {
        "agents": [{"id": "worker"}, {"id": "lead"}],
        "required_work_functions": ["implementation", "review"],
    }
    payload = _analysis(reporting_lines=reporting_lines)
    if len(reporting_lines) == 2:
        payload["assignments"].append({"work_function": "review", "agent_ref": "lead"})
        payload["documented_gaps"] = []

    with pytest.raises(ValueError):
        parse_team_hierarchy_analysis(payload, request)


def test_hierarchy_parser_accepts_manager_rooted_tree():
    request = {
        "agents": [{"id": "worker"}, {"id": "lead"}],
        "required_work_functions": ["planning", "implementation"],
    }
    payload = _analysis(
        reporting_lines=[
            {"agent_ref": "worker", "reports_to": "lead"},
            {"agent_ref": "lead", "reports_to": "manager"},
        ],
        documented_gaps=(),
    )
    payload["assignments"].insert(0, {"work_function": "planning", "agent_ref": "lead"})

    parsed = parse_team_hierarchy_analysis(payload, request)

    assert parsed.reporting_lines == (
        {"agent_ref": "worker", "reports_to": "lead"},
        {"agent_ref": "lead", "reports_to": "manager"},
    )


def test_hierarchy_parser_rejects_unassigned_validator_reporting_line():
    manager_id = "11111111-1111-1111-1111-111111111111"
    engineer_id = "22222222-2222-2222-2222-222222222222"
    validator_id = "33333333-3333-3333-3333-333333333333"
    request = {
        "selected_manager": {"kind": "agent", "id": manager_id},
        "agents": [{"id": manager_id}, {"id": engineer_id}, {"id": validator_id}],
        "required_work_functions": ["planning", "implementation", "summarization"],
    }
    payload = _analysis(
        reporting_lines=[
            {"agent_ref": manager_id, "reports_to": manager_id},
            {"agent_ref": engineer_id, "reports_to": manager_id},
            {"agent_ref": validator_id, "reports_to": manager_id},
        ],
        documented_gaps=(),
    )
    payload["assignments"] = [
        {"work_function": "planning", "agent_ref": manager_id},
        {"work_function": "implementation", "agent_ref": engineer_id},
        {"work_function": "summarization", "agent_ref": manager_id},
    ]

    with pytest.raises(ValueError, match="reporting line agents must be assigned"):
        parse_team_hierarchy_analysis(payload, request)


def test_hierarchy_parser_rejects_reporting_lines_for_unknown_agents():
    request = {
        "agents": [{"id": "worker"}, {"id": "idle"}],
        "required_work_functions": ["implementation"],
    }

    with pytest.raises(ValueError, match="known agents"):
        parse_team_hierarchy_analysis(
            _analysis(reporting_lines=[
                {"agent_ref": "unknown", "reports_to": "manager"},
                {"agent_ref": "worker", "reports_to": "manager"},
            ], documented_gaps=()),
            request,
        )


def test_hierarchy_parser_normalizes_selected_manager_self_report_to_manager():
    request = {
        "selected_manager": {"kind": "agent", "id": "lead"},
        "agents": [{"id": "worker"}, {"id": "lead"}],
        "required_work_functions": ["planning", "implementation"],
    }
    payload = _analysis(
        reporting_lines=[
            {"agent_ref": "worker", "reports_to": "lead"},
            {"agent_ref": "lead", "reports_to": "lead"},
        ],
        documented_gaps=(),
    )
    payload["assignments"].insert(0, {"work_function": "planning", "agent_ref": "lead"})

    parsed = parse_team_hierarchy_analysis(payload, request)

    assert parsed.reporting_lines == (
        {"agent_ref": "worker", "reports_to": "lead"},
        {"agent_ref": "lead", "reports_to": "manager"},
    )


def test_hierarchy_parser_rejects_non_manager_self_report():
    request = {
        "selected_manager": {"kind": "agent", "id": "lead"},
        "agents": [{"id": "worker"}, {"id": "lead"}],
        "required_work_functions": ["planning", "implementation"],
    }
    payload = _analysis(
        reporting_lines=[
            {"agent_ref": "worker", "reports_to": "worker"},
            {"agent_ref": "lead", "reports_to": "manager"},
        ],
        documented_gaps=(),
    )
    payload["assignments"].insert(0, {"work_function": "planning", "agent_ref": "lead"})

    with pytest.raises(ValueError, match="reporting lines"):
        parse_team_hierarchy_analysis(payload, request)


def test_hierarchy_parser_normalizes_workers_reporting_to_unassigned_manager_uuid():
    # Regression #91: selected manager is an agent that is NOT assigned any
    # work function (its UUID is not in assignments). Workers whose
    # reports_to is that manager's UUID must be normalized to "manager".
    manager_id = "lead"
    request = {
        "selected_manager": {"kind": "agent", "id": manager_id},
        "agents": [{"id": "worker"}, {"id": "worker2"}, {"id": manager_id}],
        "required_work_functions": ["planning", "implementation"],
    }
    payload = {
        "proposed_agents": [],
        "assignments": [
            {"work_function": "planning", "agent_ref": "worker"},
            {"work_function": "implementation", "agent_ref": "worker2"},
        ],
        "reporting_lines": [
            {"agent_ref": "worker", "reports_to": manager_id},
            {"agent_ref": "worker2", "reports_to": manager_id},
        ],
        "documented_gaps": [],
        "rationale": "The manager is unassigned but oversees both workers.",
        "self_review": "Checked graph and gaps.",
    }

    parsed = parse_team_hierarchy_analysis(payload, request)

    assert parsed.reporting_lines == (
        {"agent_ref": "worker", "reports_to": "manager"},
        {"agent_ref": "worker2", "reports_to": "manager"},
    )


def test_hierarchy_request_prompt_allows_defaults_and_constrains_reporting_lines():
    prompt = TeamHierarchyAnalyzer.build_request({})["messages"][0]["content"]

    assert "defaulted AgentCreate fields may be omitted" in prompt
    assert 'the literal token "manager"' in prompt
    assert "never the selected manager's UUID" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("model,uses_json_mode", [
    ("openrouter/minimax/minimax-m3", False),
    ("openai/gpt-4o-mini", True),
])
async def test_hierarchy_request_keeps_checkpoint_but_adapts_minimax_call(model, uses_json_mode):
    request = TeamHierarchyAnalyzer.build_request({
        "agents": [], "required_work_functions": [], "goal": {},
    })
    request["model"] = model
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": (
            '```json\n{"proposed_agents":[],"assignments":[],"reporting_lines":[],'
            '"documented_gaps":[],"rationale":"ok","self_review":"ok"}\n```'
        )}}]}

    await TeamHierarchyAnalyzer(completion).review_request(request)

    assert request["response_format"] == {"type": "json_object"}
    assert ("response_format" in calls[0]) is uses_json_mode


@pytest.mark.asyncio
async def test_hierarchy_prompt_and_repair_name_the_agent_create_contract():
    request = TeamHierarchyAnalyzer.build_request({
        "agents": [], "required_work_functions": [], "goal": {},
    })
    prompt = request["messages"][0]["content"]
    required = [name for name, field in AgentCreate.model_fields.items() if field.is_required()]
    allowed = list(AgentCreate.model_fields)
    assert f"required keys: {', '.join(required)}" in prompt
    assert f"allowed keys: {', '.join(allowed)}" in prompt

    responses = iter((
        {
            "proposed_agents": [{
                "proposal_id": "writer",
                "definition": {"name": "writer", "role": "writer", "model": "m", "extra": True},
            }],
            "assignments": [], "reporting_lines": [], "documented_gaps": [],
            "rationale": "r", "self_review": "s",
        },
        {
            "proposed_agents": [{
                "proposal_id": "writer",
                "definition": {"name": "writer", "role": "writer", "provider": "p", "model": "m"},
            }],
            "assignments": [], "reporting_lines": [], "documented_gaps": [],
            "rationale": "r", "self_review": "s",
        },
    ))
    requests = []

    async def completion(**call_request):
        requests.append(call_request)
        return {"choices": [{"message": {"content": json.dumps(next(responses))}}]}

    result = await TeamHierarchyAnalyzer(completion).review_request(request)

    assert result.proposed_agents[0]["definition"]["provider"] == "p"
    repair = requests[1]["messages"][-1]["content"]
    assert "missing required keys: provider; extra keys: extra" in repair


@pytest.mark.parametrize(
    "gaps",
    [(), ("review", "validation"), ("review", "review"), ("Review",)],
)
def test_hierarchy_parser_rejects_missing_extra_duplicate_or_nonexact_gaps(gaps):
    request = {
        "agents": [{"id": "worker"}],
        "required_work_functions": ["implementation", "review"],
    }

    with pytest.raises(ValueError, match="exactly match"):
        parse_team_hierarchy_analysis(
            _analysis(
                reporting_lines=[{"agent_ref": "worker", "reports_to": "manager"}],
                documented_gaps=gaps,
            ),
            request,
        )


def test_team_hierarchy_retry_checkpoint_owns_frozen_request():
    from huddleroom.schemas.orchestration import valid_lm_retry_checkpoint

    request = TeamHierarchyAnalyzer.build_request({
        "schema_version": 1,
        "goal": {},
        "selected_manager": {"kind": "none", "id": None},
        "required_work_functions": [],
        "agents": [],
    })
    checkpoint = {"kind": "team_hierarchy", "version": 1, "request": request}

    assert valid_lm_retry_checkpoint(checkpoint, "team_hierarchy") is True
    checkpoint["request"]["messages"] = []
    assert valid_lm_retry_checkpoint(checkpoint, "team_hierarchy") is False


@pytest.mark.asyncio
async def test_llm_hierarchy_receives_full_roster_without_deterministic_draft(
    db_session, test_project, test_user
):
    goal, run, agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    extra = _agent(f"extra-{uuid.uuid4()}", "researcher", ["investigation"])
    db_session.add(extra)
    await db_session.flush()
    captured = {}

    class Analyzer:
        async def review(self, payload, project=None, *, project_id=None):
            captured.update(payload)
            return TeamHierarchyAnalysis(
                proposed_agents=({
                    "proposal_id": "security-reviewer",
                    "definition": {
                        "name": "security-reviewer",
                        "role": "security reviewer",
                        "description": "Reviews security-sensitive implementation work.",
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "system_prompt": "Independently review changes and report concrete security risks.",
                        "adapter_type": "api",
                        "cli_runtime": None,
                        "capabilities": ["review"],
                        "config": {},
                    },
                },),
                    assignments=(
                        {"work_function": "planning", "agent_ref": str(agents["planning"].id)},
                        {"work_function": "implementation", "agent_ref": str(agents["implementation"].id)},
                        {"work_function": "validation", "agent_ref": "proposal:security-reviewer"},
                        {"work_function": "summarization", "agent_ref": str(agents["summarization"].id)},
                    ),
                    reporting_lines=(
                        {"agent_ref": str(agents["planning"].id), "reports_to": "manager"},
                        {"agent_ref": str(agents["implementation"].id), "reports_to": "manager"},
                        {"agent_ref": "proposal:security-reviewer", "reports_to": "manager"},
                        {"agent_ref": str(agents["summarization"].id), "reports_to": "manager"},
                    ),
                    documented_gaps=(),
                rationale="The proposed reviewer preserves independent review.",
                self_review="Checked complete definitions, references, coverage, and producer/reviewer separation.",
            )

    summary = await TeamHierarchyProcess(analyzer=Analyzer()).advance(db_session, goal, run)

    assert summary == {"process_type": "team_hierarchy", "status": "waiting_decision", "questions_created": 1}
    assert set(captured) == {
        "schema_version", "goal", "selected_manager", "required_work_functions", "agents",
        "resolved_proposal_exclusions",
    }
    assert {row["id"] for row in captured["agents"]} >= {str(agent.id) for agent in [*agents.values(), extra]}
    assert all("workload" in row for row in captured["agents"])
    assert not ({"proposal", "hierarchy", "role_to_agent", "candidate_agents"} & set(captured))
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    assert [(item.decision_key, item.authority, [option["key"] for option in item.options]) for item in decisions] == [
        ("team_hierarchy:agent:security-reviewer", "human", ["approve", "edit", "reject"])
    ]
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    assert current.outputs["analysis"]["proposed_agents"][0]["definition"]["name"] == "security-reviewer"
    assert "raw_response" not in current.outputs
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decisions[0], selected_option="approve", decided_by_user_id=test_user.id
    )
    resumed = await TeamHierarchyProcess(analyzer=Analyzer()).advance(db_session, goal, run)
    assert resumed == {"process_type": "team_hierarchy", "status": "waiting_decision", "questions_created": 1}
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    assert [item.decision_key for item in decisions] == [
        "team_hierarchy:agent:security-reviewer", "team_hierarchy:approval"
    ]
    created_id = uuid.UUID(json.loads(decisions[0].reason)["created_agent_id"])
    assert await db_session.get(Agent, created_id) is not None
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decisions[1], selected_option="approve",
        decided_by_user_id=test_user.id
    )
    process = TeamHierarchyProcess(analyzer=Analyzer())
    assert (await process.advance(db_session, goal, run))["status"] == "completed"
    assert (await process.advance(db_session, goal, run))["status"] == "completed"
    assert len(list((await db_session.execute(
        select(Agent.id).where(Agent.name == "security-reviewer")
    )).scalars())) == 1


@pytest.mark.asyncio
async def test_invalid_llm_hierarchy_creates_no_decisions_or_agent_mutations(
    db_session, test_project, test_user
):
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    before = list((await db_session.execute(select(Agent.id))).scalars())

    async def invalid_completion(**_request):
        return {"choices": [{"message": {"content": json.dumps({
            "proposed_agents": [],
            "assignments": [{"work_function": "implementation", "agent_ref": "unknown"}],
            "reporting_lines": [],
            "documented_gaps": [],
            "rationale": "Invalid reference.",
            "self_review": "Checked references.",
        })}}]}

    summary = await TeamHierarchyProcess(
        analyzer=TeamHierarchyAnalyzer(invalid_completion)
    ).advance(db_session, goal, run)

    assert summary["retryable"] is True
    assert await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id) == []
    assert list((await db_session.execute(select(Agent.id))).scalars()) == before
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    assert "unknown" not in json.dumps(current.outputs)


@pytest.mark.asyncio
async def test_failed_hierarchy_retries_frozen_request_into_proposal_decision(
    db_session, test_project, test_user
):
    goal, run, agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    requests = []
    returned_payload = {}

    async def completion(**request):
        requests.append(request)
        if len(requests) == 1:
            raise RuntimeError("provider unavailable")
        assignments = [
            {"work_function": work_function, "agent_ref": str(agents[work_function].id)}
            for work_function in ("planning", "implementation", "validation", "summarization")
        ]
        returned_payload.update({
            "proposed_agents": [{
                "proposal_id": "optional-specialist",
                "definition": {
                    "name": "optional-specialist", "role": "specialist",
                    "description": "Handles specialized implementation support.",
                    "provider": "openai", "model": "gpt-4o-mini",
                    "system_prompt": "Support implementation within assigned boundaries and report results.",
                    "adapter_type": "api", "cli_runtime": None,
                    "capabilities": ["implementation"], "config": {},
                },
            }],
            "assignments": assignments,
            "reporting_lines": [
                {"agent_ref": row["agent_ref"], "reports_to": "manager"}
                for row in assignments
            ],
            "documented_gaps": [],
            "rationale": "The active roster covers all required work.",
            "self_review": "Checked definitions, references, coverage, reporting, and independence.",
        })
        return {"choices": [{"message": {"content": json.dumps(returned_payload)}}]}

    process = TeamHierarchyProcess(analyzer=TeamHierarchyAnalyzer(completion))
    failed = await process.advance(db_session, goal, run)
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    frozen_request = current.outputs["_lm_retry"]["request"]

    retried = await process.retry_failed(db_session, goal, run, current)

    assert failed["retryable"] is True
    expected_call = {**frozen_request, "stream": True}
    if frozen_request["model"] == "openrouter/minimax/minimax-m3":
        expected_call.pop("response_format")
        assert frozen_request["response_format"] == {"type": "json_object"}
    assert requests[1] == expected_call
    assert retried == {"process_type": "team_hierarchy", "status": "waiting_decision", "questions_created": 1}
    assert "_lm_retry" not in current.outputs
    assert current.outputs["analysis"] == parse_team_hierarchy_analysis(
        returned_payload,
        json.loads(frozen_request["messages"][1]["content"]),
    ).to_dict()
    assert current.outputs["fingerprint"] == process._semantic_fingerprint({
        "input": process._strip_volatile(json.loads(frozen_request["messages"][1]["content"])),
        "analysis": current.outputs["analysis"],
    })
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    assert [decision.decision_key for decision in decisions] == [
        "team_hierarchy:agent:optional-specialist"
    ]


@pytest.mark.asyncio
async def test_failed_hierarchy_retry_that_fails_again_reparks_without_raising(
    db_session, test_project, test_user
):
    from huddleroom.schemas.orchestration import valid_lm_retry_checkpoint

    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    requests = []

    async def always_fails(**request):
        requests.append(request)
        raise RuntimeError("provider unavailable")

    process = TeamHierarchyProcess(analyzer=TeamHierarchyAnalyzer(always_fails))
    failed = await process.advance(db_session, goal, run)
    assert failed["retryable"] is True
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    assert "_lm_retry" in current.outputs

    retried = await process.retry_failed(db_session, goal, run, current)

    assert len(requests) == 2
    assert retried["retryable"] is True
    assert "error" in retried
    checkpoint = current.outputs["_lm_retry"]
    assert valid_lm_retry_checkpoint(checkpoint, "team_hierarchy") is True


COMPLETED_OUTPUT_KEYS = {
    "fingerprint",
    "weight",
    "compressed",
    "required_work_functions",
    "role_to_agent",
    "candidate_agents",
    "hierarchy",
    "missing_work_functions",
    "weak_fits",
    "responsibility_conflicts",
    "suggestions",
    "independent_verification",
    "source_agent_review_process_id",
    "review_ids",
    "approval_decision_id",
    "approval",
    "warning_ids",
    "gates",
}


@pytest.fixture
def make_goal(test_project):
    def factory(*, objective: str, weight: str, success_criteria=None, constraints=None):
        return OrchestrationGoal(
            id=uuid.uuid4(),
            project_id=test_project.id,
            objective=objective,
            success_criteria=success_criteria or [],
            constraints=constraints or {},
            weight=weight,
        )

    return factory


@pytest.mark.parametrize(
    ("objective", "weight", "expected"),
    [
        (
            "Research and document the migration",
            "standard",
            ["planning", "investigation", "summarization"],
        ),
        (
            "Implement the API",
            "substantial",
            ["planning", "implementation", "validation", "summarization"],
        ),
        (
            "Verify the report independently",
            "standard",
            ["planning", "review", "validation", "summarization"],
        ),
    ],
)
def test_infer_required_work_functions_uses_existing_profiles(make_goal, objective, weight, expected):
    assert infer_required_work_functions(make_goal(objective=objective, weight=weight)) == expected


def test_infer_required_work_functions_compresses_trivial_and_falls_back_to_implementation(make_goal):
    assert infer_required_work_functions(make_goal(objective="anything", weight="trivial")) == []
    assert infer_required_work_functions(make_goal(objective="Coordinate", weight="standard")) == [
        "planning",
        "implementation",
        "summarization",
    ]


def _agent(name: str, role: str, capabilities: list[str]) -> Agent:
    return Agent(
        name=name,
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config={"reasoning_effort": "high"},
        is_active=True,
    )


async def _review_process(db_session, goal, run, review_ids):
    process = await OrchestrationProcessService().start_process(
        db_session,
        goal.id,
        process_type="agent_definition_review",
        trigger_reason="test review",
        run_id=run.id,
    )
    await OrchestrationProcessService().complete_process(
        db_session, process, outputs={"review_ids": [str(review_id) for review_id in review_ids]}
    )
    return process


async def _review(db_session, goal, run, agent, functions):
    return await OrchestrationAgentReviewService().create_review(
        db_session,
        goal.id,
        agent_id=agent.id,
        fit_summary="Reviewed for hierarchy proposal.",
        proposed_work_functions=functions,
        approved_for_work_functions=functions,
        run_id=run.id,
    )


@pytest.mark.asyncio
async def test_proposal_uses_current_reviewed_candidates_and_distinct_verifier(
    db_session, test_project
):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Implement the API with independently verified results",
        success_criteria=[],
        constraints={},
        weight="substantial",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    implementer = _agent("implementer", "developer", ["implementation"])
    validator = _agent("validator", "validator", ["validation", "review"])
    stale_agent = _agent("stale", "developer", ["implementation"])
    db_session.add_all([run, implementer, validator, stale_agent])
    await db_session.flush()

    implementer_review = await _review(db_session, goal, run, implementer, ["implementation"])
    validator_review = await _review(db_session, goal, run, validator, ["review", "validation"])
    stale_review = await _review(db_session, goal, run, stale_agent, ["implementation"])
    review_process = await _review_process(
        db_session, goal, run, [implementer_review.id, validator_review.id]
    )

    proposal = await TeamHierarchyProcess().build_proposal(db_session, goal, run)

    assert proposal["source_agent_review_process_id"] == str(review_process.id)
    assert proposal["role_to_agent"]["implementation"] == str(implementer.id)
    assert proposal["role_to_agent"]["validation"] == str(validator.id)
    assert proposal["role_to_agent"]["validation"] != proposal["role_to_agent"]["implementation"]
    assert str(stale_review.id) not in json.dumps(proposal)
    assert proposal["candidate_agents"]["implementation"] == [str(implementer.id)]
    assert all("name" not in row and "system_prompt" not in row for row in proposal["candidate_agents"].values())
    assert proposal["review_ids"] == sorted([str(implementer_review.id), str(validator_review.id)])
    assert proposal["weight"] == "substantial"
    assert proposal["compressed"] is False
    assert proposal["independent_verification"] == {
        "required": True,
        "possible": True,
        "verifier_agent_ids": [str(validator.id)],
        "overridden_by_decision_id": None,
    }
    assert proposal["weak_fits"] == []
    assert proposal["responsibility_conflicts"] == []
    assert proposal["hierarchy"] == {
        "manager": {"kind": "none", "id": None},
        "team_leads": [],
        "contributors": [str(implementer.id)],
        "reviewers": [str(validator.id)],
        "validators": [str(validator.id)],
        "specialists": [],
    }




@pytest.mark.asyncio
async def test_substantial_without_explicit_wording_rejects_manager_as_validator(
    db_session, test_project, test_user
):
    goal, run, _agents, review_process = await _lifecycle_goal(
        db_session,
        test_project,
        test_user,
        weight="substantial",
        objective="Implement the API",
        manager="agent",
        include_verifiers=False,
    )
    manager = await db_session.get(Agent, goal.manager_agent_id)
    manager.role = "manager validator"
    manager.capabilities = ["management", "validation"]
    manager_review = await _review(
        db_session, goal, run, manager, ["validation"]
    )
    review_process.outputs = {
        "review_ids": [
            *review_process.outputs["review_ids"],
            str(manager_review.id),
        ]
    }
    await db_session.flush()

    proposal = await TeamHierarchyProcess().build_proposal(db_session, goal, run)

    assert str(manager.id) in proposal["candidate_agents"]["validation"]
    assert "validation" not in proposal["role_to_agent"]
    assert proposal["independent_verification"]["possible"] is False




@pytest.mark.asyncio
async def test_standard_prefers_safe_validator_when_independence_is_not_required(
    db_session, test_project, test_user
):
    goal, run, agents, review_process = await _lifecycle_goal(
        db_session,
        test_project,
        test_user,
        objective="Implement and validate the API",
        include_verifiers=False,
    )
    producer = agents["implementation"]
    producer.role = "developer validator"
    producer.capabilities = ["implementation", "validation"]
    producer_review = next(
        review
        for review in await OrchestrationAgentReviewService().list_reviews(
            db_session, goal.id
        )
        if review.agent_id == producer.id
    )
    producer_review.approved_for_work_functions = ["implementation", "validation"]
    validator = _agent(
        f"zzz-validator-{uuid.uuid4()}", "validator", ["validation"]
    )
    db_session.add(validator)
    await db_session.flush()
    validator_review = await _review(
        db_session, goal, run, validator, ["validation"]
    )
    review_process.outputs = {
        "review_ids": [
            *review_process.outputs["review_ids"],
            str(validator_review.id),
        ]
    }
    await db_session.flush()

    proposal = await TeamHierarchyProcess().build_proposal(db_session, goal, run)

    assert proposal["candidate_agents"]["validation"] == [
        str(producer.id),
        str(validator.id),
    ]
    assert proposal["role_to_agent"]["implementation"] == str(producer.id)
    assert proposal["role_to_agent"]["validation"] == str(validator.id)
    assert proposal["independent_verification"]["required"] is False


@pytest.mark.asyncio
async def test_proposal_keeps_weak_and_unreviewed_agents_as_suggestions_only(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Implement the API",
        success_criteria=[],
        constraints={},
        weight="standard",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    weak = _agent("weak", "generalist", [])
    unreviewed = _agent("unreviewed", "developer", ["implementation"])
    db_session.add_all([run, weak, unreviewed])
    await db_session.flush()

    weak_review = await _review(db_session, goal, run, weak, ["implementation"])
    await _review_process(db_session, goal, run, [weak_review.id])

    proposal = await TeamHierarchyProcess().build_proposal(db_session, goal, run)

    assert "implementation" not in proposal["role_to_agent"]
    assert proposal["candidate_agents"]["implementation"] == []
    assert proposal["missing_work_functions"] == ["planning", "implementation", "summarization"]
    assert proposal["weak_fits"] == [
        {"work_function": "implementation", "agent_id": str(weak.id)}
    ]
    assert [
        suggestion
        for suggestion in proposal["suggestions"]
        if suggestion["work_function"] == "implementation"
    ] == [
        {"work_function": "implementation", "agent_id": str(unreviewed.id)},
        {"work_function": "implementation", "agent_id": str(weak.id)},
    ]
    assert proposal["independent_verification"] == {
        "required": False,
        "possible": True,
        "verifier_agent_ids": [],
        "overridden_by_decision_id": None,
    }
    assert set(proposal["hierarchy"]) == {
        "manager", "team_leads", "contributors", "reviewers", "validators", "specialists"
    }


@pytest.mark.parametrize(
    ("authority_model", "agent_id", "user_id", "expected"),
    [
        ("agent_manager", uuid.uuid4(), None, "agent"),
        ("human_manager", None, uuid.uuid4(), "human"),
        ("no_manager", None, None, "none"),
    ],
)
def test_hierarchy_manager_encodes_agent_human_and_none(
    make_goal, authority_model, agent_id, user_id, expected
):
    goal = make_goal(objective="Implement", weight="standard")
    goal.authority_model = authority_model
    goal.manager_agent_id = agent_id
    goal.manager_user_id = user_id

    hierarchy = TeamHierarchyProcess()._hierarchy(goal, {})  # pylint: disable=protected-access

    assert hierarchy["manager"] == {
        "kind": expected,
        "id": str(agent_id or user_id) if agent_id or user_id else None,
    }


@pytest.mark.asyncio
async def test_proposal_orders_displayed_candidates_deterministically(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Implement the API",
        success_criteria=[],
        constraints={},
        weight="standard",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    alpha = _agent("alpha", "developer", ["implementation"])
    beta = _agent("beta", "developer", ["implementation"])
    db_session.add_all([run, alpha, beta])
    await db_session.flush()
    alpha_review = await _review(db_session, goal, run, alpha, ["implementation"])
    beta_review = await _review(db_session, goal, run, beta, ["implementation"])
    await _review_process(db_session, goal, run, [beta_review.id, alpha_review.id])

    proposal = await TeamHierarchyProcess().build_proposal(db_session, goal, run)

    assert proposal["candidate_agents"]["implementation"] == [str(alpha.id), str(beta.id)]


@pytest.mark.asyncio
async def test_independent_verifier_beats_higher_ranked_eligible_producer(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Implement the API and independently verify it",
        success_criteria=[],
        constraints={},
        weight="standard",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    producer = _agent("producer", "validator", ["implementation", "validation"])
    producer.config["memory_enabled"] = True
    verifier = _agent("verifier", "qa", ["validation"])
    db_session.add_all([run, producer, verifier])
    await db_session.flush()
    producer_review = await _review(db_session, goal, run, producer, ["implementation", "validation"])
    verifier_review = await _review(db_session, goal, run, verifier, ["review", "validation"])
    await _review_process(db_session, goal, run, [producer_review.id, verifier_review.id])

    proposal = await TeamHierarchyProcess().build_proposal(db_session, goal, run)

    assert proposal["candidate_agents"]["validation"] == [str(producer.id), str(verifier.id)]
    assert proposal["role_to_agent"]["implementation"] == str(producer.id)
    assert proposal["role_to_agent"]["validation"] == str(verifier.id)


@pytest.mark.asyncio
async def test_independent_verification_is_not_possible_without_both_verifier_roles(
    db_session, test_project
):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Implement the API and independently verify it",
        success_criteria=[],
        constraints={},
        weight="standard",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    implementer = _agent("implementer", "developer", ["implementation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([run, implementer, reviewer])
    await db_session.flush()
    implementer_review = await _review(db_session, goal, run, implementer, ["implementation"])
    reviewer_review = await _review(db_session, goal, run, reviewer, ["review"])
    await _review_process(db_session, goal, run, [implementer_review.id, reviewer_review.id])

    proposal = await TeamHierarchyProcess().build_proposal(db_session, goal, run)

    assert proposal["independent_verification"] == {
        "required": True,
        "possible": False,
        "verifier_agent_ids": [str(reviewer.id)],
        "overridden_by_decision_id": None,
    }


@pytest.mark.asyncio
async def test_non_independent_verifier_function_still_uses_unsafe_fallback(
    db_session, test_project
):
    """Regression: only "validation" requires independence here (weight
    substantial, no explicit-independence wording), so "review" must still
    fall back to its only (producer-overlapping) candidate instead of being
    wrongly forced into safe-candidates-only mode just because *some*
    function in the goal requires independence."""
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Implement and review the API",
        success_criteria=[],
        constraints={},
        weight="substantial",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    worker = _agent(
        "worker", "generalist",
        ["planning", "implementation", "review", "validation", "summarization"],
    )
    db_session.add_all([run, worker])
    await db_session.flush()
    worker_review = await _review(
        db_session, goal, run, worker,
        ["planning", "implementation", "review", "validation", "summarization"],
    )
    await _review_process(db_session, goal, run, [worker_review.id])

    proposal = await TeamHierarchyProcess().build_proposal(db_session, goal, run)

    # worker is the sole implementation candidate, so it becomes the producer.
    assert proposal["role_to_agent"]["implementation"] == str(worker.id)
    # review does not require independence: worker (a producer) is still
    # assigned via the unsafe fallback rather than left missing.
    assert proposal["role_to_agent"]["review"] == str(worker.id)
    # validation does require independence and has no safe candidate: it is
    # correctly left unassigned rather than also using worker.
    assert "validation" not in proposal["role_to_agent"]
    assert proposal["missing_work_functions"] == ["validation"]


@pytest.mark.asyncio
async def test_live_load_changes_display_order_not_canonical_proposal_data(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Implement the API",
        success_criteria=[],
        constraints={},
        weight="standard",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    preferred = _agent("preferred", "developer", ["implementation"])
    preferred.config["memory_enabled"] = True
    alternate = _agent("alternate", "developer", ["implementation"])
    db_session.add_all([run, preferred, alternate])
    await db_session.flush()
    preferred_review = await _review(db_session, goal, run, preferred, ["implementation"])
    alternate_review = await _review(db_session, goal, run, alternate, ["implementation"])
    await _review_process(db_session, goal, run, [preferred_review.id, alternate_review.id])

    before = await TeamHierarchyProcess().build_proposal(db_session, goal, run)
    db_session.add_all(
        [
            Task(
                project_id=test_project.id,
                title=f"load {index}",
                status="in_progress",
                assigned_to=preferred.id,
            )
            for index in range(10)
        ]
    )
    await db_session.flush()
    after = await TeamHierarchyProcess().build_proposal(db_session, goal, run)

    assert before["candidate_agents"]["implementation"] == [str(preferred.id), str(alternate.id)]
    assert after["candidate_agents"]["implementation"] == [str(alternate.id), str(preferred.id)]
    assert before["role_to_agent"] == after["role_to_agent"] == {"implementation": str(preferred.id)}
    for key in (
        "review_ids",
        "weight",
        "compressed",
        "hierarchy",
        "weak_fits",
        "responsibility_conflicts",
        "independent_verification",
        "missing_work_functions",
    ):
        assert before[key] == after[key]


@pytest.mark.asyncio
async def test_proposal_uses_only_the_current_agent_definition_review_process_reviews(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Implement the API",
        success_criteria=[],
        constraints={},
        weight="standard",
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    prior_agent = _agent("prior", "developer", ["implementation"])
    current_agent = _agent("current", "developer", ["implementation"])
    db_session.add_all([run, prior_agent, current_agent])
    await db_session.flush()
    prior_review = await _review(db_session, goal, run, prior_agent, ["implementation"])
    await _review_process(db_session, goal, run, [prior_review.id])
    current_review = await _review(db_session, goal, run, current_agent, ["implementation"])
    current_process = await _review_process(db_session, goal, run, [current_review.id])

    proposal = await TeamHierarchyProcess().build_proposal(db_session, goal, run)

    assert proposal["source_agent_review_process_id"] == str(current_process.id)
    assert proposal["review_ids"] == [str(current_review.id)]
    assert proposal["candidate_agents"]["implementation"] == [str(current_agent.id)]
    assert proposal["role_to_agent"]["implementation"] == str(current_agent.id)


async def _lifecycle_goal(
    db_session,
    test_project,
    test_user,
    *,
    weight="standard",
    objective="Implement the API",
    manager="human",
    include_verifiers=True,
):
    suffix = uuid.uuid4().hex[:8]
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective=objective,
        success_criteria=[],
        constraints={},
        weight=weight,
        created_by_user_id=test_user.id,
        authority_model=f"{manager}_manager" if manager != "none" else "no_manager",
        manager_user_id=test_user.id if manager == "human" else None,
    )
    manager_agent = _agent(f"manager-{suffix}", "manager", ["management"])
    if manager == "agent":
        manager_agent.id = uuid.uuid4()
        goal.manager_agent_id = manager_agent.id
        db_session.add(manager_agent)
        await db_session.flush()
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    agents = {
        "planning": _agent(f"planner-{suffix}", "planner", ["planning"]),
        "implementation": _agent(
            f"implementer-{suffix}", "developer", ["implementation"]
        ),
        "summarization": _agent(
            f"summarizer-{suffix}", "writer", ["summarization"]
        ),
    }
    if include_verifiers:
        agents.update(
            {
                "review": _agent(f"reviewer-{suffix}", "reviewer", ["review"]),
                "validation": _agent(
                    f"validator-{suffix}", "validator", ["validation"]
                ),
            }
        )
    db_session.add_all([run, *agents.values()])
    await db_session.flush()
    reviews = [
        await _review(db_session, goal, run, agent, [work_function])
        for work_function, agent in agents.items()
    ]
    review_process = await _review_process(
        db_session, goal, run, [review.id for review in reviews]
    )
    return goal, run, agents, review_process


class FrozenHierarchyAnalyzer:
    def __init__(self):
        self.calls = 0

    async def review(self, payload, project=None, *, project_id=None):
        self.calls += 1
        agents = {capability: agent["id"] for agent in payload["agents"] for capability in agent["capabilities"]}
        assignments = tuple(
            {"work_function": work, "agent_ref": agents[work]}
            for work in payload["required_work_functions"]
        )
        refs = dict.fromkeys(item["agent_ref"] for item in assignments)
        return TeamHierarchyAnalysis(
            proposed_agents=({
                "proposal_id": "optional-specialist",
                "definition": {
                    "name": "optional-specialist", "role": "specialist",
                    "description": "Supports specialized implementation work.",
                    "provider": "openai", "model": "gpt-4o-mini",
                    "system_prompt": "Handle assigned specialist work and report concrete results.",
                    "adapter_type": "api", "cli_runtime": None,
                    "capabilities": ["implementation"], "config": {},
                },
            },),
            assignments=assignments,
            reporting_lines=tuple({"agent_ref": ref, "reports_to": "manager"} for ref in refs),
            documented_gaps=(), rationale="The active roster covers the work.",
            self_review="Checked definitions, references, coverage, reporting, and independence.",
        )


def _proposed_agent(proposal_id="optional-specialist", name=None):
    return {
        "proposal_id": proposal_id,
        "definition": {
            "name": name or proposal_id, "role": "specialist",
            "description": "Supports specialized implementation work.",
            "provider": "openai", "model": "gpt-4o-mini",
            "system_prompt": "Handle assigned specialist work and report concrete results.",
            "adapter_type": "api", "cli_runtime": None,
            "capabilities": ["implementation"], "config": {},
        },
    }


def _analysis_for(payload, *, proposals=(), documented_gaps=()):
    gaps = set(documented_gaps)
    agents = {
        capability: agent["id"]
        for agent in payload["agents"]
        for capability in agent["capabilities"]
    }
    assignments = tuple(
        {"work_function": work, "agent_ref": agents[work]}
        for work in payload["required_work_functions"]
        if work not in gaps
    )
    refs = dict.fromkeys(item["agent_ref"] for item in assignments)
    return TeamHierarchyAnalysis(
        proposed_agents=tuple(proposals), assignments=assignments,
        reporting_lines=tuple({"agent_ref": ref, "reports_to": "manager"} for ref in refs),
        documented_gaps=tuple(documented_gaps), rationale="Covers the available work.",
        self_review="Checked definitions, references, coverage, reporting, and independence.",
    )


def _answer_url(project_id, goal_id, decision_id):
    return (
        f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}"
        f"/decisions/{decision_id}/answer"
    )


async def _persisted_rows(db_session, model, order_by):
    rows = (await db_session.execute(select(*model.__table__.columns).order_by(order_by))).mappings()
    return json.loads(json.dumps([dict(row) for row in rows], default=str))


@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "created"), [("edit", True), ("reject", False)])
async def test_proposal_resolution_reruns_via_answer_endpoint_without_reproposing_original(
    client, db_session, test_project, test_user, monkeypatch, answer, created
):
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    original = _proposed_agent()
    rerun_payloads = []

    async def review(_self, payload, project=None, *, project_id=None):
        if not rerun_payloads:
            rerun_payloads.append(payload)
            return _analysis_for(payload, proposals=(original,))
        rerun_payloads.append(payload)
        return parse_team_hierarchy_analysis(
            _analysis_for(payload, proposals=(original,)).to_dict(), payload
        )

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    await TeamHierarchyProcess().advance(db_session, goal, run)
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id))[0]
    body = {"selected_option": answer}
    edited_name = f"edited-{uuid.uuid4()}"
    if answer == "edit":
        body["edited_agent"] = {**original["definition"], "name": edited_name}

    response = await client.post(_answer_url(test_project.id, goal.id, decision.id), json=body)

    assert response.status_code == 200, response.text
    await db_session.refresh(decision)
    metadata = json.loads(decision.reason) if decision.reason else {}
    assert ("created_agent_id" in metadata) is created
    created_agents = list((await db_session.execute(
        select(Agent).where(Agent.name == edited_name)
    )).scalars())
    assert len(created_agents) == int(created)
    assert all(agent["name"] != original["definition"]["name"] for agent in rerun_payloads[-1]["agents"])
    if created:
        assert any(agent["id"] == metadata["created_agent_id"] for agent in rerun_payloads[-1]["agents"])
    assert rerun_payloads[-1]["resolved_proposal_exclusions"][0]["action"] == answer
    assert not [item for item in await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, goal.id, status="pending"
    )]


@pytest.mark.asyncio
async def test_hierarchy_change_request_reruns_then_fresh_approval_completes_same_process(
    client, db_session, test_project, test_user, monkeypatch
):
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )

    async def review(_self, payload, project=None, *, project_id=None):
        return _analysis_for(payload)

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    service = OrchestrationProcessService()
    await TeamHierarchyProcess().advance(db_session, goal, run)
    original = await service.get_current(db_session, goal.id, "team_hierarchy")
    original_approval = next(
        decision
        for decision in await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
        if decision.decision_key == "team_hierarchy:approval"
    )

    response = await client.post(
        _answer_url(test_project.id, goal.id, original_approval.id),
        json={"selected_option": "request_changes"},
    )

    assert response.status_code == 200, response.text
    fresh = await service.get_current(db_session, goal.id, "team_hierarchy")
    assert fresh.id != original.id
    assert original.superseded_by_id == fresh.id
    fresh_approval = next(
        decision
        for decision in await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
        if decision.decision_key == "team_hierarchy:approval"
        and decision.source_process_run_id == fresh.id
    )
    assert fresh_approval.status == "pending"

    response = await client.post(
        _answer_url(test_project.id, goal.id, fresh_approval.id),
        json={"selected_option": "approve"},
    )

    assert response.status_code == 200, response.text
    current = await service.get_current(db_session, goal.id, "team_hierarchy")
    assert current.id == fresh.id
    assert current.status == "completed"
    assert current.superseded_by_id is None
    warnings = await OrchestrationWarningService().list_warnings(db_session, goal.id, active_only=True)
    assert not [warning for warning in warnings if warning.warning_type == "team_hierarchy_stale_inputs"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("auth_enabled", "expected"), [(False, True), (True, False)])
async def test_hierarchy_anon_human_principal_depends_on_auth_setting(
    db_session, monkeypatch, auth_enabled, expected
):
    from huddleroom.config import settings

    monkeypatch.setattr(settings, "auth_enabled", auth_enabled)
    decision = OrchestrationAuthorityDecision(
        authority="human", decided_by_user_id=uuid.UUID(int=0)
    )

    assert await TeamHierarchyProcess()._principal_is_current(db_session, decision) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", [False, True], ids=["inactive", "deleted"])
async def test_hierarchy_invalid_human_approval_creates_fresh_pending_decision(
    db_session, test_project, test_user, removed
):
    goal, run, _agents, _review = await _lifecycle_goal(db_session, test_project, test_user)

    class Analyzer:
        async def review(self, payload, project=None, *, project_id=None):
            return _analysis_for(payload)

    process = TeamHierarchyProcess(analyzer=Analyzer())
    await process.advance(db_session, goal, run)
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    approval = next(item for item in decisions if item.decision_key == "team_hierarchy:approval")
    actor = User(email=f"former-{uuid.uuid4()}@example.com", hashed_password="", role="member")
    db_session.add(actor)
    await db_session.flush()
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, approval, selected_option="approve", decided_by_user_id=actor.id
    )
    if removed:
        await db_session.delete(actor)
    else:
        actor.is_active = False
    await db_session.flush()

    result = await process.advance(db_session, goal, run)

    assert result["status"] == "waiting_decision"
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    fresh = next(
        item for item in decisions
        if item.decision_key == "team_hierarchy:approval" and item.status == "pending"
    )
    assert fresh.id != approval.id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [("role", "  "), ("system_prompt", None), ("system_prompt", "  ")],
)
async def test_hierarchy_edit_rejects_blank_required_definition_without_mutation(
    client, db_session, test_project, test_user, monkeypatch, field, value
):
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    proposal = _proposed_agent()

    async def review(_self, payload, project=None, *, project_id=None):
        return _analysis_for(payload, proposals=(proposal,))

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    await TeamHierarchyProcess().advance(db_session, goal, run)
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id))[0]
    process_service = OrchestrationProcessService()
    current = await process_service.get_current(db_session, goal.id, "team_hierarchy")
    edited_agent = dict(proposal["definition"])
    if value is None:
        edited_agent.pop(field)
    else:
        edited_agent[field] = value
    before = {
        "agents": await _persisted_rows(db_session, Agent, Agent.id),
        "events": await _persisted_rows(db_session, EventLog, EventLog.seq),
        "decision": (
            decision.status,
            decision.selected_option,
            decision.reason,
            decision.decided_by_user_id,
            decision.decided_by_agent_id,
            decision.consequences,
            decision.overrides_recommendation,
            decision.created_warning_id,
            decision.decided_at,
        ),
        "process": (current.id, current.status, current.superseded_by_id),
        "outputs": json.loads(json.dumps(current.outputs)),
        "run": (
            run.status,
            run.event_cursor,
            json.loads(json.dumps(run.plan_state)),
            json.loads(json.dumps(run.active_blockers)),
            json.loads(json.dumps(run.budget_state)),
            json.loads(json.dumps(run.retry_state)),
        ),
    }

    response = await client.post(
        _answer_url(test_project.id, goal.id, decision.id),
        json={"selected_option": "edit", "edited_agent": edited_agent},
    )

    assert response.status_code == 400
    assert "complete non-empty" in response.json()["detail"]
    await db_session.refresh(decision)
    await db_session.refresh(run)
    current_after = await process_service.get_current(db_session, goal.id, "team_hierarchy")
    await db_session.refresh(current_after)
    assert await _persisted_rows(db_session, Agent, Agent.id) == before["agents"]
    assert await _persisted_rows(db_session, EventLog, EventLog.seq) == before["events"]
    assert (
        decision.status,
        decision.selected_option,
        decision.reason,
        decision.decided_by_user_id,
        decision.decided_by_agent_id,
        decision.consequences,
        decision.overrides_recommendation,
        decision.created_warning_id,
        decision.decided_at,
    ) == before["decision"]
    assert decision.status == "pending"
    assert (current_after.id, current_after.status, current_after.superseded_by_id) == before["process"]
    assert current_after.outputs == before["outputs"]
    assert (
        run.status,
        run.event_cursor,
        run.plan_state,
        run.active_blockers,
        run.budget_state,
        run.retry_state,
    ) == before["run"]


@pytest.mark.asyncio
async def test_orchestrated_hierarchy_approval_is_always_human(
    db_session, test_project, test_user, monkeypatch
):
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial", manager="agent"
    )

    async def review(_self, payload, project=None, *, project_id=None):
        return _analysis_for(payload)

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    await TeamHierarchyProcess().advance(db_session, goal, run)
    approval = next(
        item
        for item in await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
        if item.decision_key == "team_hierarchy:approval"
    )

    assert approval.authority == "human"
    assert approval.authority_agent_id is None
    assert [option["key"] for option in approval.options] == ["approve", "request_changes"]


@pytest.mark.asyncio
async def test_stale_proposal_answer_leaves_decisions_pending_without_agent_mutation(
    client, db_session, test_project, test_user, monkeypatch
):
    """Item 3, orchestrator override: a stale proposal answer attempt is
    rejected (409), but the pending decision(s) are never cancelled -- they
    stay parked for the human to act on deliberately (was: Bug #93's
    cancel-and-reconcile)."""
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    proposals = (_proposed_agent("first"), _proposed_agent("second"))

    async def review(_self, payload, project=None, *, project_id=None):
        return _analysis_for(payload, proposals=proposals)

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    await TeamHierarchyProcess().advance(db_session, goal, run)
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    before = set((await db_session.execute(select(Agent.id))).scalars())
    goal.objective = "Implement a changed API"
    await db_session.flush()

    response = await client.post(
        _answer_url(test_project.id, goal.id, decisions[0].id),
        json={"selected_option": "approve"},
    )

    assert response.status_code == 409
    assert {decision.status for decision in decisions} == {"pending"}
    assert before
    assert all("created_agent_id" not in (decision.reason or "") for decision in decisions)


@pytest.mark.asyncio
async def test_stale_proposal_answer_suggests_rerun_without_superseding_run(
    client, db_session, test_project, test_user, monkeypatch, test_engine
):
    """Item 3, orchestrator override: a stale fingerprint detected while
    answering a proposal decision raises a one-time suggestion instead of
    Bug #93's cancel-and-reconcile -- no new process run, old run untouched."""
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    goal_id = goal.id
    proposal = _proposed_agent()

    async def review(_self, payload, project=None, *, project_id=None):
        return _analysis_for(payload, proposals=(proposal,))

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    await TeamHierarchyProcess().advance(db_session, goal, run)
    decisions_before = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal_id)
    stale_decision = decisions_before[0]
    service = OrchestrationProcessService()
    old_run = await service.get_current(db_session, goal_id, "team_hierarchy")
    old_run_id = old_run.id

    # Drift the fingerprint by changing the objective
    goal.objective = "Implement a completely different API"
    await db_session.flush()

    # Answer the now-stale decision
    response = await client.post(
        _answer_url(test_project.id, goal_id, stale_decision.id),
        json={"selected_option": "approve"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "team hierarchy proposal inputs changed"

    assert stale_decision.status == "pending", "Stale decision must stay parked, never cancelled"
    assert old_run.superseded_by_id is None, "No new run is created -- suggestion only"
    assert old_run.id == old_run_id

    # The endpoint commits internally on this 409 path, ending the test
    # fixture's outer transaction -- query with an independent session.
    from sqlalchemy.ext.asyncio import async_sessionmaker

    session_factory = async_sessionmaker(test_engine, class_=type(db_session), expire_on_commit=False)
    async with session_factory() as fresh_session:
        warnings = await OrchestrationWarningService().list_warnings(fresh_session, goal_id, active_only=True)
    stale_warnings = [w for w in warnings if w.warning_type == "team_hierarchy_stale_inputs"]
    assert len(stale_warnings) == 1
    assert stale_warnings[0].source_process_run_id == old_run.id


@pytest.mark.asyncio
async def test_repeated_proposal_answer_is_conflict_and_creates_agent_once(
    client, db_session, test_project, test_user, monkeypatch
):
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    proposal = _proposed_agent()

    async def review(_self, payload, project=None, *, project_id=None):
        return _analysis_for(payload, proposals=(proposal,))

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    await TeamHierarchyProcess().advance(db_session, goal, run)
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id))[0]
    url = _answer_url(test_project.id, goal.id, decision.id)

    first = await client.post(url, json={"selected_option": "approve"})
    second = await client.post(url, json={"selected_option": "approve"})

    assert first.status_code == 200, first.text
    assert second.status_code == 409
    assert len(list((await db_session.execute(
        select(Agent.id).where(Agent.name == proposal["definition"]["name"])
    )).scalars())) == 1


@pytest.mark.asyncio
async def test_invalid_resolved_hierarchy_supersedes_old_run_and_keeps_approved_agent(
    client, db_session, test_project, test_user, monkeypatch
):
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    proposal = _proposed_agent()
    payloads = []

    async def review(_self, payload, project=None, *, project_id=None):
        payloads.append(payload)
        return (
            _analysis_for(payload, proposals=(proposal,))
            if len(payloads) == 1
            else _analysis_for(payload, proposals=(_proposed_agent("fresh-specialist"),))
        )

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    await TeamHierarchyProcess().advance(db_session, goal, run)
    service = OrchestrationProcessService()
    old_run = await service.get_current(db_session, goal.id, "team_hierarchy")
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id))[0]
    old_run.outputs["analysis"]["assignments"][0]["agent_ref"] = "proposal:missing"
    await db_session.flush()

    response = await client.post(
        _answer_url(test_project.id, goal.id, decision.id), json={"selected_option": "approve"}
    )

    assert response.status_code == 200, response.text
    current = await service.get_current(db_session, goal.id, "team_hierarchy")
    await db_session.refresh(old_run)
    await db_session.refresh(decision)
    assert old_run.superseded_by_id == current.id
    assert decision.status == "answered"
    assert await db_session.get(Agent, uuid.UUID(json.loads(decision.reason)["created_agent_id"]))
    assert not any(
        item.decision_key == "team_hierarchy:approval" and item.source_process_run_id == old_run.id
        for item in await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    )
    assert current.status == "waiting_decision"
    assert current.outputs.get("retryable") is not True
    assert any(
        item.status == "pending" and item.decision_key == "team_hierarchy:agent:fresh-specialist"
        for item in await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    )


@pytest.mark.asyncio
async def test_separate_proposal_and_hierarchy_approvals_apply_gaps_and_unused_agents(
    client, db_session, test_project, test_user, monkeypatch
):
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    proposals = (_proposed_agent("unused-one"), _proposed_agent("unused-two"))

    async def review(_self, payload, project=None, *, project_id=None):
        return _analysis_for(payload, proposals=proposals, documented_gaps=("validation",))

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    await TeamHierarchyProcess().advance(db_session, goal, run)
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    for decision in decisions:
        response = await client.post(
            _answer_url(test_project.id, goal.id, decision.id), json={"selected_option": "approve"}
        )
        assert response.status_code == 200, response.text

    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    approval = next(item for item in decisions if item.decision_key == "team_hierarchy:approval")
    assert approval.status == "pending"
    assert all(item.status == "answered" for item in decisions if item is not approval)
    response = await client.post(
        _answer_url(test_project.id, goal.id, approval.id),
        json={"selected_option": "approve_with_documented_gaps"},
    )

    assert response.status_code == 200, response.text
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    created_ids = sorted(
        json.loads(item.reason)["created_agent_id"]
        for item in decisions if item.decision_key.startswith("team_hierarchy:agent:")
    )
    assert current.status == "completed"
    assert current.outputs["documented_gaps"] == ["validation"]
    assert current.outputs["approved_but_unused_agent_ids"] == created_ids


@pytest.mark.asyncio
async def test_changed_fingerprint_suggests_rerun_without_cancelling_parked_decision(
    db_session, test_project, test_user
):
    """Item 3, orchestrator override: background input drift (goal edited
    directly, no human interaction with the parked decision) is a one-time
    suggestion -- the parked decision must never be cancelled, no new
    process run is created."""
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    analyzer = FrozenHierarchyAnalyzer()
    process = TeamHierarchyProcess(analyzer=analyzer)
    await process.advance(db_session, goal, run)
    service = OrchestrationProcessService()
    first = await service.get_current(db_session, goal.id, "team_hierarchy")
    old = (await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id))[0]

    goal.objective = "Implement the changed API boundary"
    await process.advance(db_session, goal, run)
    second = await service.get_current(db_session, goal.id, "team_hierarchy")

    assert second.id == first.id
    assert first.superseded_by_id is None
    assert old.status == "pending"
    assert analyzer.calls == 1

    warnings = await OrchestrationWarningService().list_warnings(db_session, goal.id, active_only=True)
    stale_warnings = [w for w in warnings if w.warning_type == "team_hierarchy_stale_inputs"]
    assert len(stale_warnings) == 1
    assert stale_warnings[0].source_process_run_id == first.id


async def _completed_orchestrated_hierarchy(db_session, test_project, test_user):
    goal, run, agents, _review = await _lifecycle_goal(db_session, test_project, test_user)
    process = TeamHierarchyProcess(analyzer=FrozenHierarchyAnalyzer())
    await process.advance(db_session, goal, run)
    decision_service = OrchestrationAuthorityDecisionService()
    decisions = await decision_service.list_decisions(db_session, goal.id)
    proposal_decision = next(
        item for item in decisions if item.decision_key.startswith("team_hierarchy:agent:")
    )
    await decision_service.answer_decision(
        db_session, proposal_decision, selected_option="approve", decided_by_user_id=test_user.id
    )
    await process.advance(db_session, goal, run)
    decisions = await decision_service.list_decisions(db_session, goal.id)
    approval_decision = next(
        item for item in decisions if item.decision_key == "team_hierarchy:approval"
    )
    await decision_service.answer_decision(
        db_session, approval_decision, selected_option="approve", decided_by_user_id=test_user.id
    )
    summary = await process.advance(db_session, goal, run)
    assert summary["status"] == "completed"
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    return goal, run, agents, process, current


@pytest.mark.asyncio
async def test_orchestrated_process_version_bump_alone_is_silently_absorbed(
    db_session, test_project, test_user
):
    """A pure PROCESS_VERSION bump (fingerprint *formula* changed -- dropping
    provider/model/adapter_type/cli_runtime/config from the hash) with no
    actual drift in the underlying roster/goal state is silently re-stamped
    -- no stale-inputs warning."""
    from huddleroom.services.orchestration_team_hierarchy import PROCESS_VERSION

    goal, run, agents, process, current = await _completed_orchestrated_hierarchy(
        db_session, test_project, test_user
    )
    old_formula_fingerprint = process._semantic_fingerprint(
        process._strip_volatile(await process._analysis_input(db_session, goal), extra_fields=())
    )
    current.input_snapshot = {"fingerprint": old_formula_fingerprint}
    current.process_version = PROCESS_VERSION - 1
    await db_session.flush()

    result = await process.advance(db_session, goal, run)

    assert result["status"] == "completed"
    refreshed = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    assert refreshed.id == current.id
    assert refreshed.process_version == PROCESS_VERSION
    warnings = await OrchestrationWarningService().list_warnings(db_session, goal.id, active_only=True)
    assert [w for w in warnings if w.warning_type == "team_hierarchy_stale_inputs"] == []


@pytest.mark.asyncio
async def test_orchestrated_process_version_bump_does_not_absorb_real_drift(
    db_session, test_project, test_user
):
    """A PROCESS_VERSION bump co-occurring with a genuine input change (a
    covered agent's system_prompt changed) must NOT be silently absorbed --
    it falls through to the normal stale-inputs path (one suggestion,
    baseline phase)."""
    from huddleroom.services.orchestration_team_hierarchy import PROCESS_VERSION

    goal, run, agents, process, current = await _completed_orchestrated_hierarchy(
        db_session, test_project, test_user
    )
    old_formula_fingerprint = process._semantic_fingerprint(
        process._strip_volatile(await process._analysis_input(db_session, goal), extra_fields=())
    )
    current.input_snapshot = {"fingerprint": old_formula_fingerprint}
    current.process_version = PROCESS_VERSION - 1
    await db_session.flush()

    agents["planning"].system_prompt = "Updated planning instructions."
    await db_session.flush()

    result = await process.advance(db_session, goal, run)

    assert result["status"] == "completed"
    refreshed = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    assert refreshed.id == current.id
    assert refreshed.process_version == PROCESS_VERSION
    warnings = await OrchestrationWarningService().list_warnings(db_session, goal.id, active_only=True)
    stale_warnings = [w for w in warnings if w.warning_type == "team_hierarchy_stale_inputs"]
    assert len(stale_warnings) == 1
    assert stale_warnings[0].source_process_run_id == current.id


@pytest.mark.asyncio
async def test_hierarchy_skip_is_terminal_until_force_start_begins_new_orchestration(
    db_session, test_project, test_user
):
    goal, run, _agents, _review = await _lifecycle_goal(db_session, test_project, test_user)
    service = OrchestrationProcessService()
    skipped = await service.skip_process(
        db_session, goal.id, process_type="team_hierarchy", skipped_by=f"human:{test_user.id}",
        reason="accept risk", run_id=run.id,
    )
    process = TeamHierarchyProcess(analyzer=FrozenHierarchyAnalyzer())
    assert (await process.advance(db_session, goal, run))["status"] == "skipped"

    forced = await service.start_process(
        db_session, goal.id, process_type="team_hierarchy", trigger_reason="human force-start", run_id=run.id,
    )
    result = await process.advance(db_session, goal, run)

    assert forced.id != skipped.id
    assert result["status"] == "waiting_decision"


@pytest.mark.asyncio
async def test_unchanged_retry_reuses_analysis_and_creates_no_duplicates(db_session, test_project, test_user):
    goal, run, _agents, _review = await _lifecycle_goal(db_session, test_project, test_user)
    analyzer = FrozenHierarchyAnalyzer()
    process = TeamHierarchyProcess(analyzer=analyzer)

    first = await process.advance(db_session, goal, run)
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id)
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    second = await process.advance(db_session, goal, run)

    assert first["questions_created"] == 1
    assert second["questions_created"] == 0
    assert analyzer.calls == 1
    assert await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id) == decisions
    assert await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy") is current
    assert await OrchestrationWarningService().list_warnings(db_session, goal.id) == []
    assert await OrchestrationMemoryService().list_sections(db_session, goal.project_id, goal.id) == []


@pytest.mark.asyncio
async def test_workload_drift_does_not_affect_hierarchy_fingerprint(
    db_session, test_project, test_user
):
    goal, run, agents, _review = await _lifecycle_goal(db_session, test_project, test_user)
    analyzer = FrozenHierarchyAnalyzer()
    process = TeamHierarchyProcess(analyzer=analyzer)
    await process.advance(db_session, goal, run)
    first = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    decision = (await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id))[0]
    snapshot = json.loads(json.dumps(first.outputs))
    db_session.add_all([
        Task(
            project_id=goal.project_id, title=f"new load {index}", status="in_progress",
            assigned_to=agents["implementation"].id,
        )
        for index in range(10)
    ])
    await db_session.flush()

    await process.advance(db_session, goal, run)
    current = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")

    # Workload changes (active_tasks, active_sessions, etc.) must not affect hierarchy fingerprints.
    # The process should remain the same, not be superseded by transient workload drift.
    assert current.id == first.id
    assert first.superseded_by_id is None
    assert decision.status != "cancelled"
    assert snapshot == current.outputs


@pytest.mark.asyncio
async def test_concurrent_unchanged_retries_reuse_process_and_create_no_duplicates(
    concurrent_sessions, monkeypatch
):
    first_session, second_session = concurrent_sessions
    analyzer = FrozenHierarchyAnalyzer()
    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", analyzer.review)

    async def approve(self, request, candidate_work_functions, *, project_id=None):
        return SemanticAgentAssessment(
            "approved", (), "Definition is actionable.", tuple(candidate_work_functions)
        )

    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review_request", approve)
    project = Project(name=f"hierarchy-{uuid.uuid4()}", config={})
    first_session.add(project)
    await first_session.flush()
    goal = OrchestrationGoal(
        project_id=project.id, objective="Implement the API", success_criteria=[], constraints={},
        weight="standard", authority_model="no_manager",
    )
    first_session.add(goal)
    await first_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    agents = {
        work: _agent(f"{work}-{uuid.uuid4()}", work, [work])
        for work in ("planning", "implementation", "summarization")
    }
    first_session.add_all([run, *agents.values()])
    await first_session.flush()
    reviews = [await _review(first_session, goal, run, agent, [work]) for work, agent in agents.items()]
    await _review_process(first_session, goal, run, [review.id for review in reviews])
    await TeamHierarchyProcess().advance(first_session, goal, run)
    await first_session.commit()

    async def retry(session):
        async with session.begin():
            return await TeamHierarchyProcess().advance(
                session,
                await session.get(OrchestrationGoal, goal.id),
                await session.get(OrchestrationRun, run.id),
            )

    summaries = await asyncio.gather(retry(first_session), retry(second_session))

    assert [summary["questions_created"] for summary in summaries] == [0, 0]
    assert analyzer.calls == 1
    decisions = await OrchestrationAuthorityDecisionService().list_decisions(first_session, goal.id)
    assert len(decisions) == 1
    process_runs = await OrchestrationProcessService().list_process_runs(first_session, goal.id)
    assert len([row for row in process_runs if row.process_type == "team_hierarchy"]) == 1
    assert await OrchestrationWarningService().list_warnings(first_session, goal.id) == []
    assert await OrchestrationMemoryService().list_sections(first_session, project.id, goal.id) == []


@pytest.mark.asyncio
async def test_trivial_hierarchy_completes_compressed_without_decision_or_warning(
    db_session, test_project, test_user
):
    goal, run, _agents, _review_process_row = await _lifecycle_goal(
        db_session, test_project, test_user, weight="trivial"
    )

    summary = await TeamHierarchyProcess().advance(db_session, goal, run)
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "team_hierarchy"
    )

    assert summary == {
        "process_type": "team_hierarchy",
        "status": "completed",
        "questions_created": 0,
    }
    assert current.outputs["compressed"] is True
    assert current.outputs["approval"] is None
    assert set(current.outputs) == COMPLETED_OUTPUT_KEYS
    assert current.outputs["gates"] == {
        "team_structure_reviewed": True,
        "required_work_functions_mapped": True,
        "agent_definitions_reviewed": True,
        "missing_capabilities_handled": True,
        "independent_verification_possible_or_overridden": True,
    }
    assert await OrchestrationAuthorityDecisionService().list_decisions(db_session, goal.id) == []
    assert await OrchestrationWarningService().list_warnings(db_session, goal.id) == []
    memory = await OrchestrationMemoryService().get_section(
        db_session, goal.project_id, goal.id, "team_hierarchy"
    )
    assert memory.body == (
        "Trivial goal: the human remains the implicit point of contact; "
        "no team hierarchy review was required."
    )
































@pytest.mark.asyncio
async def test_trivial_after_skipped_agent_definition_review_records_false_gate_without_decision_or_warning(
    db_session, test_project, test_user
):
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="trivial"
    )
    await OrchestrationProcessService().skip_process(
        db_session,
        goal.id,
        process_type="agent_definition_review",
        skipped_by=f"human:{test_user.id}",
        reason="accept review risk",
        run_id=run.id,
    )

    summary = await TeamHierarchyProcess().advance(db_session, goal, run)
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "team_hierarchy"
    )

    assert summary["status"] == "completed"
    assert current.outputs["gates"]["agent_definitions_reviewed"] is False
    assert await OrchestrationAuthorityDecisionService().list_decisions(
        db_session, goal.id
    ) == []
    hierarchy_warnings = [
        warning
        for warning in await OrchestrationWarningService().list_warnings(
            db_session, goal.id
        )
        if warning.warning_type.startswith("team_hierarchy_")
    ]
    assert hierarchy_warnings == []








async def _advance_trivial_prerequisites(db_session, test_project, *, advance_agent_review=True):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_agent_definition_review import (
        AgentDefinitionReviewProcess,
    )
    from huddleroom.services.orchestration_goal_definition import GoalDefinitionProcess
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await OrchestrationService().create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(
            objective="fix typo in README",
            success_criteria=[
                {"key": "fixed", "description": "typo gone", "evidence": "diff"}
            ],
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    assert (await GoalDefinitionProcess().advance(db_session, goal, run))["status"] == "completed"
    assert (await ManagerSelectionProcess().advance(db_session, goal, run))["status"] == "completed"
    if advance_agent_review:
        assert (await AgentDefinitionReviewProcess().advance(db_session, goal, run))["status"] == "completed"
    return goal, run


def _hierarchy_process_url(project_id, goal_id, action):
    return (
        f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}"
        f"/processes/team_hierarchy/{action}"
    )


def test_team_hierarchy_is_force_startable():
    from huddleroom.routers.orchestration_processes import STARTABLE_PROCESS_TYPES

    assert "team_hierarchy" in STARTABLE_PROCESS_TYPES


@pytest.mark.asyncio
async def test_force_start_team_hierarchy_requires_terminal_agent_review(
    client, auth_headers, db_session, test_project
):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, _run = await OrchestrationService().create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(
            objective="fix typo in README",
            success_criteria=[{"key": "fixed", "description": "typo gone", "evidence": "diff"}],
        ),
        created_by_user_id=None,
    )

    response = await client.post(
        _hierarchy_process_url(test_project.id, goal.id, "start"),
        json={"reason": "review the hierarchy"},
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Agent definition review must be terminal before team hierarchy"
    assert await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy") is None


@pytest.mark.asyncio
async def test_force_start_team_hierarchy_links_active_run_after_terminal_agent_review(
    client, auth_headers, db_session, test_project, safe_goal_analysis
):
    goal, run = await _advance_trivial_prerequisites(db_session, test_project)

    response = await client.post(
        _hierarchy_process_url(test_project.id, goal.id, "start"),
        json={"reason": "review the hierarchy"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["run_id"] == str(run.id)


@pytest.mark.asyncio
async def test_force_start_team_hierarchy_links_active_run_after_skipped_agent_review(
    client, auth_headers, db_session, test_project, safe_goal_analysis
):
    goal, run = await _advance_trivial_prerequisites(
        db_session, test_project, advance_agent_review=False
    )
    review_skip = await client.post(
        (
            f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}"
            "/processes/agent_definition_review/skip"
        ),
        json={"reason": "accept review risk"},
        headers=auth_headers,
    )

    assert review_skip.status_code == 200
    assert review_skip.json()["status"] == "skipped"

    response = await client.post(
        _hierarchy_process_url(test_project.id, goal.id, "start"),
        json={"reason": "review the hierarchy"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == str(run.id)




@pytest.mark.asyncio
async def test_generic_hierarchy_skip_is_human_attributed_and_force_start_supersedes_it(
    client, auth_headers, db_session, test_project, test_user, safe_goal_analysis
):
    goal, run = await _advance_trivial_prerequisites(db_session, test_project)
    skip = await client.post(
        _hierarchy_process_url(test_project.id, goal.id, "skip"),
        json={"reason": "accept hierarchy risk"},
        headers=auth_headers,
    )

    assert skip.status_code == 200
    assert skip.json()["skipped_by"] == f"human:{test_user.id}"
    memory = await OrchestrationMemoryService().get_section(
        db_session, goal.project_id, goal.id, "team_hierarchy"
    )
    assert memory.body == TEAM_HIERARCHY_SKIP_MESSAGE
    warnings = await OrchestrationWarningService().list_warnings(
        db_session, goal.id, active_only=True
    )
    assert {warning.warning_type for warning in warnings} == {
        "team_hierarchy_skipped",
        "team_hierarchy_not_reviewed",
    }
    assert {warning.source_process_run_id for warning in warnings} == {uuid.UUID(skip.json()["id"])}
    hierarchy_warning = next(
        warning for warning in warnings if warning.warning_type == "team_hierarchy_not_reviewed"
    )
    assert hierarchy_warning.message == TEAM_HIERARCHY_SKIP_MESSAGE

    restart = await client.post(
        _hierarchy_process_url(test_project.id, goal.id, "start"),
        json={"reason": "review the hierarchy after all"},
        headers=auth_headers,
    )

    assert restart.status_code == 200
    assert restart.json()["run_id"] == str(run.id)
    skipped = next(
        process
        for process in await OrchestrationProcessService().list_process_runs(db_session, goal.id)
        if process.id == uuid.UUID(skip.json()["id"])
    )
    assert skipped.superseded_by_id == uuid.UUID(restart.json()["id"])


@pytest.mark.asyncio
async def test_tick_sequences_team_hierarchy_after_agent_definition_review(
    db_session, test_project, safe_goal_analysis
):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await OrchestrationService().create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(
            objective="fix typo in README",
            success_criteria=[
                {"key": "fixed", "description": "typo gone", "evidence": "diff"}
            ],
        ),
        created_by_user_id=None,
    )
    run.baseline_authorized = True

    result = await OrchestrationService().tick(db_session, run.id)

    assert result["agent_definition_review_process"]["status"] == "completed"
    assert result["team_hierarchy_process"]["status"] == "completed"
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "team_hierarchy"
    )
    assert current.status == "completed"


@pytest.mark.asyncio
async def test_parked_team_hierarchy_keeps_tick_bookkeeping_and_blocks_progress(
    db_session, test_project, test_user, monkeypatch
):
    from huddleroom.services.event_bus import emit_event_once
    from huddleroom.services.orchestration_service import OrchestrationService

    analyzer = FrozenHierarchyAnalyzer()
    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", analyzer.review)

    async def approve(self, request, candidate_work_functions, *, project_id=None):
        return SemanticAgentAssessment(
            "approved", (), "Definition is actionable.", tuple(candidate_work_functions)
        )

    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review_request", approve)
    goal, run, _agents, _review = await _lifecycle_goal(
        db_session, test_project, test_user, weight="substantial"
    )
    process_service = OrchestrationProcessService()
    process = await process_service.start_process(
        db_session, goal.id, process_type="goal_definition", trigger_reason="test setup", run_id=run.id,
    )
    await process_service.complete_process(db_session, process)
    manager_selection_process = await process_service.start_process(
        db_session, goal.id, process_type="manager_selection", trigger_reason="test setup", run_id=run.id,
    )
    # Non-empty candidates: a manager candidate WAS offered/compared at
    # selection time, so #89's roster-gained-candidates rerun trigger
    # (which only fires when outputs["candidates"] is empty) does not apply.
    await process_service.complete_process(
        db_session,
        manager_selection_process,
        outputs={
            "selected_manager": f"human:{test_user.id}",
            "authority_model": "human_manager",
            "manager_fit_rationale": "test setup",
            "candidates": [{"key": "agent:manager", "label": "Manager", "score": 100}],
            "compressed": False,
            "gates": {"manager_selected": True, "authority_model_confirmed": True},
            "weight": goal.weight,
        },
    )
    goal.authority_model = "human_manager"
    goal.manager_user_id = test_user.id
    await db_session.flush()
    await emit_event_once(
        db_session, goal.project_id, "task.created", {"goal_id": str(goal.id)},
        dedup_key=f"hierarchy-parked:{goal.id}",
    )
    service = OrchestrationService()
    baseline_ready_values = []

    async def capture_recovery(_db, _run_id, baseline_ready=False):
        baseline_ready_values.append(baseline_ready)
        return 0

    async def report_ready(*_args, **_kwargs):
        return True

    monkeypatch.setattr(service, "recover_run", capture_recovery)
    monkeypatch.setattr(service, "_run_ready_for_final_summary_request", report_ready)
    monkeypatch.setattr(service, "_run_ready_for_completion", report_ready)

    result = await service.tick(db_session, run.id)

    assert result["team_hierarchy_process"]["status"] == "waiting_decision"
    assert baseline_ready_values == [False]
    assert result["processed_events"] == 1
    assert result["tick_emitted"] is True
    assert result["recoveries_created"] == 0
    assert result["final_summary_action_id"] is None
    assert result["completion_action_id"] is None
    assert result["run_completed"] is False




@pytest.mark.asyncio
async def test_forward_progress_gate_rejects_stale_team_hierarchy_fingerprint(
    db_session, test_project, safe_goal_analysis
):
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await _advance_trivial_prerequisites(db_session, test_project)
    assert (await TeamHierarchyProcess().advance(db_session, goal, run))["status"] == "completed"
    current = await OrchestrationProcessService().get_current(
        db_session, goal.id, "team_hierarchy"
    )
    current.outputs = {**current.outputs, "fingerprint": "stale"}
    await db_session.flush()

    assert (
        await OrchestrationService()._baseline_processes_ready_for_goal(
            db_session, goal.id, allow_heal=False
        )
        is False
    )


@pytest.mark.asyncio
async def test_forward_progress_gate_accepts_human_skipped_team_hierarchy(
    db_session, test_project, safe_goal_analysis
):
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await _advance_trivial_prerequisites(db_session, test_project)
    await OrchestrationProcessService().skip_process(
        db_session,
        goal.id,
        process_type="team_hierarchy",
        skipped_by=f"human:{uuid.uuid4()}",
        reason="accept hierarchy risk",
        run_id=run.id,
    )

    await OrchestrationService()._ensure_baseline_processes_ready(
        db_session, goal.id, allow_heal=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("hierarchy_state", ["missing", "parked"])
@pytest.mark.parametrize("action_name", ["plan", "delegate"])
async def test_direct_planning_and_delegation_require_terminal_team_hierarchy(
    db_session, test_project, hierarchy_state, action_name, safe_goal_analysis
):
    from fastapi import HTTPException

    from huddleroom.services.orchestration_service import OrchestrationService

    goal, run = await _advance_trivial_prerequisites(db_session, test_project)
    if hierarchy_state == "parked":
        hierarchy = await OrchestrationProcessService().start_process(
            db_session,
            goal.id,
            process_type="team_hierarchy",
            trigger_reason="test parked hierarchy",
            run_id=run.id,
        )
        await OrchestrationProcessService().park_process(db_session, hierarchy)

    service = OrchestrationService()
    if action_name == "plan":
        action = service.execute_request_plan_action
        request = {"agent_id": str(uuid.uuid4()), "work_function": "planning"}
    else:
        action = service.execute_create_delegation_task_action
        request = {"work_function": "implementation"}

    with pytest.raises(HTTPException) as exc_info:
        await action(
            db_session,
            run_id=run.id,
            request=request,
            idempotency_key=f"{action_name}-{hierarchy_state}-hierarchy",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "Goal definition, manager selection, agent definition review, and "
        "team hierarchy must complete before planning or delegation"
    )
