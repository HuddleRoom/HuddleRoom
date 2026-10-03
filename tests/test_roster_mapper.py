import uuid

import pytest

from huddleroom.models.agent import Agent
from huddleroom.models.memory_item import MemoryItem
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper


def _agent(
    name_prefix: str,
    role: str,
    capabilities: list[str],
    *,
    is_active: bool = True,
    config: dict | None = None,
) -> Agent:
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config=config or {},
        is_active=is_active,
    )


@pytest.mark.asyncio
async def test_rank_agents_selects_closest_fit_without_planner_role(db_session, test_project):
    product_agent = _agent("product", "product strategist", ["requirements", "roadmap"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([product_agent, reviewer])
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "planning")

    assert [fit.agent_id for fit in fits] == [product_agent.id, reviewer.id]
    assert fits[0].score > fits[1].score
    assert fits[0].weak is False
    assert "role:product" in fits[0].matched_signals


@pytest.mark.asyncio
async def test_rank_agents_ignores_inactive_agents(db_session, test_project):
    inactive_planner = _agent("planner", "planner", ["planning"], is_active=False)
    active_developer = _agent("developer", "developer", ["implementation"], is_active=True)
    db_session.add_all([inactive_planner, active_developer])
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "planning")

    assert [fit.agent_id for fit in fits] == [active_developer.id]


@pytest.mark.asyncio
async def test_rank_agents_marks_poor_matches_as_weak(db_session, test_project):
    billing_agent = _agent("billing", "finance", ["invoicing"])
    db_session.add(billing_agent)
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "planning")

    assert fits[0].agent_id == billing_agent.id
    assert fits[0].score < 45
    assert fits[0].weak is True


@pytest.mark.asyncio
async def test_rank_agents_accepts_domain_neutral_work_functions(db_session, test_project):
    data_agent = _agent("data", "operations", ["data_cleanup"])
    generalist = _agent("generalist", "developer", ["implementation"])
    db_session.add_all([data_agent, generalist])
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(
        db_session,
        test_project.id,
        "data_cleanup",
        required_capabilities=["data_cleanup"],
        domain_hints=["records", "duplicates"],
    )

    assert fits[0].agent_id == data_agent.id
    assert fits[0].work_function == "data_cleanup"
    assert "required_capability:data_cleanup" in fits[0].matched_signals


@pytest.mark.asyncio
async def test_rank_agents_prefers_idle_agent_when_fit_is_equal(db_session, test_project):
    busy = _agent("busy", "developer", ["implementation"])
    idle = _agent("idle", "developer", ["implementation"])
    db_session.add_all([busy, idle])
    await db_session.flush()

    db_session.add(
        Task(
            project_id=test_project.id,
            title="Busy task",
            status="in_progress",
            assigned_to=busy.id,
        )
    )
    db_session.add(
        Session(
            project_id=test_project.id,
            agent_id=busy.id,
            adapter_type="api",
            status="running",
        )
    )
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "implementation")

    assert fits[0].agent_id == idle.id
    busy_fit = next(fit for fit in fits if fit.agent_id == busy.id)
    assert busy_fit.load.active_tasks == 1
    assert busy_fit.load.active_sessions == 1
    assert "load_penalty:11" in busy_fit.matched_signals


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,project_id,expected_count,content_desc",
    [
        ("project", "current_project", 1, "Successful validation outcome accepted by review."),
        ("global", None, 1, "Validation completed successfully."),
        ("project", None, 0, "Validation completed successfully."),
    ],
)
async def test_rank_agents_outcome_memory_scope_variations(db_session, test_project, scope, project_id, expected_count, content_desc):
    experienced = _agent("experienced", "validator", ["validation"])
    new_validator = _agent("new", "validator", ["validation"])
    db_session.add_all([experienced, new_validator])
    await db_session.flush()

    # project_id is either current_project (as sentinel), None, or actual project
    resolved_project_id = test_project.id if project_id == "current_project" else project_id

    db_session.add(
        MemoryItem(
            agent_id=experienced.id,
            project_id=resolved_project_id,
            scope=scope,
            content=content_desc,
            shared=True,
        )
    )
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "validation")

    assert fits[0].agent_id == experienced.id
    assert fits[0].load.outcome_hint_count == expected_count
    if expected_count > 0:
        assert f"outcome_memory:{expected_count}" in fits[0].matched_signals
    else:
        assert not any(signal.startswith("outcome_memory:") for signal in fits[0].matched_signals)


@pytest.mark.asyncio
async def test_rank_agents_ignores_outcome_keywords_inside_larger_negative_words(db_session, test_project):
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    db_session.add_all(
        [
            MemoryItem(
                agent_id=validator.id,
                project_id=test_project.id,
                scope="project",
                content="Validation was unsuccessful.",
                shared=True,
            ),
            MemoryItem(
                agent_id=validator.id,
                project_id=test_project.id,
                scope="project",
                content="The expected threshold was surpassed.",
                shared=True,
            ),
            MemoryItem(
                agent_id=validator.id,
                project_id=test_project.id,
                scope="project",
                content="The required check was bypassed.",
                shared=True,
            ),
        ]
    )
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "validation")

    assert fits[0].agent_id == validator.id
    assert fits[0].load.outcome_hint_count == 0
    assert not any(signal.startswith("outcome_memory:") for signal in fits[0].matched_signals)


@pytest.mark.asyncio
async def test_rank_agents_caps_outcome_memory_rows_per_agent(db_session, test_project):
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    db_session.add_all(
        [
            MemoryItem(
                agent_id=validator.id,
                project_id=test_project.id,
                scope="project",
                content=f"Validation completed successfully {index}.",
                shared=True,
            )
            for index in range(30)
        ]
    )
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "validation")

    assert fits[0].agent_id == validator.id
    assert fits[0].load.outcome_hint_count == 25
    assert "outcome_memory:25" in fits[0].matched_signals


@pytest.mark.asyncio
async def test_rank_agents_matches_keywords_from_multiword_capabilities(db_session, test_project):
    reviewer = _agent("reviewer", "finance", ["code_review"])
    implementer = _agent("implementer", "developer", ["implementation"])
    db_session.add_all([reviewer, implementer])
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "review")

    reviewer_fit = next(fit for fit in fits if fit.agent_id == reviewer.id)
    assert "capability:code_review" in reviewer_fit.matched_signals
    assert "keyword:review" in reviewer_fit.matched_signals


@pytest.mark.asyncio
async def test_rank_agents_deduplicates_equivalent_required_capabilities(db_session, test_project):
    reviewer = _agent("reviewer", "reviewer", ["code_review"])
    db_session.add(reviewer)
    await db_session.flush()

    mapper = OrchestrationRosterMapper()
    single = await mapper.rank_agents(
        db_session,
        test_project.id,
        "review",
        required_capabilities=["code_review"],
    )
    replayed = await mapper.rank_agents(
        db_session,
        test_project.id,
        "review",
        required_capabilities=["code review", "code_review", "code-review"],
    )

    assert replayed[0].score == single[0].score
    assert replayed[0].matched_signals.count("required_capability:code_review") == 1


@pytest.mark.asyncio
async def test_loads_by_agent_ignores_active_sessions_without_agent_id(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()

    class NullSessionMapper(OrchestrationRosterMapper):
        async def _active_task_counts(self, db, project_id):  # pylint: disable=arguments-differ
            return {}

        async def _active_session_counts(self, db, project_id):  # pylint: disable=arguments-differ
            return {developer.id: 1, None: 99}

        async def _outcome_hint_counts(self, db, project_id):  # pylint: disable=arguments-differ
            return {}

    loads = await NullSessionMapper()._loads_by_agent(db_session, test_project.id)

    assert developer.id in loads
    assert loads[developer.id].active_sessions == 1
    assert None not in loads


@pytest.mark.asyncio
async def test_context_snapshot_returns_roster_envelope(db_session, test_project):
    planner = _agent("planner", "product strategist", ["requirements"])
    inactive = _agent("inactive", "developer", ["implementation"], is_active=False)
    db_session.add_all([planner, inactive])
    await db_session.flush()

    snapshot = await OrchestrationRosterMapper().context_snapshot(
        db_session,
        test_project.id,
        work_functions=("planning",),
        limit_per_function=1,
    )

    assert snapshot["active_agent_count"] == 1
    assert list(snapshot["work_functions"]) == ["planning"]
    assert snapshot["work_functions"]["planning"][0]["agent_id"] == str(planner.id)


@pytest.mark.asyncio
async def test_roster_fit_to_dict_includes_capabilities(db_session, test_project):
    developer = _agent("developer", "developer", ["implementation", "python"])
    db_session.add(developer)
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "implementation")

    assert fits[0].capabilities == ["implementation", "python"]
    assert fits[0].to_dict()["capabilities"] == ["implementation", "python"]


@pytest.mark.asyncio
async def test_rank_agents_does_not_match_keywords_inside_larger_words(db_session, test_project):
    contestant = _agent("contestant", "contestant", ["billing"])
    tester = _agent("tester", "tester", ["validation"])
    db_session.add_all([contestant, tester])
    await db_session.flush()

    fits = await OrchestrationRosterMapper().rank_agents(db_session, test_project.id, "validation")

    contestant_fit = next(fit for fit in fits if fit.agent_id == contestant.id)
    tester_fit = next(fit for fit in fits if fit.agent_id == tester.id)
    assert "keyword:test" not in contestant_fit.matched_signals
    assert tester_fit.score > contestant_fit.score


@pytest.mark.asyncio
async def test_decision_context_includes_roster_snapshot(db_session, test_project):
    validator = _agent("validator", "qa reviewer", ["validation", "review"])
    db_session.add(validator)
    await db_session.flush()

    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Prove a goal with independent validation",
            success_criteria=[{"key": "validated", "description": "Independent validation accepts the work."}],
        ),
        created_by_user_id=None,
    )

    context = await service._decision_context(db_session, goal, run)

    assert context["roster"]["active_agent_count"] >= 1
    validation_fits = context["roster"]["work_functions"]["validation"]
    validator_fit = next(fit for fit in validation_fits if fit["agent_id"] == str(validator.id))
    assert validator_fit["work_function"] == "validation"
    assert validator_fit["score"] >= 45
    assert validator_fit["weak"] is False
