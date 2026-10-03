import pytest


def test_merge_answer_drops_assumption_matching_answered_destination():
    from huddleroom.models.orchestration import OrchestrationGoal
    from huddleroom.services.orchestration_goal_definition import _merge_answer

    goal = OrchestrationGoal(objective="Build a dashboard", original_request="Build a dashboard")
    context = {
        "assumptions": [
            {"text": "Assume qualified stage only", "destination": "success_criteria", "round": 0},
        ]
    }

    _merge_answer(
        goal, context, "round:0:0:success_criteria", "Which stages?", "Qualified and closed-won"
    )

    assert context["assumptions"] == []


def test_merge_answer_keeps_assumption_with_different_destination():
    from huddleroom.models.orchestration import OrchestrationGoal
    from huddleroom.services.orchestration_goal_definition import _merge_answer

    goal = OrchestrationGoal(objective="Build a dashboard", original_request="Build a dashboard")
    context = {
        "assumptions": [
            {"text": "Assume weekly cadence", "destination": "objective", "round": 0},
        ]
    }

    _merge_answer(
        goal, context, "round:0:0:success_criteria", "Which stages?", "Qualified and closed-won"
    )

    assert context["assumptions"] == [
        {"text": "Assume weekly cadence", "destination": "objective", "round": 0},
    ]


def test_merge_answer_does_not_crash_on_raw_string_assumption():
    from huddleroom.models.orchestration import OrchestrationGoal
    from huddleroom.services.orchestration_goal_definition import _merge_answer

    goal = OrchestrationGoal(objective="Build a dashboard", original_request="Build a dashboard")
    context = {"assumptions": ["Assume default timezone is UTC"]}

    _merge_answer(
        goal, context, "round:0:0:success_criteria", "Which stages?", "Qualified and closed-won"
    )

    assert context["assumptions"] == ["Assume default timezone is UTC"]


@pytest.mark.parametrize(
    ("original_request", "expected"),
    [(None, None), ("  Make a sales dashboard.  ", "Make a sales dashboard.")],
)
def test_goal_create_preserves_none_and_trims_original_request(original_request, expected):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate

    goal = OrchestrationGoalCreate(objective="Build a dashboard", original_request=original_request)

    assert goal.original_request == expected


def test_goal_create_rejects_whitespace_only_original_request():
    from pydantic import ValidationError

    from huddleroom.schemas.orchestration import OrchestrationGoalCreate

    with pytest.raises(ValidationError, match="original request is required"):
        OrchestrationGoalCreate(objective="Build a dashboard", original_request="   \n\t ")


class StubGoalAnalyzer:
    def __init__(self, *results):
        self.results = list(results)
        self.snapshots = []

    def build_request(self, snapshot, project=None):
        self.snapshots.append(snapshot)
        return {"model": "stub", "messages": []}

    async def analyze_request(self, request, *, project_id=None):
        return self.results.pop(0)


@pytest.mark.asyncio
async def test_create_goal_persists_original_request_and_keeps_it_after_objective_changes(
    db_session, test_project
):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, _run = await OrchestrationService().create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(
            objective="Build the clarified dashboard",
            original_request="Make a dashboard that shows our sales pipeline.",
        ),
        created_by_user_id=None,
    )

    goal.objective = "Build the sales dashboard with filtering"
    await db_session.flush()
    await db_session.refresh(goal)

    assert goal.original_request == "Make a dashboard that shows our sales pipeline."
    with pytest.raises(ValueError, match="immutable"):
        goal.original_request = "Replace the intake text"


@pytest.mark.asyncio
async def test_create_goal_uses_objective_when_original_request_is_omitted(db_session, test_project):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    goal, _run = await OrchestrationService().create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(objective="Build the clarified dashboard"),
        created_by_user_id=None,
    )

    assert goal.original_request == "Build the clarified dashboard"


@pytest.mark.asyncio
async def test_goal_definition_keeps_original_request_in_the_answered_objective_snapshot_and_memory(
    db_session, test_project, test_user
):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_goal_analyzer import GoalAnalysis, GoalQuestion
    from huddleroom.services.orchestration_goal_definition import GoalDefinitionProcess
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
    from huddleroom.services.orchestration_service import OrchestrationService

    original_request = "Make a dashboard that shows our sales pipeline."
    goal, run = await OrchestrationService().create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(objective="Build a dashboard", original_request=original_request),
        created_by_user_id=None,
    )
    run.baseline_authorized = True
    analyzer = StubGoalAnalyzer(
        GoalAnalysis((), (GoalQuestion("Which pipeline stages?", "Changes scope", "objective"),), False),
        GoalAnalysis((), (), False),
    )
    process = GoalDefinitionProcess(analyzer=analyzer)

    await process.advance(db_session, goal, run)
    decision = (await process.decision_service.list_decisions(db_session, goal.id))[0]
    await process.decision_service.answer_decision(
        db_session,
        decision,
        selected_option="Build a dashboard for qualified and closed-won stages",
        decided_by_user_id=test_user.id,
    )
    await process.advance(db_session, goal, run)

    # Objective stays immutable; clarification goes to objective_notes
    assert analyzer.snapshots[1]["objective"] == "Build a dashboard"
    assert analyzer.snapshots[1].get("orchestrator_context", {}).get("objective_notes") == ["Build a dashboard for qualified and closed-won stages"]
    assert analyzer.snapshots[1]["original_request"] == original_request
    assert goal.original_request == original_request
    memory = await OrchestrationMemoryService().get_section(
        db_session, test_project.id, goal.id, "goal_definition"
    )
    assert original_request in memory.body


@pytest.mark.asyncio
async def test_decision_context_contains_the_original_request(db_session, test_project):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    original_request = "Make a dashboard that shows our sales pipeline."
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        test_project.id,
        OrchestrationGoalCreate(objective="Build a dashboard", original_request=original_request),
        created_by_user_id=None,
    )

    context = await service._decision_context(db_session, goal, run)

    assert context["goal"]["original_request"] == original_request
