import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def seeded_project(db_session, test_project):
    """Seed >cap goals/decisions/meetings/events so caps/truncation are exercised."""
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingDecision
    from huddleroom.models.event_log import EventLog
    from huddleroom.models.orchestration import OrchestrationDecision, OrchestrationGoal, OrchestrationRun

    long_text = "x" * 500

    goals = []
    for i in range(7):
        goal = OrchestrationGoal(
            project_id=test_project.id,
            objective=f"{long_text}-{i}",
            status="active" if i < 5 else "completed",
        )
        db_session.add(goal)
        goals.append(goal)
    await db_session.flush()

    runs = []
    for goal in goals:
        run = OrchestrationRun(goal_id=goal.id)
        db_session.add(run)
        runs.append(run)
    await db_session.flush()

    for run in runs:
        for j in range(3):
            db_session.add(
                OrchestrationDecision(
                    run_id=run.id,
                    decision_type="test_decision",
                    reason=f"{long_text}-{j}",
                )
            )

    meeting_decisions = []
    for i in range(7):
        meeting = Meeting(
            project_id=test_project.id,
            title=f"Meeting {i}",
            meeting_type="standard",
            status="concluded",
        )
        db_session.add(meeting)
        await db_session.flush()
        agenda_item = MeetingAgendaItem(meeting_id=meeting.id, order=0, title="Item")
        db_session.add(agenda_item)
        await db_session.flush()
        decision = MeetingDecision(
            meeting_id=meeting.id,
            agenda_item_id=agenda_item.id,
            title="Decision",
            question=f"{long_text}-{i}",
            chosen_option="A",
            rationale=f"{long_text}-{i}",
            decided_by="orchestrator",
        )
        db_session.add(decision)
        meeting_decisions.append(decision)

    for i in range(15):
        db_session.add(
            EventLog(project_id=test_project.id, event_type=f"test.event.{i}", payload={})
        )

    await db_session.flush()
    return test_project


@pytest.mark.asyncio
async def test_build_advisor_context_respects_caps_and_truncation(db_session, seeded_project):
    from huddleroom.services.orchestration_project_advisor_context import (
        GOAL_PREFACE_LIMIT,
        OPEN_GOALS_LIMIT,
        RECENT_DECISIONS_LIMIT,
        RECENT_EVENTS_LIMIT,
        RECENT_MEETING_DECISIONS_LIMIT,
        _TEXT_TRUNCATE_LIMIT,
        build_advisor_context,
    )

    context = await build_advisor_context(db_session, seeded_project.id)

    assert len(context["open_goals"]) == OPEN_GOALS_LIMIT
    for goal in context["open_goals"]:
        assert len(goal["objective"]) <= _TEXT_TRUNCATE_LIMIT
        assert set(goal) == {"id", "objective", "status", "weight", "current_step"}

    assert len(context["recent_decisions"]) == RECENT_DECISIONS_LIMIT
    for decision in context["recent_decisions"]:
        assert len(decision["summary"]) <= _TEXT_TRUNCATE_LIMIT

    assert len(context["recent_meeting_decisions"]) == RECENT_MEETING_DECISIONS_LIMIT
    for md in context["recent_meeting_decisions"]:
        assert len(md["question"]) <= _TEXT_TRUNCATE_LIMIT
        assert len(md["rationale"]) <= _TEXT_TRUNCATE_LIMIT

    assert len(context["goal_prefaces"]) == GOAL_PREFACE_LIMIT
    for gp in context["goal_prefaces"]:
        assert "goal_id" in gp and "preface" in gp

    assert len(context["recent_events"]) == RECENT_EVENTS_LIMIT


@pytest.mark.asyncio
async def test_build_advisor_context_empty_project_yields_empty_lists(db_session, test_project):
    from huddleroom.services.orchestration_project_advisor_context import build_advisor_context

    context = await build_advisor_context(db_session, test_project.id)

    assert context == {
        "open_goals": [],
        "recent_decisions": [],
        "recent_meeting_decisions": [],
        "goal_prefaces": [],
        "recent_events": [],
    }


@pytest.mark.asyncio
async def test_recent_decisions_include_answered_authority_decisions(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationDecision, OrchestrationGoal, OrchestrationRun
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_project_advisor_context import build_advisor_context

    goal = OrchestrationGoal(project_id=test_project.id, objective="Advisor decision context")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()

    execution = OrchestrationDecision(run_id=run.id, decision_type="execution", reason="Continue work")
    answered = OrchestrationAuthorityDecision(
        goal_id=goal.id,
        run_id=run.id,
        decision_key="team_hierarchy",
        title="Approve hierarchy",
        status="answered",
        authority="human",
        question="Approve the proposed hierarchy?",
        selected_option="approve",
        reason="The hierarchy has the required three agents.",
    )
    pending = OrchestrationAuthorityDecision(
        goal_id=goal.id,
        run_id=run.id,
        decision_key="pending_question",
        title="Pending question",
        authority="human",
        question="Still waiting?",
    )
    db_session.add_all([execution, answered, pending])
    await db_session.flush()

    decisions = (await build_advisor_context(db_session, test_project.id))["recent_decisions"]
    by_id = {decision["id"]: decision for decision in decisions}

    assert by_id[str(execution.id)]["goal_id"] == str(goal.id)
    assert by_id[str(answered.id)]["goal_id"] == str(goal.id)
    assert str(pending.id) not in by_id


@pytest.mark.asyncio
async def test_recent_decisions_merge_sources_with_global_cap_and_stable_order(db_session, test_project):
    from datetime import datetime, timedelta, timezone
    from uuid import UUID

    from huddleroom.models.orchestration import OrchestrationDecision, OrchestrationGoal, OrchestrationRun
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_project_advisor_context import (
        RECENT_DECISIONS_LIMIT,
        _TEXT_TRUNCATE_LIMIT,
        build_advisor_context,
    )

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    goal = OrchestrationGoal(project_id=test_project.id, objective="Mixed decision context")
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()

    executions = [
        OrchestrationDecision(
            id=UUID(int=10 + index),
            run_id=run.id,
            decision_type="execution",
            reason="continue",
            created_at=base + timedelta(minutes=index + 1),
        )
        for index in range(5)
    ]
    execution_tie = OrchestrationDecision(
        id=UUID(int=1),
        run_id=run.id,
        decision_type="execution",
        reason="tie",
        created_at=base + timedelta(minutes=19),
    )
    authorities = [
        OrchestrationAuthorityDecision(
            id=UUID(int=20 + index),
            goal_id=goal.id,
            run_id=run.id,
            decision_key=f"authority-{index}",
            title=f"Authority {index}",
            status="answered",
            authority="human",
            question="Approve?",
            selected_option="approve",
            asked_at=base,
            created_at=base,
            decided_at=base + timedelta(minutes=index + 6),
        )
        for index in range(5)
    ]
    authority_tie = OrchestrationAuthorityDecision(
        id=UUID(int=2),
        goal_id=goal.id,
        run_id=run.id,
        decision_key="authority-tie",
        title="Authority tie",
        status="answered",
        authority="human",
        question="Approve?",
        selected_option="approve",
        asked_at=base,
        created_at=base,
        decided_at=base + timedelta(minutes=19),
    )
    latest = OrchestrationAuthorityDecision(
        id=UUID(int=25),
        goal_id=goal.id,
        run_id=run.id,
        decision_key="latest",
        title="T" * 300,
        status="answered",
        authority="human",
        question="Approve?",
        selected_option="approve",
        reason="R" * 300,
        asked_at=base,
        created_at=base,
        decided_at=base + timedelta(minutes=20),
    )
    excluded = [
        OrchestrationAuthorityDecision(
            id=UUID(int=value),
            goal_id=goal.id,
            run_id=run.id,
            decision_key=f"excluded-{value}",
            title="Excluded",
            status=status,
            authority="human",
            question="Approve?",
            asked_at=base,
            created_at=base,
            decided_at=base + timedelta(minutes=30),
        )
        for value, status in ((30, "cancelled"), (31, "expired"))
    ]
    db_session.add_all(executions + [execution_tie] + authorities + [authority_tie, latest] + excluded)
    await db_session.flush()

    decisions = (await build_advisor_context(db_session, test_project.id))["recent_decisions"]

    assert [decision["id"] for decision in decisions] == [
        str(UUID(int=value)) for value in (25, 2, 1, 24, 23, 22, 21, 20, 14, 13)
    ]
    assert len(decisions) == RECENT_DECISIONS_LIMIT
    assert len(decisions[0]["summary"]) <= _TEXT_TRUNCATE_LIMIT
    assert {str(UUID(int=30)), str(UUID(int=31))}.isdisjoint({decision["id"] for decision in decisions})
