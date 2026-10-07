import uuid

import pytest
import pytest_asyncio
from sqlalchemy import inspect


async def _table_names(engine):
    async with engine.connect() as conn:
        return await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())


@pytest_asyncio.fixture
async def orch_goal(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the widget",
        success_criteria=[{"key": "works", "description": "widget works"}],
    )
    db_session.add(goal)
    await db_session.flush()
    return goal


@pytest_asyncio.fixture
async def orch_run(db_session, orch_goal):
    from huddleroom.models.orchestration import OrchestrationRun

    run = OrchestrationRun(goal_id=orch_goal.id)
    db_session.add(run)
    await db_session.flush()
    return run


@pytest_asyncio.fixture
async def process_run(db_session, orch_goal):
    from huddleroom.models.orchestration_process import OrchestrationProcessRun

    row = OrchestrationProcessRun(
        goal_id=orch_goal.id,
        process_type="agent_definition_review",
        trigger_reason="team hierarchy review",
    )
    db_session.add(row)
    await db_session.flush()
    return row


@pytest.mark.asyncio
async def test_agent_review_insert_defaults_and_links(
    db_session, orch_goal, orch_run, process_run, test_agent
):
    from huddleroom.models.orchestration_process import OrchestrationAgentReview

    review = OrchestrationAgentReview(
        goal_id=orch_goal.id,
        run_id=orch_run.id,
        agent_id=test_agent.id,
        source_process_run_id=process_run.id,
        definition_snapshot={"role": "developer"},
        fit_summary="Adequate fit for implementation work.",
    )
    db_session.add(review)
    await db_session.flush()

    assert review.id is not None
    assert review.goal_id == orch_goal.id
    assert review.run_id == orch_run.id
    assert review.agent_id == test_agent.id
    assert review.source_process_run_id == process_run.id
    assert review.review_context is None
    assert review.definition_snapshot == {"role": "developer"}
    assert review.proposed_work_functions == []
    assert review.strengths == []
    assert review.risks == []
    assert review.recommended_changes == []
    assert review.approved_for_work_functions == []
    assert review.created_at is not None
    assert review.updated_at is not None


@pytest.mark.asyncio
async def test_agent_review_stores_work_function_lists(db_session, orch_goal, test_agent):
    from huddleroom.models.orchestration_process import OrchestrationAgentReview

    review = OrchestrationAgentReview(
        goal_id=orch_goal.id,
        agent_id=test_agent.id,
        definition_snapshot={"role": "developer"},
        fit_summary="Strong planner despite the developer role name.",
        proposed_work_functions=["planner", "implementer"],
        strengths=["clear planning instructions"],
        risks=["no verification discipline"],
        recommended_changes=["add explicit review checklist to system prompt"],
        approved_for_work_functions=["planner"],
    )
    db_session.add(review)
    await db_session.flush()

    assert review.proposed_work_functions == ["planner", "implementer"]
    assert review.strengths == ["clear planning instructions"]
    assert review.risks == ["no verification discipline"]
    assert review.recommended_changes == ["add explicit review checklist to system prompt"]
    assert review.approved_for_work_functions == ["planner"]


@pytest_asyncio.fixture
async def other_goal_and_run(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Different goal",
        success_criteria=[{"key": "other", "description": "other goal works"}],
    )
    db_session.add(goal)
    await db_session.flush()
    run = OrchestrationRun(goal_id=goal.id)
    db_session.add(run)
    await db_session.flush()
    return goal, run


@pytest.mark.asyncio
async def test_create_review_builds_snapshot_from_agent(db_session, orch_goal, test_agent):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    svc = OrchestrationAgentReviewService()
    review = await svc.create_review(
        db_session,
        orch_goal.id,
        agent_id=test_agent.id,
        fit_summary="Solid fit for implementation work.",
        proposed_work_functions=["implementer"],
    )

    snap = review.definition_snapshot
    assert snap["name"] == test_agent.name
    assert snap["role"] == "developer"
    assert snap["description"] is None
    assert snap["system_prompt"] is None
    assert snap["provider"] == "openai"
    assert snap["model"] == "gpt-4o-mini"
    assert snap["adapter_type"] == "api"
    assert snap["cli_runtime"] is None
    assert snap["capabilities"] == []
    assert snap["config"] == {}
    assert snap["is_active"] in (True, 1)  # SQLite may return int


@pytest.mark.asyncio
async def test_snapshot_survives_agent_change(db_session, orch_goal, test_agent):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    svc = OrchestrationAgentReviewService()
    review = await svc.create_review(
        db_session,
        orch_goal.id,
        agent_id=test_agent.id,
        fit_summary="Reviewed as developer.",
    )

    snapshot_capabilities = review.definition_snapshot["capabilities"]
    snapshot_config = review.definition_snapshot["config"]

    # Scalar reassignment on the live agent must not reach the snapshot.
    test_agent.role = "reviewer"
    # In-place mutation of the agent's JSON columns must not reach the
    # snapshot either — a shallow copy would still share these nested
    # list/dict objects with the live Agent instance (reviewer-flagged gap:
    # a scalar-only check like the role reassignment above cannot catch this).
    test_agent.capabilities.append("new_capability")
    test_agent.config["temperature"] = 0.9
    await db_session.flush()

    assert review.definition_snapshot["role"] == "developer"
    assert review.definition_snapshot["capabilities"] == snapshot_capabilities
    assert "new_capability" not in review.definition_snapshot["capabilities"]
    assert review.definition_snapshot["config"] == snapshot_config
    assert "temperature" not in review.definition_snapshot["config"]


@pytest.mark.asyncio
async def test_create_review_unknown_agent_rejected(db_session, orch_goal):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    svc = OrchestrationAgentReviewService()
    with pytest.raises(ValueError, match="does not exist"):
        await svc.create_review(
            db_session,
            orch_goal.id,
            agent_id=uuid.uuid4(),
            fit_summary="x",
        )


@pytest.mark.asyncio
async def test_create_review_empty_fit_summary_rejected(db_session, orch_goal, test_agent):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    svc = OrchestrationAgentReviewService()
    with pytest.raises(ValueError, match="fit_summary"):
        await svc.create_review(
            db_session,
            orch_goal.id,
            agent_id=test_agent.id,
            fit_summary="   ",
        )


@pytest.mark.asyncio
async def test_create_review_cross_goal_run_rejected(
    db_session, orch_goal, other_goal_and_run, test_agent
):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    _, other_run = other_goal_and_run
    svc = OrchestrationAgentReviewService()
    with pytest.raises(ValueError, match="does not belong to goal"):
        await svc.create_review(
            db_session,
            orch_goal.id,
            agent_id=test_agent.id,
            fit_summary="x",
            run_id=other_run.id,
        )


@pytest.mark.asyncio
async def test_create_review_cross_goal_process_run_rejected(
    db_session, orch_goal, other_goal_and_run, test_agent
):
    from huddleroom.models.orchestration_process import OrchestrationProcessRun
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    other_goal, _ = other_goal_and_run
    foreign_process_run = OrchestrationProcessRun(
        goal_id=other_goal.id,
        process_type="agent_definition_review",
        trigger_reason="x",
    )
    db_session.add(foreign_process_run)
    await db_session.flush()

    svc = OrchestrationAgentReviewService()
    with pytest.raises(ValueError, match="does not belong to goal"):
        await svc.create_review(
            db_session,
            orch_goal.id,
            agent_id=test_agent.id,
            fit_summary="x",
            source_process_run_id=foreign_process_run.id,
        )


@pytest.mark.asyncio
async def test_create_review_approved_must_be_subset_of_proposed(
    db_session, orch_goal, test_agent
):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    svc = OrchestrationAgentReviewService()
    with pytest.raises(ValueError, match="not among proposed"):
        await svc.create_review(
            db_session,
            orch_goal.id,
            agent_id=test_agent.id,
            fit_summary="x",
            proposed_work_functions=["implementer"],
            approved_for_work_functions=["reviewer"],
        )


@pytest.mark.asyncio
async def test_create_review_idempotent_per_process_run(
    db_session, orch_goal, process_run, test_agent
):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    svc = OrchestrationAgentReviewService()
    first = await svc.create_review(
        db_session,
        orch_goal.id,
        agent_id=test_agent.id,
        fit_summary="First evaluation.",
        source_process_run_id=process_run.id,
    )
    second = await svc.create_review(
        db_session,
        orch_goal.id,
        agent_id=test_agent.id,
        fit_summary="Retried evaluation, different text.",
        source_process_run_id=process_run.id,
    )

    assert second.id == first.id
    assert second.fit_summary == "First evaluation."  # retry returns existing row unchanged


@pytest.mark.asyncio
async def test_create_review_standalone_not_deduped(db_session, orch_goal, test_agent):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    svc = OrchestrationAgentReviewService()
    first = await svc.create_review(
        db_session, orch_goal.id, agent_id=test_agent.id, fit_summary="First standalone."
    )
    second = await svc.create_review(
        db_session, orch_goal.id, agent_id=test_agent.id, fit_summary="Second standalone."
    )

    assert second.id != first.id


@pytest.mark.asyncio
async def test_approve_work_functions_updates_column(db_session, orch_goal, test_agent):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    svc = OrchestrationAgentReviewService()
    review = await svc.create_review(
        db_session,
        orch_goal.id,
        agent_id=test_agent.id,
        fit_summary="Can plan and implement.",
        proposed_work_functions=["planner", "implementer"],
    )
    assert review.approved_for_work_functions == []

    review = await svc.approve_work_functions(
        db_session, review, work_functions=["implementer"]
    )
    assert review.approved_for_work_functions == ["implementer"]

    # Re-approval overwrites (full-replace, matching Phase 1 PUT semantics).
    review = await svc.approve_work_functions(
        db_session, review, work_functions=["planner", "implementer"]
    )
    assert review.approved_for_work_functions == ["planner", "implementer"]


@pytest.mark.asyncio
async def test_approve_work_functions_rejects_non_proposed(db_session, orch_goal, test_agent):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    svc = OrchestrationAgentReviewService()
    review = await svc.create_review(
        db_session,
        orch_goal.id,
        agent_id=test_agent.id,
        fit_summary="x",
        proposed_work_functions=["implementer"],
    )
    with pytest.raises(ValueError, match="not among proposed"):
        await svc.approve_work_functions(db_session, review, work_functions=["reviewer"])


@pytest.mark.asyncio
async def test_approve_work_functions_rejects_empty(db_session, orch_goal, test_agent):
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    svc = OrchestrationAgentReviewService()
    review = await svc.create_review(
        db_session,
        orch_goal.id,
        agent_id=test_agent.id,
        fit_summary="x",
        proposed_work_functions=["implementer"],
    )
    with pytest.raises(ValueError, match="must not be empty"):
        await svc.approve_work_functions(db_session, review, work_functions=[])


@pytest.mark.asyncio
async def test_list_reviews_filters_by_agent(db_session, orch_goal, test_agent):
    from huddleroom.models.agent import Agent
    from huddleroom.services.orchestration_agent_review_service import (
        OrchestrationAgentReviewService,
    )

    other_agent = Agent(
        name=f"other-agent-{uuid.uuid4()}",
        role="reviewer",
        provider="anthropic",
        model="claude-sonnet-5-5",
        adapter_type="api",
        capabilities=[],
        config={},
    )
    db_session.add(other_agent)
    await db_session.flush()

    svc = OrchestrationAgentReviewService()
    await svc.create_review(
        db_session, orch_goal.id, agent_id=test_agent.id, fit_summary="Dev review."
    )
    await svc.create_review(
        db_session, orch_goal.id, agent_id=other_agent.id, fit_summary="Reviewer review."
    )

    all_reviews = await svc.list_reviews(db_session, orch_goal.id)
    assert len(all_reviews) == 2

    filtered = await svc.list_reviews(db_session, orch_goal.id, agent_id=other_agent.id)
    assert len(filtered) == 1
    assert filtered[0].agent_id == other_agent.id
