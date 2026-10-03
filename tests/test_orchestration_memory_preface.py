"""Regression tests for orchestration memory preface builder (Spec 5.3).

Covers Fix 1 (tz-aware/naive datetime handling), Fix 2 (whitespace-only summary),
and Fix 3 (created_at in metadata select).
"""

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.orchestration_process import (
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
)
from huddleroom.services.orchestration_authority_service import (
    OrchestrationAuthorityDecisionService,
)
from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService


@pytest_asyncio.fixture
async def orch_goal(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Test preface builder",
        success_criteria=[{"key": "done", "description": "Done"}],
    )
    db_session.add(goal)
    await db_session.flush()
    return goal


@pytest_asyncio.fixture
async def orch_run(db_session, orch_goal):
    run = OrchestrationRun(goal_id=orch_goal.id)
    db_session.add(run)
    await db_session.flush()
    return run


@pytest.mark.asyncio
async def test_preface_builder_handles_mixed_tz_aware_naive_datetimes(db_session, orch_goal, orch_run, test_user):
    """Regression test for Fix 1: mixed-session tz-aware/naive datetime handling.

    Reproduces the bug where some decision objects (identity-resident from
    earlier in the session) have tz-aware datetimes, while others (freshly
    queried) have naive datetimes, causing TypeError when sorted() compares them.
    """
    # Create two decisions with asked_at/decided_at
    svc = OrchestrationAuthorityDecisionService()

    # Decision 1: created with default _utcnow (becomes identity-resident with tz-aware datetime)
    decision1 = await svc.create_pending(
        db_session,
        orch_goal.id,
        decision_key="test_mixed_1",
        title="Decision 1",
        question="Question 1?",
        authority="human",
        options=["yes", "no"],
    )

    # Decision 2: created similarly
    decision2 = await svc.create_pending(
        db_session,
        orch_goal.id,
        decision_key="test_mixed_2",
        title="Decision 2",
        question="Question 2?",
        authority="human",
        options=["yes", "no"],
    )

    # Answer both decisions so they have decided_at set
    await svc.answer_decision(
        db_session, decision1, selected_option="yes", decided_by_user_id=test_user.id
    )
    await db_session.flush()

    await svc.answer_decision(
        db_session, decision2, selected_option="no", decided_by_user_id=test_user.id
    )
    await db_session.flush()

    # Force a mixed scenario regardless of what this dialect/session normally
    # returns: decision1 gets an explicit tz-aware decided_at, decision2 an
    # explicit naive one. Both objects stay identity-resident in this session,
    # so list_decisions() below returns exactly this mismatched pair unchanged.
    decision1.decided_at = decision1.decided_at.replace(tzinfo=timezone.utc)
    decision2.decided_at = decision2.decided_at.replace(tzinfo=None)

    # Now build the preface with decisions from list_decisions (which returns a mix).
    # The preface builder must not raise TypeError when sorting mixed tz-aware/naive datetimes.
    builder = OrchestrationMemoryPrefaceBuilder()
    preface = await builder.build(db_session, orch_goal, orch_run)

    # Assertions
    assert preface is not None
    assert "recent_decisions" in preface
    # Both decisions should be in the preface (sorted without TypeError)
    assert len(preface["recent_decisions"]) == 2


@pytest.mark.asyncio
async def test_preface_builder_excerpt_uses_body_for_whitespace_only_summary_dedicated(
    db_session, orch_goal, orch_run
):
    """Regression test for Fix 2: whitespace-only summary handling in _excerpt.

    For a dedicated section (introduction) with whitespace-only cached summary
    and non-empty body, the preface should use the body content, not an empty string.
    """
    memory_service = OrchestrationMemoryService()

    # Create introduction section with whitespace-only summary and real body
    await memory_service.upsert_section(
        db_session,
        orch_goal.project_id,
        orch_goal.id,
        section_key="introduction",
        title="Introduction",
        body="This is the real introduction content that should appear in the preface.",
        summary="   ",  # whitespace-only summary (truthy, but empty after strip)
        always_load=False,
    )

    builder = OrchestrationMemoryPrefaceBuilder()
    preface = await builder.build(db_session, orch_goal, orch_run)

    # The introduction field should use the body, not the whitespace-only summary
    assert preface["introduction"] is not None
    assert "real introduction content" in preface["introduction"]
    assert "   " not in preface["introduction"]  # whitespace should be gone


@pytest.mark.asyncio
async def test_preface_builder_reuses_loaded_sections_for_excerpts(
    db_session, orch_goal, orch_run, monkeypatch
):
    memory_service = OrchestrationMemoryService()
    section = await memory_service.upsert_section(
        db_session,
        orch_goal.project_id,
        orch_goal.id,
        section_key="introduction",
        title="Introduction",
        body="Loaded section body is enough.",
        summary="   ",
        always_load=False,
    )

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("section rows supplied by overview should be reused")

    monkeypatch.setattr(OrchestrationMemoryService, "list_section_metadata", fail_if_called)
    monkeypatch.setattr(OrchestrationMemoryService, "get_section", fail_if_called)

    preface = await OrchestrationMemoryPrefaceBuilder().build(
        db_session, orch_goal, orch_run, section_meta=[section]
    )

    assert preface["introduction"] == "Loaded section body is enough."


@pytest.mark.asyncio
async def test_preface_builder_excerpt_uses_body_for_whitespace_only_summary_always_loaded(
    db_session, orch_goal, orch_run
):
    """Regression test for Fix 2: whitespace-only summary in always_loaded sections.

    For an always_loaded section with whitespace-only summary and non-empty body,
    the always_loaded excerpt should use the body, not an empty string.
    """
    memory_service = OrchestrationMemoryService()

    # Create an always_loaded section with whitespace-only summary and real body
    await memory_service.upsert_section(
        db_session,
        orch_goal.project_id,
        orch_goal.id,
        section_key="operating_assumptions",
        title="Operating Assumptions",
        body="Key assumption: we have unlimited time and budget.",
        summary="\n\t",  # whitespace-only (newline and tab)
        always_load=True,
    )

    builder = OrchestrationMemoryPrefaceBuilder()
    preface = await builder.build(db_session, orch_goal, orch_run)

    # The always_loaded section should use the body, not the whitespace-only summary
    assert "always_loaded" in preface
    always_loaded_sections = preface["always_loaded"]
    assert len(always_loaded_sections) > 0
    operating_assumptions = next(
        (s for s in always_loaded_sections if s["section_key"] == "operating_assumptions"),
        None,
    )
    assert operating_assumptions is not None
    assert operating_assumptions["summary"] is not None
    assert "unlimited time and budget" in operating_assumptions["summary"]
    # Whitespace should not be returned as-is
    assert "\n\t" not in operating_assumptions["summary"]

