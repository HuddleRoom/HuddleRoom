"""Test suite for concurrent upsert_section handling.

Tests that OrchestrationMemoryService.upsert_section properly handles
both IntegrityError (check-then-insert race) and OperationalError
(SQLite WAL cross-session race) when two concurrent writers attempt
to create the same (goal_id, section_key).
"""

import asyncio
import uuid

import pytest
import pytest_asyncio

from huddleroom.models.orchestration import OrchestrationGoal
from huddleroom.models.project import Project
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService


@pytest_asyncio.fixture
async def orch_goal(db_session, test_project):
    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Test upsert idempotence",
        success_criteria=[{"key": "done", "description": "Done"}],
    )
    db_session.add(goal)
    await db_session.flush()
    return goal


@pytest.mark.asyncio
async def test_upsert_section_idempotent_same_session(db_session, orch_goal):
    """Verify upsert_section is idempotent when called twice in same session.

    This tests the SAVEPOINT+catch-IntegrityError pattern works correctly
    for the common case where two calls race within a single session context.
    """
    svc = OrchestrationMemoryService()
    section_key = f"idempotent_test_{uuid.uuid4()}"
    project_id = orch_goal.project_id
    goal_id = orch_goal.id

    # First upsert (create)
    result1 = await svc.upsert_section(
        db_session,
        project_id,
        goal_id,
        section_key=section_key,
        title="Section v1",
        body="Body v1",
    )

    # Second upsert with same key (should update)
    result2 = await svc.upsert_section(
        db_session,
        project_id,
        goal_id,
        section_key=section_key,
        title="Section v2",
        body="Body v2",
    )

    # Both should succeed
    assert result1 is not None
    assert result2 is not None

    # Second call should have updated the existing row
    assert result1.id == result2.id
    assert result2.title == "Section v2"
    assert result2.body == "Body v2"

    # Verify only one row exists
    final_section = await svc.get_section(db_session, project_id, goal_id, section_key)
    assert final_section.id == result1.id


@pytest.mark.asyncio
async def test_upsert_section_concurrent_insert(concurrent_sessions):
    """Two sessions race to insert the same (goal_id, section_key).

    Exercises the exception handling in upsert_section by having two
    independent database sessions attempt to create the same
    (goal_id, section_key) pair concurrently. One should create, the other
    should catch IntegrityError/OperationalError and update instead.
    """
    session1, session2 = concurrent_sessions
    svc = OrchestrationMemoryService()

    # Create project and goal using session1, commit so both sessions can see it
    project = Project(
        name="Concurrent Test Project",
        description="For concurrent upsert testing",
        config={},
    )
    session1.add(project)
    await session1.flush()

    goal = OrchestrationGoal(
        project_id=project.id,
        objective="Concurrent insert test",
        success_criteria=[{"key": "done", "description": "Done"}],
    )
    session1.add(goal)
    await session1.commit()

    section_key = "concurrent_insert_section"

    async def upsert_on_session(session, session_num):
        # Commit inline (not deferred until after both race): the atomic
        # upsert holds SQLite's single writer lock until commit, so if both
        # transactions stayed open until after gather(), the winner would
        # never release the lock for the loser to proceed on -- a
        # self-inflicted deadlock, not a real concurrency bug (Finding 6).
        row = await svc.upsert_section(
            session,
            project.id,
            goal.id,
            section_key=section_key,
            title=f"Concurrent Section v{session_num}",
            body=f"Body from concurrent session {session_num}",
        )
        await session.commit()
        return row

    # Race both sessions to insert the same section key
    results = await asyncio.gather(
        upsert_on_session(session1, 1),
        upsert_on_session(session2, 2),
    )

    # Both calls should succeed (neither should raise)
    assert results[0] is not None
    assert results[1] is not None

    # Verify exactly one row exists with this (goal_id, section_key)
    final_section = await svc.get_section(
        session1, project.id, goal.id, section_key
    )
    assert final_section is not None

    # Verify both sessions see the same section
    final_section2 = await svc.get_section(
        session2, project.id, goal.id, section_key
    )
    assert final_section2 is not None
    assert final_section.id == final_section2.id

    # Verify only one section row exists for this goal
    sections = await svc.list_sections(session1, project.id, goal.id)
    assert len(sections) == 1
    assert sections[0].id == final_section.id
