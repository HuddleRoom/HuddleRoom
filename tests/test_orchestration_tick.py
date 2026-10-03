import uuid
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")

from huddleroom.models.project import Project
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.models.event_log import EventLog
from sqlalchemy import func, select


async def _make_run(db, project_id):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Orchestration tick",
            success_criteria=[{"key": "done", "description": "Done"}],
        ),
        created_by_user_id=None,
    )
    return service, goal, run


async def _count_tick_events(db, project_id):
    result = await db.execute(
        select(func.count(EventLog.id)).where(
            EventLog.project_id == project_id,
            EventLog.event_type == "orchestration.tick",
        )
    )
    return result.scalar() or 0


async def _seq(db, bus_event) -> int:
    """Return EventLog.seq for the given BusEvent (id is now a UNIQUE non-PK column)."""
    row = (await db.execute(
        select(EventLog).where(EventLog.id == bus_event.id)
    )).scalar_one()
    return row.seq


@pytest.mark.asyncio
async def test_tick_advances_cursor_and_emits(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)

    # Seed one non-orchestration project event.
    seeded, created = await emit_event_once(
        db_session, test_project.id, "task.created", {"n": 1}, dedup_key="seed-1"
    )
    assert created is True

    result = await service.tick(db_session, run.id)

    assert result["processed_events"] == 1
    assert result["tick_emitted"] is True
    assert result["event_cursor"] == await _seq(db_session, seeded)
    assert run.event_cursor == await _seq(db_session, seeded)
    assert await _count_tick_events(db_session, test_project.id) == 1


@pytest.mark.asyncio
async def test_tick_processes_multiple_events_and_lands_on_last(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)

    seeded = []
    for i in range(3):
        ev, created = await emit_event_once(
            db_session, test_project.id, "task.created", {"n": i}, dedup_key=f"seed-{i}"
        )
        assert created is True
        seeded.append(ev)

    result = await service.tick(db_session, run.id)

    assert result["processed_events"] == 3
    # Cursor must land on the last event, exercising the emitted_at/id ordering
    # and tiebreak branch in _new_events.
    assert result["event_cursor"] == await _seq(db_session, seeded[-1])
    assert run.event_cursor == await _seq(db_session, seeded[-1])

    # A follow-up tick with no new events is a no-op from the multi-event cursor.
    followup = await service.tick(db_session, run.id)
    assert followup["processed_events"] == 0
    assert followup["tick_emitted"] is False


@pytest.mark.asyncio
async def test_repeated_tick_without_new_events_is_noop(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)

    first = await service.tick(db_session, run.id)
    assert first["tick_emitted"] is True
    cursor_after_first = run.event_cursor
    ticks_after_first = await _count_tick_events(db_session, test_project.id)

    second = await service.tick(db_session, run.id)
    assert second["tick_emitted"] is False
    assert second["processed_events"] == 0
    assert run.event_cursor == cursor_after_first
    assert await _count_tick_events(db_session, test_project.id) == ticks_after_first


@pytest.mark.asyncio
async def test_new_events_respects_batch_limit(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)

    seeded = []
    for i in range(3):
        ev, created = await emit_event_once(
            db_session, test_project.id, "task.created", {"n": i}, dedup_key=f"batch-{i}"
        )
        assert created is True
        seeded.append(ev)

    first_batch = await service._new_events(db_session, test_project.id, run.event_cursor, limit=2)
    assert [event.seq for event in first_batch] == [await _seq(db_session, seeded[0]), await _seq(db_session, seeded[1])]

    second_batch = await service._new_events(db_session, test_project.id, first_batch[-1].seq, limit=2)
    assert [event.seq for event in second_batch] == [await _seq(db_session, seeded[2])]


@pytest.mark.asyncio
async def test_tick_ignores_orchestration_events(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)

    # An orchestration.* event must never count as new input work.
    await emit_event_once(
        db_session, test_project.id, "orchestration.something", {}, dedup_key="orch-1"
    )

    result = await service.tick(db_session, run.id)
    assert result["processed_events"] == 0


@pytest.mark.asyncio
async def test_tick_skips_inactive_run(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    run.status = "paused"
    await db_session.flush()

    result = await service.tick(db_session, run.id)
    assert result["tick_emitted"] is False
    assert result["processed_events"] == 0
    assert result["status"] == "paused"
    assert result["agent_definition_review_process"] is None
    assert result["team_hierarchy_process"] is None
    assert await _count_tick_events(db_session, test_project.id) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["completed", "cancelled"])
async def test_tick_skips_terminal_run(db_session, test_project, terminal_status):
    service, goal, run = await _make_run(db_session, test_project.id)
    run.status = terminal_status
    await db_session.flush()

    result = await service.tick(db_session, run.id)
    assert result["tick_emitted"] is False
    assert result["processed_events"] == 0
    assert result["status"] == terminal_status
    assert result["team_hierarchy_process"] is None
    assert await _count_tick_events(db_session, test_project.id) == 0


@pytest.mark.asyncio
async def test_tick_processes_blocked_run(db_session, test_project):
    # "blocked" is a tickable status (spec §6.3), unlike paused/completed/cancelled.
    service, goal, run = await _make_run(db_session, test_project.id)
    run.status = "blocked"
    # Create an actual blocker so status persists across tick (not reset to "running").
    run.active_blockers = [{"kind": "test_blocker", "reason": "Testing blocked status persistence"}]
    await db_session.flush()

    seeded, _ = await emit_event_once(
        db_session, test_project.id, "task.created", {"n": 1}, dedup_key="blocked-1"
    )
    result = await service.tick(db_session, run.id)
    assert result["status"] == "blocked"
    assert result["tick_emitted"] is True
    assert result["processed_events"] == 1
    assert result["event_cursor"] == await _seq(db_session, seeded)


@pytest.mark.asyncio
async def test_tick_unknown_run_raises_404(db_session):
    from fastapi import HTTPException

    service = OrchestrationService()
    with pytest.raises(HTTPException) as exc:
        await service.tick(db_session, uuid.uuid4())
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_tick_missing_goal_raises_404(db_session, test_project, monkeypatch):
    from fastapi import HTTPException

    service, goal, run = await _make_run(db_session, test_project.id)
    original_get = db_session.get

    async def fake_get(model, ident, *args, **kwargs):
        if model.__name__ == "OrchestrationGoal" and ident == run.goal_id:
            return None
        return await original_get(model, ident, *args, **kwargs)

    monkeypatch.setattr(db_session, "get", fake_get)

    with pytest.raises(HTTPException) as exc:
        await service.tick(db_session, run.id)

    assert exc.value.status_code == 404
    assert exc.value.detail == "Orchestration goal not found"


def _goal_payload():
    return {
        "objective": "Ship it",
        "success_criteria": [{"key": "done", "description": "Done"}],
        "constraints": {},
        "budget": {},
    }


@pytest.mark.asyncio
async def test_force_tick_endpoint_first_tick_emits_then_noop(client, test_project):
    created = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json=_goal_payload(),
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["run"]["id"]

    first = await client.post(f"/api/v1/projects/{test_project.id}/orchestration/runs/{run_id}/tick")
    assert first.status_code == 200, first.text
    first_body = first.json()
    assert first_body["run_id"] == run_id
    assert first_body["status"] == "running"
    assert first_body["tick_emitted"] is True

    second = await client.post(f"/api/v1/projects/{test_project.id}/orchestration/runs/{run_id}/tick")
    assert second.status_code == 200
    assert second.json()["tick_emitted"] is False


@pytest.mark.asyncio
async def test_force_tick_unknown_run_returns_404(client):
    resp = await client.post(f"/api/v1/projects/{uuid.uuid4()}/orchestration/runs/{uuid.uuid4()}/tick")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_force_tick_run_from_another_project_returns_404(client, db_session, test_project):
    other_project = Project(name=f"other-{uuid.uuid4()}", description="other", config={})
    db_session.add(other_project)
    await db_session.flush()
    _, _, run = await _make_run(db_session, test_project.id)

    resp = await client.post(f"/api/v1/projects/{other_project.id}/orchestration/runs/{run.id}/tick")

    assert resp.status_code == 404


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_force_tick_requires_auth_when_enabled(client, test_project, auth_headers):
    from huddleroom.config import Settings

    created = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals",
        json=_goal_payload(),
        headers=auth_headers,
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["run"]["id"]

    with patch("huddleroom.dependencies.settings", Settings(auth_enabled=True)):
        resp = await client.post(f"/api/v1/projects/{test_project.id}/orchestration/runs/{run_id}/tick")

    assert resp.status_code == 401
