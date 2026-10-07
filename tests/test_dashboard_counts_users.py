import uuid
import pytest
from huddleroom.models.task import Task
from huddleroom.models.session import Session
from huddleroom.models.meeting import Meeting


@pytest.mark.parametrize("status_filter", ["ready", None])
@pytest.mark.asyncio
async def test_task_count(client, test_project, db_session, status_filter):
    task_status = status_filter if status_filter else "backlog"
    task = Task(project_id=test_project.id, title="count-task", status=task_status)
    db_session.add(task)
    await db_session.flush()

    if status_filter:
        resp = await client.get(f"/api/v1/projects/{test_project.id}/tasks/count?status={status_filter}")
    else:
        resp = await client.get(f"/api/v1/projects/{test_project.id}/tasks/count")

    assert resp.status_code == 200
    assert resp.json()["count"] >= 1


@pytest.mark.asyncio
async def test_session_count(client, test_project, test_agent, db_session):
    session = Session(
        project_id=test_project.id,
        agent_id=test_agent.id,
        adapter_type="api",
        status="running",
        origin="dashboard",
    )
    db_session.add(session)
    await db_session.flush()
    resp = await client.get(f"/api/v1/sessions/count?project_id={test_project.id}&status=running")
    assert resp.status_code == 200
    assert resp.json()["count"] >= 1


@pytest.mark.asyncio
async def test_meeting_count(client, test_project, db_session):
    meeting = Meeting(
        project_id=test_project.id,
        title="count-meeting",
        meeting_type="planning",
        status="active",
        participant_agent_ids=[],
        participant_user_ids=[],
        max_duration_minutes=30,
    )
    db_session.add(meeting)
    await db_session.flush()
    resp = await client.get(f"/api/v1/projects/{test_project.id}/meetings/count?status=active")
    assert resp.status_code == 200
    assert resp.json()["count"] >= 1


@pytest.mark.asyncio
async def test_graph_run_count(client, test_project):
    resp = await client.get(f"/api/v1/projects/{test_project.id}/graph-runs/count")
    assert resp.status_code == 200
    assert resp.json()["count"] >= 0


@pytest.mark.asyncio
async def test_user_list_admin(client, db_session):
    """Anon user (admin) can list users."""
    resp = await client.get("/api/v1/users")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_user_list_non_admin(client, test_user, auth_headers):
    """Non-admin user gets 403."""
    resp = await client.get("/api/v1/users", headers=auth_headers)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_anon_user_seed_for_fk_safety(db_session):
    """Verify anon user can be inserted and used as FK target."""
    import uuid as uuid_mod
    import sqlalchemy as sa
    from huddleroom.models.user import User
    anon_id = uuid_mod.UUID("00000000-0000-0000-0000-000000000000")
    await db_session.execute(
        sa.text(
            "INSERT OR IGNORE INTO users (id, email, hashed_password, display_name, role, is_active, created_at, updated_at) "
            "VALUES (:id, 'anon@local', '', 'Anonymous', 'admin', 1, datetime('now'), datetime('now'))"
        ).bindparams(sa.bindparam("id", value=anon_id, type_=sa.Uuid()))
    )
    await db_session.flush()
    result = await db_session.execute(
        sa.select(User).where(User.id == anon_id)
    )
    user = result.scalar_one_or_none()
    assert user is not None
    assert user.email == "anon@local"
    assert user.role == "admin"
