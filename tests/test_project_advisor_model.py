import os
import sys
import sqlite3
import subprocess

import pytest


def _alembic(env, *command):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *command],
        check=False, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr


def test_migration_045_round_trip_creates_advisor_table(tmp_path):
    db_path = tmp_path / "project_advisor.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}

    _alembic(env, "upgrade", "044")
    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "project_advisor_turns" not in tables

    _alembic(env, "upgrade", "045")
    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(project_advisor_turns)")}
        assert columns == {
            "id", "project_id", "actor_id", "question", "answer", "citations",
            "off_topic", "tokens_used", "status", "error", "created_at",
        }

    _alembic(env, "downgrade", "044")
    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "project_advisor_turns" not in tables


@pytest.mark.asyncio
async def test_advisor_allowance_used_sums_settled_rows(db_session, test_user, test_project):
    from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn, advisor_allowance_used

    db_session.add_all([
        ProjectAdvisorTurn(
            project_id=test_project.id, actor_id=test_user.id,
            question="q1", answer="a1", status="completed", tokens_used=100,
        ),
        ProjectAdvisorTurn(
            project_id=test_project.id, actor_id=test_user.id,
            question="q2", answer="a2", status="completed", tokens_used=50,
        ),
        ProjectAdvisorTurn(
            project_id=test_project.id, actor_id=test_user.id,
            question="q3", status="pending", tokens_used=999,
        ),
        ProjectAdvisorTurn(
            project_id=test_project.id, actor_id=test_user.id,
            question="q4", status="failed", tokens_used=999, error="boom",
        ),
    ])
    await db_session.flush()

    assert await advisor_allowance_used(db_session, test_project.id, test_user.id) == 1_149


@pytest.mark.asyncio
async def test_advisor_allowance_used_defaults_to_zero_with_no_rows(db_session, test_user, test_project):
    from huddleroom.models.orchestration_advisor import advisor_allowance_used

    assert await advisor_allowance_used(db_session, test_project.id, test_user.id) == 0
