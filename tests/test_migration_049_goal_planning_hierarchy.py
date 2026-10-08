"""Migration 049 (goal-shape provenance + successor-run lineage). Raw SQL seeds; no ORM writes."""
import os
import sqlite3
import subprocess
import sys
import uuid

import pytest
import sqlalchemy as sa

from huddleroom.models import orchestration as models

NOW = "2026-10-08T00:00:00+00:00"
GOAL_COLUMNS = ("goal_type_source", "goal_type_rationale", "planning_mode", "planning_rationale")
RUN_COLUMNS = (
    "supersedes_run_id", "terminal_reason", "goal_type_snapshot", "planning_mode_snapshot",
    "goal_contract_fingerprint",
)


def _alembic_result(env, *command):
    return subprocess.run(
        [sys.executable, "-m", "alembic", *command], check=False, capture_output=True, text=True, env=env,
    )


def _alembic(env, *command):
    result = _alembic_result(env, *command)
    assert result.returncode == 0, result.stderr


def _env(tmp_path):
    db_path = tmp_path / "migration-049.db"
    return db_path, {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}


def _columns(db_path, table):
    return {column["name"] for column in sa.inspect(sa.create_engine(f"sqlite:///{db_path}")).get_columns(table)}


def _goal(conn, goal_type="outcome", status="active", **extra):
    values = {
        "id": uuid.uuid4().hex, "project_id": uuid.uuid4().hex, "objective": goal_type, "original_request": goal_type,
        "success_criteria": "[]", "constraints": "{}", "budget": "{}", "orchestrator_context": "{}",
        "status": status, "weight": "standard", "explicit_multi_work_function": 0, "goal_type": goal_type,
        "created_at": NOW, "updated_at": NOW, **extra,
    }
    conn.execute(
        f"INSERT INTO orchestration_goals ({', '.join(values)}) VALUES ({', '.join('?' * len(values))})",
        tuple(values.values()),
    )
    return values["id"]


def _run(conn, goal_id, status="running", run_id=None, **extra):
    values = {
        "id": run_id or uuid.uuid4().hex, "goal_id": goal_id, "status": status, "plan_state": '{"legacy":true}',
        "active_blockers": "[]", "budget_state": '{"tokens":7}', "retry_state": "{}", "phase": "baseline",
        "started_at": NOW, "created_at": NOW, "updated_at": NOW, **extra,
    }
    conn.execute(
        f"INSERT INTO orchestration_runs ({', '.join(values)}) VALUES ({', '.join('?' * len(values))})",
        tuple(values.values()),
    )
    return values["id"]


def _action(conn, run_id):
    conn.execute(
        "INSERT INTO orchestration_actions (id, run_id, idempotency_key, action_type, status, request, "
        "created_at, updated_at) VALUES (?, ?, 'k', 'noop', 'completed', '{}', ?, ?)",
        (uuid.uuid4().hex, run_id, NOW, NOW),
    )


def _dump(conn, table):
    conn.row_factory = sqlite3.Row
    return {row["id"]: dict(row) for row in conn.execute(f"SELECT * FROM {table}")}


def _seed_legacy(conn):
    outcome = _goal(conn, "outcome", "active")
    roadmap = _goal(conn, "roadmap", "paused")
    continuous = _goal(conn, "continuous", "active", continuous_policy='{"version":1}')
    for goal_id, status in ((outcome, "running"), (roadmap, "paused"), (continuous, "running")):
        _run(conn, goal_id, status)
    _run(conn, outcome, "cancelled")
    _action(conn, _run(conn, roadmap, "cancelled"))
    conn.commit()
    return {"outcome": outcome, "roadmap": roadmap, "continuous": continuous}


def test_upgrade_preserves_existing_rows_and_backfills_shape(tmp_path):
    db_path, env = _env(tmp_path)
    _alembic(env, "upgrade", "048")
    with sqlite3.connect(db_path) as conn:
        ids = _seed_legacy(conn)
        goals_before, runs_before = _dump(conn, "orchestration_goals"), _dump(conn, "orchestration_runs")
        actions_before = conn.execute("SELECT count(*) FROM orchestration_actions").fetchone()[0]
    _alembic(env, "upgrade", "049")

    with sqlite3.connect(db_path) as conn:
        goals_after, runs_after = _dump(conn, "orchestration_goals"), _dump(conn, "orchestration_runs")
        assert conn.execute("SELECT count(*) FROM orchestration_actions").fetchone()[0] == actions_before
        index_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'uq_orch_runs_one_active_per_goal'"
        ).fetchone()[0]
    assert "WHERE" in index_sql
    mode = {"outcome": "implementation", "roadmap": "roadmap", "continuous": "continuous_policy"}
    assert set(goals_after) == set(goals_before) and set(runs_after) == set(runs_before)
    for goal_id, before in goals_before.items():
        after = goals_after[goal_id]
        assert {key: after[key] for key in before} == before
        assert after["goal_type_source"] == "legacy"
        assert after["goal_type_rationale"] == "Existing goal type preserved by the planning-hierarchy migration."
        assert after["planning_rationale"] == (
            "Planning mode mapped from the existing goal type by the planning-hierarchy migration."
        )
        assert after["planning_mode"] == mode[before["goal_type"]]
    for run_id, before in runs_before.items():
        after = runs_after[run_id]
        assert {key: after[key] for key in before} == before
        goal = goals_after[before["goal_id"]]
        assert (after["goal_type_snapshot"], after["planning_mode_snapshot"]) == (
            goal["goal_type"], goal["planning_mode"],
        )
        assert after["goal_contract_fingerprint"] is after["supersedes_run_id"] is after["terminal_reason"] is None
    assert {goal["goal_type"] for goal in goals_after.values()} == set(ids) and len(runs_after) == 5


@pytest.fixture
def migrated(tmp_path):
    db_path, env = _env(tmp_path)
    _alembic(env, "upgrade", "049")
    conn = sqlite3.connect(db_path)
    yield conn
    conn.close()


def test_goal_constraints_reject_invalid_shape(migrated):
    for extra in (
        {"goal_type_source": "bogus"},
        {"goal_type_rationale": "  "},
        {"planning_rationale": ""},
        {"planning_mode": "bogus"},
        {"goal_type": "roadmap", "planning_mode": "implementation"},
    ):
        with pytest.raises(sqlite3.IntegrityError):
            _goal(migrated, **extra)
    _goal(migrated, "outcome", planning_mode="delivery")
    # Omitted columns take the legacy server defaults.
    default_id = _goal(migrated, "outcome")
    assert migrated.execute(
        "SELECT goal_type_source, planning_mode FROM orchestration_goals WHERE id = ?", (default_id,)
    ).fetchone() == ("legacy", "implementation")


def test_run_constraints_reject_invalid_lineage(migrated):
    goal_id = _goal(migrated)
    run_id = uuid.uuid4().hex
    with pytest.raises(sqlite3.IntegrityError):
        _run(migrated, goal_id, "cancelled", run_id=run_id, supersedes_run_id=run_id)
    old = _run(migrated, goal_id, "cancelled")
    _run(migrated, goal_id, "cancelled", supersedes_run_id=old)
    with pytest.raises(sqlite3.IntegrityError):
        _run(migrated, goal_id, "cancelled", supersedes_run_id=old)
    with pytest.raises(sqlite3.IntegrityError):
        _run(migrated, goal_id, "running", terminal_reason="reclassified")
    with pytest.raises(sqlite3.IntegrityError):
        _run(migrated, goal_id, "cancelled", terminal_reason="bogus")
    _run(migrated, goal_id, "cancelled", terminal_reason="reclassified")
    with pytest.raises(sqlite3.IntegrityError):
        _run(migrated, goal_id, "cancelled", goal_type_snapshot="bogus")
    with pytest.raises(sqlite3.IntegrityError):
        _run(migrated, goal_id, "cancelled", planning_mode_snapshot="bogus")


def test_clean_round_trip_removes_columns(tmp_path):
    db_path, env = _env(tmp_path)
    _alembic(env, "upgrade", "048")
    with sqlite3.connect(db_path) as conn:
        _seed_legacy(conn)
    _alembic(env, "upgrade", "049")
    _alembic(env, "downgrade", "048")
    assert not set(GOAL_COLUMNS) & _columns(db_path, "orchestration_goals")
    assert not set(RUN_COLUMNS) & _columns(db_path, "orchestration_runs")
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM orchestration_runs").fetchone()[0] == 5
        assert "WHERE" in conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'uq_orch_runs_one_active_per_goal'"
        ).fetchone()[0]


def _successor_lineage(conn):
    goal_id = _goal(conn)
    _run(conn, goal_id, "cancelled", run_id=(old := uuid.uuid4().hex))
    _run(conn, goal_id, "running", supersedes_run_id=old)


def _recursive_child(conn):
    parent = _goal(conn, "roadmap", planning_mode="roadmap")
    _goal(
        conn, "roadmap", planning_mode="roadmap", parent_goal_id=parent, roadmap_version_id=uuid.uuid4().hex,
        roadmap_item_key="a", parent_contract_snapshot="{}", goal_delta="{}",
    )


@pytest.mark.parametrize("seed", [_successor_lineage, _recursive_child])
def test_downgrade_refuses_when_new_lineage_exists(tmp_path, seed):
    db_path, env = _env(tmp_path)
    _alembic(env, "upgrade", "049")
    with sqlite3.connect(db_path) as conn:
        seed(conn)
        conn.commit()
    result = _alembic_result(env, "downgrade", "048")
    assert result.returncode != 0 and "049 downgrade refused" in result.stderr
    assert set(GOAL_COLUMNS) <= _columns(db_path, "orchestration_goals")
    assert set(RUN_COLUMNS) <= _columns(db_path, "orchestration_runs")


def test_models_match_migration(migrated, tmp_path):
    db_path = migrated.execute("PRAGMA database_list").fetchone()[2]
    assert set(GOAL_COLUMNS) <= _columns(db_path, "orchestration_goals")
    assert set(RUN_COLUMNS) <= _columns(db_path, "orchestration_runs")
    goal_table, run_table = models.OrchestrationGoal.__table__, models.OrchestrationRun.__table__
    assert set(GOAL_COLUMNS) <= set(goal_table.c.keys()) and set(RUN_COLUMNS) <= set(run_table.c.keys())
    migration_sql = "".join(row[0] or "" for row in migrated.execute(
        "SELECT sql FROM sqlite_master WHERE tbl_name IN ('orchestration_goals', 'orchestration_runs')"
    ))
    for table, names in (
        (goal_table, ("ck_orch_goals_goal_type_source", "ck_orch_goals_planning_mode",
                      "ck_orch_goals_type_rationale_nonblank", "ck_orch_goals_planning_rationale_nonblank",
                      "ck_orch_goals_planning_mode_fits_type")),
        (run_table, ("uq_orch_runs_supersedes_run_id", "ck_orch_runs_not_own_successor",
                     "ck_orch_runs_terminal_reason", "ck_orch_runs_goal_type_snapshot",
                     "ck_orch_runs_planning_mode_snapshot")),
    ):
        assert set(names) <= {constraint.name for constraint in table.constraints}
        assert all(name in migration_sql for name in names)
    assert "fk_orch_runs_supersedes_run_id" in migration_sql
    assert models.PLANNING_MODE_FITS_TYPE_CHECK.replace("'", "").replace(" ", "") in migration_sql.replace(
        "'", ""
    ).replace(" ", "")
