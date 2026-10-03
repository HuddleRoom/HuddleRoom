import os
import sqlite3
import subprocess
import sys

import pytest


def _alembic(env, *command):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *command], check=False,
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr


def test_migration_038_round_trip_enforces_continuous_and_roadmap_lineage(tmp_path):
    db_path = tmp_path / "migration-038.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "037")
    _alembic(env, "upgrade", "038")

    with sqlite3.connect(db_path) as connection:
        goal_columns = {row[1] for row in connection.execute("PRAGMA table_info(orchestration_goals)")}
        run_columns = {row[1] for row in connection.execute("PRAGMA table_info(orchestration_runs)")}
        reservation_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(orchestration_budget_reservations)")
        }
        goal_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_goals'"
        ).fetchone()[0]
        run_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_runs'"
        ).fetchone()[0]
        reservation_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_budget_reservations'"
        ).fetchone()[0]
        assert {"continuous_policy", "continuous_state", "continuous_origin_key"} <= goal_columns
        assert "cycle_key" in run_columns
        assert "continuous_origin_key" in reservation_columns
        assert "uq_orch_goals_parent_origin_key" in goal_sql
        assert "waiting_activation" in run_sql
        assert "ck_orch_budget_reservation_lineage" in reservation_sql
        assert "roadmap_version_id IS NOT NULL" in goal_sql

        connection.execute(
            "INSERT INTO projects (id, name, config, status) VALUES (?, ?, ?, ?)",
            ("a" * 32, "migration", "{}", "active"),
        )
        for goal_id, parent_id, origin in (("b" * 32, None, None), ("c" * 32, "b" * 32, "slot")):
            connection.execute(
                """INSERT INTO orchestration_goals
                (id, project_id, objective, original_request, success_criteria, constraints, budget,
                 orchestrator_context, status, weight, explicit_multi_work_function, goal_type,
                 parent_goal_id, parent_contract_snapshot, goal_delta, continuous_origin_key,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, '{}', '{}', '{}', '{}', 'active', 'standard', 0, 'continuous',
                        ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                (
                    goal_id, "a" * 32, goal_id, goal_id, parent_id,
                    None if parent_id is None else "{}", None if parent_id is None else "{}", origin,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_goals
                (id, project_id, objective, original_request, success_criteria, constraints, budget,
                 orchestrator_context, status, weight, explicit_multi_work_function, goal_type,
                 parent_goal_id, parent_contract_snapshot, goal_delta, continuous_origin_key,
                 created_at, updated_at)
                VALUES (?, ?, 'duplicate', 'duplicate', '{}', '{}', '{}', '{}', 'active', 'standard', 0,
                        'continuous', ?, '{}', '{}', 'slot', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                ("d" * 32, "a" * 32, "b" * 32),
            )
        connection.execute(
            """INSERT INTO orchestration_goals
            (id, project_id, objective, original_request, success_criteria, constraints, budget,
             orchestrator_context, status, weight, explicit_multi_work_function, goal_type,
             parent_goal_id, roadmap_version_id, roadmap_item_key, parent_contract_snapshot, goal_delta,
             created_at, updated_at)
            VALUES (?, ?, 'legacy', 'legacy', '{}', '{}', '{}', '{}', 'active', 'standard', 0, 'roadmap',
                    ?, ?, 'legacy-item', '{}', '{}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
            ("7" * 32, "a" * 32, "b" * 32, "8" * 32),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_goals
                (id, project_id, objective, original_request, success_criteria, constraints, budget,
                 orchestrator_context, status, weight, explicit_multi_work_function, goal_type,
                 parent_goal_id, roadmap_version_id, roadmap_item_key, parent_contract_snapshot, goal_delta,
                 created_at, updated_at)
                VALUES (?, ?, 'broken legacy', 'broken legacy', '{}', '{}', '{}', '{}', 'active', 'standard',
                        0, 'roadmap', ?, ?, 'broken-item', NULL, '{}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                ("9" * 32, "a" * 32, "b" * 32, "8" * 32),
            )
        connection.execute(
            """INSERT INTO orchestration_runs
            (id, goal_id, status, phase, plan_state, active_blockers, budget_state, retry_state,
             started_at, created_at, updated_at, cycle_key)
            VALUES (?, ?, 'completed', 'completed', '{}', '[]', '{}', '{}', CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '2026-09-06T10:00:00Z')""",
            ("e" * 32, "b" * 32),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_runs
                (id, goal_id, status, phase, plan_state, active_blockers, budget_state, retry_state,
                 started_at, created_at, updated_at, cycle_key)
                VALUES (?, ?, 'completed', 'completed', '{}', '[]', '{}', '{}', CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '2026-09-06T10:00:00Z')""",
                ("f" * 32, "b" * 32),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_budget_reservations
                (id, parent_goal_id, roadmap_item_id, child_goal_id, allocation, settled_spend,
                 measurement_complete, status, created_at, continuous_origin_key)
                VALUES (?, ?, ?, ?, '{}', '{}', 1, 'active', CURRENT_TIMESTAMP, 'slot')""",
                ("1" * 32, "b" * 32, "0" * 32, "c" * 32),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_budget_reservations
                (id, parent_goal_id, child_goal_id, allocation, settled_spend, measurement_complete, status,
                 created_at)
                VALUES (?, ?, ?, '{}', '{}', 1, 'active', CURRENT_TIMESTAMP)""",
                ("2" * 32, "b" * 32, "c" * 32),
            )

    _alembic(env, "downgrade", "037")
    with sqlite3.connect(db_path) as connection:
        goal_columns = {row[1] for row in connection.execute("PRAGMA table_info(orchestration_goals)")}
        run_columns = {row[1] for row in connection.execute("PRAGMA table_info(orchestration_runs)")}
    assert "continuous_policy" not in goal_columns
    assert "cycle_key" not in run_columns
