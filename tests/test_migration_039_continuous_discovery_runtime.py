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


def test_migration_039_round_trip_enforces_discovery_lineage(tmp_path):
    db_path = tmp_path / "migration-039.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "038")
    _alembic(env, "upgrade", "039")

    with sqlite3.connect(db_path) as connection:
        reservation_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(orchestration_budget_reservations)")
        }
        candidate_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(orchestration_continuous_candidates)")
        }
        reservation_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_budget_reservations'"
        ).fetchone()[0]
        candidate_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_continuous_candidates'"
        ).fetchone()[0]
        assert {"discovery_run_id", "child_goal_id"} <= reservation_columns
        assert {"parent_goal_id", "source_run_id", "origin_key", "position", "snapshot", "child_goal_id"} <= candidate_columns
        assert "uq_orch_budget_discovery_run" in reservation_sql
        assert "needs_attention" in reservation_sql
        assert "uq_orch_continuous_candidate_parent_origin" in candidate_sql
        assert "uq_orch_continuous_candidate_run_position" in candidate_sql
        assert "uq_orch_continuous_candidate_child" in candidate_sql

        connection.execute(
            "INSERT INTO projects (id, name, config, status) VALUES (?, ?, ?, ?)",
            ("a" * 32, "migration", "{}", "active"),
        )
        connection.execute(
            """INSERT INTO orchestration_goals
            (id, project_id, objective, original_request, success_criteria, constraints, budget,
             orchestrator_context, status, weight, explicit_multi_work_function, goal_type,
             created_at, updated_at)
            VALUES (?, ?, 'parent', 'parent', '{}', '{}', '{}', '{}', 'active', 'standard', 0,
                    'continuous', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
            ("b" * 32, "a" * 32),
        )
        for run_id, key in (("c" * 32, "first"), ("d" * 32, "second")):
            connection.execute(
                """INSERT INTO orchestration_runs
                (id, goal_id, status, phase, plan_state, active_blockers, budget_state, retry_state,
                 started_at, created_at, updated_at, cycle_key)
                VALUES (?, ?, 'completed', 'completed', '{}', '[]', '{}', '{}', CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)""",
                (run_id, "b" * 32, key),
            )
        connection.execute(
            """INSERT INTO orchestration_runs
            (id, goal_id, status, phase, plan_state, active_blockers, budget_state, retry_state,
             started_at, created_at, updated_at, cycle_key)
            VALUES (?, ?, 'completed', 'completed', '{}', '[]', '{}', '{}', CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'third')""",
            ("e" * 32, "b" * 32),
        )
        for reservation_id, roadmap_item_id, origin_key, discovery_run_id, child_goal_id in (
            ("f" * 32, "1" * 32, None, None, "2" * 32),
            ("g" * 32, None, "origin", None, "3" * 32),
            ("h" * 32, None, None, "d" * 32, None),
        ):
            connection.execute(
                """INSERT INTO orchestration_budget_reservations
                (id, parent_goal_id, roadmap_item_id, continuous_origin_key, discovery_run_id,
                 child_goal_id, allocation, settled_spend, measurement_complete, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, '{}', '{}', 1, 'active', CURRENT_TIMESTAMP)""",
                (reservation_id, "b" * 32, roadmap_item_id, origin_key, discovery_run_id, child_goal_id),
            )
        for reservation_id, roadmap_item_id, origin_key, discovery_run_id, child_goal_id in (
            ("i" * 32, None, None, None, None),
            ("j" * 32, "4" * 32, "mixed", None, "5" * 32),
            ("k" * 32, None, "mixed-source", "e" * 32, None),
            ("l" * 32, None, None, "e" * 32, "6" * 32),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    """INSERT INTO orchestration_budget_reservations
                    (id, parent_goal_id, roadmap_item_id, continuous_origin_key, discovery_run_id,
                     child_goal_id, allocation, settled_spend, measurement_complete, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, '{}', '{}', 1, 'active', CURRENT_TIMESTAMP)""",
                    (reservation_id, "b" * 32, roadmap_item_id, origin_key, discovery_run_id, child_goal_id),
                )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_budget_reservations
                (id, parent_goal_id, discovery_run_id, child_goal_id, allocation, settled_spend,
                 measurement_complete, status, settlement_reason, settled_at, created_at)
                VALUES (?, ?, ?, NULL, '{}', '{}', 1, 'settled', 'invalid', CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP)""",
                ("m" * 32, "b" * 32, "e" * 32),
            )
        connection.execute(
            """INSERT INTO orchestration_budget_reservations
            (id, parent_goal_id, discovery_run_id, child_goal_id, allocation, settled_spend,
             measurement_complete, status, settlement_reason, settled_at, created_at)
            VALUES (?, ?, ?, NULL, '{}', '{}', 1, 'settled', 'needs_attention', CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP)""",
            ("e" * 32, "b" * 32, "c" * 32),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_budget_reservations
                (id, parent_goal_id, discovery_run_id, child_goal_id, allocation, settled_spend,
                 measurement_complete, status, created_at)
                VALUES (?, ?, ?, NULL, '{}', '{}', 1, 'active', CURRENT_TIMESTAMP)""",
                ("f" * 32, "b" * 32, "c" * 32),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_budget_reservations
                (id, parent_goal_id, discovery_run_id, child_goal_id, allocation, settled_spend,
                 measurement_complete, status, created_at)
                VALUES (?, ?, ?, ?, '{}', '{}', 1, 'active', CURRENT_TIMESTAMP)""",
                ("f" * 32, "b" * 32, "d" * 32, "b" * 32),
            )
        connection.execute(
            """INSERT INTO orchestration_continuous_candidates
            (id, parent_goal_id, source_run_id, origin_key, position, snapshot, child_goal_id, created_at)
            VALUES (?, ?, ?, 'source:1', 0, '{}', NULL, CURRENT_TIMESTAMP)""",
            ("1" * 32, "b" * 32, "c" * 32),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_continuous_candidates
                (id, parent_goal_id, source_run_id, origin_key, position, snapshot, child_goal_id, created_at)
                VALUES (?, ?, ?, 'source:1', 0, '{}', NULL, CURRENT_TIMESTAMP)""",
                ("2" * 32, "b" * 32, "d" * 32),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_continuous_candidates
                (id, parent_goal_id, source_run_id, origin_key, position, snapshot, child_goal_id, created_at)
                VALUES (?, ?, ?, 'source:2', 0, '{}', NULL, CURRENT_TIMESTAMP)""",
                ("3" * 32, "b" * 32, "c" * 32),
            )
        connection.execute(
            """INSERT INTO orchestration_continuous_candidates
            (id, parent_goal_id, source_run_id, origin_key, position, snapshot, child_goal_id, created_at)
            VALUES (?, ?, ?, 'source:2', 1, '{}', ?, CURRENT_TIMESTAMP)""",
            ("4" * 32, "b" * 32, "d" * 32, "b" * 32),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_continuous_candidates
                (id, parent_goal_id, source_run_id, origin_key, position, snapshot, child_goal_id, created_at)
                VALUES (?, ?, ?, 'source:3', 1, '{}', ?, CURRENT_TIMESTAMP)""",
                ("5" * 32, "b" * 32, "c" * 32, "b" * 32),
            )

    _alembic(env, "downgrade", "038")
    with sqlite3.connect(db_path) as connection:
        reservation_columns = {
            row[1]: row for row in connection.execute("PRAGMA table_info(orchestration_budget_reservations)")
        }
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        reservation_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_budget_reservations'"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO orchestration_budget_reservations
                (id, parent_goal_id, roadmap_item_id, child_goal_id, allocation, settled_spend,
                 measurement_complete, status, settlement_reason, settled_at, created_at)
                VALUES (?, ?, ?, ?, '{}', '{}', 1, 'settled', 'needs_attention', CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP)""",
                ("n" * 32, "b" * 32, "7" * 32, "8" * 32),
            )
    assert "discovery_run_id" not in reservation_columns
    assert reservation_columns["child_goal_id"][3] == 1
    assert "orchestration_continuous_candidates" not in tables
    assert "continuous_origin_key IS NOT NULL" in reservation_sql
    assert "needs_attention" not in reservation_sql
