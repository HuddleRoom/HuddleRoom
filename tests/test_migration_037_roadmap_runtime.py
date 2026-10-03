import os
import sqlite3
import subprocess
import sys


def test_migration_037_round_trip(tmp_path):
    db_path = tmp_path / "migration-037.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    for command in (("upgrade", "036"), ("upgrade", "037")):
        result = subprocess.run(
            [sys.executable, "-m", "alembic", *command],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr

    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        goal_columns = {row[1] for row in connection.execute("PRAGMA table_info(orchestration_goals)")}
        goal_indexes = {row[1] for row in connection.execute("PRAGMA index_list(orchestration_goals)")}
        goal_foreign_keys = {row[3] for row in connection.execute("PRAGMA foreign_key_list(orchestration_goals)")}
        version_indexes = {
            row[1] for row in connection.execute("PRAGMA index_list(orchestration_roadmap_versions)")
        }
        item_foreign_keys = {
            row[3] for row in connection.execute("PRAGMA foreign_key_list(orchestration_roadmap_items)")
        }
        reservation_foreign_keys = {
            row[3] for row in connection.execute("PRAGMA foreign_key_list(orchestration_budget_reservations)")
        }
        goal_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_goals'"
        ).fetchone()[0]
        item_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_roadmap_items'"
        ).fetchone()[0]
        reservation_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='orchestration_budget_reservations'"
        ).fetchone()[0]

    assert {"orchestration_roadmap_versions", "orchestration_roadmap_items", "orchestration_budget_reservations"} <= tables
    assert {"parent_goal_id", "roadmap_version_id", "roadmap_item_key", "parent_contract_snapshot", "goal_delta"} <= goal_columns
    assert "idx_orch_goals_parent_status" in goal_indexes
    assert {"parent_goal_id", "roadmap_version_id"} <= goal_foreign_keys
    assert "idx_orch_roadmap_versions_goal_created" in version_indexes
    assert {"goal_id", "first_version_id", "task_id", "child_goal_id", "gate_id"} <= item_foreign_keys
    assert {"parent_goal_id", "roadmap_item_id", "child_goal_id"} <= reservation_foreign_keys
    assert "ck_orch_goals_child_lineage_complete" in goal_sql
    assert "ck_orch_roadmap_items_target_matches_type" in item_sql
    assert "ck_orch_budget_reservations_settlement" in reservation_sql

    downgraded = subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "036"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert downgraded.returncode == 0, downgraded.stderr

    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        goal_columns = {row[1] for row in connection.execute("PRAGMA table_info(orchestration_goals)")}

    assert "orchestration_roadmap_versions" not in tables
    assert "parent_goal_id" not in goal_columns
