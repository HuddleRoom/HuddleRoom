"""Regression test for repairing the legacy warning agent-review column."""

import os
import sqlite3
import subprocess
import sys


def test_migration_032_repairs_legacy_warning_agent_review_column_and_preserves_value(tmp_path):
    db_path = tmp_path / "migration-032.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    env = {**os.environ, "RALLY_DATABASE_URL": db_url}

    at_031 = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "031"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert at_031.returncode == 0, at_031.stderr

    review_id = "review-value"
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "ALTER TABLE orchestration_warnings "
            "RENAME COLUMN source_agent_review_id TO related_agent_review_id"
        )
        connection.execute("DROP INDEX idx_orch_warnings_source_agent_review")
        connection.execute(
            """
            INSERT INTO orchestration_warnings
                (id, goal_id, warning_type, severity, message,
                 related_agent_review_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("warning-value", "goal-value", "test", "warning", "test", review_id, "now", "now"),
        )

    upgraded = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert upgraded.returncode == 0, upgraded.stderr

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(orchestration_warnings)")}
        assert "source_agent_review_id" in columns
        assert "related_agent_review_id" not in columns
        value = connection.execute(
            "SELECT source_agent_review_id FROM orchestration_warnings WHERE id = ?",
            ("warning-value",),
        ).fetchone()[0]

    assert value == review_id

    downgraded = subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "031"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert downgraded.returncode == 0, downgraded.stderr

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(orchestration_warnings)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(orchestration_warnings)")}
        assert "source_agent_review_id" in columns
        assert "related_agent_review_id" not in columns
        assert "idx_orch_warnings_source_agent_review" in indexes
        value = connection.execute(
            "SELECT source_agent_review_id FROM orchestration_warnings WHERE id = ?",
            ("warning-value",),
        ).fetchone()[0]

    assert value == review_id


def test_migration_032_round_trip_keeps_canonical_warning_agent_review_column(tmp_path):
    db_path = tmp_path / "migration-032-round-trip.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    env = {**os.environ, "RALLY_DATABASE_URL": db_url}

    for command in (("upgrade", "031"), ("upgrade", "032")):
        result = subprocess.run(
            [sys.executable, "-m", "alembic", *command],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(orchestration_warnings)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(orchestration_warnings)")}

    assert "source_agent_review_id" in columns
    assert "related_agent_review_id" not in columns
    assert "idx_orch_warnings_source_agent_review" in indexes

    downgraded = subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "031"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert downgraded.returncode == 0, downgraded.stderr

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(orchestration_warnings)")}
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(orchestration_warnings)")}

    assert "source_agent_review_id" in columns
    assert "related_agent_review_id" not in columns
    assert "idx_orch_warnings_source_agent_review" in indexes
