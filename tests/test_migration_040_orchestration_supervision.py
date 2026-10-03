import os
import sqlite3
import subprocess
import sys
import uuid

import pytest
import sqlalchemy as sa
from pydantic import ValidationError


def _alembic(env: dict[str, str], *command: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def _legacy_rows(connection: sqlite3.Connection) -> dict[str, str]:
    """Seed durable pre-040 facts which must survive the additive upgrade."""
    ids = {name: str(uuid.uuid4()) for name in ("active", "paused", "stopped", "cancelled")}
    for name, goal_status, run_status, continuous_state in (
        ("active", "active", "running", None),
        ("paused", "paused", "paused", None),
        ("stopped", "active", "completed", '{"state":"stopped"}'),
        ("cancelled", "cancelled", "cancelled", None),
    ):
        goal_id = ids[name]
        connection.execute(
            """
            INSERT INTO orchestration_goals (
                id, project_id, objective, original_request, success_criteria, constraints, budget,
                orchestrator_context, status, weight, explicit_multi_work_function, goal_type,
                continuous_state, created_at, updated_at
            ) VALUES (?, ?, ?, ?, '[]', '{}', '{}', '{}', ?, 'standard', 0, 'outcome', ?,
                      '2026-09-09T00:00:00+00:00', '2026-09-09T00:00:00+00:00')
            """,
            (goal_id, str(uuid.uuid4()), name, name, goal_status, continuous_state),
        )
        connection.execute(
            """
            INSERT INTO orchestration_runs (
                id, goal_id, status, event_cursor, plan_state, active_blockers, budget_state,
                retry_state, phase, started_at, completed_at, created_at, updated_at
            ) VALUES (?, ?, ?, NULL, '{"legacy":true}', '[]', '{"tokens":7}', '{"attempt":1}',
                      'baseline', '2026-09-09T00:00:00+00:00', NULL,
                      '2026-09-09T00:00:00+00:00', '2026-09-09T00:00:00+00:00')
            """,
            (str(uuid.uuid4()), goal_id, run_status),
        )
    connection.commit()
    return ids


def test_upgrade_adds_supervision_facts_and_preserves_legacy_control_state(tmp_path):
    """Removing an additive field or rewriting a legacy control state must fail here."""
    db_path = tmp_path / "migration-040.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "039")
    with sqlite3.connect(db_path) as connection:
        ids = _legacy_rows(connection)
    _alembic(env, "upgrade", "040")

    with sqlite3.connect(db_path) as connection:
        inspector = sa.inspect(sa.create_engine(f"sqlite:///{db_path}"))
        run_columns = {column["name"] for column in inspector.get_columns("orchestration_runs")}
        action_columns = {column["name"] for column in inspector.get_columns("orchestration_actions")}
        decision_columns = {
            column["name"] for column in inspector.get_columns("orchestration_authority_decisions")
        }
        memory_columns = {
            column["name"] for column in inspector.get_columns("orchestration_memory_sections")
        }
        wait_columns = {column["name"] for column in inspector.get_columns("orchestration_waits")}

        assert "supervision_state" in run_columns
        assert {"dispatch_contract", "budget_ledger"} <= action_columns
        assert {
            "runtime_identity",
            "contract_version",
            "continuation",
            "continuation_action_id",
            "continuation_applied_at",
        } <= decision_columns
        assert {"fact_status", "provenance"} <= memory_columns
        assert {
            "run_id", "wait_key", "owner", "awaited_event", "due_recheck_at", "fallback",
            "status", "cleared_by_event_id", "created_at", "cleared_at",
        } <= wait_columns
        assert "orchestration_scheduler_state" in inspector.get_table_names()
        column_defaults = {
            table: {
                column["name"]: (column["default"], column["nullable"])
                for column in inspector.get_columns(table)
            }
            for table in (
                "orchestration_runs",
                "orchestration_actions",
                "orchestration_memory_sections",
                "orchestration_waits",
            )
        }
        assert column_defaults["orchestration_runs"]["supervision_state"] == ("'{}'", False)
        assert column_defaults["orchestration_actions"]["dispatch_contract"] == ("'{}'", False)
        assert column_defaults["orchestration_actions"]["budget_ledger"] == ("'{}'", False)
        assert column_defaults["orchestration_memory_sections"]["fact_status"] == ("'unverified'", False)
        assert column_defaults["orchestration_memory_sections"]["provenance"] == ("'{}'", False)
        assert column_defaults["orchestration_waits"]["status"] == ("'open'", False)

        rows = connection.execute(
            """
            SELECT g.id, g.status, g.continuous_state, r.status, r.plan_state, r.retry_state,
                   r.budget_state, r.supervision_state
            FROM orchestration_goals AS g
            JOIN orchestration_runs AS r ON r.goal_id = g.id
            ORDER BY g.objective
            """
        ).fetchall()
        assert rows == [
            (ids["active"], "active", None, "running", '{"legacy":true}', '{"attempt":1}', '{"tokens":7}', '{}'),
            (
                ids["cancelled"], "cancelled", None, "cancelled", '{"legacy":true}',
                '{"attempt":1}', '{"tokens":7}', '{}',
            ),
            (ids["paused"], "paused", None, "paused", '{"legacy":true}', '{"attempt":1}', '{"tokens":7}', '{}'),
            (
                ids["stopped"], "active", '{"state":"stopped"}', "completed", '{"legacy":true}',
                '{"attempt":1}', '{"tokens":7}', '{}',
            ),
        ]


def test_upgrade_round_trip_restores_039_schema(tmp_path):
    """A downgrade must remove only the additive supervision contract."""
    db_path = tmp_path / "migration-040-round-trip.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "040")
    _alembic(env, "downgrade", "039")

    engine = sa.create_engine(f"sqlite:///{db_path}")
    inspector = sa.inspect(engine)
    assert "orchestration_waits" not in inspector.get_table_names()
    assert "orchestration_scheduler_state" not in inspector.get_table_names()
    assert "supervision_state" not in {
        column["name"] for column in inspector.get_columns("orchestration_runs")
    }
    assert not {"dispatch_contract", "budget_ledger"} & {
        column["name"] for column in inspector.get_columns("orchestration_actions")
    }


def test_supervision_constraints_and_defaults_are_enforced_after_upgrade(tmp_path):
    """Changing wait lifecycle or allowing duplicate open waits must fail here."""
    db_path = tmp_path / "migration-040-constraints.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "040")

    with sqlite3.connect(db_path) as connection:
        run_id = str(uuid.uuid4())
        wait_id = str(uuid.uuid4())
        connection.execute(
            """
            INSERT INTO orchestration_waits (
                id, run_id, wait_key, owner, awaited_event, due_recheck_at, fallback, status, created_at
            ) VALUES (?, ?, 'approval', '{}', '{}', '2026-09-09T00:00:00+00:00', '{}', 'open',
                      '2026-09-09T00:00:00+00:00')
            """,
            (wait_id, run_id),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO orchestration_waits (
                    id, run_id, wait_key, owner, awaited_event, due_recheck_at, fallback, status, created_at
                ) VALUES (?, ?, 'approval', '{}', '{}', '2026-09-09T00:00:00+00:00', '{}', 'open',
                          '2026-09-09T00:00:00+00:00')
                """,
                (str(uuid.uuid4()), run_id),
            )
        connection.execute(
            """
            INSERT INTO orchestration_waits (
                id, run_id, wait_key, owner, awaited_event, due_recheck_at, fallback, status, created_at
            ) VALUES (?, ?, 'approval', '{}', '{}', '2026-09-09T00:00:00+00:00', '{}', 'cleared',
                      '2026-09-09T00:00:00+00:00')
            """,
            (str(uuid.uuid4()), run_id),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO orchestration_waits (
                    id, run_id, wait_key, owner, awaited_event, due_recheck_at, fallback, status, created_at
                ) VALUES (?, ?, 'bad-status', '{}', '{}', '2026-09-09T00:00:00+00:00', '{}', 'unknown',
                          '2026-09-09T00:00:00+00:00')
                """,
                (str(uuid.uuid4()), run_id),
            )

        indexes = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'orchestration_waits'"
            )
        }
        assert "WHERE status = 'open'" in indexes["uq_orch_waits_run_key_open"]

        scheduler_columns = {
            row[1]: row[5]
            for row in connection.execute("PRAGMA table_info(orchestration_scheduler_state)")
        }
        assert scheduler_columns["name"] == 1
        connection.execute(
            "INSERT INTO orchestration_scheduler_state (pass_key, updated_at) "
            "VALUES ('slot', '2026-09-09T00:00:00+00:00')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO orchestration_scheduler_state (name, pass_key, updated_at) "
                "VALUES ('other', 'slot', '2026-09-09T00:00:00+00:00')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO orchestration_scheduler_state (name, pass_key, updated_at) "
                "VALUES ('other', 'other-slot', '2026-09-09T00:00:00+00:00')"
            )


def test_models_export_the_same_durable_contract():
    """Dropping model metadata or a cross-dialect partial index must fail here."""
    from huddleroom.models import OrchestrationSchedulerState, OrchestrationWait
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationRun, WAIT_STATUS_VALUES
    from huddleroom.models.orchestration_memory import MEMORY_FACT_STATUS_VALUES, OrchestrationMemorySection
    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision

    assert WAIT_STATUS_VALUES == ("open", "cleared")
    assert MEMORY_FACT_STATUS_VALUES == ("unverified", "accepted", "superseded")
    assert OrchestrationWait.__tablename__ == "orchestration_waits"
    assert OrchestrationSchedulerState.__tablename__ == "orchestration_scheduler_state"
    assert OrchestrationSchedulerState.__table__.c.name.server_default.arg == "supervision"
    assert {"supervision_state"} <= set(OrchestrationRun.__table__.c.keys())
    assert {"dispatch_contract", "budget_ledger"} <= set(OrchestrationAction.__table__.c.keys())
    assert {"fact_status", "provenance"} <= set(OrchestrationMemorySection.__table__.c.keys())
    assert {
        "runtime_identity", "contract_version", "continuation", "continuation_action_id",
        "continuation_applied_at",
    } <= set(OrchestrationAuthorityDecision.__table__.c.keys())

    wait_index = next(
        index for index in OrchestrationWait.__table__.indexes
        if index.name == "uq_orch_waits_run_key_open"
    )
    assert str(wait_index.dialect_options["sqlite"]["where"]) == "status = 'open'"
    assert str(wait_index.dialect_options["postgresql"]["where"]) == "status = 'open'"


@pytest.mark.parametrize(
    "field_name",
    [
        "orchestration_reconcile_interval_seconds",
        "orchestration_event_coalesce_seconds",
        "orchestration_semantic_progress_seconds",
        "orchestration_sweep_goal_limit",
        "orchestration_sweep_seconds_limit",
    ],
)
def test_settings_reject_zero_supervision_limits(field_name):
    """Removing positive validation from any bounded scheduler policy must fail here."""
    from huddleroom.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field_name: 0})
