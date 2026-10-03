import os
import subprocess
import sys

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError


EXPECTED_GATE_STATUS_CHECK = "status IN ('open', 'accepted', 'failed')"
EXPECTED_EVIDENCE_VERDICT_CHECK = "verdict IN ('candidate', 'accepted', 'rejected')"
EXPECTED_AGENT_SUGGESTION_STATUS_CHECK = "status IN ('open', 'accepted', 'dismissed')"


def _foreign_key_contracts(inspector, table_name):
    return {
        tuple(foreign_key["constrained_columns"]): {
            "referred_table": foreign_key["referred_table"],
            "referred_columns": tuple(foreign_key["referred_columns"]),
            "ondelete": foreign_key.get("options", {}).get("ondelete"),
        }
        for foreign_key in inspector.get_foreign_keys(table_name)
    }


def _check_sqltexts(inspector, table_name):
    return {
        check["name"]: check.get("sqltext")
        for check in inspector.get_check_constraints(table_name)
    }


def _indexes(inspector, table_name):
    return {
        index["name"]: tuple(index["column_names"])
        for index in inspector.get_indexes(table_name)
    }


async def _create_goal_run(db_session, project_id, objective, success_criteria):
    from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun

    goal = OrchestrationGoal(
        project_id=project_id,
        objective=objective,
        success_criteria=success_criteria,
        constraints={},
        budget={},
    )
    db_session.add(goal)
    await db_session.flush()

    run = OrchestrationRun(
        goal_id=goal.id,
        event_cursor=None,
        plan_state={},
        active_blockers=[],
        budget_state={},
        retry_state={},
    )
    db_session.add(run)
    await db_session.flush()
    return goal, run


def test_migration_018_creates_orchestration_gates_schema_contract(tmp_path):
    db_path = tmp_path / "migrated.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "RALLY_DATABASE_URL": db_url},
    )
    assert result.returncode == 0, result.stderr

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        inspector = sa.inspect(conn)
        assert {
            "orchestration_gates",
            "orchestration_evidence",
            "orchestration_agent_suggestions",
        } <= set(inspector.get_table_names())

        gate_columns = {column["name"] for column in inspector.get_columns("orchestration_gates")}
        assert gate_columns == {
            "id",
            "run_id",
            "success_criterion_key",
            "gate_type",
            "required_evidence",
            "status",
            "failure_reason",
            "created_at",
            "updated_at",
            "accepted_at",
            "failed_at",
        }

        evidence_columns = {column["name"] for column in inspector.get_columns("orchestration_evidence")}
        assert evidence_columns == {
            "id",
            "run_id",
            "gate_id",
            "source_type",
            "source_id",
            "observed_event_id",
            "producer_agent_id",
            "verdict",
            "metadata",
            "created_at",
            "updated_at",
        }

        suggestion_columns = {
            column["name"] for column in inspector.get_columns("orchestration_agent_suggestions")
        }
        assert suggestion_columns == {
            "id",
            "run_id",
            "missing_work_function",
            "reason",
            "suggested_role",
            "suggested_capabilities",
            "suggested_adapter_type",
            "suggested_model",
            "suggested_system_prompt_outline",
            "status",
            "created_at",
            "updated_at",
        }

        gate_fks = _foreign_key_contracts(inspector, "orchestration_gates")
        evidence_fks = _foreign_key_contracts(inspector, "orchestration_evidence")
        suggestion_fks = _foreign_key_contracts(inspector, "orchestration_agent_suggestions")
        assert gate_fks == {
            ("run_id",): {
                "referred_table": "orchestration_runs",
                "referred_columns": ("id",),
                "ondelete": "CASCADE",
            },
        }
        assert evidence_fks == {
            ("run_id",): {
                "referred_table": "orchestration_runs",
                "referred_columns": ("id",),
                "ondelete": "CASCADE",
            },
            ("gate_id",): {
                "referred_table": "orchestration_gates",
                "referred_columns": ("id",),
                "ondelete": "CASCADE",
            },
            ("observed_event_id",): {
                "referred_table": "event_log",
                "referred_columns": ("id",),
                "ondelete": "SET NULL",
            },
            ("producer_agent_id",): {
                "referred_table": "agents",
                "referred_columns": ("id",),
                "ondelete": "SET NULL",
            },
        }
        assert suggestion_fks == {
            ("run_id",): {
                "referred_table": "orchestration_runs",
                "referred_columns": ("id",),
                "ondelete": "CASCADE",
            },
        }

        gate_checks = _check_sqltexts(inspector, "orchestration_gates")
        evidence_checks = _check_sqltexts(inspector, "orchestration_evidence")
        suggestion_checks = _check_sqltexts(inspector, "orchestration_agent_suggestions")
        assert "ck_orchestration_gates_status" in gate_checks
        assert "ck_orchestration_evidence_verdict" in evidence_checks
        assert "ck_orchestration_agent_suggestions_status" in suggestion_checks
        if gate_checks["ck_orchestration_gates_status"] is not None:
            assert gate_checks["ck_orchestration_gates_status"] == EXPECTED_GATE_STATUS_CHECK
        if evidence_checks["ck_orchestration_evidence_verdict"] is not None:
            assert evidence_checks["ck_orchestration_evidence_verdict"] == EXPECTED_EVIDENCE_VERDICT_CHECK
        if suggestion_checks["ck_orchestration_agent_suggestions_status"] is not None:
            assert (
                suggestion_checks["ck_orchestration_agent_suggestions_status"]
                == EXPECTED_AGENT_SUGGESTION_STATUS_CHECK
            )

        gate_indexes = _indexes(inspector, "orchestration_gates")
        evidence_indexes = _indexes(inspector, "orchestration_evidence")
        suggestion_indexes = _indexes(inspector, "orchestration_agent_suggestions")
        assert gate_indexes == {
            "idx_orch_gates_run_status": ("run_id", "status"),
            "idx_orch_gates_run_criterion": ("run_id", "success_criterion_key"),
        }
        assert evidence_indexes == {
            "idx_orch_evidence_gate_created": ("gate_id", "created_at"),
            "idx_orch_evidence_run_source": ("run_id", "source_type", "source_id"),
            "idx_orch_evidence_observed_event": ("observed_event_id",),
            "idx_orch_evidence_producer_agent": ("producer_agent_id",),
        }
        assert suggestion_indexes == {
            "idx_orch_agent_suggestions_run_status": ("run_id", "status"),
            "idx_orch_agent_suggestions_missing_work_function": ("missing_work_function",),
        }


def test_orchestration_gates_model_timestamps_match_migration_timezone_contract():
    from huddleroom.models.orchestration import OrchestrationAgentSuggestion, OrchestrationEvidence, OrchestrationGate

    assert OrchestrationEvidence.__mapper__.attrs.evidence_metadata.columns[0].name == "metadata"
    assert OrchestrationGate.__table__.c.created_at.type.timezone is True
    assert OrchestrationGate.__table__.c.updated_at.type.timezone is True
    assert OrchestrationGate.__table__.c.accepted_at.type.timezone is True
    assert OrchestrationGate.__table__.c.failed_at.type.timezone is True
    assert OrchestrationEvidence.__table__.c.created_at.type.timezone is True
    assert OrchestrationEvidence.__table__.c.updated_at.type.timezone is True
    assert OrchestrationAgentSuggestion.__table__.c.created_at.type.timezone is True
    assert OrchestrationAgentSuggestion.__table__.c.updated_at.type.timezone is True


@pytest.mark.asyncio
async def test_gate_evidence_and_agent_suggestion_records_link_to_run(db_session, test_project, test_agent):
    from huddleroom.models.orchestration import OrchestrationAgentSuggestion, OrchestrationEvidence, OrchestrationGate

    _, run = await _create_goal_run(
        db_session,
        project_id=test_project.id,
        objective="Ship proof records",
        success_criteria=[{"key": "reviewed", "description": "Independent review evidence exists."}],
    )

    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="reviewed",
        gate_type="review_accepted",
        required_evidence={"min_count": 1, "required_source_types": ["task"]},
    )
    db_session.add(gate)
    await db_session.flush()

    evidence = OrchestrationEvidence(
        run_id=run.id,
        gate_id=gate.id,
        source_type="task",
        source_id=None,
        observed_event_id=None,
        producer_agent_id=test_agent.id,
        verdict="accepted",
        evidence_metadata={"summary": "Review approved."},
    )
    suggestion = OrchestrationAgentSuggestion(
        run_id=run.id,
        missing_work_function="validation",
        reason="No active agent has independent validation capability.",
        suggested_role="validator",
        suggested_capabilities=["tests", "review"],
        suggested_adapter_type="api",
        suggested_model="gpt-4o-mini",
        suggested_system_prompt_outline="Validate completed work without producing implementation artifacts.",
    )
    db_session.add_all([evidence, suggestion])
    await db_session.flush()

    assert gate.run_id == run.id
    assert evidence.run_id == run.id
    assert evidence.gate_id == gate.id
    assert evidence.producer_agent_id == test_agent.id
    assert evidence.evidence_metadata == {"summary": "Review approved."}
    assert suggestion.run_id == run.id
    assert suggestion.suggested_capabilities == ["tests", "review"]


@pytest.mark.asyncio
async def test_orchestration_gate_status_constraints(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationAgentSuggestion, OrchestrationEvidence, OrchestrationGate

    _, run = await _create_goal_run(
        db_session,
        project_id=test_project.id,
        objective="Reject invalid proof record statuses",
        success_criteria=[{"key": "proof", "description": "Proof records reject invalid states."}],
    )

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                OrchestrationGate(
                    run_id=run.id,
                    success_criterion_key="proof",
                    gate_type="validation_passed",
                    required_evidence={},
                    status="not-real",
                )
            )
            await db_session.flush()

    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="proof",
        gate_type="validation_passed",
        required_evidence={},
    )
    db_session.add(gate)
    await db_session.flush()

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                OrchestrationEvidence(
                    run_id=run.id,
                    gate_id=gate.id,
                    source_type="task",
                    source_id=None,
                    observed_event_id=None,
                    producer_agent_id=None,
                    verdict="not-real",
                    evidence_metadata={},
                )
            )
            await db_session.flush()

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                OrchestrationAgentSuggestion(
                    run_id=run.id,
                    missing_work_function="validation",
                    reason="No safe fit exists.",
                    suggested_role="validator",
                    suggested_capabilities=["tests"],
                    suggested_adapter_type="api",
                    suggested_model="gpt-4o-mini",
                    suggested_system_prompt_outline="Validate work independently.",
                    status="not-real",
                )
            )
            await db_session.flush()
