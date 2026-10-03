"""m7_orchestration_gates_evidence_agent_suggestions

Revision ID: 018
Revises: 017
Create Date: 2026-07-01

"""
from alembic import op
import sqlalchemy as sa

revision = "018"
down_revision = "017"
branch_labels = None
depends_on = None


def _column_check(column_name: str, values: tuple[str, ...]) -> str:
    return f"{column_name} IN ({', '.join(repr(value) for value in values)})"


GATE_STATUS_VALUES = ("open", "accepted", "failed")
EVIDENCE_VERDICT_VALUES = ("candidate", "accepted", "rejected")
AGENT_SUGGESTION_STATUS_VALUES = ("open", "accepted", "dismissed")
GATE_STATUS_CHECK = _column_check("status", GATE_STATUS_VALUES)
EVIDENCE_VERDICT_CHECK = _column_check("verdict", EVIDENCE_VERDICT_VALUES)
AGENT_SUGGESTION_STATUS_CHECK = _column_check("status", AGENT_SUGGESTION_STATUS_VALUES)


def upgrade() -> None:
    op.create_table(
        "orchestration_gates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("success_criterion_key", sa.String(length=100), nullable=False),
        sa.Column("gate_type", sa.String(length=100), nullable=False),
        sa.Column("required_evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default=sa.text("'open'")),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(GATE_STATUS_CHECK, name="ck_orchestration_gates_status"),
    )
    op.create_index("idx_orch_gates_run_status", "orchestration_gates", ["run_id", "status"])
    op.create_index(
        "idx_orch_gates_run_criterion",
        "orchestration_gates",
        ["run_id", "success_criterion_key"],
    )

    op.create_table(
        "orchestration_evidence",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("gate_id", sa.Uuid(), sa.ForeignKey("orchestration_gates.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_type", sa.String(length=100), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=True),
        sa.Column("observed_event_id", sa.Uuid(), sa.ForeignKey("event_log.id", ondelete="SET NULL"), nullable=True),
        sa.Column("producer_agent_id", sa.Uuid(), sa.ForeignKey("agents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("verdict", sa.String(length=50), nullable=False, server_default=sa.text("'candidate'")),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(EVIDENCE_VERDICT_CHECK, name="ck_orchestration_evidence_verdict"),
    )
    op.create_index("idx_orch_evidence_gate_created", "orchestration_evidence", ["gate_id", "created_at"])
    op.create_index(
        "idx_orch_evidence_run_source",
        "orchestration_evidence",
        ["run_id", "source_type", "source_id"],
    )
    op.create_index("idx_orch_evidence_observed_event", "orchestration_evidence", ["observed_event_id"])
    op.create_index("idx_orch_evidence_producer_agent", "orchestration_evidence", ["producer_agent_id"])

    op.create_table(
        "orchestration_agent_suggestions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), sa.ForeignKey("orchestration_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("missing_work_function", sa.String(length=100), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("suggested_role", sa.String(length=100), nullable=True),
        sa.Column("suggested_capabilities", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("suggested_adapter_type", sa.String(length=100), nullable=True),
        sa.Column("suggested_model", sa.String(length=100), nullable=True),
        sa.Column("suggested_system_prompt_outline", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False, server_default=sa.text("'open'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(AGENT_SUGGESTION_STATUS_CHECK, name="ck_orchestration_agent_suggestions_status"),
    )
    op.create_index(
        "idx_orch_agent_suggestions_run_status",
        "orchestration_agent_suggestions",
        ["run_id", "status"],
    )
    op.create_index(
        "idx_orch_agent_suggestions_missing_work_function",
        "orchestration_agent_suggestions",
        ["missing_work_function"],
    )


def downgrade() -> None:
    op.drop_table("orchestration_agent_suggestions")
    op.drop_table("orchestration_evidence")
    op.drop_table("orchestration_gates")
