"""Persist orchestration steering ledger records.

Revision ID: 043
Revises: 042
Create Date: 2026-09-16
"""
# pylint: disable=invalid-name
import sqlalchemy as sa
from alembic import op


revision = "043"
down_revision = "042"
branch_labels = None
depends_on = None


REQUEST_STATES = "'pending', 'being_considered', 'applied', 'deferred', 'rejected', 'superseded', 'needs_clarification', 'withdrawn'"


def upgrade() -> None:
    op.create_table(
        "orchestration_steering_state",
        sa.Column("id", sa.Uuid(), primary_key=True), sa.Column("goal_id", sa.Uuid(), nullable=False),
        sa.Column("inbox_version", sa.Integer(), nullable=False, server_default="0"), sa.Column("direction_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["goal_id"], ["orchestration_goals.id"], name="fk_orch_steering_state_goal_id", ondelete="CASCADE"),
        sa.UniqueConstraint("goal_id", name="uq_orch_steering_state_goal"), sa.CheckConstraint("inbox_version >= 0 AND direction_version >= 0", name="ck_orch_steering_state_versions"),
    )
    op.create_table(
        "orchestration_steering_proposals",
        sa.Column("id", sa.Uuid(), primary_key=True), sa.Column("response_id", sa.Uuid(), nullable=False), sa.Column("goal_id", sa.Uuid(), nullable=False), sa.Column("actor_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="proposed"), sa.Column("draft", sa.JSON(), nullable=False), sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True), sa.Column("promoted_request_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["response_id"], ["orchestration_conversation_responses.id"], name="fk_orch_steering_proposals_response_id", ondelete="CASCADE"), sa.ForeignKeyConstraint(["goal_id"], ["orchestration_goals.id"], name="fk_orch_steering_proposals_goal_id", ondelete="CASCADE"), sa.ForeignKeyConstraint(["actor_id"], ["users.id"], name="fk_orch_steering_proposals_actor_id", ondelete="RESTRICT"),
        sa.UniqueConstraint("response_id", name="uq_orch_steering_proposals_response"), sa.CheckConstraint("status IN ('proposed', 'dismissed', 'promoted')", name="ck_orch_steering_proposals_status"), sa.CheckConstraint("(status = 'proposed' AND dismissed_at IS NULL AND promoted_request_id IS NULL) OR (status = 'dismissed' AND dismissed_at IS NOT NULL AND promoted_request_id IS NULL) OR (status = 'promoted' AND dismissed_at IS NULL AND promoted_request_id IS NOT NULL)", name="ck_orch_steering_proposals_lifecycle"),
    )
    op.create_index("idx_orch_steering_proposals_goal_status", "orchestration_steering_proposals", ["goal_id", "status"])
    op.create_table(
        "orchestration_steering_requests",
        sa.Column("id", sa.Uuid(), primary_key=True), sa.Column("goal_id", sa.Uuid(), nullable=False), sa.Column("actor_id", sa.Uuid(), nullable=False), sa.Column("client_request_id", sa.Uuid(), nullable=False), sa.Column("sequence", sa.Integer(), nullable=False), sa.Column("submitted_run_id", sa.Uuid(), nullable=True),
        sa.Column("directive", sa.String(4000), nullable=False), sa.Column("target_type", sa.String(32), nullable=False), sa.Column("target_id", sa.String(255), nullable=False), sa.Column("scope", sa.String(32), nullable=False), sa.Column("lifetime", sa.String(32), nullable=False), sa.Column("impact_summary", sa.String(1000), nullable=False), sa.Column("source_proposal_id", sa.Uuid(), nullable=True), sa.Column("supersedes_request_id", sa.Uuid(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"), sa.Column("reason_code", sa.String(64), nullable=False), sa.Column("contract_version", sa.String(96), nullable=False), sa.Column("plan_version", sa.String(96), nullable=False), sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False), sa.Column("considered_at", sa.DateTime(timezone=True), nullable=True), sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["goal_id"], ["orchestration_goals.id"], name="fk_orch_steering_requests_goal_id", ondelete="CASCADE"), sa.ForeignKeyConstraint(["actor_id"], ["users.id"], name="fk_orch_steering_requests_actor_id", ondelete="RESTRICT"), sa.ForeignKeyConstraint(["source_proposal_id"], ["orchestration_steering_proposals.id"], name="fk_orch_steering_requests_source_proposal_id", ondelete="SET NULL"), sa.ForeignKeyConstraint(["supersedes_request_id"], ["orchestration_steering_requests.id"], name="fk_orch_steering_requests_supersedes_request_id", ondelete="SET NULL", use_alter=True),
        sa.UniqueConstraint("goal_id", "actor_id", "client_request_id", name="uq_orch_steering_requests_goal_actor_client"), sa.UniqueConstraint("goal_id", "sequence", name="uq_orch_steering_requests_goal_sequence"), sa.CheckConstraint(f"status IN ({REQUEST_STATES})", name="ck_orch_steering_requests_status"), sa.CheckConstraint("target_type IN ('goal', 'plan_item', 'task')", name="ck_orch_steering_requests_target_type"), sa.CheckConstraint("length(directive) BETWEEN 1 AND 4000", name="ck_orch_steering_requests_directive_length"), sa.CheckConstraint("(scope = 'item' AND lifetime = 'selected_item') OR (scope = 'run' AND lifetime = 'remaining_current_run') OR (scope = 'goal' AND lifetime = 'future_runs')", name="ck_orch_steering_requests_scope_lifetime"), sa.CheckConstraint("sequence >= 1", name="ck_orch_steering_requests_sequence"),
    )
    op.create_index("idx_orch_steering_requests_goal_status_sequence", "orchestration_steering_requests", ["goal_id", "status", "sequence"])
    op.create_table(
        "orchestration_steering_transitions",
        sa.Column("id", sa.Uuid(), primary_key=True), sa.Column("request_id", sa.Uuid(), nullable=False), sa.Column("sequence", sa.Integer(), nullable=False), sa.Column("from_status", sa.String(32), nullable=True), sa.Column("to_status", sa.String(32), nullable=False), sa.Column("reason_code", sa.String(64), nullable=False), sa.Column("actor", sa.String(64), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["request_id"], ["orchestration_steering_requests.id"], name="fk_orch_steering_transitions_request_id", ondelete="CASCADE"), sa.UniqueConstraint("request_id", "sequence", name="uq_orch_steering_transitions_request_sequence"), sa.CheckConstraint("sequence >= 1", name="ck_orch_steering_transitions_sequence"), sa.CheckConstraint(f"to_status IN ({REQUEST_STATES}) AND (from_status IS NULL OR from_status IN ({REQUEST_STATES}))", name="ck_orch_steering_transitions_status"),
    )
    op.create_index("idx_orch_steering_transitions_request_sequence", "orchestration_steering_transitions", ["request_id", "sequence"])
    op.create_table(
        "orchestration_steering_result_links",
        sa.Column("id", sa.Uuid(), primary_key=True), sa.Column("request_id", sa.Uuid(), nullable=False), sa.Column("decision_id", sa.Uuid(), nullable=False), sa.Column("action_id", sa.Uuid(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["request_id"], ["orchestration_steering_requests.id"], name="fk_orch_steering_result_links_request_id", ondelete="CASCADE"), sa.ForeignKeyConstraint(["decision_id"], ["orchestration_decisions.id"], name="fk_orch_steering_result_links_decision_id", ondelete="RESTRICT"), sa.ForeignKeyConstraint(["action_id"], ["orchestration_actions.id"], name="fk_orch_steering_result_links_action_id", ondelete="RESTRICT"), sa.UniqueConstraint("request_id", "decision_id", "action_id", name="uq_orch_steering_result_links_request_decision_action"),
    )


def downgrade() -> None:
    op.drop_table("orchestration_steering_result_links")
    op.drop_index("idx_orch_steering_transitions_request_sequence", table_name="orchestration_steering_transitions")
    op.drop_table("orchestration_steering_transitions")
    op.drop_index("idx_orch_steering_requests_goal_status_sequence", table_name="orchestration_steering_requests")
    op.drop_table("orchestration_steering_requests")
    op.drop_index("idx_orch_steering_proposals_goal_status", table_name="orchestration_steering_proposals")
    op.drop_table("orchestration_steering_proposals")
    op.drop_table("orchestration_steering_state")
