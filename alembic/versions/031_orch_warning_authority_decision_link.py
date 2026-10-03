"""Link orchestration warnings to authority decisions.

Revision ID: 031
Revises: 030
Create Date: 2026-07-27

"""
# pylint: disable=invalid-name,no-member
from alembic import op
import sqlalchemy as sa


revision = "031"
down_revision = "030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("orchestration_warnings") as batch:
        batch.add_column(sa.Column("related_authority_decision_id", sa.Uuid(), nullable=True))
        batch.create_foreign_key(
            "fk_orch_warnings_related_authority_decision_id",
            "orchestration_authority_decisions",
            ["related_authority_decision_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index(
            "idx_orch_warnings_related_authority_decision",
            ["related_authority_decision_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("orchestration_warnings") as batch:
        batch.drop_index("idx_orch_warnings_related_authority_decision")
        batch.drop_constraint("fk_orch_warnings_related_authority_decision_id", type_="foreignkey")
        batch.drop_column("related_authority_decision_id")
