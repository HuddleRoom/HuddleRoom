"""Swap event_log PK from UUID id to monotonic seq; change event_cursor to BigInteger.

Revision ID: 019
Revises: 018
Create Date: 2026-07-01 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from huddleroom.config import settings


revision = "019"
down_revision = "018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    op.execute(sa.text("UPDATE orchestration_runs SET event_cursor = NULL"))

    if settings.is_sqlite:
        conn.execute(sa.text("PRAGMA foreign_keys=OFF"))
        try:
            conn.execute(sa.text("""
                CREATE TABLE event_log_new (
                    seq  INTEGER PRIMARY KEY AUTOINCREMENT,
                    id   TEXT NOT NULL,
                    project_id   TEXT NOT NULL,
                    event_type   TEXT NOT NULL,
                    dedup_key    TEXT,
                    payload      JSON NOT NULL DEFAULT '{}',
                    source       TEXT NOT NULL DEFAULT 'system',
                    emitted_at   DATETIME NOT NULL
                )
            """))
            conn.execute(sa.text("""
                INSERT INTO event_log_new
                    (id, project_id, event_type, dedup_key, payload, source, emitted_at)
                SELECT id, project_id, event_type, dedup_key, payload, source, emitted_at
                FROM event_log
                ORDER BY emitted_at, id
            """))
            conn.execute(sa.text("DROP INDEX IF EXISTS idx_event_log_project_emitted"))
            conn.execute(sa.text("DROP INDEX IF EXISTS idx_event_log_event_type"))
            conn.execute(sa.text("DROP INDEX IF EXISTS uq_event_log_project_dedup_key"))
            conn.execute(sa.text("DROP TABLE event_log"))
            conn.execute(sa.text("ALTER TABLE event_log_new RENAME TO event_log"))
            conn.execute(sa.text("CREATE UNIQUE INDEX uq_event_log_id ON event_log (id)"))
            conn.execute(sa.text("CREATE INDEX idx_event_log_project_emitted ON event_log (project_id, emitted_at)"))
            conn.execute(sa.text("CREATE INDEX idx_event_log_event_type ON event_log (event_type)"))
            conn.execute(sa.text("CREATE UNIQUE INDEX uq_event_log_project_dedup_key ON event_log (project_id, dedup_key)"))
            with op.batch_alter_table("orchestration_runs") as batch_op:
                batch_op.alter_column("event_cursor", type_=sa.BigInteger(), existing_type=sa.Uuid(), nullable=True)
        finally:
            conn.execute(sa.text("PRAGMA foreign_keys=ON"))
    else:
        op.add_column("event_log", sa.Column("seq", sa.BigInteger(), nullable=True))
        conn.execute(sa.text("""
            WITH ordered AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY emitted_at, id) AS rn FROM event_log
            )
            UPDATE event_log SET seq = ordered.rn FROM ordered WHERE event_log.id = ordered.id
        """))
        conn.execute(sa.text("CREATE SEQUENCE event_log_seq_seq"))
        conn.execute(sa.text("SELECT setval('event_log_seq_seq', COALESCE((SELECT MAX(seq) FROM event_log), 0) + 1, false)"))
        op.alter_column("event_log", "seq", nullable=False, server_default=sa.text("nextval('event_log_seq_seq')"))
        op.create_unique_constraint("uq_event_log_id", "event_log", ["id"])
        # Drop FK before swapping PK — Postgres rejects dropping a PK while a FK references it.
        # Use dynamic lookup because the constraint was auto-named by Postgres.
        conn.execute(sa.text("""
            DO $$ DECLARE fk text; BEGIN
                SELECT conname INTO fk FROM pg_constraint
                WHERE conrelid='orchestration_evidence'::regclass
                  AND confrelid='event_log'::regclass AND contype='f';
                IF fk IS NOT NULL THEN
                    EXECUTE 'ALTER TABLE orchestration_evidence DROP CONSTRAINT '||quote_ident(fk);
                END IF;
            END $$
        """))
        op.drop_constraint("event_log_pkey", "event_log", type_="primary")
        op.create_primary_key("event_log_pkey", "event_log", ["seq"])
        op.create_foreign_key(
            "orchestration_evidence_observed_event_id_fkey",
            "orchestration_evidence", "event_log",
            ["observed_event_id"], ["id"], ondelete="SET NULL",
        )
        op.alter_column("orchestration_runs", "event_cursor", type_=sa.BigInteger(), existing_type=sa.Uuid(), nullable=True, postgresql_using="NULL::bigint")


def downgrade() -> None:
    conn = op.get_bind()
    op.execute(sa.text("UPDATE orchestration_runs SET event_cursor = NULL"))

    if settings.is_sqlite:
        conn.execute(sa.text("PRAGMA foreign_keys=OFF"))
        try:
            with op.batch_alter_table("orchestration_runs") as batch_op:
                batch_op.alter_column("event_cursor", type_=sa.Uuid(), existing_type=sa.BigInteger(), nullable=True)
            conn.execute(sa.text("""
                CREATE TABLE event_log_old (
                    id          TEXT PRIMARY KEY,
                    project_id  TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    dedup_key   TEXT,
                    payload     JSON NOT NULL DEFAULT '{}',
                    source      TEXT NOT NULL DEFAULT 'system',
                    emitted_at  DATETIME NOT NULL
                )
            """))
            conn.execute(sa.text("""
                INSERT INTO event_log_old
                    (id, project_id, event_type, dedup_key, payload, source, emitted_at)
                SELECT id, project_id, event_type, dedup_key, payload, source, emitted_at
                FROM event_log ORDER BY seq
            """))
            conn.execute(sa.text("DROP INDEX IF EXISTS uq_event_log_id"))
            conn.execute(sa.text("DROP INDEX IF EXISTS idx_event_log_project_emitted"))
            conn.execute(sa.text("DROP INDEX IF EXISTS idx_event_log_event_type"))
            conn.execute(sa.text("DROP INDEX IF EXISTS uq_event_log_project_dedup_key"))
            conn.execute(sa.text("DROP TABLE event_log"))
            conn.execute(sa.text("ALTER TABLE event_log_old RENAME TO event_log"))
            conn.execute(sa.text("CREATE INDEX idx_event_log_project_emitted ON event_log (project_id, emitted_at)"))
            conn.execute(sa.text("CREATE INDEX idx_event_log_event_type ON event_log (event_type)"))
            conn.execute(sa.text("CREATE UNIQUE INDEX uq_event_log_project_dedup_key ON event_log (project_id, dedup_key)"))
        finally:
            conn.execute(sa.text("PRAGMA foreign_keys=ON"))
    else:
        op.alter_column("orchestration_runs", "event_cursor", type_=sa.Uuid(), existing_type=sa.BigInteger(), nullable=True, postgresql_using="NULL::uuid")
        op.drop_constraint("event_log_pkey", "event_log", type_="primary")
        op.create_primary_key("event_log_pkey", "event_log", ["id"])
        # FK is backed by uq_event_log_id; drop it before dropping that constraint, restore after.
        op.drop_constraint(
            "orchestration_evidence_observed_event_id_fkey",
            "orchestration_evidence", type_="foreignkey",
        )
        op.drop_constraint("uq_event_log_id", "event_log", type_="unique")
        op.create_foreign_key(
            "orchestration_evidence_observed_event_id_fkey",
            "orchestration_evidence", "event_log",
            ["observed_event_id"], ["id"], ondelete="SET NULL",
        )
        conn.execute(sa.text("ALTER TABLE event_log ALTER COLUMN seq DROP DEFAULT"))
        conn.execute(sa.text("DROP SEQUENCE IF EXISTS event_log_seq_seq"))
        op.drop_column("event_log", "seq")
