"""Read-only durable-state oracle for live acceptance runs.

Readers are plain SELECTs ordered by rowid. Nothing in this module writes to the
database under inspection; connections are opened with mode=ro and query_only.

Column ids are stored as 32-char hex (SQLAlchemy Uuid on SQLite). Idempotency and
dedup keys are built by the product from dashed UUID strings, so keys are always
built with _dash(), never _hex().
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

JSON_COLUMNS: dict[str, tuple[str, ...]] = {
    "orchestration_runs": ("plan_state", "active_blockers", "budget_state", "retry_state", "supervision_state"),
    "orchestration_actions": ("request", "dispatch_contract", "budget_ledger"),
    "orchestration_authority_decisions": ("continuation", "options"),
    "orchestration_process_runs": ("input_snapshot", "outputs"),
    "event_log": ("payload",),
    "sessions": ("input_context", "metadata"),
    "tasks": ("depends_on", "trigger", "metadata"),
}

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "orchestration_runs": (
        "id", "goal_id", "status", "pause_requested", "control_version", "baseline_authorized", "phase",
        "plan_state", "active_blockers", "budget_state", "retry_state", "supervision_state",
    ),
    "orchestration_actions": (
        "id", "run_id", "idempotency_key", "action_type", "status", "request", "dispatch_contract",
        "budget_ledger", "created_at",
    ),
    "orchestration_authority_decisions": (
        "id", "goal_id", "run_id", "status", "selected_option", "reason", "decided_by_user_id",
        "continuation_action_id", "continuation_applied_at", "source_process_run_id", "options", "continuation",
    ),
    "orchestration_process_runs": (
        "id", "goal_id", "run_id", "process_type", "status", "outputs", "input_snapshot", "created_at",
    ),
    "event_log": ("seq", "project_id", "event_type", "dedup_key", "payload", "emitted_at"),
    "sessions": ("id", "task_id", "status", "created_at", "input_context", "metadata"),
    "tasks": ("id", "project_id", "status", "metadata", "depends_on", "trigger"),
    # tasks have no goal_id column; roadmap items and metadata.orchestration.goal_id link them to goals.
    "orchestration_roadmap_items": ("goal_id", "task_id"),
}

for _table, _cols in JSON_COLUMNS.items():
    _undeclared = set(_cols) - set(REQUIRED_COLUMNS.get(_table, ()))
    if _undeclared:
        raise RuntimeError(f"JSON columns not declared as required for {_table}: {sorted(_undeclared)}")


class OracleSchemaError(RuntimeError):
    """Raised when the database lacks a table or column the oracle reads."""


class OracleDecodeError(ValueError):
    """Raised when a JSON column holds text that does not parse."""


def _hex(value: Any) -> str:
    return uuid.UUID(str(value)).hex


def _dash(value: Any) -> str:
    return str(uuid.UUID(str(value)))


def _dict_row(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: value for col, value in zip(cursor.description, row)}


def _json_default(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(f"oracle dump cannot encode {type(value).__name__}")


def open_ro(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    # quote() keeps paths containing '?', '#' or spaces intact inside the URI.
    conn = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
    conn.row_factory = _dict_row
    conn.execute("PRAGMA query_only=1")
    return conn


def _canonical_hash(db_path: str | Path) -> str:
    conn = open_ro(db_path)
    try:
        digest = hashlib.sha256()
        for line in conn.iterdump():
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()
    finally:
        conn.close()


def snapshot(db_path: str | Path, dest: str | Path) -> str:
    """Copy db_path to dest via the SQLite backup API; return sha256 of the canonical dump."""
    dest = Path(dest)
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{dest}{suffix}").unlink(missing_ok=True)
    src = open_ro(db_path)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return _canonical_hash(dest)


class Oracle:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        self._conn = open_ro(self.db_path)
        self._check_schema()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Oracle":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check_schema(self) -> None:
        for table, columns in REQUIRED_COLUMNS.items():
            present = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if not present:
                raise OracleSchemaError(f"table {table!r} is missing from {self.db_path}")
            missing = [c for c in columns if c not in present]
            if missing:
                raise OracleSchemaError(f"table {table!r} is missing column(s): {', '.join(missing)}")

    def _rows(self, table: str, clauses: tuple[str, ...] = (), params: tuple = ()) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY rowid"
        rows = self._conn.execute(sql, params).fetchall()
        for row in rows:
            for column in JSON_COLUMNS.get(table, ()):
                raw = row.get(column)
                if isinstance(raw, str):
                    try:
                        row[column] = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise OracleDecodeError(
                            f"{table}.{column} (row id={row.get('id')}) is not valid JSON: {exc}"
                        ) from exc
        return rows

    def run_state(self, run_id: Any) -> dict[str, Any] | None:
        rows = self._rows("orchestration_runs", ("id = ?",), (_hex(run_id),))
        return rows[0] if rows else None

    def runs(self, goal_id: Any) -> list[dict[str, Any]]:
        return self._rows("orchestration_runs", ("goal_id = ?",), (_hex(goal_id),))

    def actions(self, run_id: Any, action_type: str | None = None) -> list[dict[str, Any]]:
        clauses, params = ["run_id = ?"], [_hex(run_id)]
        if action_type is not None:
            clauses.append("action_type = ?")
            params.append(action_type)
        return self._rows("orchestration_actions", tuple(clauses), tuple(params))

    def pause_control(self, run_id: Any, request_id: Any = None) -> list[dict[str, Any]]:
        """Pause acceptance rows; key is run:<run>:pause:<request> with dashed UUIDs."""
        key = f"run:{_dash(run_id)}:pause:"
        if request_id is None:
            return self._rows(
                "orchestration_actions",
                ("run_id = ?", "action_type = 'pause_control'", "substr(idempotency_key, 1, ?) = ?"),
                (_hex(run_id), len(key), key),
            )
        return self._rows(
            "orchestration_actions",
            ("run_id = ?", "action_type = 'pause_control'", "idempotency_key = ?"),
            (_hex(run_id), key + _dash(request_id)),
        )

    def answer_receipt(self, run_id: Any, decision_ids: Any) -> dict[str, Any]:
        """Find the apply_authority_answer receipt for one decision or a batch.

        Key mirrors orchestration_answer_receipts.py: 'apply_authority_answer:' + sorted dashed ids,
        sha256 hex when longer than 255 chars. If no key match, scan the run's receipts whose
        request mentions every dashed id. decision.continuation_action_id is compared against the
        receipt and reported as link_mismatch; it is never used to fail the lookup, because a later
        decision_continuation action legitimately overwrites that link.
        """
        if isinstance(decision_ids, (str, uuid.UUID)):
            decision_ids = [decision_ids]
        dashed = sorted(_dash(d) for d in decision_ids)
        key = "apply_authority_answer:" + ":".join(dashed)
        if len(key) > 255:
            key = "apply_authority_answer:" + hashlib.sha256(key.encode()).hexdigest()
        base = ("run_id = ?", "action_type = 'apply_authority_answer'")
        receipts = self._rows("orchestration_actions", base + ("idempotency_key = ?",), (_hex(run_id), key))
        matched_by = "key" if receipts else None
        if not receipts:
            clauses = base + tuple("instr(request, ?) > 0" for _ in dashed)
            receipts = self._rows("orchestration_actions", clauses, (_hex(run_id), *dashed))
            matched_by = "scan" if receipts else None
        link_mismatch = False
        if receipts:
            found_id = receipts[0]["id"]
            for d in dashed:
                linked = self._conn.execute(
                    "SELECT continuation_action_id FROM orchestration_authority_decisions WHERE id = ?",
                    (_hex(d),),
                ).fetchone()
                if linked is None or linked["continuation_action_id"] != found_id:
                    link_mismatch = True
        return {"receipts": receipts, "matched_by": matched_by, "link_mismatch": link_mismatch}

    def decisions(self, goal_id: Any) -> list[dict[str, Any]]:
        return self._rows("orchestration_authority_decisions", ("goal_id = ?",), (_hex(goal_id),))

    def process_runs(self, goal_id: Any = None, run_id: Any = None) -> list[dict[str, Any]]:
        if (goal_id is None) == (run_id is None):
            raise ValueError("pass exactly one of goal_id or run_id")
        if goal_id is not None:
            return self._rows("orchestration_process_runs", ("goal_id = ?",), (_hex(goal_id),))
        return self._rows("orchestration_process_runs", ("run_id = ?",), (_hex(run_id),))

    def event_log(self, project_id: Any, event_type_prefix: str | None = None) -> list[dict[str, Any]]:
        clauses, params = ["project_id = ?"], [_hex(project_id)]
        if event_type_prefix is not None:
            # substr() instead of LIKE: no wildcard or case-folding surprises.
            clauses.append("substr(event_type, 1, ?) = ?")
            params += [len(event_type_prefix), event_type_prefix]
        return self._rows("event_log", tuple(clauses), tuple(params))

    def event_by_dedup(self, project_id: Any, dedup_key: str) -> list[dict[str, Any]]:
        return self._rows("event_log", ("project_id = ?", "dedup_key = ?"), (_hex(project_id), dedup_key))

    def sessions(self, task_id: Any = None) -> list[dict[str, Any]]:
        if task_id is None:
            return self._rows("sessions")
        return self._rows("sessions", ("task_id = ?",), (_hex(task_id),))

    def tasks(self, goal_id: Any = None) -> list[dict[str, Any]]:
        if goal_id is None:
            return self._rows("tasks")
        return self._rows(
            "tasks",
            ("id IN (SELECT task_id FROM orchestration_roadmap_items WHERE goal_id = ? AND task_id IS NOT NULL) "
             "OR json_extract(metadata, '$.orchestration.goal_id') = ?",),
            (_hex(goal_id), _dash(goal_id)),
        )

    def report_consumed(self, run_id: Any) -> list[dict[str, Any]]:
        prefix = f"run:{_dash(run_id)}:kind:report_consumed:task:"
        return self._rows(
            "orchestration_actions",
            ("run_id = ?", "substr(idempotency_key, 1, ?) = ?"),
            (_hex(run_id), len(prefix), prefix),
        )

    def dump(self, label: str, dest_dir: str | Path, *, run_id: Any = None, goal_id: Any = None,
             project_id: Any = None, task_id: Any = None) -> Path:
        def opt(reader, value):
            return reader(value) if value is not None else None

        # One read transaction so every reader sees the same committed state.
        self._conn.execute("BEGIN")
        try:
            payload = {
                "run_state": opt(self.run_state, run_id),
                "runs": opt(self.runs, goal_id),
                "actions": opt(self.actions, run_id),
                "pause_control": opt(self.pause_control, run_id),
                "report_consumed": opt(self.report_consumed, run_id),
                "decisions": opt(self.decisions, goal_id),
                "process_runs": (
                    self.process_runs(goal_id=goal_id) if goal_id is not None
                    else self.process_runs(run_id=run_id) if run_id is not None
                    else None
                ),
                "event_log": opt(self.event_log, project_id),
                "sessions": opt(self.sessions, task_id),
                "tasks": opt(self.tasks, goal_id),
            }
        finally:
            self._conn.execute("ROLLBACK")
        out_dir = Path(dest_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"oracle-{label}.json"
        text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
        out.write_text(text + "\n", encoding="utf-8")
        return out
