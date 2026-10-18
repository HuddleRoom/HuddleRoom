import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from tests.live_acceptance.oracle import (
    Oracle,
    OracleDecodeError,
    OracleSchemaError,
    _json_default,
    open_ro,
    snapshot,
)

pytestmark = pytest.mark.live

HUDDLE_ROOT = Path(__file__).resolve().parents[2]
NOW = "2026-10-18 12:00:00"


def _id() -> str:
    return uuid.uuid4().hex


def _dash() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "oracle_selfcheck.db"
    env = os.environ.copy()
    env["RALLY_DATABASE_URL"] = f"sqlite+aiosqlite:///{path}"
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=HUDDLE_ROOT, env=env, check=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()
    return path


def _connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _seed(path) -> dict[str, str]:
    """Column ids are hex (as stored); product-built keys and dedup keys use dashed UUIDs."""
    names = ("project", "goal", "agent", "task", "meta_task", "run", "process", "decision", "decision2",
             "session", "pause_action", "admit_action", "consumed_action", "answer_action", "roadmap_item",
             "version")
    ids = {k: _id() for k in names}
    dashed = {k: str(uuid.UUID(ids[k])) for k in ("goal", "run", "task", "decision")}
    dashed["request"] = _dash()
    conn = _connect(path)
    with conn:
        conn.execute("INSERT INTO projects (id, name, config, created_at, updated_at) VALUES (?,?,?,?,?)",
                     (ids["project"], "oracle-selfcheck", "{}", NOW, NOW))
        conn.execute(
            "INSERT INTO orchestration_goals (id, project_id, objective, original_request, success_criteria, "
            "constraints, budget, orchestrator_context, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ids["goal"], ids["project"], "objective", "request", "[]", "{}", "{}", "{}", NOW, NOW))
        conn.execute(
            "INSERT INTO agents (id, name, role, provider, model, capabilities, config, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (ids["agent"], "dev", "developer", "codex", "model", "[]", "{}", NOW, NOW))
        conn.execute(
            "INSERT INTO tasks (id, project_id, title, metadata, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (ids["task"], ids["project"], "task", json.dumps({"k": "v"}), NOW, NOW))
        conn.execute(
            "INSERT INTO tasks (id, project_id, title, metadata, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (ids["meta_task"], ids["project"], "meta task",
             json.dumps({"orchestration": {"goal_id": dashed["goal"]}}), NOW, NOW))
        conn.execute(
            "INSERT INTO sessions (id, task_id, agent_id, project_id, adapter_type, input_context, metadata, "
            "created_at) VALUES (?,?,?,?,?,?,?,?)",
            (ids["session"], ids["task"], ids["agent"], ids["project"], "cli", "{}", json.dumps({"s": 1}), NOW))
        conn.execute(
            "INSERT INTO orchestration_runs (id, goal_id, status, pause_requested, control_version, plan_state, "
            "active_blockers, budget_state, retry_state, supervision_state, phase, baseline_authorized, "
            "started_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ids["run"], ids["goal"], "running", 0, 1, json.dumps({"p": 1}), "[]", "{}", "{}", "{}",
             "baseline", 0, NOW, NOW, NOW))
        run = dashed["run"]
        for key, action_type, idem in (
            ("pause_action", "pause_control", f"run:{run}:pause:{dashed['request']}"),
            ("admit_action", "admit_control_call", f"run:{run}:admit:1"),
            ("consumed_action", "record_event",
             f"run:{run}:kind:report_consumed:task:{dashed['task']}"),
            ("answer_action", "apply_authority_answer",
             f"apply_authority_answer:{dashed['decision']}"),
        ):
            conn.execute(
                "INSERT INTO orchestration_actions (id, run_id, idempotency_key, action_type, request, "
                "dispatch_contract, budget_ledger, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ids[key], ids["run"], idem, action_type, json.dumps({"target": key}),
                 json.dumps({"state": "applied"}), "{}", "completed", NOW, NOW))
        conn.execute(
            "INSERT INTO orchestration_process_runs (id, goal_id, run_id, process_type, process_version, status, "
            "trigger_reason, input_snapshot, outputs, started_at, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (ids["process"], ids["goal"], ids["run"], "planner", 1, "completed", "test", "{}",
             json.dumps({"ok": True}), NOW, NOW, NOW))
        conn.execute(
            "INSERT INTO orchestration_authority_decisions (id, goal_id, run_id, decision_key, title, authority, "
            "question, options, status, continuation, continuation_action_id, asked_at, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ids["decision"], ids["goal"], ids["run"], "dk-1", "title", "human", "q?",
             json.dumps(["approve", "reject"]), "pending", json.dumps({"c": 1}), ids["answer_action"],
             NOW, NOW, NOW))
        conn.execute(
            "INSERT INTO orchestration_authority_decisions (id, goal_id, run_id, decision_key, title, authority, "
            "question, options, status, asked_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (ids["decision2"], ids["goal"], ids["run"], "dk-2", "title2", "human", "q2?",
             json.dumps([]), "pending", NOW, NOW, NOW))
        conn.execute(
            "INSERT INTO event_log (seq, id, project_id, event_type, dedup_key, payload, source, emitted_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (1, _id(), ids["project"], "authority.decision_resolved",
             f"authority.decision_resolved:{dashed['decision']}", json.dumps({"d": 1}), "test", NOW))
        conn.execute(
            "INSERT INTO event_log (seq, id, project_id, event_type, dedup_key, payload, source, emitted_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (2, _id(), ids["project"], "run.paused", "k2", json.dumps({"d": 2}), "test", NOW))
    # Roadmap versions need an artifact chain; this one row is inserted with FK enforcement off.
    conn.execute("PRAGMA foreign_keys=OFF")
    with conn:
        conn.execute(
            "INSERT INTO orchestration_roadmap_items (id, goal_id, first_version_id, item_key, unit_type, "
            "item_snapshot, task_id, gate_id, released_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (ids["roadmap_item"], ids["goal"], ids["version"], "item-1", "task", "{}", ids["task"], _id(), NOW))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.close()
    return {**ids, "run_dashed": dashed["run"], "goal_dashed": dashed["goal"],
            "task_dashed": dashed["task"], "decision_dashed": dashed["decision"],
            "decision2_dashed": str(uuid.UUID(ids["decision2"])),
            "request_dashed": dashed["request"]}


def _add_action(path, run_id, idem, action_type, request=None) -> str:
    action_id = _id()
    conn = _connect(path)
    with conn:
        conn.execute(
            "INSERT INTO orchestration_actions (id, run_id, idempotency_key, action_type, request, "
            "dispatch_contract, budget_ledger, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (action_id, run_id, idem, action_type, json.dumps(request or {}), "{}", "{}", "completed", NOW, NOW))
    conn.close()
    return action_id


def _add_decision(path, ids, n, goal_id, run_id) -> str:
    decision_id = _id()
    conn = _connect(path)
    with conn:
        conn.execute(
            "INSERT INTO orchestration_authority_decisions (id, goal_id, run_id, decision_key, title, authority, "
            "question, options, status, asked_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (decision_id, goal_id, run_id, f"batch-{n}-{decision_id}", "t", "human", "q", "[]", "pending",
             NOW, NOW, NOW))
    conn.close()
    return str(uuid.UUID(decision_id))


def test_readers_return_rowid_order_and_decode_json(db):
    ids = _seed(db)
    with Oracle(db) as oracle:
        run = oracle.run_state(ids["run"])
        assert run["id"] == ids["run"]
        assert run["plan_state"] == {"p": 1}
        assert oracle.run_state(_id()) is None

        assert [r["id"] for r in oracle.runs(ids["goal"])] == [ids["run"]]

        actions = oracle.actions(ids["run"])
        assert [a["id"] for a in actions] == [
            ids["pause_action"], ids["admit_action"], ids["consumed_action"], ids["answer_action"]]
        assert actions[0]["dispatch_contract"] == {"state": "applied"}
        assert actions[0]["request"] == {"target": "pause_action"}
        assert [a["id"] for a in oracle.actions(ids["run"], action_type="admit_control_call")] == [ids["admit_action"]]

        decisions = oracle.decisions(ids["goal"])
        assert [d["id"] for d in decisions] == [ids["decision"], ids["decision2"]]
        assert decisions[0]["options"] == ["approve", "reject"]
        assert decisions[0]["continuation"] == {"c": 1}
        assert decisions[1]["continuation"] is None

        assert oracle.process_runs(goal_id=ids["goal"])[0]["outputs"] == {"ok": True}
        assert oracle.process_runs(run_id=ids["run"])[0]["id"] == ids["process"]
        with pytest.raises(ValueError):
            oracle.process_runs(goal_id=ids["goal"], run_id=ids["run"])

        events = oracle.event_log(ids["project"])
        assert [e["event_type"] for e in events] == ["authority.decision_resolved", "run.paused"]
        assert events[0]["payload"] == {"d": 1}
        assert [e["event_type"] for e in oracle.event_log(ids["project"], "authority.")] == [
            "authority.decision_resolved"]

        assert oracle.sessions(ids["task"])[0]["metadata"] == {"s": 1}
        assert [s["id"] for s in oracle.sessions()] == [ids["session"]]
        assert oracle.tasks()[0]["metadata"] == {"k": "v"}
        assert oracle.tasks(_id()) == []


def test_product_dashed_keys_are_found(db):
    ids = _seed(db)
    with Oracle(db) as oracle:
        assert [a["id"] for a in oracle.report_consumed(ids["run"])] == [ids["consumed_action"]]
        assert [a["id"] for a in oracle.pause_control(ids["run"])] == [ids["pause_action"]]
        assert [a["id"] for a in oracle.pause_control(ids["run"], ids["request_dashed"])] == [ids["pause_action"]]
        assert oracle.pause_control(ids["run"], _id()) == []

        receipt = oracle.answer_receipt(ids["run"], ids["decision_dashed"])
        assert [a["id"] for a in receipt["receipts"]] == [ids["answer_action"]]
        assert receipt["matched_by"] == "key"
        assert receipt["link_mismatch"] is False
        none = oracle.answer_receipt(ids["run"], ids["decision2_dashed"])
        assert none == {"receipts": [], "matched_by": None, "link_mismatch": False}

        found = oracle.event_by_dedup(ids["project"], f"authority.decision_resolved:{ids['decision_dashed']}")
        assert [e["event_type"] for e in found] == ["authority.decision_resolved"]
        assert oracle.event_by_dedup(ids["project"], "authority.decision_resolved:nope") == []


def test_tasks_by_goal_union_of_roadmap_link_and_metadata(db):
    ids = _seed(db)
    with Oracle(db) as oracle:
        assert [t["id"] for t in oracle.tasks(ids["goal"])] == [ids["task"], ids["meta_task"]]


def test_missing_column_raises_clearly(tmp_path, db):
    _seed(db)
    broken = tmp_path / "broken.db"
    snapshot(db, broken)
    conn = sqlite3.connect(broken)
    try:
        conn.execute("ALTER TABLE orchestration_runs DROP COLUMN pause_requested")
        conn.commit()
    except sqlite3.OperationalError as exc:
        pytest.skip(f"sqlite cannot drop column here: {exc}")
    finally:
        conn.close()
    with pytest.raises(OracleSchemaError, match="pause_requested"):
        Oracle(broken)


def test_corrupt_json_names_table_and_column(tmp_path, db):
    ids = _seed(db)
    broken = tmp_path / "corrupt.db"
    snapshot(db, broken)
    conn = sqlite3.connect(broken)
    conn.execute("UPDATE event_log SET payload = '{bad' WHERE event_type = 'run.paused'")
    conn.commit()
    conn.close()
    with Oracle(broken) as oracle:
        with pytest.raises(OracleDecodeError, match="event_log.payload"):
            oracle.event_log(ids["project"])


def test_dump_writes_all_readers_and_rejects_unknown_types(db, tmp_path):
    ids = _seed(db)
    with Oracle(db) as oracle:
        path = oracle.dump("selfcheck", tmp_path / "dump", run_id=ids["run"], goal_id=ids["goal"],
                           project_id=ids["project"], task_id=ids["task"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "oracle-selfcheck.json"
    assert data["run_state"]["id"] == ids["run"]
    assert len(data["decisions"]) == 2
    assert len(data["tasks"]) == 2
    assert len(data["report_consumed"]) == 1
    with pytest.raises(TypeError):
        _json_default(object())


def test_snapshot_under_open_write_txn_is_stable(db, tmp_path):
    ids = _seed(db)
    writer = _connect(db)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO event_log (seq, id, project_id, event_type, dedup_key, payload, source, emitted_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (3, _id(), ids["project"], "uncommitted", "k3", "{}", "test", NOW))

        first = snapshot(db, tmp_path / "snap-a.db")
        second = snapshot(db, tmp_path / "snap-b.db")
        assert first == second

        with Oracle(tmp_path / "snap-a.db") as snap:
            assert [r["id"] for r in snap.runs(ids["goal"])] == [ids["run"]]
            assert all(e["event_type"] != "uncommitted" for e in snap.event_log(ids["project"]))
    finally:
        writer.rollback()
        writer.close()


def test_wal_readers_see_last_committed_state(db):
    ids = _seed(db)
    writer = _connect(db)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO event_log (seq, id, project_id, event_type, dedup_key, payload, source, emitted_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (3, _id(), ids["project"], "pending.only", "k3", "{}", "test", NOW))
        with Oracle(db) as oracle:
            assert [e["event_type"] for e in oracle.event_log(ids["project"])] == [
                "authority.decision_resolved", "run.paused"]
            writer.commit()
            assert [e["event_type"] for e in oracle.event_log(ids["project"])][-1] == "pending.only"
    finally:
        writer.close()


def test_answer_receipt_survives_continuation_overwrite(db):
    ids = _seed(db)
    cont = _add_action(db, ids["run"], f"run:{ids['run_dashed']}:decision_continuation:x", "decision_continuation")
    conn = _connect(db)
    with conn:
        conn.execute("UPDATE orchestration_authority_decisions SET continuation_action_id = ? WHERE id = ?",
                     (cont, ids["decision"]))
    conn.close()
    with Oracle(db) as oracle:
        r = oracle.answer_receipt(ids["run"], ids["decision_dashed"])
    assert [a["id"] for a in r["receipts"]] == [ids["answer_action"]]
    assert r["matched_by"] == "key"
    assert r["link_mismatch"] is True


def test_answer_receipt_batch_is_order_insensitive(db):
    ids = _seed(db)
    d1, d2 = ids["decision_dashed"], ids["decision2_dashed"]
    key = "apply_authority_answer:" + ":".join(sorted([d1, d2]))
    batch = _add_action(db, ids["run"], key, "apply_authority_answer", {"decisions": [d1, d2]})
    with Oracle(db) as oracle:
        r = oracle.answer_receipt(ids["run"], [d2, d1])
    assert [a["id"] for a in r["receipts"]] == [batch]
    assert r["matched_by"] == "key"
    assert r["link_mismatch"] is True


def test_answer_receipt_hashed_key_for_large_batch(db):
    ids = _seed(db)
    decisions = [_add_decision(db, ids, n, ids["goal"], ids["run"]) for n in range(8)]
    plain = "apply_authority_answer:" + ":".join(sorted(decisions))
    assert len(plain) > 255
    key = "apply_authority_answer:" + hashlib.sha256(plain.encode()).hexdigest()
    action = _add_action(db, ids["run"], key, "apply_authority_answer", {})
    with Oracle(db) as oracle:
        r = oracle.answer_receipt(ids["run"], decisions)
    assert [a["id"] for a in r["receipts"]] == [action]
    assert r["matched_by"] == "key"


def test_answer_receipt_scan_fallback(db):
    ids = _seed(db)
    d2 = ids["decision2_dashed"]  # no key-matched receipt exists for this decision
    action = _add_action(db, ids["run"], "custom-key", "apply_authority_answer", {"decision_ids": [d2]})
    with Oracle(db) as oracle:
        r = oracle.answer_receipt(ids["run"], d2)
    assert [a["id"] for a in r["receipts"]] == [action]
    assert r["matched_by"] == "scan"


def test_open_ro_rejects_writes(db):
    _seed(db)
    conn = open_ro(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO projects (id, name, config, created_at, updated_at) "
                         "VALUES (?,?,?,?,?)", (_id(), "x", "{}", NOW, NOW))
    finally:
        conn.close()
