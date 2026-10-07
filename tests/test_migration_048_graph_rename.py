"""Migration 048 (protocol -> graph rename). Raw SQL only; must not import huddleroom.models."""
import json
import os
import sqlite3
import subprocess
import sys
import uuid

OLD_TABLES = ("protocols", "protocol_instances", "protocol_transitions", "protocol_timeouts")
NEW_TABLES = ("graphs", "graph_runs", "graph_run_steps", "graph_run_timeouts")
NOW = "2026-10-07T00:00:00+00:00"


def _alembic(env, *command):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *command],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def _id():
    return uuid.uuid4().hex


def _insert(conn, table, **values):
    """Insert a row, filling any other NOT NULL column without a default."""
    for _, name, ctype, notnull, default, pk in conn.execute(f"PRAGMA table_info({table})").fetchall():
        if name in values or not notnull or default is not None or (pk and ctype == "INTEGER"):
            continue
        ctype = ctype.upper()
        values[name] = (
            "{}" if "JSON" in ctype else NOW if "DATE" in ctype else 0 if ctype in ("INTEGER", "BOOLEAN") else "x"
        )
    values = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in values.items()}
    conn.execute(
        f"INSERT INTO {table} ({', '.join(f'"{k}"' for k in values)}) VALUES ({', '.join('?' * len(values))})",
        tuple(values.values()),
    )


def _seed(conn):
    ids = {name: _id() for name in ("proj", "graph", "run", "step", "timeout", "task", "session",
                                     "meeting", "agenda", "action_item", "ki", "user", "oruns",
                                     "oaction", "oevidence", "decision", "chain", "plain_user",
                                     "bare_graph", "plain_meeting", "plain_run", "disp_run")}
    definition = {
        "name": "bug_fix",
        "initial_state": "triage",
        "terminal_states": {"success": ["done"], "failure": []},
        "states": {
            "triage": {
                "on_enter": [{"action_type": "update_protocol_context", "note": "protocol notes stay"}],
                "transitions": [
                    {
                        "to": "done",
                        "name": "finish",
                        "actions": [
                            {"message": "{{protocol_name}} in {{ current_state }} for {{protocol_instance.id}}"
                                        " see {{dashboard_url}}/protocol-instances/{{protocol_instance.id}}"},
                        ],
                    }
                ],
            },
            "done": {},
        },
    }
    _insert(conn, "protocols", id=ids["graph"], project_id=ids["proj"], name="bug_fix",
            definition=definition, triggers=[{"event_type": "protocol.completed"}],
            loaded_from="/ws/protocols/bug_fix.yaml", created_at=NOW, updated_at=NOW)
    _insert(conn, "protocol_instances", id=ids["run"], protocol_id=ids["graph"], project_id=ids["proj"],
            current_state="triage", last_transitioned_at=NOW, context={"protocol_name": "bug_fix"},
            started_at=NOW, created_at=NOW, updated_at=NOW)
    _insert(conn, "protocol_transitions", id=ids["step"], protocol_instance_id=ids["run"], from_state="",
            to_state="triage", transition_name="manual_advance", transitioned_at=NOW)
    _insert(conn, "protocol_timeouts", id=ids["timeout"], protocol_instance_id=ids["run"],
            state_name="triage", timeout_action="escalate", expires_at=NOW, created_at=NOW)
    _insert(conn, "tasks", id=ids["task"], project_id=ids["proj"], title="Protocol task",
            description="the protocol handbook is free text", protocol_instance_id=ids["run"],
            metadata={"protocol": "bug_fix", "origin": "protocol"})
    _insert(conn, "sessions", id=ids["session"], agent_id=_id(), project_id=ids["proj"],
            adapter_type="x", origin="protocol", protocol_instance_id=ids["run"],
            metadata={"protocol_instance_id": ids["run"]})
    _insert(conn, "meetings", id=ids["meeting"], project_id=ids["proj"], title="m", meeting_type="x",
            participant_agent_ids=[], participant_user_ids=[], source_protocol_instance_id=ids["run"],
            created_at=NOW, updated_at=NOW)
    _insert(conn, "meeting_agenda_items", id=ids["agenda"], meeting_id=ids["meeting"], creates_protocol=1)
    _insert(conn, "meeting_action_items", id=ids["action_item"], meeting_id=ids["meeting"],
            creates_protocol=1, protocol_instance_id=ids["run"])
    _insert(conn, "knowledge_items", id=ids["ki"], content="c", content_type="note",
            provenance_type="protocol", provenance_protocol_instance_id=ids["run"],
            metadata={"protocol_id": ids["graph"]})
    _insert(conn, "users", id=ids["user"], email="protocol-system@local", display_name="Protocol System")
    _insert(conn, "users", id=ids["plain_user"], email="bob@example.com", display_name="Protocol Bob")
    _insert(conn, "orchestration_runs", id=ids["oruns"], goal_id=_id(),
            supervision_state={"protocols": [{"protocol_id": "p"}], "protocol_instances": [], "note": "protocol"})
    _insert(conn, "orchestration_actions", id=ids["oaction"], run_id=ids["oruns"], idempotency_key="k",
            action_type="start_protocol", target_type="protocol_instance",
            request={"protocol_id": ids["graph"], "subject_type": "task"},
            dispatch_contract={"allowed": {"start_protocol": ["protocol_id"]}})
    _insert(conn, "orchestration_evidence", id=ids["oevidence"], run_id=ids["oruns"], gate_id=_id(),
            source_type="protocol_transition",
            metadata={"protocol_transition_id": ids["step"], "from_state": "a", "to_state": "b"})
    _insert(conn, "orchestration_decisions", id=ids["decision"], run_id=ids["oruns"],
            parsed_decision={"disposition": "protocol", "action_type": "start_protocol"},
            input_snapshot={"protocol_instances": [{"current_state": "x", "protocol_instance_id": "1"}]})
    _insert(conn, "escalation_chains", id=ids["chain"],
            definition={"steps": [{"message": "{{protocol_instance.id}} {{protocol_name}} protocol"}]},
            steps=[{"message": "{{ current_state }}"}])
    _insert(conn, "event_log", id=_id(), project_id=ids["proj"], event_type="protocol.state_transitioned",
            source="protocol",
            payload={"protocol_instance_id": ids["run"], "from_state": "a", "to_state": "b",
                     "protocol_instance": {"protocol_id": "x", "current_state": "b"},
                     "note": "the protocol was fine", "initial_state": "a"})
    # No "protocol" word anywhere: must still be converted (graphs.definition is unconditional).
    _insert(conn, "protocols", id=ids["bare_graph"], project_id=ids["proj"], name="bare",
            definition={"initial_state": "a", "terminal_states": {"success": ["b"]},
                        "states": {"a": {"transitions": [{"to": "b"}]}, "b": {}}},
            created_at=NOW, updated_at=NOW)
    # current_state outside graph context (meeting/orchestration meaning) must stay untouched.
    _insert(conn, "meetings", id=ids["plain_meeting"], project_id=ids["proj"], title="m2", meeting_type="x",
            participant_agent_ids=[], participant_user_ids=[], created_at=NOW, updated_at=NOW,
            resume_state={"current_state": "paused", "from_state": "x"})
    _insert(conn, "orchestration_runs", id=ids["plain_run"], goal_id=_id(),
            plan_state={"current_state": "planning", "transition_name": "t"})
    _insert(conn, "event_log", id=_id(), project_id=ids["proj"], event_type="meeting.state",
            source="system", payload={"current_state": "active", "to_state": "done"})
    _insert(conn, "event_log", id=_id(), project_id=ids["proj"], event_type="task.created", source="system",
            payload={"title": "plain", "current_state": "kept"})
    # Stored supervision disposition: action_type is rewritten only under "disposition".
    _insert(conn, "orchestration_runs", id=ids["disp_run"], goal_id=_id(),
            supervision_state={"assessment": {"disposition": {"action_type": "protocol", "reason": "r",
                                                              "request": {"protocol_id": "p"}}},
                               "other": {"action_type": "protocol"}})
    _insert(conn, "event_log", id=_id(), project_id=ids["proj"], event_type="orchestration.protocol_started",
            source="system", dedup_key="orchestration.protocol_started:action:abc", payload={})
    conn.commit()
    return ids


def _dump(conn):
    """Order-stable snapshot of every row (JSON parsed) in tables touched by the migration."""
    snap = {}
    for table in ("event_log", "tasks", "sessions", "knowledge_items", "users", "orchestration_actions",
                  "orchestration_evidence", "orchestration_decisions", "orchestration_runs",
                  "escalation_chains", "meetings", "meeting_agenda_items", "meeting_action_items"):
        cur = conn.execute(f"SELECT * FROM {table} ORDER BY 1")
        names = [d[0] for d in cur.description]
        snap[table] = [dict(zip(names, row)) for row in cur.fetchall()]
    return json.loads(json.dumps(snap, default=str))  # normalise


def _cols(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _indexes(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


def _graph_fk_violations(conn):
    # Seed rows use fake parents elsewhere; only graph-table links must be intact.
    return [v for v in conn.execute("PRAGMA foreign_key_check") if v[2] in OLD_TABLES + NEW_TABLES]


def _loads(conn, table, column, row_id):
    return json.loads(conn.execute(f"SELECT {column} FROM {table} WHERE id = ?", (row_id,)).fetchone()[0])


def test_upgrade_renames_schema_converts_data_and_round_trips(tmp_path):
    db_path = tmp_path / "migration-048.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "047")
    with sqlite3.connect(db_path) as conn:
        ids = _seed(conn)
        before = _dump(conn)
        legacy_indexes = _indexes(conn)

    _alembic(env, "upgrade", "048")
    with sqlite3.connect(db_path) as conn:
        tables = _tables(conn)
        assert set(NEW_TABLES) <= tables and not set(OLD_TABLES) & tables
        assert {"graph_id", "current_node", "last_stepped_at"} <= _cols(conn, "graph_runs")
        assert {"graph_run_id", "from_node", "to_node", "edge_name", "stepped_at"} <= _cols(conn, "graph_run_steps")
        assert {"graph_run_id", "node_name"} <= _cols(conn, "graph_run_timeouts")
        assert "graph_run_id" in _cols(conn, "tasks") and "graph_run_id" in _cols(conn, "sessions")
        assert "source_graph_run_id" in _cols(conn, "meetings")
        assert "creates_graph" in _cols(conn, "meeting_agenda_items")
        assert {"creates_graph", "graph_run_id"} <= _cols(conn, "meeting_action_items")
        assert "provenance_graph_run_id" in _cols(conn, "knowledge_items")

        indexes = _indexes(conn)
        assert {"idx_graphs_active", "idx_gr_project_status", "idx_gr_graph", "idx_grs_run",
                "idx_grs_run_ts", "idx_grt_expires", "idx_grt_run", "idx_tasks_graph_run"} <= indexes
        assert not {i for i in indexes if i.startswith(("idx_pi_", "idx_pt_", "idx_pto_", "idx_protocols"))}
        assert not {i for i in indexes if "protocol" in i}

        assert _graph_fk_violations(conn) == []
        fks = {(r[2], r[3]) for r in conn.execute("PRAGMA foreign_key_list(graph_runs)")}
        assert ("graphs", "graph_id") in fks
        fks = {r[2] for r in conn.execute("PRAGMA foreign_key_list(graph_run_steps)")}
        assert fks == {"graph_runs"}
        for table in NEW_TABLES:
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == (2 if table == "graphs" else 1)

        # Graph definition: keys, action type, template tokens converted; prose untouched.
        definition = _loads(conn, "graphs", "definition", ids["graph"])
        assert definition["start_node"] == "triage"
        assert definition["terminal_nodes"] == {"success": ["done"], "failure": []}
        assert set(definition["nodes"]) == {"triage", "done"}
        triage = definition["nodes"]["triage"]
        assert triage["on_enter"][0]["action_type"] == "update_graph_context"
        assert triage["on_enter"][0]["note"] == "protocol notes stay"
        edge = triage["edges"][0]
        assert "transitions" not in triage and edge["to"] == "done"
        assert edge["actions"][0]["message"] == (
            "{{graph_name}} in {{ current_node }} for {{graph_run.id}}"
            " see {{dashboard_url}}/graph-runs/{{graph_run.id}}"
        )
        assert _loads(conn, "graphs", "triggers", ids["graph"]) == [{"event_type": "graph.run_completed"}]
        assert conn.execute("SELECT loaded_from FROM graphs").fetchone()[0] == "/ws/graphs/bug_fix.yaml"
        assert _loads(conn, "graph_runs", "context", ids["run"]) == {"graph_name": "bug_fix"}

        # Scalars.
        assert conn.execute("SELECT origin FROM sessions").fetchone()[0] == "graph"
        assert conn.execute("SELECT provenance_type FROM knowledge_items").fetchone()[0] == "graph"
        assert conn.execute("SELECT title, description FROM tasks").fetchone() == (
            "Graph task", "the protocol handbook is free text")
        assert conn.execute("SELECT email, display_name FROM users WHERE id=?", (ids["user"],)).fetchone() == (
            "graph-system@local", "Graph System")
        assert conn.execute("SELECT email, display_name FROM users WHERE id=?", (ids["plain_user"],)).fetchone() == (
            "bob@example.com", "Protocol Bob")
        assert conn.execute("SELECT action_type, target_type FROM orchestration_actions").fetchone() == (
            "start_graph", "graph_run")
        assert conn.execute("SELECT source_type FROM orchestration_evidence").fetchone()[0] == "graph_run_step"

        # JSON sweep across tables.
        assert _loads(conn, "tasks", "metadata", ids["task"]) == {"graph": "bug_fix", "origin": "graph"}
        assert _loads(conn, "sessions", "metadata", ids["session"]) == {"graph_run_id": ids["run"]}
        assert _loads(conn, "knowledge_items", "metadata", ids["ki"]) == {"graph_id": ids["graph"]}
        assert _loads(conn, "orchestration_runs", "supervision_state", ids["oruns"]) == {
            "graphs": [{"graph_id": "p"}], "graph_runs": [], "note": "protocol"}
        assert _loads(conn, "orchestration_actions", "request", ids["oaction"]) == {
            "graph_id": ids["graph"], "subject_type": "task"}
        assert _loads(conn, "orchestration_actions", "dispatch_contract", ids["oaction"]) == {
            "allowed": {"start_graph": ["graph_id"]}}
        assert _loads(conn, "orchestration_evidence", "metadata", ids["oevidence"]) == {
            "graph_run_step_id": ids["step"], "from_node": "a", "to_node": "b"}
        assert _loads(conn, "orchestration_decisions", "parsed_decision", ids["decision"]) == {
            "disposition": "graph", "action_type": "start_graph"}
        assert _loads(conn, "orchestration_decisions", "input_snapshot", ids["decision"]) == {
            "graph_runs": [{"current_node": "x", "graph_run_id": "1"}]}
        assert _loads(conn, "escalation_chains", "definition", ids["chain"]) == {
            "steps": [{"message": "{{graph_run.id}} {{graph_name}} protocol"}]}
        assert _loads(conn, "escalation_chains", "steps", ids["chain"]) == [{"message": "{{ current_node }}"}]

        bare = _loads(conn, "graphs", "definition", ids["bare_graph"])
        assert bare == {"start_node": "a", "terminal_nodes": {"success": ["b"]},
                        "nodes": {"a": {"edges": [{"to": "b"}]}, "b": {}}}
        assert _loads(conn, "meetings", "resume_state", ids["plain_meeting"]) == {
            "current_state": "paused", "from_state": "x"}
        assert _loads(conn, "orchestration_runs", "plan_state", ids["plain_run"]) == {
            "current_state": "planning", "transition_name": "t"}
        assert json.loads(conn.execute(
            "SELECT payload FROM event_log WHERE event_type = 'meeting.state'").fetchone()[0]) == {
            "current_state": "active", "to_state": "done"}

        # Disposition converted; same key outside "disposition" untouched; dedup_key prefix renamed.
        assert _loads(conn, "orchestration_runs", "supervision_state", ids["disp_run"]) == {
            "assessment": {"disposition": {"action_type": "graph", "reason": "r",
                                           "request": {"graph_id": "p"}}},
            "other": {"action_type": "protocol"}}
        assert conn.execute("SELECT dedup_key FROM event_log WHERE event_type = 'orchestration.graph_started'"
                            ).fetchone()[0] == "orchestration.graph_started:action:abc"

        # Event log: type, source, payload keys; unrelated events and free text untouched.
        event = conn.execute("SELECT event_type, source, payload FROM event_log WHERE seq = 1").fetchone()
        assert event[:2] == ("graph.run_advanced", "graph")
        assert json.loads(event[2]) == {
            "graph_run_id": ids["run"], "from_node": "a", "to_node": "b",
            "graph_run": {"graph_id": "x", "current_node": "b"},
            "note": "the protocol was fine", "start_node": "a"}
        other = conn.execute("SELECT event_type, source, payload FROM event_log WHERE event_type = 'task.created'").fetchone()
        assert (other[0], other[1], json.loads(other[2])) == (
            "task.created", "system", {"title": "plain", "current_state": "kept"})

    _alembic(env, "downgrade", "047")
    with sqlite3.connect(db_path) as conn:
        assert set(OLD_TABLES) <= _tables(conn) and not set(NEW_TABLES) & _tables(conn)
        assert "protocol_instance_id" in _cols(conn, "protocol_transitions")
        assert "creates_protocol" in _cols(conn, "meeting_action_items")
        assert _indexes(conn) == legacy_indexes
        assert _graph_fk_violations(conn) == []
        assert _dump(conn) == before
        definition = _loads(conn, "protocols", "definition", ids["graph"])
        assert definition["initial_state"] == "triage" and "transitions" in definition["states"]["triage"]
        assert conn.execute("SELECT loaded_from FROM protocols").fetchone()[0] == "/ws/protocols/bug_fix.yaml"

    _alembic(env, "upgrade", "048")  # re-applies cleanly after a downgrade


def test_upgrade_resumes_after_partial_schema_run(tmp_path):
    """DDL autocommits: a table already renamed (version still 047) must not break the re-run."""
    db_path = tmp_path / "migration-048-partial.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "047")
    with sqlite3.connect(db_path) as conn:
        ids = _seed(conn)
        conn.execute("DROP INDEX idx_protocols_active")
        conn.execute("ALTER TABLE protocols RENAME TO graphs")
        conn.execute("ALTER TABLE tasks RENAME COLUMN protocol_instance_id TO graph_run_id")
        conn.commit()

    _alembic(env, "upgrade", "048")
    with sqlite3.connect(db_path) as conn:
        assert set(NEW_TABLES) <= _tables(conn) and not set(OLD_TABLES) & _tables(conn)
        assert "graph_run_id" in _cols(conn, "tasks") and "graph_id" in _cols(conn, "graph_runs")
        assert {"idx_graphs_active", "idx_gr_project_status", "idx_gr_graph", "idx_grs_run", "idx_grs_run_ts",
                "idx_grt_expires", "idx_grt_run", "idx_tasks_graph_run"} <= _indexes(conn)
        assert not {i for i in _indexes(conn) if "protocol" in i}
        assert _loads(conn, "graphs", "definition", ids["graph"])["start_node"] == "triage"


def test_downgrade_resumes_after_partial_schema_run(tmp_path):
    db_path = tmp_path / "migration-048-partial-down.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{db_path}"}
    _alembic(env, "upgrade", "047")
    with sqlite3.connect(db_path) as conn:
        _seed(conn)
        legacy_indexes = _indexes(conn)
    _alembic(env, "upgrade", "048")
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_graphs_active")
        conn.execute("ALTER TABLE graphs RENAME TO protocols")
        conn.execute("ALTER TABLE tasks RENAME COLUMN graph_run_id TO protocol_instance_id")
        conn.commit()

    _alembic(env, "downgrade", "047")
    with sqlite3.connect(db_path) as conn:
        assert set(OLD_TABLES) <= _tables(conn) and not set(NEW_TABLES) & _tables(conn)
        assert conn.execute("SELECT loaded_from FROM protocols").fetchone()[0] == "/ws/protocols/bug_fix.yaml"
        assert _indexes(conn) == legacy_indexes
