"""Rename the "protocol" vocabulary to "graph" (tables, columns, indexes, stored data).

Revision ID: 048
Revises: 047
Create Date: 2026-10-07

Clean break, no compatibility aliases.  Schema changes use native
``ALTER TABLE ... RENAME TABLE/COLUMN`` (never batch mode: recreating a parent table with
foreign keys enabled can cascade-delete its children).  Stored data is rewritten ONLY through
the explicit key/value maps below; free text is never search-and-replaced.

Rewrite rule.  Protocol-specific names (``protocol_id``, ``protocol_instance_id``,
``protocol_transition_id``, ``protocol_name``, ``protocol_instance(s)``, ``protocols``,
``protocol``, ``start_protocol`` ...) are renamed in every JSON column.  Generic state names
(``current_state``, ``from_state``, ``to_state``, ``initial_state``, ``transition_name``,
``state_name``) may mean something else elsewhere (meeting/orchestration state), so they are
renamed ONLY in graph context: any JSON column of graphs/graph_runs/graph_run_steps/
graph_run_timeouts, ``event_log.payload`` of graph events, and any dict that has a graph
identifier key beside them or sits under a ``protocol_instance(s)``/``protocols`` key.
``graphs.definition`` is rewritten for every row, unconditionally (idempotent).  Other JSON is
prefiltered by any legacy token, then rewritten only through the explicit maps.

Downgrade applies the inverse maps.  Schema is restored exactly; data is best effort, because
the inverse of a generic key such as ``graph`` cannot distinguish rows that were never legacy
protocol data.  The migration imports no application models.
"""
# pylint: disable=invalid-name
import json
import re

import sqlalchemy as sa
from alembic import op


revision = "048"
down_revision = "047"
branch_labels = None
depends_on = None

# (old table, new table)
TABLES = [
    ("protocols", "graphs"),
    ("protocol_instances", "graph_runs"),
    ("protocol_transitions", "graph_run_steps"),
    ("protocol_timeouts", "graph_run_timeouts"),
]

# new table -> {old column: new column}
COLUMNS = {
    "graph_runs": {
        "protocol_id": "graph_id",
        "current_state": "current_node",
        "last_transitioned_at": "last_stepped_at",
    },
    "graph_run_steps": {
        "protocol_instance_id": "graph_run_id",
        "from_state": "from_node",
        "to_state": "to_node",
        "transition_name": "edge_name",
        "transitioned_at": "stepped_at",
    },
    "graph_run_timeouts": {"protocol_instance_id": "graph_run_id", "state_name": "node_name"},
    "tasks": {"protocol_instance_id": "graph_run_id"},
    "sessions": {"protocol_instance_id": "graph_run_id"},
    "meetings": {"source_protocol_instance_id": "source_graph_run_id"},
    "meeting_agenda_items": {"creates_protocol": "creates_graph"},
    "meeting_action_items": {
        "creates_protocol": "creates_graph",
        "protocol_instance_id": "graph_run_id",
    },
    "knowledge_items": {"provenance_protocol_instance_id": "provenance_graph_run_id"},
}

# (old index, new index, old table, new table, legacy column names or None).  Columns are the
# pre-048 names (mapped through COLUMNS going forward) so a crashed run's dropped index can be
# rebuilt.  None = not present in a fresh 047 schema; only re-created if found and dropped here.
INDEXES = [
    ("idx_protocols_active", "idx_graphs_active", "protocols", "graphs", ["project_id", "is_active"]),
    ("uq_protocols_project_name_version", "uq_graphs_project_name_version", "protocols", "graphs", None),
    ("idx_pi_project_status", "idx_gr_project_status", "protocol_instances", "graph_runs", ["project_id", "status"]),
    ("idx_pi_protocol", "idx_gr_graph", "protocol_instances", "graph_runs", ["protocol_id"]),
    ("idx_pt_instance", "idx_grs_run", "protocol_transitions", "graph_run_steps", ["protocol_instance_id"]),
    ("idx_pt_instance_ts", "idx_grs_run_ts", "protocol_transitions", "graph_run_steps",
     ["protocol_instance_id", "transitioned_at"]),
    ("idx_pto_expires", "idx_grt_expires", "protocol_timeouts", "graph_run_timeouts", ["expires_at"]),
    ("idx_pto_instance", "idx_grt_run", "protocol_timeouts", "graph_run_timeouts", ["protocol_instance_id"]),
    ("idx_tasks_protocol_instance", "idx_tasks_graph_run", "tasks", "tasks", ["protocol_instance_id"]),
]

# Persisted event types (event_log.event_type and exact string values inside JSON).
EVENT_TYPES = {
    "protocol.instance_started": "graph.run_started",
    "protocol.state_transitioned": "graph.run_advanced",
    "protocol.completed": "graph.run_completed",
    "protocol.failed": "graph.run_failed",
    "protocol.escalated": "graph.run_escalated",
    "protocol.external_resolution": "graph.run_external_resolution",
    "orchestration.protocol_started": "orchestration.graph_started",
    "orchestration.protocol_start_requested": "orchestration.graph_start_requested",
}

# Exact string values rewritten anywhere inside swept JSON.
JSON_VALUES = {
    **EVENT_TYPES,
    "start_protocol": "start_graph",
    "protocol_instance": "graph_run",
    "protocol_transition": "graph_run_step",
    "update_protocol_context": "update_graph_context",
    # Key names listed as values (e.g. dispatch-contract allowed-key lists).
    "protocol_id": "graph_id",
    "protocol_instance_id": "graph_run_id",
}

# The bare value "protocol" is only rewritten under these JSON keys.
JSON_SCOPED_VALUES = {
    "protocol": "graph",
}
JSON_SCOPED_KEYS = {"disposition", "source", "origin", "provenance_type", "source_type", "target_type"}

# Exact JSON object keys.
JSON_KEYS = {
    "protocol_instance_id": "graph_run_id",
    "protocol_transition_id": "graph_run_step_id",
    "protocol_name": "graph_name",
    "protocol_id": "graph_id",
    "protocol_instance": "graph_run",
    "start_protocol": "start_graph",  # action type used as a dict key (dispatch contracts)
    "protocol": "graph",
    "protocol_instances": "graph_runs",
    "protocols": "graphs",
    "creates_protocol": "creates_graph",
    "source_protocol_instance_id": "source_graph_run_id",
    "provenance_protocol_instance_id": "provenance_graph_run_id",
}

# Generic names that other subsystems may use with a non-graph meaning; renamed ONLY inside a
# graph context (see module docstring).
STATE_KEYS = {
    "current_state": "current_node",
    "from_state": "from_node",
    "to_state": "to_node",
    "initial_state": "start_node",
    "transition_name": "edge_name",
    "state_name": "node_name",
}
# A dict holding one of these keys is a graph dict; a value under a container key is graph context.
ID_MARKERS = {"protocol_instance_id", "protocol_id", "protocol_transition_id", "protocol_name"}
CONTAINER_MARKERS = {"protocol_instance", "protocol_instances", "protocols"}
# JSON columns of these tables are entirely graph data.
GRAPH_TABLES = {"graphs", "graph_runs", "graph_run_steps", "graph_run_timeouts"}

# graphs.definition structure: top-level keys, then per-node keys.
DEFINITION_TOP_KEYS = {
    "states": "nodes",
    "initial_state": "start_node",
    "terminal_states": "terminal_nodes",
}
DEFINITION_NODE_KEYS = {"transitions": "edges"}

# Template variables: {{ name }} heads, plus the one dashboard path.
TEMPLATE_TOKENS = {
    "protocol_instance_id": "graph_run_id",
    "protocol_instance": "graph_run",
    "protocol_name": "graph_name",
    "current_state": "current_node",
}
TEMPLATE_PATHS = {"/protocol-instances/": "/graph-runs/"}

# (table, column, old, new): whole-value scalar updates.
SCALARS = [
    *(("event_log", "event_type", old, new) for old, new in EVENT_TYPES.items()),
    ("event_log", "source", "protocol", "graph"),
    ("sessions", "origin", "protocol", "graph"),
    ("knowledge_items", "provenance_type", "protocol", "graph"),
    ("orchestration_actions", "action_type", "start_protocol", "start_graph"),
    ("orchestration_actions", "target_type", "protocol_instance", "graph_run"),
    ("orchestration_evidence", "source_type", "protocol_transition", "graph_run_step"),
    ("orchestration_evidence", "source_type", "protocol_instance", "graph_run"),
    ("tasks", "title", "Protocol task", "Graph task"),
    ("tasks", "title", "Protocol session", "Graph session"),
]

SYSTEM_USER = ("protocol-system@local", "Protocol System", "graph-system@local", "Graph System")
BATCH = 500


def _invert(mapping: dict[str, str]) -> dict[str, str]:
    return {new: old for old, new in mapping.items()}


# ---------------------------------------------------------------- JSON rewriting

def _rewrite(obj, keys, values, scoped, tokens=None, ctx=None):
    """Recursively rewrite keys/values from explicit maps; tokens only for template strings.

    ``ctx`` is None for no graph context, else (state_keys, id_markers, containers, active).
    State-key renames only apply while ``active``.
    """
    if isinstance(obj, dict):
        active = bool(ctx and (ctx[3] or ctx[1] & obj.keys()))
        out = {}
        for key, value in obj.items():
            new_key = keys.get(key) or (ctx[0].get(key) if active else None) or key
            if new_key != key and new_key in obj:
                new_key = key
            if isinstance(value, str) and key in JSON_SCOPED_KEYS and value in scoped:
                out[new_key] = scoped[value]
                continue
            # Scoped: action_type "protocol" is rewritten only inside a dict under "disposition".
            if key == "disposition" and isinstance(value, dict) and value.get("action_type") in scoped:
                value = {**value, "action_type": scoped[value["action_type"]]}
            child = (ctx[0], ctx[1], ctx[2], active or key in ctx[2]) if ctx else None
            out[new_key] = _rewrite(value, keys, values, scoped, tokens, child)
        return out
    if isinstance(obj, list):
        return [_rewrite(item, keys, values, scoped, tokens, ctx) for item in obj]
    if isinstance(obj, str):
        obj = values.get(obj, obj)
        return tokens(obj) if tokens else obj
    return obj


def _token_rewriter(token_map: dict[str, str], path_map: dict[str, str]):
    pattern = re.compile(
        r"(\{\{\s*)(" + "|".join(sorted(map(re.escape, token_map), key=len, reverse=True)) + r")\b"
    )

    def apply(text: str) -> str:
        text = pattern.sub(lambda m: m.group(1) + token_map[m.group(2)], text)
        for old, new in path_map.items():
            text = text.replace(old, new)
        return text

    return apply


def _rewrite_definition(definition, forward: bool):
    top = DEFINITION_TOP_KEYS if forward else _invert(DEFINITION_TOP_KEYS)
    node_keys = DEFINITION_NODE_KEYS if forward else _invert(DEFINITION_NODE_KEYS)
    out = _rewrite(definition, *_maps(forward), tokens=_tokens(forward), ctx=_ctx(forward, True))
    if not isinstance(out, dict):
        return out
    out = {top.get(key, key): value for key, value in out.items()}
    nodes = out.get("nodes" if forward else "states")
    if isinstance(nodes, dict):
        for name, node in list(nodes.items()):
            if isinstance(node, dict):
                nodes[name] = {node_keys.get(key, key): value for key, value in node.items()}
    return out


def _maps(forward: bool):
    if forward:
        return JSON_KEYS, JSON_VALUES, JSON_SCOPED_VALUES
    return _invert(JSON_KEYS), _invert(JSON_VALUES), _invert(JSON_SCOPED_VALUES)


def _ctx(forward: bool, active: bool):
    if forward:
        return (STATE_KEYS, ID_MARKERS, CONTAINER_MARKERS, active)
    return (
        _invert(STATE_KEYS),
        {JSON_KEYS[k] for k in ID_MARKERS},
        {JSON_KEYS[k] for k in CONTAINER_MARKERS},
        active,
    )


def _tokens(forward: bool):
    if forward:
        return _token_rewriter(TEMPLATE_TOKENS, TEMPLATE_PATHS)
    return _token_rewriter(_invert(TEMPLATE_TOKENS), _invert(TEMPLATE_PATHS))


def _sweep_column(bind, table, column, pk, transform, needles, extra=None, extra_where=None) -> None:
    """Rewrite one JSON column in pk-ordered batches.

    Only rows whose text contains one of ``needles`` (or matching ``extra_where``) are read;
    ``extra`` is a second column handed to ``transform`` (event_log.event_type).
    """
    col, key = f'"{column}"', f'"{pk}"'
    clauses = [f"CAST({col} AS TEXT) LIKE :n{i}" for i in range(len(needles))] if needles else []
    params: dict = {f"n{i}": f"%{n}%" for i, n in enumerate(needles or [])}
    if extra_where and clauses:
        clauses.append(extra_where)
    where = f"AND ({' OR '.join(clauses)})" if clauses else ""
    select = f'{key}, {col}, {f"{chr(34)}{extra}{chr(34)}" if extra else "NULL"}'
    last = None
    while True:
        after = f"AND {key} > :last" if last is not None else ""
        if last is not None:
            params["last"] = last
        rows = bind.execute(
            sa.text(
                f'SELECT {select} FROM "{table}" WHERE {col} IS NOT NULL {where} {after} '
                f"ORDER BY {key} LIMIT {BATCH}"
            ),
            params,
        ).fetchall()
        if not rows:
            return
        for row_pk, raw, extra_value in rows:
            last = row_pk
            if not isinstance(raw, str):
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            new = transform(data, extra_value)
            if new != data:
                bind.execute(
                    sa.text(f'UPDATE "{table}" SET {col} = :v WHERE {key} = :pk'),
                    {"v": json.dumps(new), "pk": row_pk},
                )


def _rewrite_data(bind, forward: bool) -> None:
    keys, values, scoped = _maps(forward)
    tokens = _tokens(forward)
    event_types = set(EVENT_TYPES) if forward else set(EVENT_TYPES.values())

    def swept(active):
        return lambda data, _: _rewrite(data, keys, values, scoped, ctx=_ctx(forward, active))

    def event_payload(data, event_type):
        return _rewrite(data, keys, values, scoped, ctx=_ctx(forward, event_type in event_types))

    token_only = lambda data, _: _rewrite(data, {}, {}, {}, tokens=tokens)  # noqa: E731
    definition = lambda data, _: _rewrite_definition(data, forward)  # noqa: E731
    # Prefilter: any legacy (or, going back, new) token from the maps.  Rows are only rewritten
    # through the explicit maps, so a broad prefilter cannot change unrelated content.
    needles = (["protocol", *STATE_KEYS] if forward
               else ["graph", *STATE_KEYS.values()])
    event_where = ("event_type LIKE 'protocol.%' OR event_type LIKE 'orchestration.protocol%'" if forward
                   else "event_type LIKE 'graph.%' OR event_type LIKE 'orchestration.graph_%'")

    inspector = sa.inspect(bind)
    for table in inspector.get_table_names():
        if table == "alembic_version":
            continue
        pk_cols = inspector.get_pk_constraint(table)["constrained_columns"]
        if len(pk_cols) != 1:
            continue
        for col in inspector.get_columns(table):
            if not isinstance(col["type"], sa.JSON):
                continue
            name, pk = col["name"], pk_cols[0]
            if table == "graphs" and name == "definition":
                _sweep_column(bind, table, name, pk, definition, None)  # every row, idempotent
            elif table == "escalation_chains" and name in ("definition", "steps"):
                _sweep_column(bind, table, name, pk, token_only, None)
            elif table == "event_log" and name == "payload":
                _sweep_column(bind, table, name, pk, event_payload, needles, "event_type", event_where)
            else:
                _sweep_column(bind, table, name, pk, swept(table in GRAPH_TABLES), needles)

    for table, column, old, new in SCALARS:
        old, new = (old, new) if forward else (new, old)
        bind.execute(sa.text(f"UPDATE {table} SET {column} = :new WHERE {column} = :old"), {"new": new, "old": old})

    # event_log.dedup_key is "<event_type>:<suffix>"; only rewrite the event-type prefix.
    for old, new in EVENT_TYPES.items():
        old, new = (old, new) if forward else (new, old)
        bind.execute(
            sa.text("UPDATE event_log SET dedup_key = :new || substr(dedup_key, :n) WHERE dedup_key LIKE :pat ESCAPE '\\'"),
            {"new": new + ":", "n": len(old) + 2, "pat": old.replace("_", "\\_") + ":%"},
        )

    old_email, old_name, new_email, new_name = SYSTEM_USER
    if not forward:
        old_email, old_name, new_email, new_name = new_email, new_name, old_email, old_name
    bind.execute(
        sa.text("UPDATE users SET display_name = :n WHERE email = :e AND display_name = :o"),
        {"n": new_name, "e": old_email, "o": old_name},
    )
    # Skip if the target address already exists (unique email); never merge accounts.
    bind.execute(
        sa.text(
            "UPDATE users SET email = :new WHERE email = :old "
            "AND NOT EXISTS (SELECT 1 FROM users WHERE email = :new)"
        ),
        {"new": new_email, "old": old_email},
    )

    src, dst = ("protocols", "graphs") if forward else ("graphs", "protocols")
    # Resumable: after a partial schema step the table may already carry either name.
    present = set(sa.inspect(bind).get_table_names())
    graph_table = "graphs" if "graphs" in present else "protocols"
    for row_pk, path in bind.execute(
        sa.text(f"SELECT id, loaded_from FROM {graph_table} WHERE loaded_from IS NOT NULL")
    ).fetchall():
        parts = path.split("/")
        if src in parts:
            bind.execute(
                sa.text(f"UPDATE {graph_table} SET loaded_from = :p WHERE id = :id"),
                {"p": "/".join(dst if part == src else part for part in parts), "id": row_pk},
            )


# ---------------------------------------------------------------- schema

def _rename_schema(bind, forward: bool) -> None:
    """Idempotent: SQLite DDL autocommits, so a re-run after a partial failure must skip done steps."""
    colmap = {t: (m if forward else _invert(m)) for t, m in COLUMNS.items()}

    def tables() -> set[str]:
        return set(sa.inspect(bind).get_table_names())  # fresh inspector: no stale cache

    def columns(table: str) -> set[str]:
        return {c["name"] for c in sa.inspect(bind).get_columns(table)}

    # 1. Capture and drop indexes under their current names (SQLite cannot rename an index).
    pending = []
    for old, new, old_table, new_table, legacy_cols in INDEXES:
        src_name, dst_name = (old, new) if forward else (new, old)
        src_table, dst_table = (old_table, new_table) if forward else (new_table, old_table)
        # The index lives on whichever name the table currently has.
        for table in (src_table, dst_table):
            if table not in tables():
                continue
            found = next((i for i in sa.inspect(bind).get_indexes(table) if i["name"] == src_name), None)
            if found is not None:
                op.drop_index(src_name, table_name=table)
                if legacy_cols is None:
                    pending.append((dst_name, dst_table, new_table, list(found["column_names"]), bool(found["unique"])))
                break

    # 2. Tables and columns (native RENAME keeps rows and foreign keys intact).  Columns live on
    # the new-named table, so go forward: tables then columns; backward: columns then tables.
    def rename_tables() -> None:
        for old, new in TABLES:
            src, dst = (old, new) if forward else (new, old)
            present = tables()
            if src in present and dst not in present:
                op.execute(f"ALTER TABLE {src} RENAME TO {dst}")

    def rename_columns() -> None:
        for table, mapping in colmap.items():
            if table not in tables():
                continue
            for src, dst in mapping.items():
                present = columns(table)
                if src in present and dst not in present:
                    op.execute(f"ALTER TABLE {table} RENAME COLUMN {src} TO {dst}")

    for step in (rename_tables, rename_columns) if forward else (rename_columns, rename_tables):
        step()

    # 3. Create every target index from the static list (survives a crashed earlier run that
    # already dropped it), then the captured ones that have no static definition.
    def ensure(name, table, cols, unique=False) -> None:
        if table not in tables() or not set(cols) <= columns(table):
            return
        if name not in {i["name"] for i in sa.inspect(bind).get_indexes(table)}:
            op.create_index(name, table, cols, unique=unique)

    for old, new, old_table, new_table, legacy_cols in INDEXES:
        if legacy_cols is None:
            continue
        dst_name, dst_table = (new, new_table) if forward else (old, old_table)
        mapping = COLUMNS.get(new_table, {})
        ensure(dst_name, dst_table, [mapping.get(c, c) for c in legacy_cols] if forward else legacy_cols)
    for dst_name, dst_table, new_table, cols, unique in pending:
        ensure(dst_name, dst_table, [colmap.get(new_table, {}).get(c, c) for c in cols], unique)


def upgrade() -> None:
    bind = op.get_bind()
    _rename_schema(bind, True)
    _rewrite_data(bind, True)


def downgrade() -> None:
    """Best-effort data downgrade; see module docstring."""
    bind = op.get_bind()
    _rewrite_data(bind, False)
    _rename_schema(bind, False)
