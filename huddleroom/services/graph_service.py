from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.graph import Graph, GraphRun

logger = logging.getLogger(__name__)

_LEGACY_KEYS = ("states", "initial_state", "terminal_states")

# ponytail: legacy mappings for validation error messages; removed after migration 048
_LEGACY_EVENTS = {
    "protocol.instance_started": "graph.run_started",
    "protocol.state_transitioned": "graph.run_advanced",
    "protocol.completed": "graph.run_completed",
    "protocol.failed": "graph.run_failed",
    "protocol.escalated": "graph.run_escalated",
    "protocol.external_resolution": "graph.run_external_resolution",
    "orchestration.protocol_started": "orchestration.graph_started",
    "orchestration.protocol_start_requested": "orchestration.graph_start_requested",
}

_LEGACY_ACTION_TYPES = {
    "update_protocol_context": "update_graph_context",
    "start_protocol": "start_graph",
}


def _check_action_type(action_type: Any, location: str) -> None:
    """Validate action_type is string and not a legacy protocol action."""
    if not isinstance(action_type, str):
        raise ValueError(
            f"action_type must be a string, got {type(action_type).__name__} ({location})"
        )
    if action_type in _LEGACY_ACTION_TYPES:
        new_type = _LEGACY_ACTION_TYPES[action_type]
        raise ValueError(
            f"Action type '{action_type}' has been renamed to '{new_type}' ({location})"
        )


def _check_trigger_event(trigger_event: Any, location: str) -> None:
    """Validate trigger_event is string and not a legacy protocol event."""
    if not isinstance(trigger_event, str):
        raise ValueError(
            f"trigger_event must be a string, got {type(trigger_event).__name__} ({location})"
        )
    if trigger_event in _LEGACY_EVENTS:
        new_event = _LEGACY_EVENTS[trigger_event]
        raise ValueError(
            f"Trigger event '{trigger_event}' has been renamed to '{new_event}' ({location})"
        )
    if trigger_event.startswith("protocol.") or trigger_event.startswith("orchestration.protocol_"):
        raise ValueError(
            f"Trigger event '{trigger_event}' uses legacy 'protocol' terminology ({location})"
        )


def _check_event_type(event_type: Any, location: str) -> None:
    """Validate event_type is string and not a legacy protocol event."""
    if not isinstance(event_type, str):
        raise ValueError(
            f"event_type must be a string, got {type(event_type).__name__} ({location})"
        )
    if event_type in _LEGACY_EVENTS:
        new_event = _LEGACY_EVENTS[event_type]
        raise ValueError(
            f"Event type '{event_type}' has been renamed to '{new_event}' ({location})"
        )
    if event_type.startswith("protocol.") or event_type.startswith("orchestration.protocol_"):
        raise ValueError(
            f"Event type '{event_type}' uses legacy 'protocol' terminology ({location})"
        )


def _check_legacy_template_tokens(definition: dict, location: str = "definition") -> None:
    """Reject legacy template tokens in string values (skip description fields)."""
    legacy_pattern = re.compile(r"\{\{\s*(protocol_\w*|current_state\b)")

    def check_value(value: Any, path: str) -> None:
        """Recursively check values, skipping description fields."""
        if isinstance(value, dict):
            for key, val in value.items():
                # Skip description fields (free text)
                if key == "description":
                    continue
                check_value(val, f"{path}.{key}" if path else key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                check_value(item, f"{path}[{i}]")
        elif isinstance(value, str):
            # Check for legacy template tokens
            if legacy_pattern.search(value):
                if "protocol_" in value:
                    raise ValueError(
                        f"Template tokens use legacy 'protocol_' prefix; use 'graph_' instead ({path})"
                    )
                if "current_state" in value:
                    raise ValueError(
                        f"Template token '{{{{current_state' has been renamed to '{{{{current_node' ({path})"
                    )
            # Check for legacy URL paths
            if "/protocol-instances/" in value:
                raise ValueError(
                    f"URL path '/protocol-instances/' has been renamed to '/graph-runs/' ({path})"
                )

    check_value(definition, "")


def validate_graph_definition(definition: dict) -> None:
    """Raise ValueError if a graph definition is malformed or uses legacy protocol keys."""
    if not isinstance(definition, dict):
        raise ValueError("Graph definition must be a mapping")

    # Check for legacy template tokens early
    _check_legacy_template_tokens(definition)

    legacy = [k for k in _LEGACY_KEYS if k in definition]
    nodes = definition.get("nodes")
    if isinstance(nodes, dict):
        legacy += [
            f"nodes.{n}.transitions"
            for n, d in nodes.items()
            if isinstance(d, dict) and "transitions" in d
        ]
    if legacy:
        raise ValueError(
            "Graph definition uses legacy keys: " + ", ".join(legacy)
            + " (use nodes, start_node, terminal_nodes, and node-level edges)"
        )
    if not isinstance(nodes, dict):
        raise ValueError("Graph definition requires 'nodes' (mapping)")
    if not definition.get("start_node"):
        raise ValueError("Graph definition requires 'start_node'")

    # Check top-level triggers for legacy event_type
    triggers = definition.get("triggers", [])
    if isinstance(triggers, list):
        for i, trigger in enumerate(triggers):
            if isinstance(trigger, dict) and "event_type" in trigger:
                _check_event_type(trigger["event_type"], f"triggers[{i}].event_type")

    # Check nodes for legacy action types and trigger events
    if isinstance(nodes, dict):
        for node_name, node_def in nodes.items():
            if not isinstance(node_def, dict):
                continue

            # Check on_enter actions
            on_enter = node_def.get("on_enter", [])
            if isinstance(on_enter, list):
                for i, action in enumerate(on_enter):
                    if isinstance(action, dict) and "action_type" in action:
                        _check_action_type(action["action_type"], f"nodes.{node_name}.on_enter[{i}].action_type")

            # Check timeout.action
            timeout = node_def.get("timeout")
            if isinstance(timeout, dict) and "action" in timeout:
                timeout_action = timeout["action"]
                if isinstance(timeout_action, str):
                    if timeout_action in _LEGACY_ACTION_TYPES:
                        new_action = _LEGACY_ACTION_TYPES[timeout_action]
                        raise ValueError(
                            f"Timeout action '{timeout_action}' has been renamed to '{new_action}' "
                            f"(nodes.{node_name}.timeout.action)"
                        )
                else:
                    raise ValueError(
                        f"timeout.action must be a string, got {type(timeout_action).__name__} "
                        f"(nodes.{node_name}.timeout.action)"
                    )

            # Check edges for legacy trigger events and actions
            edges = node_def.get("edges", [])
            if isinstance(edges, list):
                for i, edge in enumerate(edges):
                    if not isinstance(edge, dict):
                        continue

                    if "trigger_event" in edge:
                        _check_trigger_event(edge["trigger_event"], f"nodes.{node_name}.edges[{i}].trigger_event")

                    # Check actions in edge (if present)
                    edge_actions = edge.get("actions", [])
                    if isinstance(edge_actions, list):
                        for j, action in enumerate(edge_actions):
                            if isinstance(action, dict) and "action_type" in action:
                                _check_action_type(action["action_type"], f"nodes.{node_name}.edges[{i}].actions[{j}].action_type")


def _extract_triggers(definition: dict) -> list:
    return definition.get("triggers", [])


class GraphService:
    async def load_from_yaml(
        self,
        db: AsyncSession,
        path: Path,
        project_id: uuid.UUID | None = None,
    ) -> Graph:
        text = path.read_text()
        definition: dict[str, Any] = yaml.safe_load(text)
        validate_graph_definition(definition)

        name = definition["name"]
        version = str(definition.get("version", "1.0"))
        triggers = _extract_triggers(definition)
        escalation_chain = definition.get("escalation_chain")

        result = await db.execute(
            select(Graph).where(
                Graph.project_id.is_(None) if project_id is None else Graph.project_id == project_id,
                Graph.name == name,
                Graph.version == version,
            )
        )
        graph = result.scalar_one_or_none()
        if graph is None:
            graph = Graph(
                project_id=project_id,
                name=name,
                version=version,
                description=definition.get("description"),
                definition=definition,
                triggers=triggers,
                escalation_chain=escalation_chain,
                loaded_from=str(path),
            )
            db.add(graph)
        else:
            graph.definition = definition
            graph.triggers = triggers
            graph.escalation_chain = escalation_chain
            graph.loaded_from = str(path)

        await db.flush()
        return graph

    async def load_all_from_workspace(
        self, db: AsyncSession, workspace_dir: Path, project_id: uuid.UUID | None = None
    ) -> list[Graph]:
        if (workspace_dir / "protocols").exists():
            logger.warning(
                "workspace/protocols/ is no longer supported; rename to workspace/graphs/ and convert keys"
            )
        graphs_dir = workspace_dir / "graphs"
        if not graphs_dir.exists():
            return []
        loaded = []
        for yaml_file in sorted(graphs_dir.glob("*.yaml")):
            # ponytail: savepoint per file; rollback on exception keeps session usable for next file
            try:
                async with db.begin_nested():
                    graph = await self.load_from_yaml(db, yaml_file, project_id=project_id)
                # Only append after clean exit (no exception)
                loaded.append(graph)
            except Exception:
                logger.exception("Failed to load graph %s", yaml_file)
                # Savepoint is auto-rolled back on exception exit; continue to next file
        return loaded

    async def get(self, db: AsyncSession, graph_id: uuid.UUID) -> Graph | None:
        result = await db.execute(select(Graph).where(Graph.id == graph_id))
        return result.scalar_one_or_none()

    async def get_by_name(
        self, db: AsyncSession, name: str, project_id: uuid.UUID | None = None
    ) -> Graph | None:
        result = await db.execute(
            select(Graph).where(
                Graph.name == name,
                Graph.project_id.is_(None) if project_id is None else Graph.project_id == project_id,
                Graph.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        db: AsyncSession,
        project_id: uuid.UUID | None = None,
        active_only: bool = True,
    ) -> list[Graph]:
        q = select(Graph).where(
            Graph.project_id.is_(None) if project_id is None else Graph.project_id == project_id
        )
        if active_only:
            q = q.where(Graph.is_active.is_(True))
        result = await db.execute(q)
        return list(result.scalars().all())

    async def get_active_graphs_for_event(
        self, db: AsyncSession, event_type: str, project_id: uuid.UUID | None = None
    ) -> list[Graph]:
        graphs = await self.list(db, project_id=project_id, active_only=True)
        matching = []
        for p in graphs:
            for trigger in p.triggers:
                if trigger.get("event_type") == event_type:
                    matching.append(p)
                    break
        return matching

    async def create_run(
        self,
        db: AsyncSession,
        graph: Graph,
        project_id: uuid.UUID,
        start_node: str,
        linked_task_id: uuid.UUID | None = None,
        artifact_id: uuid.UUID | None = None,
        triggering_event_id: uuid.UUID | None = None,
    ) -> GraphRun:
        run = GraphRun(
            graph_id=graph.id,
            project_id=project_id,
            current_node=start_node,
            status="active",
            linked_task_id=linked_task_id,
            artifact_id=artifact_id,
            triggering_event_id=triggering_event_id,
            actor_assignments={},
            context={},
        )
        db.add(run)
        await db.flush()
        return run
