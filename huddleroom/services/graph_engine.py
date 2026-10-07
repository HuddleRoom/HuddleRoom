from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.artifact import Artifact
from huddleroom.models.graph import Graph, GraphRun, GraphRunStep, GraphRunTimeout
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.action_executor import ActionExecutor
from huddleroom.services.agent_service import GLOBAL_PROJECT_ID
from huddleroom.services.actor_resolver import ActorResolver
from huddleroom.services.escalation_service import EscalationChainService
from huddleroom.services.event_bus import BusEvent, EventBusService, emit_event, emit_event_once
from huddleroom.services.guard_evaluator import GuardEvaluator
from huddleroom.services.graph_service import GraphService
from huddleroom.services.project_service import ProjectService
from huddleroom.services.template_resolver import TemplateResolver

logger = logging.getLogger(__name__)

_DURATION_MAP = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_duration(duration_str: str) -> int:
    value = (duration_str or "1h").strip().lower()
    for suffix, multiplier in _DURATION_MAP.items():
        if value.endswith(suffix):
            try:
                return int(value[:-1]) * multiplier
            except ValueError:
                return 3600
    return 3600


class GraphEngineService:
    def __init__(self, _bus: EventBusService | None = None) -> None:
        self._bus = _bus
        self._graph_service = GraphService()
        self._actor_resolver = ActorResolver()
        self._guard_evaluator = GuardEvaluator()
        self._action_executor = ActionExecutor(_bus=_bus)
        self._template_resolver = TemplateResolver()

    async def process_event(self, db: AsyncSession, event: BusEvent) -> None:
        if event.project_id == GLOBAL_PROJECT_ID:
            return
        project_service = ProjectService()
        await project_service.lock_workspace_boundary(db, event.project_id)
        await project_service.require_runnable_project(db, event.project_id)
        enriched = BusEvent(
            id=event.id,
            project_id=event.project_id,
            event_type=event.event_type,
            payload=await self._enrich_event_payload(db, event.project_id, event.payload, event.event_type),
            source=event.source,
            emitted_at=event.emitted_at,
        )
        await self._check_triggers(db, enriched)
        await self._evaluate_active_runs(db, enriched)
        derived = await self._maybe_auto_emit_review_graph_event(db, enriched)
        if derived is not None:
            await self._check_triggers(db, derived)
            await self._evaluate_active_runs(db, derived)

    async def _enrich_event_payload(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        payload: dict,
        event_type: str | None = None,
    ) -> dict:
        enriched = dict(payload or {})
        raw_payload = dict(payload or {})
        metadata = {}
        payload_metadata = dict(enriched.get("metadata") or {})

        task_id = self._uuid_from_payload(enriched.get("task_id"))
        artifact_id = self._uuid_from_payload(enriched.get("artifact_id"))
        session_id = self._uuid_from_payload(enriched.get("session_id"))
        graph_run_id = self._uuid_from_payload(enriched.get("graph_run_id"))
        canonical_graph_run_id = graph_run_id

        session = None
        session_payload = None
        if session_id is not None:
            result = await db.execute(select(Session).where(Session.id == session_id, Session.project_id == project_id))
            session = result.scalar_one_or_none()
            if session is not None:
                task_id = task_id or session.task_id
                graph_run_id = graph_run_id or session.graph_run_id
                canonical_graph_run_id = canonical_graph_run_id or session.graph_run_id
                metadata = {**metadata, **(session.metadata_ or {})}
                enriched["session_id"] = str(session.id)
                session_payload = {
                    "id": str(session.id),
                    "task_id": str(session.task_id) if session.task_id else None,
                    "agent_id": str(session.agent_id),
                    "project_id": str(session.project_id),
                    "graph_run_id": (
                        str(session.graph_run_id) if session.graph_run_id else None
                    ),
                    "status": session.status,
                    "metadata": session.metadata_ or {},
                    "origin": session.origin,
                }
                enriched["session"] = session_payload

        task = None
        if task_id is not None:
            result = await db.execute(select(Task).where(Task.id == task_id, Task.project_id == project_id))
            task = result.scalar_one_or_none()
            if task is not None:
                graph_run_id = graph_run_id or task.graph_run_id
                canonical_graph_run_id = canonical_graph_run_id or task.graph_run_id
                metadata = {**(task.metadata_ or {}), **metadata}
                enriched["task_id"] = str(task.id)
                enriched["task"] = {
                    "id": str(task.id),
                    "project_id": str(task.project_id),
                    "parent_id": str(task.parent_id) if task.parent_id else None,
                    "graph_run_id": str(task.graph_run_id) if task.graph_run_id else None,
                    "title": task.title,
                    "status": task.status,
                    "priority": task.priority,
                    "assigned_to": str(task.assigned_to) if task.assigned_to else None,
                    "metadata": task.metadata_ or {},
                }
                if "status" in enriched and "new_status" not in enriched:
                    enriched["new_status"] = enriched["status"]

        artifact = None
        if artifact_id is not None:
            result = await db.execute(select(Artifact).where(Artifact.id == artifact_id, Artifact.project_id == project_id))
            artifact = result.scalar_one_or_none()
            if artifact is not None:
                task_id = task_id or artifact.linked_task_id
                artifact_metadata = dict(artifact.metadata_ or {})
                artifact_metadata.setdefault("type", artifact.artifact_type)
                if artifact.linked_task_id is not None:
                    artifact_metadata.setdefault("task_id", str(artifact.linked_task_id))
                metadata = {**artifact_metadata, **metadata}
                enriched["artifact_id"] = str(artifact.id)
                enriched["artifact"] = {
                    "id": str(artifact.id),
                    "project_id": str(artifact.project_id),
                    "linked_task_id": str(artifact.linked_task_id) if artifact.linked_task_id else None,
                    "name": artifact.name,
                    "artifact_type": artifact.artifact_type,
                    "status": artifact.status,
                    "path": artifact.path,
                    "url": artifact.url,
                    "metadata": artifact.metadata_ or {},
                }

        run = None
        run_graph = None
        if graph_run_id is not None:
            result = await db.execute(
                select(GraphRun).where(
                    GraphRun.id == graph_run_id,
                    GraphRun.project_id == project_id,
                )
            )
            run = result.scalar_one_or_none()
        if run is None and task_id is not None:
            result = await db.execute(
                select(GraphRun).where(
                    GraphRun.project_id == project_id,
                    GraphRun.linked_task_id == task_id,
                    GraphRun.status == "active",
                )
            )
            run = result.scalars().first()
        if run is None and artifact_id is not None:
            result = await db.execute(
                select(GraphRun).where(
                    GraphRun.project_id == project_id,
                    GraphRun.artifact_id == artifact_id,
                    GraphRun.status == "active",
                )
            )
            run = result.scalars().first()
        if run is not None:
            graph_result = await db.execute(select(Graph).where(Graph.id == run.graph_id))
            run_graph = graph_result.scalar_one_or_none()
            if canonical_graph_run_id is not None:
                enriched["graph_run_id"] = str(canonical_graph_run_id)
            enriched["graph_run"] = {
                "id": str(run.id),
                "graph_id": str(run.graph_id),
                "project_id": str(run.project_id),
                "linked_task_id": str(run.linked_task_id) if run.linked_task_id else None,
                "artifact_id": str(run.artifact_id) if run.artifact_id else None,
                "current_node": run.current_node,
                "status": run.status,
                "context": run.context or {},
            }
            if self._should_enrich_review_session(event_type, session, session_payload, run, run_graph):
                session_payload["output"] = session.output
                enriched["review_outcome"] = self._parse_review_outcome(session.output)

        enriched["metadata"] = {**metadata, **payload_metadata}
        enriched["raw_payload"] = raw_payload

        # Write CLI session output into graph run context for template/guard access
        if (
            session is not None
            and run is not None
            and session.status == "completed"
            and session.output
        ):
            run.context = {
                **(run.context or {}),
                "last_cli_output": session.output,
            }
            await db.flush()

        return enriched

    async def _check_triggers(self, db: AsyncSession, event: BusEvent) -> None:
        scoped = await self._graph_service.get_active_graphs_for_event(db, event.event_type, project_id=event.project_id)
        global_graphs = await self._graph_service.get_active_graphs_for_event(db, event.event_type, project_id=None)
        graphs = {graph.id: graph for graph in [*scoped, *global_graphs]}.values()

        for graph in graphs:
            for trigger in graph.triggers:
                if trigger.get("event_type") != event.event_type:
                    continue
                if not self._guard_evaluator.evaluate(trigger.get("conditions") or {}, event.payload):
                    continue
                await self.start_run(db, graph, event)
                break

    async def start_run(
        self,
        db: AsyncSession,
        graph: Graph,
        event: BusEvent,
    ) -> GraphRun:
        definition = graph.definition or {}
        start_node = definition.get("start_node", "")
        actor_assignments = await self._actor_resolver.resolve_all_slots(
            db,
            definition.get("actors") or {},
            triggering_event_payload=event.payload,
        )

        linked_task_id = self._uuid_from_payload(event.payload.get("task_id"))
        artifact_id = self._uuid_from_payload(event.payload.get("artifact_id"))
        existing = await self._find_existing_active_run(
            db,
            graph_id=graph.id,
            project_id=event.project_id,
            artifact_id=artifact_id,
            linked_task_id=linked_task_id,
            triggering_event_id=event.id,
        )
        if existing is not None:
            logger.info(
                "Skipping duplicate graph run for graph=%s project=%s artifact=%s task=%s",
                graph.name,
                event.project_id,
                artifact_id,
                linked_task_id,
            )
            return existing

        await ProjectService().require_runnable_project(db, event.project_id)
        run = await self._graph_service.create_run(
            db,
            graph,
            project_id=event.project_id,
            start_node=start_node,
            linked_task_id=linked_task_id,
            artifact_id=artifact_id,
            triggering_event_id=event.id,
        )
        run.actor_assignments = {
            role: assignment for role, assignment in actor_assignments.items() if assignment is not None
        }
        run.context = {
            **(run.context or {}),
            "trigger_event_type": event.event_type,
        }
        await db.flush()

        await emit_event(
            db,
            event.project_id,
            "graph.run_started",
            {
                "graph_run_id": str(run.id),
                "graph_name": graph.name,
                "start_node": start_node,
                "project_id": str(event.project_id),
            },
            _bus=self._bus,
        )

        await self._record_step(
            db,
            run,
            from_node="",
            to_node=start_node,
            trigger_event_id=event.id,
            trigger_reason="event",
        )

        node_def = (definition.get("nodes") or {}).get(start_node, {})
        await self._execute_on_enter(db, node_def, run)
        await self._register_timeout(db, node_def, run)
        return run

    async def _find_existing_active_run(
        self,
        db: AsyncSession,
        graph_id: uuid.UUID,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID | None,
        linked_task_id: uuid.UUID | None,
        triggering_event_id: uuid.UUID | None,
    ) -> GraphRun | None:
        conditions = [
            GraphRun.graph_id == graph_id,
            GraphRun.project_id == project_id,
            GraphRun.status == "active",
        ]
        if artifact_id is not None:
            conditions.append(GraphRun.artifact_id == artifact_id)
        elif linked_task_id is not None:
            conditions.append(GraphRun.linked_task_id == linked_task_id)
        elif triggering_event_id is not None:
            conditions.append(GraphRun.triggering_event_id == triggering_event_id)
        else:
            return None

        result = await db.execute(select(GraphRun).where(*conditions))
        return result.scalars().first()

    async def _evaluate_active_runs(self, db: AsyncSession, event: BusEvent) -> None:
        result = await db.execute(
            select(GraphRun).where(
                GraphRun.project_id == event.project_id,
                GraphRun.status == "active",
            )
        )
        for run in result.scalars().all():
            await self._evaluate_run(db, run, event)

    async def _evaluate_run(self, db: AsyncSession, run: GraphRun, event: BusEvent) -> None:
        event_run_id = self._uuid_from_payload(event.payload.get("graph_run_id"))
        if event_run_id is not None and event_run_id != run.id:
            return

        result = await db.execute(select(Graph).where(Graph.id == run.graph_id))
        graph = result.scalar_one_or_none()
        if graph is None:
            return

        nodes = (graph.definition or {}).get("nodes") or {}
        current_node_def = nodes.get(run.current_node) or {}
        for edge in current_node_def.get("edges") or []:
            if edge.get("trigger_event") != event.event_type:
                continue
            resolved_guard = await self._resolve_guard(db, edge.get("guard") or {}, run)
            if not self._guard_evaluator.evaluate(resolved_guard, event.payload):
                continue
            await self._fire_edge(
                db,
                run,
                graph,
                from_node=run.current_node,
                to_node=edge["to"],
                edge_name=edge.get("name"),
                actions=edge.get("actions") or [],
                trigger_event_id=event.id,
                trigger_reason="event",
            )
            break

    async def process_timeouts(self, db: AsyncSession) -> int:
        now = datetime.now(timezone.utc)
        result = await db.execute(
            select(GraphRunTimeout).where(
                GraphRunTimeout.resolved.is_(False),
                GraphRunTimeout.expires_at <= now,
            )
        )
        timeouts = result.scalars().all()
        for timeout in timeouts:
            await self._process_timeout(db, timeout, now)
        return len(timeouts)

    async def _process_timeout(self, db: AsyncSession, timeout: GraphRunTimeout, now: datetime) -> None:
        result = await db.execute(
            select(GraphRun).where(
                GraphRun.id == timeout.graph_run_id,
                GraphRun.status == "active",
            )
        )
        run = result.scalar_one_or_none()
        if run is None:
            timeout.resolved = True
            timeout.resolved_at = now
            await db.flush()
            return

        await ProjectService().require_runnable_project(db, run.project_id)
        graph_result = await db.execute(select(Graph).where(Graph.id == run.graph_id))
        graph = graph_result.scalar_one_or_none()
        graph_name = graph.name if graph is not None else ""
        new_step = (run.escalation_step or 0) + 1

        emitted_alert = False
        if graph is not None and graph.escalation_chain:
            escalation_service = EscalationChainService()
            chain = await escalation_service.get_by_name(db, graph.escalation_chain, project_id=run.project_id)
            if chain is None:
                chain = await escalation_service.get_by_name(db, graph.escalation_chain, project_id=None)
            steps = chain.steps if chain is not None else []
            run.escalation_step = min(new_step, max(len(steps), 1))
            step_def = next((step for step in steps if step.get("step") == run.escalation_step), None)
            if step_def and step_def.get("action") == "human_notify":
                await escalation_service.notify_humans(
                    db,
                    run.project_id,
                    run.id,
                    graph_name,
                    run.current_node,
                    step_def.get("message_template", ""),
                )
                emitted_alert = True

        else:
            run.escalation_step = new_step

        if not emitted_alert:
            await emit_event(
                db,
                run.project_id,
                "graph.run_escalated",
                {
                    "graph_run_id": str(run.id),
                    "graph_name": graph_name,
                    "current_node": run.current_node,
                    "timeout_id": str(timeout.id),
                    "timeout_action": timeout.timeout_action,
                    "escalation_step": run.escalation_step,
                },
                _bus=self._bus,
            )

        timeout.resolved = True
        timeout.resolved_at = now
        await db.flush()

    async def _resolve_guard(self, db: AsyncSession, guard: dict, run: GraphRun) -> dict:
        resolved = {}
        for key, value in guard.items():
            if isinstance(value, str):
                resolved[key] = await self._template_resolver.resolve(db, value, run)
            elif isinstance(value, dict):
                resolved[key] = await self._template_resolver.resolve_dict(db, value, run)
            else:
                resolved[key] = value
        return resolved

    async def _fire_edge(
        self,
        db: AsyncSession,
        run: GraphRun,
        graph: Graph,
        from_node: str,
        to_node: str,
        edge_name: str | None,
        actions: list[dict],
        trigger_event_id: uuid.UUID | None = None,
        trigger_reason: str = "event",
    ) -> None:
        await ProjectService().require_runnable_project(db, run.project_id)
        await self._resolve_timeout(db, run)

        now = datetime.now(timezone.utc)
        run.current_node = to_node
        run.last_stepped_at = now

        terminal_nodes = (graph.definition or {}).get("terminal_nodes") or {}
        if to_node in (terminal_nodes.get("success") or []):
            run.status = "completed"
            run.completed_at = now
        elif to_node in (terminal_nodes.get("failure") or []):
            run.status = "failed"
            run.completed_at = now
        await db.flush()

        executed_actions = await self._action_executor.execute_all(db, actions, run)
        step = await self._record_step(
            db,
            run,
            from_node=from_node,
            to_node=to_node,
            edge_name=edge_name,
            trigger_event_id=trigger_event_id,
            trigger_reason=trigger_reason,
            actions_executed=executed_actions,
        )

        node_def = ((graph.definition or {}).get("nodes") or {}).get(to_node, {})
        await self._execute_on_enter(db, node_def, run)
        await self._register_timeout(db, node_def, run)

        event_type = "graph.run_advanced"
        if run.status == "completed":
            event_type = "graph.run_completed"
        elif run.status == "failed":
            event_type = "graph.run_failed"

        await emit_event(
            db,
            run.project_id,
            event_type,
            {
                "graph_run_id": str(run.id),
                "graph_run_step_id": str(step.id),
                "graph_name": graph.name,
                "from_node": from_node,
                "to_node": to_node,
                "edge_name": edge_name,
                "project_id": str(run.project_id),
            },
            _bus=self._bus,
        )

    async def _execute_on_enter(self, db: AsyncSession, node_def: dict, run: GraphRun) -> None:
        on_enter = node_def.get("on_enter") or []
        if on_enter:
            await self._action_executor.execute_all(db, on_enter, run)

    async def _record_step(
        self,
        db: AsyncSession,
        run: GraphRun,
        from_node: str,
        to_node: str,
        edge_name: str | None = None,
        trigger_event_id: uuid.UUID | None = None,
        trigger_reason: str | None = None,
        actions_executed: list | None = None,
    ) -> GraphRunStep:
        step = GraphRunStep(
            graph_run_id=run.id,
            from_node=from_node,
            to_node=to_node,
            edge_name=edge_name,
            trigger_event_id=trigger_event_id,
            trigger_reason=trigger_reason,
            actions_executed=actions_executed or [],
        )
        db.add(step)
        await db.flush()
        return step

    async def _register_timeout(self, db: AsyncSession, node_def: dict, run: GraphRun) -> None:
        timeout = node_def.get("timeout")
        if not timeout:
            return
        record = GraphRunTimeout(
            graph_run_id=run.id,
            node_name=run.current_node,
            timeout_action=timeout.get("action", "escalate"),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=_parse_duration(timeout.get("duration", "1h"))),
        )
        db.add(record)
        await db.flush()

    async def _resolve_timeout(self, db: AsyncSession, run: GraphRun) -> None:
        result = await db.execute(
            select(GraphRunTimeout).where(
                GraphRunTimeout.graph_run_id == run.id,
                GraphRunTimeout.resolved.is_(False),
            )
        )
        for timeout in result.scalars().all():
            timeout.resolved = True
            timeout.resolved_at = datetime.now(timezone.utc)
        await db.flush()

    async def advance_manually(
        self,
        db: AsyncSession,
        run_or_id: GraphRun | uuid.UUID,
        to_node: str,
        reason: str | None = None,
    ) -> GraphRun:
        from fastapi import HTTPException

        if isinstance(run_or_id, GraphRun):
            run = run_or_id
            if run.status != "active":
                raise HTTPException(status_code=409, detail=f"Graph run is {run.status}")
        else:
            result = await db.execute(select(GraphRun).where(GraphRun.id == run_or_id))
            run = result.scalar_one_or_none()
            if run is None:
                raise HTTPException(status_code=404, detail="Graph run not found")
            if run.status != "active":
                raise HTTPException(status_code=409, detail=f"Graph run is {run.status}")

        graph_result = await db.execute(select(Graph).where(Graph.id == run.graph_id))
        graph = graph_result.scalar_one_or_none()
        if graph is None:
            raise HTTPException(status_code=404, detail="Graph not found")
        nodes = (graph.definition or {}).get("nodes") or {}
        if to_node not in nodes:
            raise HTTPException(status_code=400, detail=f"Unknown graph node: {to_node}")
        await self._fire_edge(
            db,
            run,
            graph,
            from_node=run.current_node,
            to_node=to_node,
            edge_name="manual_advance",
            actions=[],
            trigger_reason=reason or "manual",
        )
        return run

    def _uuid_from_payload(self, value: str | None) -> uuid.UUID | None:
        if not value:
            return None
        try:
            return uuid.UUID(str(value))
        except ValueError:
            return None

    def _parse_review_outcome(self, raw_output: str | None) -> dict:
        normalized_output = (raw_output or "").upper()
        if self._is_changes_requested_review_output(normalized_output):
            verdict = "changes_requested"
        elif self._is_approved_review_output(normalized_output):
            verdict = "approved"
        else:
            verdict = "commented"
        return {
            "verdict": verdict,
            "raw_output": raw_output,
        }

    def _should_enrich_review_session(
        self,
        event_type: str | None,
        session: Session | None,
        session_payload: dict | None,
        run: GraphRun | None,
        graph: Graph | None,
    ) -> bool:
        return (
            event_type == "session.completed"
            and session is not None
            and session.origin == "graph"
            and session_payload is not None
            and run is not None
            and run.current_node == "ready_for_review"
            and graph is not None
            and graph.name == "code_review"
            and self._session_matches_assigned_review_actor(run, session)
        )

    def _is_approved_review_output(self, normalized_output: str) -> bool:
        if not normalized_output:
            return False
        if re.search(r"\b(?:DO\s+NOT|NOT)\s+APPROVE(?:D)?\b", normalized_output):
            return False
        if re.search(r"\bDISAPPROVE\b", normalized_output):
            return False
        return re.search(r"\bAPPROVE(?:D)?\b", normalized_output) is not None

    def _is_changes_requested_review_output(self, normalized_output: str) -> bool:
        if not normalized_output:
            return False
        return re.search(r"\b(?:CHANGES[_\s]+REQUESTED|REQUEST(?:ED)?\s+CHANGES)\b", normalized_output) is not None

    def _session_matches_assigned_review_actor(self, run: GraphRun, session: Session) -> bool:
        reviewer_assignment = (run.actor_assignments or {}).get("reviewer") or {}
        if reviewer_assignment.get("kind") != "agent":
            return False
        reviewer_id = self._uuid_from_payload(reviewer_assignment.get("id"))
        return reviewer_id is not None and reviewer_id == session.agent_id

    async def _maybe_auto_emit_review_graph_event(self, db: AsyncSession, event: BusEvent) -> BusEvent | None:
        if event.event_type != "session.completed":
            return None

        outcome = event.payload.get("review_outcome") or {}
        event_type = self._review_graph_event_type(outcome.get("verdict"))
        if event_type is None:
            return None

        session_id = self._uuid_from_payload(event.payload.get("session_id"))
        graph_run_id = self._graph_run_uuid_from_payload(event.payload)
        artifact_id = self._artifact_uuid_from_payload(event.payload)
        if session_id is None or graph_run_id is None or artifact_id is None:
            return None

        result = await db.execute(select(Session).where(Session.id == session_id, Session.project_id == event.project_id))
        session = result.scalar_one_or_none()
        if session is None:
            return None
        run_result = await db.execute(
            select(GraphRun).where(
                GraphRun.id == graph_run_id,
                GraphRun.project_id == event.project_id,
            )
        )
        run = run_result.scalar_one_or_none()
        if run is None or not self._session_matches_assigned_review_actor(run, session):
            return None

        payload = {
            "artifact_id": str(artifact_id),
            "graph_run_id": str(graph_run_id),
            "session_id": str(session.id),
        }
        if event.payload.get("task_id"):
            payload["task_id"] = event.payload["task_id"]
        if outcome:
            payload["review_outcome"] = outcome

        emitted, created = await emit_event_once(
            db,
            event.project_id,
            event_type,
            payload,
            source="graph",
            dedup_key=self._derived_review_event_dedup_key(session.id, event_type),
            _bus=self._bus,
        )
        session.metadata_ = {
            **(session.metadata_ or {}),
            "auto_review_graph_event": {
                "event_type": event_type,
                "event_id": str(emitted.id),
                "graph_run_id": str(graph_run_id),
            },
        }
        await db.flush()
        if not created:
            return None
        return emitted

    def _review_graph_event_type(self, verdict: str | None) -> str | None:
        if verdict == "approved":
            return "review.approved"
        if verdict == "changes_requested":
            return "review.changes_requested"
        return None

    def _derived_review_event_dedup_key(self, session_id: uuid.UUID, event_type: str) -> str:
        return f"derived_review:{event_type}:{session_id}"

    def _graph_run_uuid_from_payload(self, payload: dict) -> uuid.UUID | None:
        graph_run_id = self._uuid_from_payload(payload.get("graph_run_id"))
        if graph_run_id is not None:
            return graph_run_id
        run = payload.get("graph_run") or {}
        return self._uuid_from_payload(run.get("id"))

    def _artifact_uuid_from_payload(self, payload: dict) -> uuid.UUID | None:
        artifact_id = self._uuid_from_payload(payload.get("artifact_id"))
        if artifact_id is not None:
            return artifact_id
        run = payload.get("graph_run") or {}
        return self._uuid_from_payload(run.get("artifact_id"))
