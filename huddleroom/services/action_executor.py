from __future__ import annotations

import logging
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.graph import GraphRun
from huddleroom.services.channel_service import ChannelService
from huddleroom.services.event_bus import EventBusService, emit_event
from huddleroom.services.message_service import MessageService
from huddleroom.services.template_resolver import TemplateResolver

logger = logging.getLogger(__name__)


class ActionExecutor:
    def __init__(self, _bus: EventBusService | None = None) -> None:
        self._bus = _bus
        self._resolver = TemplateResolver()
        self._message_service = MessageService()
        self._channel_service = ChannelService()

    async def _r(self, db: AsyncSession, value: str | None, run: GraphRun) -> str:
        """Resolve template variables in an action parameter string."""
        if not value or "{{" not in value:
            return value or ""
        return await self._resolver.resolve(db, value, run)

    async def execute(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        action_type = action.get("action_type")
        resolved = await self._resolver.resolve_dict(db, action, run)

        handler = {
            "emit_event": self._emit_event,
            "assign_task": self._assign_task,
            "create_session": self._create_session,
            "post_message": self._post_message,
            "notify_actor": self._notify_actor,
            "record_decision": self._record_decision,
            "complete_task": self._complete_task,
            "update_graph_context": self._update_context,
            "set_artifact_status": self._set_artifact_status,
            "trigger_escalation": self._trigger_escalation,
            "start_meeting": self._start_meeting_stub,
        }.get(action_type)
        if handler is None:
            logger.warning("Unknown graph action_type: %s", action_type)
            return {"action_type": action_type, "status": "unknown"}
        return await handler(db, resolved, run)

    async def execute_all(self, db: AsyncSession, actions: list[dict], run: GraphRun) -> list[dict]:
        results = []
        for action in actions:
            results.append(await self.execute(db, action, run))
        return results

    async def _emit_event(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        from huddleroom.models.artifact import Artifact
        from huddleroom.models.graph import Graph
        from huddleroom.models.task import Task

        payload = dict(action.get("payload_overrides") or {})
        metadata = dict(payload.get("metadata") or {})

        graph_result = await db.execute(select(Graph).where(Graph.id == run.graph_id))
        graph = graph_result.scalar_one_or_none()
        if graph is not None:
            metadata.setdefault("graph", graph.name)

        if run.linked_task_id is not None:
            task_result = await db.execute(select(Task).where(Task.id == run.linked_task_id))
            task = task_result.scalar_one_or_none()
            if task is not None:
                metadata = {**(task.metadata_ or {}), **metadata}
            metadata["task_id"] = str(run.linked_task_id)
            payload["task_id"] = str(run.linked_task_id)

        if run.artifact_id is not None:
            artifact_result = await db.execute(select(Artifact).where(Artifact.id == run.artifact_id))
            artifact = artifact_result.scalar_one_or_none()
            if artifact is not None:
                artifact_metadata = dict(artifact.metadata_ or {})
                artifact_metadata.setdefault("type", artifact.artifact_type)
                metadata = {**artifact_metadata, **metadata}
            metadata["artifact_id"] = str(run.artifact_id)
            payload["artifact_id"] = str(run.artifact_id)

        metadata["graph_run_id"] = str(run.id)
        payload["graph_run_id"] = str(run.id)
        payload["metadata"] = metadata
        await emit_event(
            db,
            run.project_id,
            action["event_type"],
            payload,
            source="graph",
            _bus=self._bus,
        )
        return {"action_type": "emit_event", "event_type": action["event_type"]}

    async def _assign_task(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        from huddleroom.models.task import Task

        assigned_to = self._actor_uuid(run, action.get("to_actor"))
        task_title = await self._r(db, action.get("task_title", "Graph task"), run)
        task_description = await self._r(db, action.get("task_description"), run)
        task = Task(
            project_id=run.project_id,
            title=task_title,
            description=task_description,
            assigned_to=assigned_to,
            graph_run_id=run.id,
            status="ready",
        )
        db.add(task)
        await db.flush()
        await emit_event(
            db,
            run.project_id,
            "task.created",
            {
                "task_id": str(task.id),
                "title": task.title,
                "graph_run_id": str(run.id),
                "project_id": str(run.project_id),
            },
            _bus=self._bus,
        )
        return {"action_type": "assign_task", "task_id": str(task.id)}

    async def _create_session(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        from huddleroom.models.task import Task
        from huddleroom.schemas.session import SessionCreate
        from huddleroom.services.session_service import SessionService

        assigned_to = self._actor_uuid(run, action.get("actor"))
        if assigned_to is None:
            return {"action_type": "create_session", "status": "actor_not_found"}

        task_title = await self._r(db, action.get("task_title", "Graph session"), run)
        task_description = await self._r(db, action.get("task_description"), run)
        task = Task(
            project_id=run.project_id,
            title=task_title,
            description=task_description,
            assigned_to=assigned_to,
            graph_run_id=run.id,
            status="ready",
        )
        db.add(task)
        await db.flush()

        result = {"action_type": "create_session", "task_id": str(task.id)}
        session = await SessionService().create(
            db,
            SessionCreate(
                agent_id=assigned_to,
                task_id=task.id,
                project_id=run.project_id,
                graph_run_id=run.id,
                origin="graph",
            ),
        )
        result["session_id"] = str(session.id)
        return result

    async def _post_message(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        from huddleroom.schemas.channel import ChannelCreate

        channel_name = action.get("channel", "general")
        from huddleroom.models.channel import Channel

        channel_result = await db.execute(
            select(Channel).where(
                Channel.project_id == run.project_id,
                Channel.name == channel_name,
            )
        )
        channel = channel_result.scalar_one_or_none()
        if channel is None:
            channel = await self._channel_service.create(
                db,
                run.project_id,
                ChannelCreate(name=channel_name, channel_type="general"),
            )
        content = await self._r(db, action.get("template", ""), run)
        message = await self._message_service.create(
            db,
            channel.id,
            content,
            sender_user_id=await self._ensure_system_user_id(db, run),
            metadata={"graph_run_id": str(run.id)},
        )
        return {"action_type": "post_message", "channel": channel_name, "message_id": str(message.id)}

    async def _notify_actor(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        from huddleroom.schemas.channel import ChannelCreate

        actor_role = action.get("actor")
        slot = (run.actor_assignments or {}).get(actor_role) if actor_role else None
        if not slot:
            return {"action_type": "notify_actor", "status": "actor_not_found"}

        dm_name = f"dm-{slot['id']}"
        from huddleroom.models.channel import Channel

        result = await db.execute(
            select(Channel).where(
                Channel.project_id == run.project_id,
                Channel.name == dm_name,
            )
        )
        channel = result.scalar_one_or_none()
        if channel is None:
            channel = await self._channel_service.create(
                db,
                run.project_id,
                ChannelCreate(name=dm_name, channel_type="general"),
            )
        message_text = await self._r(db, action.get("message", ""), run)
        await self._message_service.create(
            db,
            channel.id,
            message_text,
            sender_user_id=await self._ensure_system_user_id(db, run),
            metadata={"graph_run_id": str(run.id), "actor_role": actor_role},
        )
        return {"action_type": "notify_actor", "actor_role": actor_role, "channel": dm_name}

    async def _record_decision(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        from huddleroom.models.knowledge_item import KnowledgeItem

        summary = await self._r(db, action.get("summary", ""), run)
        rationale = await self._r(db, action.get("rationale", ""), run)
        item = KnowledgeItem(
            project_id=run.project_id,
            title=summary[:200] or None,
            content=f"Decision: {summary}\n\nRationale: {rationale}",
            content_type="decision",
            provenance_type="graph",
            provenance_graph_run_id=run.id,
            metadata_={"graph_run_id": str(run.id)},
        )
        db.add(item)
        await db.flush()
        return {"action_type": "record_decision", "knowledge_item_id": str(item.id)}

    async def _complete_task(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        from fastapi import HTTPException
        from huddleroom.models.task import Task
        from huddleroom.services.task_service import TaskService

        task_id_str = await self._r(db, action.get("task_id"), run)
        if not task_id_str:
            task_id_str = str(run.linked_task_id) if run.linked_task_id else ""
        if not task_id_str:
            return {"action_type": "complete_task", "status": "no_task_id"}
        try:
            task_id = uuid.UUID(task_id_str)
        except ValueError:
            return {"action_type": "complete_task", "status": "invalid_task_id"}

        result = await db.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()
        if task is None:
            return {"action_type": "complete_task", "status": "task_not_found", "task_id": task_id_str}
        if task.status in ("done", "cancelled"):
            return {"action_type": "complete_task", "task_id": task_id_str}

        task_service = TaskService()
        try:
            if task.status == "backlog":
                task = await task_service.transition_status(db, task.project_id, task.id, "ready")
            if task.status in ("failed", "blocked", "ready"):
                task = await task_service.transition_status(db, task.project_id, task.id, "in_progress")
            if task.status == "in_progress":
                await task_service.transition_status(db, task.project_id, task.id, "done")
            else:
                return {"action_type": "complete_task", "status": "invalid_transition", "task_id": task_id_str}
        except HTTPException as exc:
            from huddleroom.services.session_service import SessionClaimAttention, SessionService
            if isinstance(exc, SessionClaimAttention):
                await SessionService.persist_claim_attention(db, exc)
                return {"action_type": "complete_task", "status": "claim_refused", "task_id": task_id_str}
            if exc.status_code == 409:
                return {"action_type": "complete_task", "status": "invalid_transition", "task_id": task_id_str}
            raise
        return {"action_type": "complete_task", "task_id": task_id_str}

    async def _update_context(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        key = action.get("key")
        if key:
            run.context = {**(run.context or {}), key: action.get("value")}
            await db.flush()
        return {"action_type": "update_graph_context", "key": key}

    async def _set_artifact_status(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        from huddleroom.models.artifact import Artifact

        artifact_id_str = action.get("artifact_id") or (str(run.artifact_id) if run.artifact_id else "")
        try:
            artifact_id = uuid.UUID(artifact_id_str)
        except (TypeError, ValueError):
            return {"action_type": "set_artifact_status", "status": "invalid_id"}

        result = await db.execute(select(Artifact).where(Artifact.id == artifact_id))
        artifact = result.scalar_one_or_none()
        if artifact is not None:
            artifact.status = action.get("status", artifact.status)
            await db.flush()
        return {"action_type": "set_artifact_status", "artifact_id": artifact_id_str}

    async def _trigger_escalation(self, db: AsyncSession, action: dict, run: GraphRun) -> dict:
        await emit_event(
            db,
            run.project_id,
            "graph.run_escalated",
            {
                "graph_run_id": str(run.id),
                "chain_name": action.get("chain_name"),
                "current_node": run.current_node,
            },
            _bus=self._bus,
        )
        return {"action_type": "trigger_escalation", "chain_name": action.get("chain_name")}

    async def _start_meeting_stub(self, _db: AsyncSession, _action: dict, run: GraphRun) -> dict:
        logger.info("start_meeting deferred for graph run %s", run.id)
        return {"action_type": "start_meeting", "status": "deferred_m4"}

    def _actor_uuid(self, run: GraphRun, actor_role: str | None) -> uuid.UUID | None:
        if not actor_role:
            return None
        slot = (run.actor_assignments or {}).get(actor_role)
        if not slot or slot.get("kind") != "agent":
            return None
        try:
            return uuid.UUID(str(slot["id"]))
        except (KeyError, TypeError, ValueError):
            return None

    def _system_user_id(self, run: GraphRun) -> uuid.UUID | None:
        raw_id = (run.context or {}).get("system_user_id")
        if raw_id is None:
            return None
        try:
            return uuid.UUID(str(raw_id))
        except (TypeError, ValueError):
            return None

    async def _ensure_system_user_id(self, db: AsyncSession, run: GraphRun) -> uuid.UUID:
        existing_id = self._system_user_id(run)
        if existing_id is not None:
            return existing_id

        from huddleroom.models.user import User

        result = await db.execute(select(User).where(User.email == "graph-system@local"))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(
                email="graph-system@local",
                hashed_password="!",
                display_name="Graph System",
                role="system",
            )
            db.add(user)
            try:
                await db.flush()
            except IntegrityError:
                await db.rollback()
                result = await db.execute(select(User).where(User.email == "graph-system@local"))
                user = result.scalar_one()
        run.context = {**(run.context or {}), "system_user_id": str(user.id)}
        await db.flush()
        return user.id
