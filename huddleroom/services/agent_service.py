from __future__ import annotations

import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from huddleroom.models.agent import Agent
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.schemas.agent import AgentCreate, AgentUpdate, AgentContextResponse, AgentResponse, AgentTaskSummary
from huddleroom.services.event_bus import emit_event

GLOBAL_PROJECT_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


def _has_cli_runtime(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _effective_cli_runtime(cli_runtime: str | None, config: dict) -> object:
    return config["cli_runtime"] if "cli_runtime" in config else cli_runtime


class AgentService:
    async def create(self, db: AsyncSession, data: AgentCreate) -> Agent:
        if data.adapter_type == "cli" and not _has_cli_runtime(
            _effective_cli_runtime(data.cli_runtime, data.config)
        ):
            raise HTTPException(status_code=422, detail="CLI agents require cli_runtime")
        agent = Agent(
            name=data.name,
            role=data.role,
            description=data.description,
            provider=data.provider,
            model=data.model,
            system_prompt=data.system_prompt,
            adapter_type=data.adapter_type,
            cli_runtime=data.cli_runtime,
            capabilities=data.capabilities,
            config=data.config,
        )
        db.add(agent)
        await db.flush()
        await emit_event(db, GLOBAL_PROJECT_ID, "agent.created", {
            "agent_id": str(agent.id),
            "name": agent.name,
            "role": agent.role,
        })
        project_result = await db.execute(select(Project.id))
        for project_id in project_result.scalars().all():
            await emit_event(db, project_id, "agent.created", {
                "agent_id": str(agent.id),
                "name": agent.name,
                "role": agent.role,
            })
        return agent

    async def get(self, db: AsyncSession, agent_id: uuid.UUID) -> Agent | None:
        result = await db.execute(select(Agent).where(Agent.id == agent_id))
        return result.scalar_one_or_none()

    async def get_or_404(self, db: AsyncSession, agent_id: uuid.UUID) -> Agent:
        agent = await self.get(db, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent

    async def list(
        self,
        db: AsyncSession,
        role: str | None = None,
        is_active: bool | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> tuple[list[Agent], str | None]:
        from datetime import datetime
        from sqlalchemy import or_, and_
        query = select(Agent).order_by(Agent.created_at.desc(), Agent.id.desc()).limit(limit + 1)
        if role is not None:
            query = query.where(Agent.role == role)
        if is_active is not None:
            query = query.where(Agent.is_active == is_active)
        if cursor:
            try:
                cursor_dt_str, cursor_id_str = cursor.split("__", 1)
                cursor_dt = datetime.fromisoformat(cursor_dt_str)
                cursor_id = uuid.UUID(cursor_id_str)
            except (ValueError, AttributeError) as exc:
                raise HTTPException(status_code=400, detail="Invalid cursor") from exc
            query = query.where(
                or_(
                    Agent.created_at < cursor_dt,
                    and_(Agent.created_at == cursor_dt, Agent.id < cursor_id),
                )
            )
        result = await db.execute(query)
        items = list(result.scalars().all())
        next_cursor = None
        if len(items) > limit:
            items = items[:limit]
            last = items[-1]
            next_cursor = f"{last.created_at.isoformat()}__{last.id}"
        return items, next_cursor

    async def update(self, db: AsyncSession, agent_id: uuid.UUID, data: AgentUpdate) -> Agent:
        agent = await self.get_or_404(db, agent_id)
        resulting_adapter_type = data.adapter_type if "adapter_type" in data.model_fields_set else agent.adapter_type
        runtime_is_supplied = "cli_runtime" in data.model_fields_set
        config_is_supplied = "config" in data.model_fields_set and data.config is not None
        resulting_config = data.config if config_is_supplied else agent.config
        if resulting_adapter_type == "cli" and (
            (data.adapter_type == "cli" and agent.adapter_type != "cli")
            or runtime_is_supplied
            or config_is_supplied
        ) and not _has_cli_runtime(_effective_cli_runtime(
            data.cli_runtime if runtime_is_supplied else agent.cli_runtime, resulting_config
        )):
            raise HTTPException(status_code=422, detail="CLI agents require cli_runtime")
        scalar_fields = {"name", "role", "description", "provider", "model", "system_prompt",
                         "adapter_type", "cli_runtime"}
        json_fields = {"capabilities", "config"}
        for field in data.model_fields_set & scalar_fields:
            setattr(agent, field, getattr(data, field))
        for field in data.model_fields_set & json_fields:
            val = getattr(data, field)
            if val is not None:
                setattr(agent, field, val)
        if data.adapter_type and data.adapter_type != "cli":
            # Prevent stale CLI-only state from surviving an adapter switch.
            agent.cli_runtime = None
            if "cli_runtime" in agent.config:
                agent.config = {
                    key: value for key, value in agent.config.items() if key != "cli_runtime"
                }
        # is_active is boolean — False is a valid value, must check for None explicitly
        if data.is_active is not None:
            agent.is_active = data.is_active
        await db.flush()
        return agent

    async def delete(self, db: AsyncSession, agent_id: uuid.UUID) -> None:
        agent = await self.get_or_404(db, agent_id)
        agent.is_active = False
        await db.flush()

    async def build_context(self, db: AsyncSession, agent_id: uuid.UUID) -> AgentContextResponse:
        agent = await self.get_or_404(db, agent_id)
        # Get assigned tasks
        result = await db.execute(
            select(Task).where(
                Task.assigned_to == agent_id,
                Task.status.in_(["backlog", "ready", "in_progress", "blocked"]),
            ).limit(20)
        )
        tasks = result.scalars().all()
        task_summaries = [
            AgentTaskSummary(id=t.id, title=t.title, status=t.status, priority=t.priority)
            for t in tasks
        ]
        return AgentContextResponse(
            agent=AgentResponse.model_validate(agent),
            current_tasks=task_summaries,
            pending_meetings=[],   # Not included in the agent context yet.
            recent_knowledge=[],   # Not included in the agent context yet.
            active_protocol_instances=[],  # Not included in the agent context yet.
        )
