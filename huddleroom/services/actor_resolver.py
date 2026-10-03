from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent


class ActorResolver:
    def _agent_has_capabilities(self, agent: Agent, required_capabilities: list[str]) -> bool:
        return all(capability in (agent.capabilities or []) for capability in required_capabilities)

    async def resolve_role(
        self,
        db: AsyncSession,
        role_value: str,
        prefer_agent_id: uuid.UUID | None = None,
    ) -> Agent | None:
        result = await db.execute(
            select(Agent).where(
                Agent.role == role_value,
                Agent.is_active.is_(True),
            )
        )
        agents = list(result.scalars().all())
        if not agents:
            return None
        if prefer_agent_id is not None:
            for agent in agents:
                if agent.id == prefer_agent_id:
                    return agent
        return sorted(agents, key=lambda agent: agent.name)[0]

    async def resolve_auto(
        self,
        db: AsyncSession,
        required_capabilities: list[str],
        prefer_agent_id: uuid.UUID | None = None,
    ) -> Agent | None:
        result = await db.execute(select(Agent).where(Agent.is_active.is_(True)))
        agents = list(result.scalars().all())
        matching = [
            agent
            for agent in agents
            if self._agent_has_capabilities(agent, required_capabilities)
        ]
        if not matching:
            return None
        if prefer_agent_id is not None:
            for agent in matching:
                if agent.id == prefer_agent_id:
                    return agent
        return sorted(matching, key=lambda agent: agent.name)[0]

    async def resolve_actor_slot(
        self,
        db: AsyncSession,
        slot_def: dict,
        triggering_event_payload: dict | None = None,
        prefer_agent_id: uuid.UUID | None = None,
    ) -> dict | None:
        assignment = slot_def.get("assignment", "auto")

        if assignment == "manual":
            return None

        if assignment == "role":
            agent = await self.resolve_role(db, slot_def.get("role_value", ""), prefer_agent_id=prefer_agent_id)
            if agent is None:
                return None
            return {"kind": "agent", "id": str(agent.id), "name": agent.name}

        if assignment == "auto":
            required_capabilities = slot_def.get("required_capabilities", [])
            if triggering_event_payload:
                author_agent_id = triggering_event_payload.get("author_agent_id")
                if author_agent_id:
                    try:
                        preferred_id = uuid.UUID(str(author_agent_id))
                    except (TypeError, ValueError):
                        preferred_id = None
                    if preferred_id is not None:
                        result = await db.execute(
                            select(Agent).where(
                                Agent.id == preferred_id,
                                Agent.is_active.is_(True),
                            )
                        )
                        event_agent = result.scalar_one_or_none()
                        if event_agent is not None and self._agent_has_capabilities(event_agent, required_capabilities):
                            return {"kind": "agent", "id": str(event_agent.id), "name": event_agent.name}

            agent = await self.resolve_auto(
                db,
                required_capabilities,
                prefer_agent_id=prefer_agent_id,
            )
            if agent is None:
                return None
            return {"kind": "agent", "id": str(agent.id), "name": agent.name}

        return None

    async def resolve_all_slots(
        self,
        db: AsyncSession,
        actors_def: dict,
        triggering_event_payload: dict | None = None,
    ) -> dict:
        assignments = {}
        for role_name, slot_def in actors_def.items():
            assignments[role_name] = await self.resolve_actor_slot(
                db,
                slot_def,
                triggering_event_payload=triggering_event_payload,
            )
        return assignments
