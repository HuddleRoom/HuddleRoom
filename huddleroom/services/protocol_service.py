from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.protocol import Protocol, ProtocolInstance


def _extract_triggers(definition: dict) -> list:
    return definition.get("triggers", [])


class ProtocolService:
    async def load_from_yaml(
        self,
        db: AsyncSession,
        path: Path,
        project_id: uuid.UUID | None = None,
    ) -> Protocol:
        text = path.read_text()
        definition: dict[str, Any] = yaml.safe_load(text)

        name = definition["name"]
        version = str(definition.get("version", "1.0"))
        triggers = _extract_triggers(definition)
        escalation_chain = definition.get("escalation_chain")

        result = await db.execute(
            select(Protocol).where(
                Protocol.project_id.is_(None) if project_id is None else Protocol.project_id == project_id,
                Protocol.name == name,
                Protocol.version == version,
            )
        )
        proto = result.scalar_one_or_none()
        if proto is None:
            proto = Protocol(
                project_id=project_id,
                name=name,
                version=version,
                description=definition.get("description"),
                definition=definition,
                triggers=triggers,
                escalation_chain=escalation_chain,
                loaded_from=str(path),
            )
            db.add(proto)
        else:
            proto.definition = definition
            proto.triggers = triggers
            proto.escalation_chain = escalation_chain
            proto.loaded_from = str(path)

        await db.flush()
        return proto

    async def load_all_from_workspace(
        self, db: AsyncSession, workspace_dir: Path, project_id: uuid.UUID | None = None
    ) -> list[Protocol]:
        protocols_dir = workspace_dir / "protocols"
        if not protocols_dir.exists():
            return []
        loaded = []
        for yaml_file in sorted(protocols_dir.glob("*.yaml")):
            proto = await self.load_from_yaml(db, yaml_file, project_id=project_id)
            loaded.append(proto)
        return loaded

    async def get(self, db: AsyncSession, protocol_id: uuid.UUID) -> Protocol | None:
        result = await db.execute(select(Protocol).where(Protocol.id == protocol_id))
        return result.scalar_one_or_none()

    async def get_by_name(
        self, db: AsyncSession, name: str, project_id: uuid.UUID | None = None
    ) -> Protocol | None:
        result = await db.execute(
            select(Protocol).where(
                Protocol.name == name,
                Protocol.project_id.is_(None) if project_id is None else Protocol.project_id == project_id,
                Protocol.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        db: AsyncSession,
        project_id: uuid.UUID | None = None,
        active_only: bool = True,
    ) -> list[Protocol]:
        q = select(Protocol).where(
            Protocol.project_id.is_(None) if project_id is None else Protocol.project_id == project_id
        )
        if active_only:
            q = q.where(Protocol.is_active.is_(True))
        result = await db.execute(q)
        return list(result.scalars().all())

    async def get_active_protocols_for_event(
        self, db: AsyncSession, event_type: str, project_id: uuid.UUID | None = None
    ) -> list[Protocol]:
        protocols = await self.list(db, project_id=project_id, active_only=True)
        matching = []
        for p in protocols:
            for trigger in p.triggers:
                if trigger.get("event_type") == event_type:
                    matching.append(p)
                    break
        return matching

    async def create_instance(
        self,
        db: AsyncSession,
        protocol: Protocol,
        project_id: uuid.UUID,
        initial_state: str,
        linked_task_id: uuid.UUID | None = None,
        artifact_id: uuid.UUID | None = None,
        triggering_event_id: uuid.UUID | None = None,
    ) -> ProtocolInstance:
        instance = ProtocolInstance(
            protocol_id=protocol.id,
            project_id=project_id,
            current_state=initial_state,
            status="active",
            linked_task_id=linked_task_id,
            artifact_id=artifact_id,
            triggering_event_id=triggering_event_id,
            actor_assignments={},
            context={},
        )
        db.add(instance)
        await db.flush()
        return instance
