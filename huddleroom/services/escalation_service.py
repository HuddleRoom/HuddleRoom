from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.escalation import EscalationChain
from huddleroom.services.event_bus import emit_event


def _extract_steps(definition: dict) -> list:
    return definition.get("steps", [])


class EscalationChainService:
    async def load_from_yaml(
        self,
        db: AsyncSession,
        path: Path,
        project_id: uuid.UUID | None = None,
    ) -> EscalationChain:
        """Load escalation chain from YAML file and upsert by (project_id, name)."""
        text = path.read_text()
        definition: dict[str, Any] = yaml.safe_load(text)

        name = definition["name"]
        description = definition.get("description")
        steps = _extract_steps(definition)

        # Query by (project_id, name)
        result = await db.execute(
            select(EscalationChain).where(
                EscalationChain.project_id.is_(None) if project_id is None else EscalationChain.project_id == project_id,
                EscalationChain.name == name,
            )
        )
        chain = result.scalar_one_or_none()
        if chain is None:
            chain = EscalationChain(
                project_id=project_id,
                name=name,
                description=description,
                definition=definition,
                steps=steps,
                is_active=True,
            )
            db.add(chain)
        else:
            chain.description = description
            chain.definition = definition
            chain.steps = steps

        await db.flush()
        return chain

    async def load_all_from_workspace(
        self, db: AsyncSession, workspace_dir: Path, project_id: uuid.UUID | None = None
    ) -> list[EscalationChain]:
        """Load all escalation chains from workspace_dir/escalation_chains/*.yaml."""
        chains_dir = workspace_dir / "escalation_chains"
        if not chains_dir.exists():
            return []
        loaded = []
        for yaml_file in sorted(chains_dir.glob("*.yaml")):
            chain = await self.load_from_yaml(db, yaml_file, project_id=project_id)
            loaded.append(chain)
        return loaded

    async def get_by_name(
        self, db: AsyncSession, name: str, project_id: uuid.UUID | None = None
    ) -> EscalationChain | None:
        """Get escalation chain by name, filtered by project."""
        result = await db.execute(
            select(EscalationChain).where(
                EscalationChain.name == name,
                EscalationChain.project_id.is_(None) if project_id is None else EscalationChain.project_id == project_id,
                EscalationChain.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        db: AsyncSession,
        project_id: uuid.UUID | None = None,
        active_only: bool = True,
    ) -> list[EscalationChain]:
        """List escalation chains, filtered by project."""
        q = select(EscalationChain).where(
            EscalationChain.project_id.is_(None) if project_id is None else EscalationChain.project_id == project_id
        )
        if active_only:
            q = q.where(EscalationChain.is_active.is_(True))
        result = await db.execute(q)
        return list(result.scalars().all())

    async def notify_humans(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        protocol_instance_id: uuid.UUID,
        protocol_name: str,
        current_state: str,
        message_template: str,
    ) -> None:
        """Replace template vars and emit system.escalation_alert event."""
        # Build the message string from template
        msg = (
            message_template
            .replace("{{protocol_name}}", protocol_name)
            .replace("{{current_state}}", current_state)
            .replace("{{protocol_instance_id}}", str(protocol_instance_id))
        )

        # Emit event
        await emit_event(
            db,
            project_id=project_id,
            event_type="system.escalation_alert",
            payload={
                "protocol_instance_id": str(protocol_instance_id),
                "protocol_name": protocol_name,
                "current_state": current_state,
                "message": msg,
            },
            source="escalation_service",
        )
