from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.protocol import ProtocolInstance

_VAR_RE = re.compile(r"\{\{([^}]+)\}\}")


class TemplateResolver:
    async def resolve(self, db: AsyncSession, template: str, instance: ProtocolInstance) -> str:
        artifact = None
        task = None

        async def _lookup(variable: str) -> str:
            nonlocal artifact, task

            if variable.startswith("protocol_instance."):
                attr = variable.removeprefix("protocol_instance.")
                value = getattr(instance, attr, None)
                if value is None:
                    value = (instance.context or {}).get(attr)
                return str(value) if value is not None else f"{{{{{variable}}}}}"

            if variable.startswith("artifact."):
                if artifact is None and instance.artifact_id:
                    from huddleroom.models.artifact import Artifact

                    result = await db.execute(select(Artifact).where(Artifact.id == instance.artifact_id))
                    artifact = result.scalar_one_or_none()
                if artifact is None:
                    return f"{{{{{variable}}}}}"

                attr = variable.removeprefix("artifact.")
                if attr.startswith("metadata."):
                    value = (artifact.metadata_ or {}).get(attr.removeprefix("metadata."))
                else:
                    value = getattr(artifact, attr, None)
                    if value is None:
                        value = (artifact.metadata_ or {}).get(attr)
                return str(value) if value is not None else f"{{{{{variable}}}}}"

            if variable.startswith("task."):
                if task is None and instance.linked_task_id:
                    from huddleroom.models.task import Task

                    result = await db.execute(select(Task).where(Task.id == instance.linked_task_id))
                    task = result.scalar_one_or_none()
                if task is None:
                    return f"{{{{{variable}}}}}"

                attr = variable.removeprefix("task.")
                value = getattr(task, attr, None)
                if value is None and attr.startswith("metadata."):
                    value = (task.metadata_ or {}).get(attr.removeprefix("metadata."))
                return str(value) if value is not None else f"{{{{{variable}}}}}"

            value = (instance.context or {}).get(variable)
            return str(value) if value is not None else f"{{{{{variable}}}}}"

        result = template
        for match in _VAR_RE.finditer(template):
            variable = match.group(1).strip()
            replacement = await _lookup(variable)
            result = result.replace(match.group(0), replacement, 1)
        return result

    async def resolve_dict(self, db: AsyncSession, data: dict, instance: ProtocolInstance) -> dict:
        resolved = {}
        for key, value in data.items():
            if isinstance(value, str):
                resolved[key] = await self.resolve(db, value, instance)
            elif isinstance(value, dict):
                resolved[key] = await self.resolve_dict(db, value, instance)
            elif isinstance(value, list):
                resolved[key] = await self.resolve_list(db, value, instance)
            else:
                resolved[key] = value
        return resolved

    async def resolve_list(self, db: AsyncSession, items: list, instance: ProtocolInstance) -> list:
        resolved = []
        for value in items:
            if isinstance(value, str):
                resolved.append(await self.resolve(db, value, instance))
            elif isinstance(value, dict):
                resolved.append(await self.resolve_dict(db, value, instance))
            elif isinstance(value, list):
                resolved.append(await self.resolve_list(db, value, instance))
            else:
                resolved.append(value)
        return resolved
