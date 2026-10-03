from __future__ import annotations

import re
import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Row
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.orchestration import (
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.services.event_bus import emit_event_once

_SECTION_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")


class OrchestrationMemoryService:
    """CRUD for orchestrator-only memory sections (Spec section 5).

    Agents have no path into this service: it is wired only to the
    project-scoped human REST router and, in later phases, the orchestrator
    tick loop. Do not expose it through agent self-service or tool executors.
    """

    async def reconcile_accepted_sources(self, db: AsyncSession, goal_id: uuid.UUID, run_id: uuid.UUID) -> int:
        """Promote only one explicit accepted source from this exact run."""
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, goal_id)
        if run is None or goal is None or run.goal_id != goal.id:
            return 0
        sections = list(await db.scalars(select(OrchestrationMemorySection).where(
            OrchestrationMemorySection.project_id == goal.project_id,
            OrchestrationMemorySection.goal_id == goal.id,
            # Legacy sections predate run linkage; their accepted source still
            # has to prove this exact run below.
            or_(OrchestrationMemorySection.run_id.is_(None), OrchestrationMemorySection.run_id == run.id),
            OrchestrationMemorySection.fact_status == "unverified")))
        unresolved = 0
        promoted = False
        for section in sections:
            provenance = section.provenance if isinstance(section.provenance, dict) else {}
            sources = [(kind, provenance.get(f"{kind}_id")) for kind in ("decision", "evidence") if f"{kind}_id" in provenance]
            if len(sources) != 1:
                unresolved += 1; continue
            kind, source_id = sources[0]
            try:
                source_id = uuid.UUID(str(source_id))
            except (TypeError, ValueError):
                unresolved += 1; continue
            if kind == "decision":
                accepted = await db.scalar(
                    select(OrchestrationDecision.id)
                    .join(OrchestrationRun, OrchestrationRun.id == OrchestrationDecision.run_id)
                    .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
                    .where(
                        OrchestrationDecision.id == source_id,
                        OrchestrationDecision.run_id == run.id,
                        OrchestrationDecision.validator_status == "accepted",
                        OrchestrationGoal.id == goal.id,
                        OrchestrationGoal.project_id == goal.project_id,
                    )
                )
            else:
                accepted = await db.scalar(
                    select(OrchestrationEvidence.id)
                    .join(OrchestrationGate, OrchestrationGate.id == OrchestrationEvidence.gate_id)
                    .join(OrchestrationRun, OrchestrationRun.id == OrchestrationEvidence.run_id)
                    .join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id)
                    .where(
                        OrchestrationEvidence.id == source_id,
                        OrchestrationEvidence.run_id == run.id,
                        OrchestrationEvidence.verdict == "accepted",
                        OrchestrationGate.run_id == run.id,
                        OrchestrationGoal.id == goal.id,
                        OrchestrationGoal.project_id == goal.project_id,
                    )
                )
            if accepted is None:
                unresolved += 1
            else:
                section.fact_status = "accepted"
                promoted = True
        await db.flush()
        if promoted and not unresolved:
            event, _ = await emit_event_once(
                db, goal.project_id, "memory.upgrade_resolved",
                {"goal_id": str(goal.id), "run_id": str(run.id),
                 "source": "accepted_source_reconciliation"},
                source="orchestrator",
                dedup_key=f"memory.upgrade_resolved:{run.id}:accepted_source_reconciliation",
            )
            # This Task 9 boundary produces the exact event that resolves its
            # aggregate wait; do not broaden Task 8's scheduler event surface.
            from huddleroom.services.orchestration_supervision import OrchestrationSupervisionService
            await OrchestrationSupervisionService(None).clear_matching_waits(
                db, run, event_type=event.event_type, event_id=event.id, matcher=event.payload,
            )
        return unresolved

    async def upsert_section(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        *,
        section_key: str,
        title: str,
        body: str,
        summary: str | None = None,
        section_type: str = "text",
        always_load: bool = False,
        toc_order: int = 0,
        run_id: uuid.UUID | None = None,
        created_by: str = "orchestrator",
        event_id: uuid.UUID | None = None,
        fact_status: str | None = None,
        provenance: dict | None = None,
    ) -> OrchestrationMemorySection:
        if not _SECTION_KEY_RE.fullmatch(section_key):
            raise ValueError(f"invalid section_key: {section_key!r}")

        # ponytail: Atomic upsert via dialect-specific on_conflict_do_update;
        # eliminates SQLITE_BUSY_SNAPSHOT race by making insert+update atomic.
        now = func.now()  # pylint: disable=not-callable

        insert_provenance = {
            "created_by": created_by,
            "run_id": str(run_id) if run_id else None,
            "event_id": str(event_id) if event_id else None,
            **(provenance or {}),
        }

        # Values for INSERT (includes all fields)
        insert_values = {
            "project_id": project_id,
            "goal_id": goal_id,
            "run_id": run_id,
            "section_key": section_key,
            "title": title,
            "body": body,
            "summary": summary,
            "section_type": section_type,
            "always_load": always_load,
            "toc_order": toc_order,
            "created_by": created_by,
            "created_from_event_id": event_id,
            "created_at": now,
            "updated_at": now,
            "fact_status": fact_status or "unverified",
            "provenance": insert_provenance,
        }

        # Values for UPDATE on conflict (same as _apply_update but with updated_at)
        update_values = {
            "title": title,
            "body": body,
            "summary": summary,
            "section_type": section_type,
            "always_load": always_load,
            "toc_order": toc_order,
            "created_by": created_by,
            "updated_from_event_id": event_id,
            "updated_at": now,
        }
        if run_id is not None:
            update_values["run_id"] = run_id
        if fact_status is not None:
            update_values["fact_status"] = fact_status
        if provenance is not None:
            update_values["provenance"] = insert_provenance

        # Build dialect-specific insert statement with atomic conflict handling.
        # .returning(...) fetches the post-upsert row directly from this same
        # statement instead of a second SELECT, so there's no need to expire
        # the session (a blanket db.expire_all() would also invalidate every
        # other ORM object the caller has loaded in this transaction, e.g. the
        # goal/run being advanced by the same tick).
        bind = db.get_bind()
        dialect_name = bind.dialect.name
        if dialect_name == "sqlite":
            stmt = sqlite_insert(OrchestrationMemorySection).values(**insert_values)
            stmt = stmt.on_conflict_do_update(
                index_elements=["project_id", "goal_id", "section_key"],
                set_=update_values,
            )
        elif dialect_name == "postgresql":
            stmt = postgresql_insert(OrchestrationMemorySection).values(**insert_values)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_orch_memory_sections_project_goal_key",
                set_=update_values,
            )
        else:
            raise ValueError(f"Unsupported database dialect: {dialect_name}")

        stmt = stmt.returning(OrchestrationMemorySection)
        result = await db.execute(stmt, execution_options={"populate_existing": True})
        return result.scalars().one()

    async def get_section(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
        section_key: str,
    ) -> OrchestrationMemorySection | None:
        result = await db.execute(
            select(OrchestrationMemorySection).where(
                OrchestrationMemorySection.project_id == project_id,
                OrchestrationMemorySection.goal_id == goal_id,
                OrchestrationMemorySection.section_key == section_key,
            )
        )
        return result.scalar_one_or_none()

    async def list_sections(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
    ) -> list[OrchestrationMemorySection]:
        result = await db.execute(
            select(OrchestrationMemorySection)
            .where(
                OrchestrationMemorySection.project_id == project_id,
                OrchestrationMemorySection.goal_id == goal_id,
            )
            .order_by(
                OrchestrationMemorySection.toc_order.asc(),
                OrchestrationMemorySection.created_at.asc(),
                OrchestrationMemorySection.id.asc(),
            )
        )
        return list(result.scalars().all())

    async def list_section_metadata(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
    ) -> list[Row]:
        """Metadata only (no `body`) — for TOC / always-loaded construction
        without paying to load section bodies that may be arbitrarily long
        (spec 5.4). Same order as list_sections; callers needing an excerpt
        fetch that one section's body via get_section."""
        result = await db.execute(
            select(
                OrchestrationMemorySection.id,
                OrchestrationMemorySection.section_key,
                OrchestrationMemorySection.title,
                OrchestrationMemorySection.toc_order,
                OrchestrationMemorySection.always_load,
                OrchestrationMemorySection.summary,
                OrchestrationMemorySection.created_at,
            )
            .where(
                OrchestrationMemorySection.project_id == project_id,
                OrchestrationMemorySection.goal_id == goal_id,
            )
            .order_by(
                OrchestrationMemorySection.toc_order.asc(),
                OrchestrationMemorySection.created_at.asc(),
                OrchestrationMemorySection.id.asc(),
            )
        )
        return list(result.all())

    async def get_always_loaded(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        goal_id: uuid.UUID,
    ) -> list[OrchestrationMemorySection]:
        result = await db.execute(
            select(OrchestrationMemorySection)
            .where(
                OrchestrationMemorySection.project_id == project_id,
                OrchestrationMemorySection.goal_id == goal_id,
                OrchestrationMemorySection.always_load.is_(True),
            )
            .order_by(
                OrchestrationMemorySection.toc_order.asc(),
                OrchestrationMemorySection.created_at.asc(),
                OrchestrationMemorySection.id.asc(),
            )
        )
        return list(result.scalars().all())
