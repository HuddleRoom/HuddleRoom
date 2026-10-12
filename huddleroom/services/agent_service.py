from __future__ import annotations

import copy
import dataclasses
import uuid
from types import SimpleNamespace

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
    """Agent CRUD. Every roster write goes through AgentRosterMutationCoordinator;
    the ``_*_row`` helpers are the private validated writes it runs under its locks."""

    @staticmethod
    def validate_create(data: AgentCreate) -> None:
        if data.adapter_type == "cli" and not _has_cli_runtime(
            _effective_cli_runtime(data.cli_runtime, data.config)
        ):
            raise HTTPException(status_code=422, detail="CLI agents require cli_runtime")

    @staticmethod
    def state_from_create(data: AgentCreate, agent_id: uuid.UUID | None = None):
        from huddleroom.services.orchestration_roster_mutation import CanonicalAgentState

        return CanonicalAgentState(
            agent_id or uuid.uuid4(), data.name, data.role, data.description, data.provider, data.model,
            data.system_prompt, data.adapter_type, data.cli_runtime, tuple(data.capabilities),
            copy.deepcopy(data.config), True,
        )

    async def create(self, db: AsyncSession, data: AgentCreate) -> Agent:
        self.validate_create(data)
        from huddleroom.services.orchestration_roster_mutation import ValidatedRosterDelta

        state = self.state_from_create(data)

        async def build():
            async def mutation(session, delta, _plan):
                return await self._create_row(session, delta.creates[0])
            return ValidatedRosterDelta(creates=(state,)), mutation

        await self._publish(db, build)
        return await db.get(Agent, state.id, populate_existing=True)

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
        await self._publish_state(db, agent_id, lambda before: self._updated_state(before, data))
        return await db.get(Agent, agent_id, populate_existing=True)

    async def delete(self, db: AsyncSession, agent_id: uuid.UUID) -> None:
        await self._publish_state(db, agent_id, lambda before: dataclasses.replace(before, is_active=False))

    @staticmethod
    def _apply_update(target, data: AgentUpdate) -> None:
        """Validate and apply AgentUpdate semantics to an Agent or a canonical copy."""
        resulting_adapter_type = data.adapter_type if "adapter_type" in data.model_fields_set else target.adapter_type
        runtime_is_supplied = "cli_runtime" in data.model_fields_set
        config_is_supplied = "config" in data.model_fields_set and data.config is not None
        resulting_config = data.config if config_is_supplied else target.config
        if resulting_adapter_type == "cli" and (
            (data.adapter_type == "cli" and target.adapter_type != "cli")
            or runtime_is_supplied
            or config_is_supplied
        ) and not _has_cli_runtime(_effective_cli_runtime(
            data.cli_runtime if runtime_is_supplied else target.cli_runtime, resulting_config
        )):
            raise HTTPException(status_code=422, detail="CLI agents require cli_runtime")
        scalar_fields = {"name", "role", "description", "provider", "model", "system_prompt",
                         "adapter_type", "cli_runtime"}
        json_fields = {"capabilities", "config"}
        for field in data.model_fields_set & scalar_fields:
            setattr(target, field, getattr(data, field))
        for field in data.model_fields_set & json_fields:
            val = getattr(data, field)
            if val is not None:
                setattr(target, field, val)
        if data.adapter_type and data.adapter_type != "cli":
            # Prevent stale CLI-only state from surviving an adapter switch.
            target.cli_runtime = None
            if "cli_runtime" in target.config:
                target.config = {
                    key: value for key, value in target.config.items() if key != "cli_runtime"
                }
        # is_active is boolean — False is a valid value, must check for None explicitly
        if data.is_active is not None:
            target.is_active = data.is_active

    def _updated_state(self, before, data: AgentUpdate):
        target = SimpleNamespace(**{
            field.name: copy.deepcopy(getattr(before, field.name)) for field in dataclasses.fields(before)
        })
        target.capabilities = list(target.capabilities)
        self._apply_update(target, data)
        return dataclasses.replace(
            before, **{**vars(target), "capabilities": tuple(target.capabilities)}
        )

    async def _publish_state(self, db: AsyncSession, agent_id: uuid.UUID, transform) -> None:
        from huddleroom.services.orchestration_roster_mutation import (
            RosterStateChanged,
            ValidatedRosterDelta,
            _canonical_agent,
        )

        async def build():
            agent = await db.get(Agent, agent_id, populate_existing=True)
            if agent is None:
                raise HTTPException(status_code=404, detail="Agent not found")
            before = _canonical_agent(agent)
            after = transform(before)
            if after == before:
                return None
            if not before.is_active and not after.is_active:
                # Inactive agents are outside every roster fingerprint (
                # projection), so no goal locks are needed -- but the edit is still
                # serialized on the roster lock and re-read under it.
                await self._write_inactive(db, agent_id, before, after)
                return None

            async def mutation(session, _delta, _plan):
                row = await session.get(Agent, agent_id, populate_existing=True)
                if row is None or _canonical_agent(row) != before:
                    raise RosterStateChanged()
                self._write_row(row, after)
                await session.flush()

            delta = (
                ValidatedRosterDelta(updates=(after,)) if before.is_active
                else ValidatedRosterDelta(creates=(after,))  # reactivation joins the active roster
            )
            return delta, mutation

        await self._publish(db, build)

    @staticmethod
    async def _write_inactive(db: AsyncSession, agent_id: uuid.UUID, before, after) -> None:
        from huddleroom.services.orchestration_roster_mutation import (
            AgentRosterWriteLock,
            RosterStateChanged,
            _canonical_agent,
            holds_goal_lock,
            require_lock_introspection,
            transaction_shape,
        )

        require_lock_introspection(db)
        if holds_goal_lock():
            raise RuntimeError("the agent roster lock must be acquired before any goal lock")
        owns = transaction_shape(db) in {"none", "autobegin"}
        async with AgentRosterWriteLock().acquire():
            row = await db.get(Agent, agent_id, populate_existing=True)
            if row is None or _canonical_agent(row) != before:
                raise RosterStateChanged()  # e.g. reactivated meanwhile: re-plan via the coordinator
            AgentService._write_row(row, after)
            await db.flush()
            if owns:
                await db.commit()

    async def _publish(self, db: AsyncSession, build, *, attempts: int = 3):
        """Public mutations re-plan against fresh state if a concurrent change won."""
        from huddleroom.services.orchestration_roster_mutation import RosterStateChanged

        for attempt in range(attempts):
            try:
                built = await build()
                if built is None:
                    return None
                delta, mutation = built
                return await self._coordinate(db, delta, None, mutation, origin_goal_id=None, provenance={})
            except RosterStateChanged:
                if attempt == attempts - 1:
                    raise
        return None

    @staticmethod
    async def _coordinate(
        db: AsyncSession, delta, expected_plan, mutation, *, origin_goal_id, provenance: dict, origin_run_id=None,
    ):
        from huddleroom.services.orchestration_roster_mutation import (
            AgentRosterMutationCoordinator,
            transaction_shape,
        )

        coordinator = AgentRosterMutationCoordinator()
        # Compatibility only: an explicit/nested outer transaction that this code
        # never commits (test fixtures). No production path reaches it: request
        # sessions (get_db), tick, debug step and the decision routes are AUTOBEGIN
        # and hand off outside their savepoints; production db.begin() callers
        # (scheduler graph timeouts, supersession, conversation services, startup
        # seeding) never mutate agents.
        if transaction_shape(db) in {"explicit", "nested"}:
            async with coordinator.outer_transaction_guard(
                db, delta, expected_plan, origin_goal_id=origin_goal_id, origin_run_id=origin_run_id
            ) as locked:
                value = await mutation(db, locked.delta, locked.plan)
                await coordinator.invalidate_collateral(db, locked.plan, provenance)
                return value
        result = await coordinator.apply(
            db, delta, expected_plan, mutation, origin_goal_id=origin_goal_id, provenance=provenance,
            origin_run_id=origin_run_id,
        )
        return result.value

    async def _create_row(self, db: AsyncSession, state) -> Agent:
        agent = Agent(
            id=state.id,
            name=state.name,
            role=state.role,
            description=state.description,
            provider=state.provider,
            model=state.model,
            system_prompt=state.system_prompt,
            adapter_type=state.adapter_type,
            cli_runtime=state.cli_runtime,
            capabilities=list(state.capabilities),
            config=copy.deepcopy(state.config),
            is_active=state.is_active,
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

    @staticmethod
    def _write_row(agent: Agent, state) -> None:
        for field in dataclasses.fields(state):
            if field.name == "id":
                continue
            value = getattr(state, field.name)
            value = list(value) if field.name == "capabilities" else copy.deepcopy(value)
            if getattr(agent, field.name) != value:
                setattr(agent, field.name, value)

    async def _apply_delta_rows(self, db: AsyncSession, delta, _plan=None) -> list[uuid.UUID]:
        for state in delta.creates:
            await self._create_row(db, state)
        for state in delta.updates:
            row = await db.get(Agent, state.id, populate_existing=True)
            self._write_row(row, state)
        for agent_id in delta.deletes:
            (await db.get(Agent, agent_id, populate_existing=True)).is_active = False
        await db.flush()
        return [state.id for state in delta.creates + delta.updates] + list(delta.deletes)

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
            active_graph_runs=[],  # Not included in the agent context yet.
        )


async def apply_prepared_roster_mutation(db: AsyncSession, prepared, *, provenance: dict | None = None):
    """Roster-first application of a reserved PreparedRosterMutation (no goal lock held)."""
    service = AgentService()
    return await service._coordinate(  # pylint: disable=protected-access
        db, prepared.delta, prepared.expected_plan, service._apply_delta_rows,  # pylint: disable=protected-access
        origin_goal_id=prepared.origin_goal_id, provenance=provenance or {},
        origin_run_id=prepared.origin_run_id,
    )


async def hand_off_roster_delta(db: AsyncSession, delta, *, goal_id: uuid.UUID, run_id: uuid.UUID,
                                process_id: uuid.UUID, decision_ids) -> None:
    """Reserve an approved process roster effect.

    Under a goal lock this raises RosterHandoffRequired so the entry point
    commits, releases the goal lock, applies roster-first and resumes. Without a
    goal lock (direct process calls) it applies now, roster-first.
    """
    from huddleroom.services.orchestration_roster_mutation import (
        PreparedRosterMutation,
        RosterHandoffRequired,
        fresh_preview,
        holds_goal_lock,
        require_lock_introspection,
    )

    require_lock_introspection(db)  # never apply inline under an undetectable goal lock
    await db.flush()
    plan = await fresh_preview(db, delta, origin_goal_id=goal_id, refresh_all=False)
    prepared = PreparedRosterMutation(goal_id, run_id, process_id, tuple(decision_ids), delta, plan)
    if holds_goal_lock():
        raise RosterHandoffRequired(prepared)
    await apply_prepared_roster_mutation(db, prepared)


async def complete_roster_handoff(db: AsyncSession, prepared, resume, *, attempts: int = 3):
    """Apply a reserved roster effect roster-first, then re-enter the origin goal.

    ``resume(prepared)`` reacquires the origin goal lock, revalidates and
    continues; it may raise RosterHandoffRequired again with a fresh reservation.

    Legacy process handoffs only: a stale apply is swallowed and the process
    re-derives its reservation from fresh state (up to ``attempts``). Approved
    recovery must NOT use this: it calls apply_prepared_roster_mutation
    directly and lets RosterStateChanged propagate (fail ``state_changed`` and
    re-diagnose), because its approval binds the exact sealed plan.
    """
    from huddleroom.services.orchestration_roster_mutation import RosterHandoffRequired, RosterStateChanged

    for _ in range(attempts):
        try:
            await apply_prepared_roster_mutation(db, prepared)
        except RosterStateChanged:
            pass  # resume re-derives the reservation from fresh state (or finds it applied)
        try:
            return await resume(prepared)
        except RosterHandoffRequired as again:
            prepared = again.prepared
    raise HTTPException(status_code=409, detail="agent roster kept changing; retry")
