from __future__ import annotations

import uuid
from copy import deepcopy

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import SessionTransactionOrigin

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_effectiveness_review import EffectivenessReviewProcess
from huddleroom.services.orchestration_goal_closeout import GoalCloseoutProcess
from huddleroom.services.orchestration_goal_definition import GoalDefinitionProcess
from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import LLM_DECISION_RUN_STATUSES, OrchestrationService
from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService
from huddleroom.services.project_service import ProjectService
from huddleroom.schemas.orchestration import valid_lm_retry_checkpoint

SUPPORTED_PROCESS_TYPES = frozenset({
    "goal_definition", "manager_selection", "agent_definition_review",
    "team_hierarchy", "effectiveness_review", "goal_closeout",
})

_PREDECESSOR = {
    "manager_selection": "goal_definition",
    "agent_definition_review": "manager_selection",
    "team_hierarchy": "agent_definition_review",
}


class OrchestrationDebugService:
    """Advance exactly one baseline process on demand, outside of tick().

    Reuses tick()'s per-process invocation shape (goal lock, workspace boundary
    lock, predecessor gate, single advance() call) but never chains into the next
    process, ingests events, validates gates, creates delegations, or completes
    the run/goal -- tick() stays the only place that does all of that together.
    """

    async def step(self, db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID, process_type: str) -> dict:
        if process_type not in SUPPORTED_PROCESS_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported process type '{process_type}'")

        orchestration_service = OrchestrationService()

        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = (
            transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
        )

        if await orchestration_service.get_goal(db, project_id, goal_id) is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")

        async with orchestration_service._lock_goal_for_baseline_transition(db, goal_id):
            goal = await orchestration_service.get_goal(db, project_id, goal_id)
            if goal is None:
                raise HTTPException(status_code=404, detail="Orchestration goal not found")
            # get_goal's plain select returns the identity-map instance without
            # repopulating it, so force a fresh read inside the lock -- otherwise a
            # concurrent pause/cancel committed between the pre-lock check and lock
            # acquisition would be missed (same guarantee tick() gets via
            # populate_existing=True).
            await db.refresh(goal)
            if goal.status not in {"active", "blocked"}:
                raise HTTPException(status_code=409, detail=f"goal is '{goal.status}'; cannot advance a process")

            run = await orchestration_service.get_active_run_for_goal(db, project_id, goal_id)
            if run is None:
                raise HTTPException(status_code=409, detail="goal has no active run to advance")
            await db.refresh(run)
            if run.status not in LLM_DECISION_RUN_STATUSES:
                raise HTTPException(status_code=409, detail=f"active run is '{run.status}'; not tickable")

            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)

            current = await OrchestrationProcessService().get_current(db, goal.id, process_type)
            if current is not None and isinstance((current.outputs or {}).get("_lm_retry"), dict):
                raise HTTPException(status_code=409, detail="Retry the failed LM request via baseline/retry")
            if current is not None and current.status == "waiting_decision":
                nested = await db.begin_nested()
                try:
                    process_summary = await self._rerun_waiting_process(db, goal, run, current)
                except Exception:
                    await nested.rollback()
                    raise
                await nested.commit()
            else:
                process_summary = await self._advance(db, goal, run, process_type)

            await db.flush()
            if not caller_owns_transaction:
                await db.commit()

        return {"goal_id": goal.id, "run_id": run.id, "process_type": process_type, "process": process_summary}

    async def retry_failed(
        self, db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID, process_type: str
    ) -> dict:
        """Retry one persisted LM request while holding the baseline goal lock."""
        if process_type not in SUPPORTED_PROCESS_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported process type '{process_type}'")

        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = (
            transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
        )
        orchestration_service = OrchestrationService()

        async with orchestration_service._lock_goal_for_baseline_transition(db, goal_id):
            goal = await orchestration_service.get_goal(db, project_id, goal_id)
            run = await orchestration_service.get_active_run_for_goal(db, project_id, goal_id)
            current = await OrchestrationProcessService().get_current(db, goal_id, process_type)
            if goal is not None:
                await db.refresh(goal)
            if run is not None:
                await db.refresh(run)
            if current is not None:
                await db.refresh(current)
            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)
            checkpoint = (current.outputs or {}).get("_lm_retry") if current is not None else None
            expected_kind = {
                "goal_definition": "goal_analysis",
                "manager_selection": "manager_selection",
                "agent_definition_review": "agent_definition_review",
                "team_hierarchy": "team_hierarchy",
                "effectiveness_review": "effectiveness_review",
            }.get(process_type)
            if (
                goal is None
                or run is None
                or current is None
                or goal.status not in {"active", "blocked"}
                or run.status not in LLM_DECISION_RUN_STATUSES
                or current.superseded_by_id is not None
                or current.run_id != run.id
                or current.status != "running"
                or not isinstance(checkpoint, dict)
                or not valid_lm_retry_checkpoint(checkpoint, process_type)
                or checkpoint.get("kind") != expected_kind
            ):
                raise HTTPException(status_code=409, detail="No retryable LM request")
            decisions = await OrchestrationAuthorityDecisionService().list_decisions(
                db, goal.id, status="pending"
            )
            if any(decision.source_process_run_id == current.id for decision in decisions):
                raise HTTPException(status_code=409, detail="No retryable LM request")

            try:
                if process_type == "goal_definition":
                    process_summary = await GoalDefinitionProcess().retry_failed(db, goal, run, current)
                elif process_type == "manager_selection":
                    process_summary = await ManagerSelectionProcess().retry_failed(db, goal, run, current)
                elif process_type == "agent_definition_review":
                    process_summary = await AgentDefinitionReviewProcess().retry_failed(db, goal, run, current)
                elif process_type == "team_hierarchy":
                    process_summary = await TeamHierarchyProcess().retry_failed(db, goal, run, current)
                elif process_type == "effectiveness_review":
                    process_summary = await EffectivenessReviewProcess().retry_failed(db, goal, run, current)
                else:
                    raise HTTPException(status_code=409, detail="This process has no retryable LM request")
            except ValueError as exc:
                raise HTTPException(status_code=409, detail="No retryable LM request") from exc

            await db.flush()
            if not caller_owns_transaction:
                await db.commit()

        return {
            "goal_id": goal.id,
            "run_id": run.id,
            "process_type": process_type,
            "process": process_summary,
        }

    async def rerun_last(
        self, db: AsyncSession, project_id: uuid.UUID, goal_id: uuid.UUID, process_type: str | None = None
    ) -> dict:
        """Rerun a waiting baseline process, or the latest terminal one when
        none is waiting. The create+advance is one savepoint: any failure
        leaves no half-created successor and the prior current row
        un-superseded.

        For the fingerprinted types (team_hierarchy, agent_definition_review)
        the copied input_snapshot is a drift-detection fingerprint, not replay
        input: if world state has changed since the original completion, that
        type's advance() recomputes against current state and internally
        re-supersedes the seeded row (a standard advance() side effect). For
        those types the rerun therefore reflects current reality rather than
        replaying the old fingerprint.
        """
        if process_type is not None and process_type not in SUPPORTED_PROCESS_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported process type '{process_type}'")

        orchestration_service = OrchestrationService()

        transaction = db.sync_session.get_transaction()
        caller_owns_transaction = (
            transaction is not None and transaction.origin != SessionTransactionOrigin.AUTOBEGIN
        )

        if await orchestration_service.get_goal(db, project_id, goal_id) is None:
            raise HTTPException(status_code=404, detail="Orchestration goal not found")

        async with orchestration_service._lock_goal_for_baseline_transition(db, goal_id):
            goal = await orchestration_service.get_goal(db, project_id, goal_id)
            if goal is None:
                raise HTTPException(status_code=404, detail="Orchestration goal not found")
            # fresh read inside lock (same rationale as step())
            await db.refresh(goal)
            if goal.status not in {"active", "blocked"}:
                raise HTTPException(status_code=409, detail=f"goal is '{goal.status}'; cannot rerun a process")

            run = await orchestration_service.get_active_run_for_goal(db, project_id, goal_id)
            if run is None:
                raise HTTPException(status_code=409, detail="goal has no active run to rerun")
            await db.refresh(run)
            if run.status not in LLM_DECISION_RUN_STATUSES:
                raise HTTPException(status_code=409, detail=f"active run is '{run.status}'; not tickable")

            if process_type is None:
                selected = await self._select_waiting_process(db, goal.id)
                if selected is None:
                    selected = await self._select_last_terminal_process(db, goal.id)
            else:
                selected = await OrchestrationProcessService().get_current(db, goal.id, process_type)
                checkpoint = (selected.outputs or {}).get("_lm_retry") if selected is not None else None
                retryable_stuck = (
                    selected is not None
                    and selected.status in {"waiting_decision", "running"}
                    and valid_lm_retry_checkpoint(checkpoint, selected.process_type)
                )
                if selected is not None and selected.status not in {"waiting_decision", "completed", "skipped"} and not retryable_stuck:
                    selected = None
            if selected is None:
                detail = "no baseline process to rerun" if process_type is None else (
                    f"'{process_type}' has no rerunnable current process"
                )
                raise HTTPException(status_code=409, detail=detail)

            await ProjectService().lock_workspace_boundary(db, project_id)
            await ProjectService().require_runnable_project(db, project_id)

            nested = await db.begin_nested()
            try:
                checkpoint = (selected.outputs or {}).get("_lm_retry")
                retryable_stuck = selected.status in {"waiting_decision", "running"} and valid_lm_retry_checkpoint(
                    checkpoint, selected.process_type
                )
                if retryable_stuck:
                    if selected.status == "running":
                        # start_process deliberately treats a running row as idempotent.
                        # This locked, savepoint-scoped placeholder lets it create the
                        # successor while retaining the source's original status/history.
                        selected.superseded_by_id = selected.id
                        await db.flush()
                    successor = await OrchestrationProcessService().start_process(
                        db,
                        goal.id,
                        process_type=selected.process_type,
                        trigger_reason=f"human requested: rerun of {selected.process_type}",
                        run_id=run.id,
                        input_snapshot=deepcopy(selected.input_snapshot),
                        process_version=selected.process_version,
                        supersede_waiting=selected.status == "waiting_decision",
                    )
                    if selected.status == "running":
                        selected.superseded_by_id = successor.id
                        await db.flush()
                    await self._cancel_pending_decisions_for_process(db, goal.id, selected)
                    process_summary = await self._advance(db, goal, run, selected.process_type)
                    warning_service = OrchestrationWarningService()
                    resolved_ids = set()
                    for warning in await warning_service.list_warnings(db, goal.id, active_only=True):
                        if warning.source_process_run_id == selected.id:
                            await warning_service.resolve_warning(
                                db,
                                warning,
                                resolved_by="orchestrator",
                                reason="successful rerun of failed LM process",
                            )
                            resolved_ids.add(str(warning.id))
                    run.active_blockers = [
                        blocker for blocker in run.active_blockers
                        if not (isinstance(blocker, dict) and str(blocker.get("warning_id")) in resolved_ids)
                    ]
                    if not run.active_blockers:
                        if goal.status == "blocked":
                            goal.status = "active"
                        if run.status == "blocked":
                            run.status = "running"
                elif selected.status == "waiting_decision":
                    process_summary = await self._rerun_waiting_process(db, goal, run, selected)
                else:
                    await self._cancel_pending_decisions_for_process(db, goal.id, selected)
                    await OrchestrationProcessService().start_process(
                        db,
                        goal.id,
                        process_type=selected.process_type,
                        trigger_reason=f"human requested: rerun of {selected.process_type}",
                        run_id=run.id,
                        input_snapshot=deepcopy(selected.input_snapshot),
                        process_version=selected.process_version,
                    )
                    process_summary = await self._advance(db, goal, run, selected.process_type)
                    if selected.status == "skipped":
                        warning_service = OrchestrationWarningService()
                        for warning in await warning_service.list_warnings(db, goal.id, active_only=True):
                            if warning.source_process_run_id == selected.id:
                                await warning_service.resolve_warning(
                                    db,
                                    warning,
                                    resolved_by="orchestrator",
                                    reason="successful rerun of skipped process",
                                )
            except Exception:
                await nested.rollback()
                raise
            await nested.commit()

            await db.flush()
            if not caller_owns_transaction:
                await db.commit()

        return {
            "goal_id": goal.id,
            "run_id": run.id,
            "process_type": selected.process_type,
            "process": process_summary,
        }

    async def _select_last_terminal_process(
        self, db: AsyncSession, goal_id: uuid.UUID
    ) -> OrchestrationProcessRun | None:
        result = await db.execute(
            select(OrchestrationProcessRun)
            .where(
                OrchestrationProcessRun.goal_id == goal_id,
                OrchestrationProcessRun.process_type.in_(SUPPORTED_PROCESS_TYPES),
                OrchestrationProcessRun.superseded_by_id.is_(None),
                OrchestrationProcessRun.status.in_(("completed", "skipped")),
            )
            .order_by(
                OrchestrationProcessRun.completed_at.desc(),
                OrchestrationProcessRun.created_at.desc(),
                OrchestrationProcessRun.id.desc(),
            )
            .limit(1)
        )
        return result.scalars().first()

    async def _select_waiting_process(
        self, db: AsyncSession, goal_id: uuid.UUID
    ) -> OrchestrationProcessRun | None:
        result = await db.execute(
            select(OrchestrationProcessRun)
            .where(
                OrchestrationProcessRun.goal_id == goal_id,
                OrchestrationProcessRun.process_type.in_(SUPPORTED_PROCESS_TYPES),
                OrchestrationProcessRun.superseded_by_id.is_(None),
                OrchestrationProcessRun.status == "waiting_decision",
            )
            .order_by(
                OrchestrationProcessRun.started_at.desc(),
                OrchestrationProcessRun.created_at.desc(),
                OrchestrationProcessRun.id.desc(),
            )
            .limit(1)
        )
        return result.scalars().first()

    async def _rerun_waiting_process(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        process_run: OrchestrationProcessRun,
    ) -> dict:
        await self._cancel_pending_decisions_for_process(db, goal.id, process_run)
        await OrchestrationProcessService().start_process(
            db,
            goal.id,
            process_type=process_run.process_type,
            trigger_reason=f"human requested: rerun of {process_run.process_type}",
            run_id=run.id,
            input_snapshot=deepcopy(process_run.input_snapshot),
            process_version=process_run.process_version,
            supersede_waiting=True,
        )
        return await self._advance(db, goal, run, process_run.process_type)

    async def _cancel_pending_decisions_for_process(
        self, db: AsyncSession, goal_id: uuid.UUID, process_run: OrchestrationProcessRun
    ) -> None:
        decision_service = OrchestrationAuthorityDecisionService()
        for decision in await decision_service.list_decisions(db, goal_id, status="pending"):
            if decision.source_process_run_id == process_run.id:
                await decision_service.cancel_decision(
                    db,
                    decision,
                    reason=f"manual rerun of process '{process_run.process_type}'",
                )

    async def advance_process(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, process_type: str
    ) -> dict:
        """Advance one already-selected baseline process without tick chaining."""
        if process_type not in SUPPORTED_PROCESS_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported process type '{process_type}'")
        predecessor = _PREDECESSOR.get(process_type)
        if predecessor is not None:
            await self._require_terminal(db, goal.id, predecessor, process_type)

        if process_type == "goal_definition":
            return await GoalDefinitionProcess().advance(db, goal, run, manual=True)
        if process_type == "manager_selection":
            return await ManagerSelectionProcess().advance(db, goal, run, manual=True)
        if process_type == "agent_definition_review":
            current = await OrchestrationProcessService().get_current(
                db, goal.id, process_type
            )
            if current is not None and isinstance((current.outputs or {}).get("_lm_retry"), dict):
                raise HTTPException(status_code=409, detail="Retry the failed LM request via baseline/retry")
            return await AgentDefinitionReviewProcess().advance(db, goal, run, manual=True)
        if process_type == "team_hierarchy":
            return await TeamHierarchyProcess().advance(db, goal, run, manual=True)
        if process_type == "effectiveness_review":
            return await EffectivenessReviewProcess().advance(db, goal, run)

        if process_type == "goal_closeout":
            orchestration_service = OrchestrationService()
            # The four predecessors must be status-terminal before closeout. Staleness
            # (a terminal-but-outdated review/hierarchy) is checked separately below —
            # it only matters when there is real work to authorize.
            status_reason = await self._predecessors_status_terminal_reason(db, goal.id)
            if status_reason is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"goal_closeout cannot advance: {status_reason}",
                )
            # (#1) A goal can reach closeout with no executable work (no gates to
            # authorize) — mark it done so the step is terminal instead of stuck at
            # "Not started". A stale predecessor is moot here: no gate ever used those
            # agents/team, so closeout bypasses the staleness gate. Returns None when
            # gates exist, deferring to the manifest path that enforces acceptance.
            no_work = await GoalCloseoutProcess().complete_no_work(db, goal, run)
            if no_work is not None:
                return no_work
            # Real work exists: now staleness matters, so enforce full readiness.
            reason = await orchestration_service._baseline_readiness_reason(db, goal.id, allow_heal=False)
            if reason is not None:
                raise HTTPException(
                    status_code=409,
                    detail=f"goal_closeout cannot advance: {reason}",
                )
            preconditions = await orchestration_service._closeout_preconditions_manifest(db, goal, run)
            return await GoalCloseoutProcess().advance(db, goal, run, preconditions=preconditions)

        raise AssertionError(f"unhandled supported process type '{process_type}'")

    async def _advance(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, process_type: str
    ) -> dict:
        return await self.advance_process(db, goal, run, process_type)

    async def _predecessors_status_terminal_reason(
        self, db: AsyncSession, goal_id: uuid.UUID
    ) -> str | None:
        """Return the first baseline predecessor that is not status-terminal, or None.

        Status-only (completed/skipped) — deliberately ignores the staleness refinements
        in _baseline_readiness_reason so a no-work closeout is not blocked by an outdated
        review/hierarchy that no executed work ever depended on.
        """
        process_service = OrchestrationProcessService()
        for predecessor_type in (
            "goal_definition",
            "manager_selection",
            "agent_definition_review",
            "team_hierarchy",
        ):
            current = await process_service.get_current(db, goal_id, predecessor_type)
            if current is None or current.status not in ("completed", "skipped"):
                return f"'{predecessor_type}' is not yet complete"
        return None

    async def _require_terminal(self, db: AsyncSession, goal_id: uuid.UUID, predecessor_type: str, process_type: str) -> None:
        current = await OrchestrationProcessService().get_current(db, goal_id, predecessor_type)
        if current is None or current.status not in ("completed", "skipped"):
            raise HTTPException(status_code=409, detail=f"'{predecessor_type}' must be terminal before '{process_type}' can advance")
