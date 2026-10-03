from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration_process import (
    PROCESS_TYPES,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_linkage import check_goal_linkage


class OrchestrationProcessService:
    """Lifecycle for deterministic baseline process runs (Spec 6, 15.2).

    Transitions: start -> running -> completed; skip creates a terminal
    skipped row directly. waiting_decision (park/resume) arrives in Phase 9.

    Starting or skipping while a *terminal* (completed/skipped) current run
    of the same type exists supersedes it (spec 6.3) — at most one current
    run per (goal, process_type), enforced here rather than by a DB partial
    unique index (single writer: the orchestrator tick). Starting or
    skipping while the current run is already in the *target* state is an
    idempotent no-op: it returns the existing row unchanged instead of
    superseding it, so a retried trigger never creates a duplicate run or
    (for skip) a duplicate skip warning (spec ~4: "all side effects remain
    code-validated and idempotent"). Skipping a *running* current run
    converts that row in place to `skipped` rather than creating a second
    row and merely superseding the first — leaving the original row
    permanently `running` (just marked superseded) would misrepresent an
    abandoned run as still active in its own status column.

    `run_id`, when given, must belong to `goal_id` (Spec Deviation 11): the
    FK alone only proves the run exists, not that it's this goal's run, so
    a run id from another goal would otherwise be recorded as if it were
    this goal's process-run context.
    """

    async def get_current(
        self, db: AsyncSession, goal_id: uuid.UUID, process_type: str
    ) -> OrchestrationProcessRun | None:
        result = await db.execute(
            select(OrchestrationProcessRun)
            .where(
                OrchestrationProcessRun.goal_id == goal_id,
                OrchestrationProcessRun.process_type == process_type,
                OrchestrationProcessRun.superseded_by_id.is_(None),
            )
            .order_by(OrchestrationProcessRun.started_at.desc())
            .limit(1)
        )
        return result.scalars().first()

    @staticmethod
    def _skip_warning_message(process_type: str, skipped_by: str, reason: str) -> str:
        if process_type == "goal_closeout":
            return (
                "Goal was completed without closeout. No completion rationale or "
                f"lessons learned were recorded for this goal. Reason: {reason}"
            )
        return f"Process '{process_type}' was skipped by {skipped_by}: {reason}"

    async def start_process(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        process_type: str,
        trigger_reason: str,
        run_id: uuid.UUID | None = None,
        input_snapshot: dict | None = None,
        process_version: int = 1,
        supersede_waiting: bool = False,
    ) -> OrchestrationProcessRun:
        if process_type not in PROCESS_TYPES:
            raise ValueError(f"unknown process_type: {process_type!r}")
        await check_goal_linkage(db, goal_id, run_id=run_id)
        current = await self.get_current(db, goal_id, process_type)
        if current is not None and (
            current.status == "running"
            or (current.status == "waiting_decision" and not supersede_waiting)
        ):
            return current  # idempotent retry: already active (running or parked), no duplicate
        # Use nested transaction to handle IntegrityError from concurrent inserts
        # via the partial unique index on (goal_id, process_type).
        nested = await db.begin_nested()
        row = OrchestrationProcessRun(
            goal_id=goal_id,
            run_id=run_id,
            process_type=process_type,
            process_version=process_version,
            status="running",
            trigger_reason=trigger_reason,
            input_snapshot=input_snapshot or {},
        )
        db.add(row)
        try:
            await db.flush()
        except IntegrityError as exc:
            # Concurrent call inserted a row for the same (goal_id, process_type)
            # before this one was visible. Rollback savepoint and retry via get_current.
            await nested.rollback()
            current = await self.get_current(db, goal_id, process_type)
            if current is not None and (
                current.status == "running"
                or (current.status == "waiting_decision" and not supersede_waiting)
            ):
                return current  # idempotent retry: another concurrent caller won the race
            # If the new row is in a terminal state (completed/skipped), we need to
            # supersede it. The partial unique index (superseded_by_id IS NULL)
            # requires the old row to go non-NULL before the new row is inserted,
            # but superseded_by_id has an immediate (non-deferrable) FK, so it
            # can't point at the new row's id before that row exists. Bridge the
            # gap with a self-reference placeholder (always FK-valid, since the
            # row already exists), then repoint it once the new row is in.
            if current is not None:
                new_row_id = uuid.uuid4()
                nested2 = await db.begin_nested()
                try:
                    current.superseded_by_id = current.id  # placeholder: clears the unique index
                    await db.flush()
                    row = OrchestrationProcessRun(
                        id=new_row_id,
                        goal_id=goal_id,
                        run_id=run_id,
                        process_type=process_type,
                        process_version=process_version,
                        status="running",
                        trigger_reason=trigger_reason,
                        input_snapshot=input_snapshot or {},
                    )
                    db.add(row)
                    await db.flush()  # new row now exists
                    current.superseded_by_id = new_row_id  # repoint to the real successor
                    await db.flush()
                    await nested2.commit()
                except IntegrityError:
                    await nested2.rollback()
                    winner = await self.get_current(db, goal_id, process_type)
                    if (
                        winner is not None
                        and winner.id != current.id
                        and winner.status in ("running", "waiting_decision")
                    ):
                        return winner
                    raise
                return row
            # No current row exists, create the new row
            row = OrchestrationProcessRun(
                goal_id=goal_id,
                run_id=run_id,
                process_type=process_type,
                process_version=process_version,
                status="running",
                trigger_reason=trigger_reason,
                input_snapshot=input_snapshot or {},
            )
            db.add(row)
            nested2 = await db.begin_nested()
            try:
                await db.flush()
                await nested2.commit()
            except IntegrityError:
                await nested2.rollback()
                raise
            return row
        await nested.commit()
        if current is not None:
            current.superseded_by_id = row.id
            await db.flush()
        return row

    async def park_process(
        self, db: AsyncSession, process_run: OrchestrationProcessRun
    ) -> OrchestrationProcessRun:
        """running -> waiting_decision (spec 6.3 park). Status-conditional
        UPDATE, same concurrency pattern as complete_process. Phase 9 builds
        the general answer->resume surface on top of these primitives."""
        if process_run.status == "waiting_decision":
            return process_run
        result = await db.execute(
            update(OrchestrationProcessRun)
            .where(
                OrchestrationProcessRun.id == process_run.id,
                OrchestrationProcessRun.status == "running",
            )
            .values(status="waiting_decision")
        )
        if result.rowcount != 1:
            # A concurrent identical park call may have already won the
            # status-conditional UPDATE between our stale in-memory read and
            # this one (review finding, MEDIUM): the row is already at the
            # target state, not conflicting, so refresh and succeed instead
            # of raising a spurious failure.
            await db.refresh(process_run)
            if process_run.status == "waiting_decision":
                return process_run
            raise ValueError(f"cannot park process run in status {process_run.status!r}")
        await db.refresh(process_run)
        return process_run

    async def resume_process(
        self, db: AsyncSession, process_run: OrchestrationProcessRun
    ) -> OrchestrationProcessRun:
        """waiting_decision -> running (spec 6.3 resume)."""
        if process_run.status == "running":
            return process_run
        result = await db.execute(
            update(OrchestrationProcessRun)
            .where(
                OrchestrationProcessRun.id == process_run.id,
                OrchestrationProcessRun.status == "waiting_decision",
            )
            .values(status="running")
        )
        if result.rowcount != 1:
            # Same race as park_process above (review finding, MEDIUM): a
            # concurrent identical resume may have already landed.
            await db.refresh(process_run)
            if process_run.status == "running":
                return process_run
            raise ValueError(f"cannot resume process run in status {process_run.status!r}")
        await db.refresh(process_run)
        return process_run

    async def skip_process(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        process_type: str,
        skipped_by: str,
        reason: str,
        run_id: uuid.UUID | None = None,
    ) -> OrchestrationProcessRun:
        result = await self._skip_process_impl(
            db,
            goal_id,
            process_type=process_type,
            skipped_by=skipped_by,
            reason=reason,
            run_id=run_id,
        )
        if process_type == "manager_selection":
            # Runs after ALL four of _skip_process_impl's internal return
            # paths, not just the common one -- a single wrap point instead
            # of an inline snippet is what guarantees the spec-8.7 warning,
            # memory update, and no_manager authority state fire on every
            # skip, including idempotent retries and the concurrent-insert
            # branches (review finding).
            from huddleroom.models.orchestration import OrchestrationGoal
            from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess

            # handle_skip derives run linkage from the skipped row's own
            # run_id internally (review finding, LOW) -- no need to look up
            # an OrchestrationRun here.
            goal = await db.get(OrchestrationGoal, goal_id)
            # Check that the returned skipped row is still current -- a stale
            # retry racing a newer completed rerun could otherwise clobber a
            # real manager selection back to no_manager (Task 8). The check
            # and handle_skip() must be atomic (review finding, MEDIUM): a
            # force-started rerun can only supersede this row by writing its
            # superseded_by_id column, so locking THIS row FOR UPDATE and
            # re-reading it after the lock is held blocks that write until
            # we commit, closing the window a plain get_current() re-check
            # would leave open between the check and handle_skip()'s writes.
            if goal is not None:
                locked = (
                    await db.execute(
                        select(OrchestrationProcessRun)
                        .where(OrchestrationProcessRun.id == result.id)
                        .with_for_update()
                        .execution_options(populate_existing=True)
                    )
                ).scalar_one_or_none()
                if (
                    locked is not None
                    and locked.superseded_by_id is None
                    and locked.status == "skipped"
                ):
                    await ManagerSelectionProcess().handle_skip(db, goal, locked)
        elif process_type == "agent_definition_review":
            from huddleroom.models.orchestration import OrchestrationGoal
            from huddleroom.services.orchestration_agent_definition_review import (
                AgentDefinitionReviewProcess,
            )

            goal = await db.get(OrchestrationGoal, goal_id)
            if goal is not None:
                locked = (
                    await db.execute(
                        select(OrchestrationProcessRun)
                        .where(OrchestrationProcessRun.id == result.id)
                        .with_for_update()
                        .execution_options(populate_existing=True)
                    )
                ).scalar_one_or_none()
                if (
                    locked is not None
                    and locked.superseded_by_id is None
                    and locked.status == "skipped"
                ):
                    await AgentDefinitionReviewProcess().handle_skip(db, goal, locked)
        elif process_type == "team_hierarchy":
            from huddleroom.models.orchestration import OrchestrationGoal
            from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess

            goal = await db.get(OrchestrationGoal, goal_id)
            if goal is not None:
                locked = (
                    await db.execute(
                        select(OrchestrationProcessRun)
                        .where(OrchestrationProcessRun.id == result.id)
                        .with_for_update()
                        .execution_options(populate_existing=True)
                    )
                ).scalar_one_or_none()
                if (
                    locked is not None
                    and locked.superseded_by_id is None
                    and locked.status == "skipped"
                ):
                    await TeamHierarchyProcess().handle_skip(db, goal, locked)
        return result

    async def _skip_process_impl(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        process_type: str,
        skipped_by: str,
        reason: str,
        run_id: uuid.UUID | None = None,
    ) -> OrchestrationProcessRun:
        """Record a skip (spec 6.5). Only the human may skip — `skipped_by`
        must be human attribution (`"human:<user_id>"`, Phase 1 convention);
        the manager/LLM may only *recommend* a skip elsewhere, never call
        this. Idempotent: repeating an identical skip returns the existing
        skipped row rather than creating a new run or a second warning.
        Creates the process's skip warning inline (spec 6.5) so a skip is
        never missing its durable warning.
        """
        if process_type not in PROCESS_TYPES:
            raise ValueError(f"unknown process_type: {process_type!r}")
        if not skipped_by.startswith("human:") or len(skipped_by) <= len("human:"):
            raise ValueError(
                'skipped_by must be non-empty human attribution ("human:<user_id>", spec 6.5)'
            )
        if not reason or not reason.strip():
            raise ValueError("reason must not be empty")
        await check_goal_linkage(db, goal_id, run_id=run_id)
        current = await self.get_current(db, goal_id, process_type)
        if current is not None and current.status == "skipped":
            return current  # idempotent retry: already skipped, no duplicate warning
        if current is not None and current.status in ("running", "waiting_decision"):
            # Convert the running/parked row in place instead of creating a second
            # row and superseding the first — that would leave the original
            # row permanently "running"/"waiting_decision" (just marked superseded), which
            # misrepresents an abandoned run as still active. Use status-conditional
            # UPDATE to ensure concurrent skip/complete calls can't both succeed on
            # the same row (Finding 2).
            result = await db.execute(
                update(OrchestrationProcessRun)
                .where(
                    OrchestrationProcessRun.id == current.id,
                    OrchestrationProcessRun.status.in_(("running", "waiting_decision")),
                )
                .values(
                    status="skipped",
                    skipped_by=skipped_by,
                    override_reason=reason,
                    completed_at=_utcnow(),
                )
            )
            if result.rowcount != 1:
                # A concurrent identical skip call may have already won the
                # status-conditional UPDATE between our stale in-memory read and
                # this one: the row is already at the target state, not conflicting,
                # so refresh and succeed instead of raising a spurious failure.
                await db.refresh(current)
                if current.status == "skipped":
                    db.add(
                        OrchestrationWarning(
                            goal_id=goal_id,
                            run_id=run_id,
                            warning_type=f"{process_type}_skipped",
                            severity="warning",
                            message=self._skip_warning_message(process_type, skipped_by, reason),
                            source_process_run_id=current.id,
                        )
                    )
                    decision_service = OrchestrationAuthorityDecisionService()
                    sourced_pending = await decision_service.list_decisions(
                        db, goal_id, status="pending"
                    )
                    for decision in sourced_pending:
                        if decision.source_process_run_id == current.id:
                            await decision_service.cancel_decision(
                                db, decision,
                                reason=f"process '{process_type}' skipped by {skipped_by}: {reason}",
                            )
                    await db.flush()
                    return current
                raise ValueError(f"cannot skip process run in status {current.status!r}")
            db.add(
                OrchestrationWarning(
                    goal_id=goal_id,
                    run_id=run_id,
                    warning_type=f"{process_type}_skipped",
                    severity="warning",
                    message=self._skip_warning_message(process_type, skipped_by, reason),
                    source_process_run_id=current.id,
                )
            )
            # Plan Deviation 13: dispose of pending decisions the skipped run
            # sourced, so a later force-started rerun never parks on stale
            # decisions from the superseded run. Generic for any process type.
            decision_service = OrchestrationAuthorityDecisionService()
            sourced_pending = await decision_service.list_decisions(
                db, goal_id, status="pending"
            )
            for decision in sourced_pending:
                if decision.source_process_run_id == current.id:
                    await decision_service.cancel_decision(
                        db, decision,
                        reason=f"process '{process_type}' skipped by {skipped_by}: {reason}",
                    )
            await db.flush()
            return current
        # Use nested transaction to handle IntegrityError from concurrent inserts
        # via the partial unique index on (goal_id, process_type).
        nested = await db.begin_nested()
        row = OrchestrationProcessRun(
            goal_id=goal_id,
            run_id=run_id,
            process_type=process_type,
            process_version=1,
            status="skipped",
            trigger_reason="skip requested",
            input_snapshot={},
            skipped_by=skipped_by,
            override_reason=reason,
            completed_at=_utcnow(),
        )
        db.add(row)
        try:
            await db.flush()
        except IntegrityError as exc:
            # Concurrent call inserted a row for the same (goal_id, process_type)
            # before this one was visible. Rollback savepoint and retry via get_current.
            await nested.rollback()
            current = await self.get_current(db, goal_id, process_type)
            if current is not None and current.status == "skipped":
                return current  # idempotent retry: already skipped by concurrent call
            # Another concurrent caller inserted a running/waiting_decision row;
            # convert it to skipped (normal flow for converting running -> skipped).
            if current is not None and current.status in ("running", "waiting_decision"):
                result = await db.execute(
                    update(OrchestrationProcessRun)
                    .where(
                        OrchestrationProcessRun.id == current.id,
                        OrchestrationProcessRun.status.in_(("running", "waiting_decision")),
                    )
                    .values(
                        status="skipped",
                        skipped_by=skipped_by,
                        override_reason=reason,
                        completed_at=_utcnow(),
                    )
                )
                if result.rowcount != 1:
                    # Same race as above: concurrent identical skip may have already
                    # landed and set status to "skipped", so refresh and check.
                    await db.refresh(current)
                    if current.status == "skipped":
                        db.add(
                            OrchestrationWarning(
                                goal_id=goal_id,
                                run_id=run_id,
                                warning_type=f"{process_type}_skipped",
                                severity="warning",
                                message=self._skip_warning_message(process_type, skipped_by, reason),
                                source_process_run_id=current.id,
                            )
                        )
                        decision_service = OrchestrationAuthorityDecisionService()
                        sourced_pending = await decision_service.list_decisions(
                            db, goal_id, status="pending"
                        )
                        for decision in sourced_pending:
                            if decision.source_process_run_id == current.id:
                                await decision_service.cancel_decision(
                                    db, decision,
                                    reason=f"process '{process_type}' skipped by {skipped_by}: {reason}",
                                )
                        await db.flush()
                        return current
                    raise ValueError(f"cannot skip process run in status {current.status!r}") from exc
                await db.refresh(current)
                db.add(
                    OrchestrationWarning(
                        goal_id=goal_id,
                        run_id=run_id,
                        warning_type=f"{process_type}_skipped",
                        severity="warning",
                        message=self._skip_warning_message(process_type, skipped_by, reason),
                        source_process_run_id=current.id,
                    )
                )
                # Plan Deviation 13: dispose of pending decisions the skipped run sourced
                decision_service = OrchestrationAuthorityDecisionService()
                sourced_pending = await decision_service.list_decisions(
                    db, goal_id, status="pending"
                )
                for decision in sourced_pending:
                    if decision.source_process_run_id == current.id:
                        await decision_service.cancel_decision(
                            db, decision,
                            reason=f"process '{process_type}' skipped by {skipped_by}: {reason}",
                        )
                await db.flush()
                return current
            # No current row exists; need to create new skipped row and potentially
            # supersede a terminal one. Same self-reference-placeholder dance as
            # start_process: superseded_by_id has an immediate FK, so the old row
            # can't point at the new row's id before that row exists, but the
            # partial unique index needs the old row non-NULL before the insert.
            if current is not None:
                new_row_id = uuid.uuid4()
                nested2 = await db.begin_nested()
                try:
                    current.superseded_by_id = current.id  # placeholder: clears the unique index
                    await db.flush()
                    row = OrchestrationProcessRun(
                        id=new_row_id,
                        goal_id=goal_id,
                        run_id=run_id,
                        process_type=process_type,
                        process_version=1,
                        status="skipped",
                        trigger_reason="skip requested",
                        input_snapshot={},
                        skipped_by=skipped_by,
                        override_reason=reason,
                        completed_at=_utcnow(),
                    )
                    db.add(row)
                    await db.flush()  # new row now exists
                    current.superseded_by_id = new_row_id  # repoint to the real successor
                    await db.flush()
                    await nested2.commit()
                except IntegrityError:
                    await nested2.rollback()
                    winner = await self.get_current(db, goal_id, process_type)
                    if (
                        winner is not None
                        and winner.id != current.id
                        and winner.status == "skipped"
                    ):
                        return winner
                    raise
            else:
                # No current row exists, create new skipped row
                row = OrchestrationProcessRun(
                    goal_id=goal_id,
                    run_id=run_id,
                    process_type=process_type,
                    process_version=1,
                    status="skipped",
                    trigger_reason="skip requested",
                    input_snapshot={},
                    skipped_by=skipped_by,
                    override_reason=reason,
                    completed_at=_utcnow(),
                )
                db.add(row)
                nested2 = await db.begin_nested()
                try:
                    await db.flush()
                    await nested2.commit()
                except IntegrityError:
                    await nested2.rollback()
                    raise
        else:
            await nested.commit()
        db.add(
            OrchestrationWarning(
                goal_id=goal_id,
                run_id=run_id,
                warning_type=f"{process_type}_skipped",
                severity="warning",
                message=self._skip_warning_message(process_type, skipped_by, reason),
                source_process_run_id=row.id,
            )
        )
        await db.flush()
        return row

    async def complete_process(
        self,
        db: AsyncSession,
        process_run: OrchestrationProcessRun,
        *,
        outputs: dict | None = None,
    ) -> OrchestrationProcessRun:
        # Status-conditional UPDATE (not a load-then-write) so two concurrent
        # complete/skip calls on the same row can't both succeed: only the first
        # to land wins the row, the second sees rowcount 0 and fails instead of
        # silently overwriting terminal data (Finding 2).
        result = await db.execute(
            update(OrchestrationProcessRun)
            .where(
                OrchestrationProcessRun.id == process_run.id,
                OrchestrationProcessRun.status == "running",
            )
            .values(status="completed", outputs=outputs or {}, completed_at=_utcnow())
        )
        if result.rowcount != 1:
            raise ValueError(f"cannot complete process run in status {process_run.status!r}")
        await db.refresh(process_run)
        return process_run

    async def list_process_runs(
        self,
        db: AsyncSession,
        goal_id: uuid.UUID,
        *,
        process_type: str | None = None,
    ) -> list[OrchestrationProcessRun]:
        query = select(OrchestrationProcessRun).where(OrchestrationProcessRun.goal_id == goal_id)
        if process_type is not None:
            query = query.where(OrchestrationProcessRun.process_type == process_type)
        query = query.order_by(
            OrchestrationProcessRun.started_at.asc(),
            OrchestrationProcessRun.created_at.asc(),
            OrchestrationProcessRun.id.asc(),
        )
        result = await db.execute(query)
        return list(result.scalars().all())
