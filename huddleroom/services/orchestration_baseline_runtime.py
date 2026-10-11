"""Interruptible provider boundary for local baseline operations."""
import asyncio
from contextlib import asynccontextmanager
from contextvars import Context, ContextVar
from dataclasses import dataclass, field
from typing import Any
from fastapi import HTTPException
from sqlalchemy import select, update, delete
from huddleroom.models.orchestration import OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationProcessRun, OrchestrationAuthorityDecision

class BaselineInterrupted(asyncio.CancelledError):
    """Reset cancellation bypasses analyzer error/repair handling."""

@dataclass
class Operation:
    db: Any
    owns_transaction: bool
    run_id: Any = None
    savepoint: Any = None
    interrupted: asyncio.Event = field(default_factory=asyncio.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)

_current = ContextVar("baseline_operation", default=None)
_active = {}
_detached = set()
_resetting = {}

def _drain(task):
    _detached.discard(task)
    if not task.cancelled():
        task.exception()

def require_baseline_available(goal_id):
    if goal_id in _resetting:
        raise HTTPException(status_code=409, detail="Baseline operation discarded by reset")

@asynccontextmanager
async def baseline_operation(db, goal_id, *, owns_transaction, tick=False):
    require_baseline_available(goal_id)
    operation = Operation(db, owns_transaction)
    token = _current.set(operation)
    _active[goal_id] = operation
    try:
        operation.run_id = await db.scalar(
            select(OrchestrationRun.id).where(OrchestrationRun.goal_id == goal_id)
            .order_by(OrchestrationRun.created_at.desc()).limit(1)
        )
        yield
        if operation.interrupted.is_set():
            raise BaselineInterrupted()
    except BaselineInterrupted:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Baseline operation discarded by reset") from None
    finally:
        _current.reset(token)
        if _active.get(goal_id) is operation:
            _active.pop(goal_id, None)
        operation.finished.set()

async def interrupt_baseline(goal_id):
    operation = _active.get(goal_id)
    if operation:
        operation.interrupted.set()
        await operation.finished.wait()

@asynccontextmanager
async def reset_baseline(goal_id):
    _resetting[goal_id] = _resetting.get(goal_id, 0) + 1
    try:
        await interrupt_baseline(goal_id)
        yield
    finally:
        _resetting[goal_id] -= 1
        if not _resetting[goal_id]:
            _resetting.pop(goal_id)

async def await_baseline_completion(completion_fn, request):
    operation = _current.get()
    if operation is None:
        return await completion_fn(**request)
    if operation.interrupted.is_set():
        raise BaselineInterrupted()
    if operation.owns_transaction:
        nested = operation.db.get_nested_transaction()
        registered = operation.savepoint
        if nested and (registered is None or nested is not registered.nested):
            # A nested transaction introduced by the caller/process is not ours.
            pass
        else:
            if nested:
                await nested.commit()
                registered.published = True
            await operation.db.flush()
            await operation.db.commit()
    # Provider-only task: application writes and streaming output stay in parent.
    values = {key: value for key, value in request.items() if key != "stream"}
    provider = asyncio.create_task(completion_fn(**values), context=Context())
    _detached.add(provider)
    provider.add_done_callback(_drain, context=Context())
    signal = asyncio.create_task(operation.interrupted.wait())
    try:
        await asyncio.wait({provider, signal}, return_when=asyncio.FIRST_COMPLETED)
        if operation.interrupted.is_set():
            raise BaselineInterrupted()
        if operation.run_id is not None and await operation.db.scalar(
            select(OrchestrationRun.id).where(OrchestrationRun.id == operation.run_id)
        ) is None:
            raise BaselineInterrupted()
        if operation.interrupted.is_set():
            raise BaselineInterrupted()
        return provider.result()
    finally:
        signal.cancel()
        await asyncio.gather(signal, return_exceptions=True)

class BaselineSavepoint:
    """Only this explicit rerun savepoint can be published at a provider boundary."""
    @classmethod
    async def create(cls, db, goal_id):
        self = cls()
        self.db, self.goal_id = db, goal_id
        processes = list((await db.scalars(select(OrchestrationProcessRun).where(
            OrchestrationProcessRun.goal_id == goal_id))).all())
        self.processes = {row.id: (row.status, row.superseded_by_id) for row in processes}
        decisions = list((await db.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == goal_id,
            OrchestrationAuthorityDecision.status == "pending"))).all())
        self.decisions = {row.id: row.reason for row in decisions}
        self.nested = await db.begin_nested()
        self.published = False
        operation = _current.get()
        if operation is not None and operation.owns_transaction:
            operation.savepoint = self
        return self

    async def commit(self):
        if self.nested.is_active:
            await self.nested.commit()

    async def rollback(self):
        if self.nested.is_active:
            await self.nested.rollback()
        elif self.published:
            # Discard any post-provider writes before restoring the pre-rerun
            # current/history and pending questions from primitive snapshots.
            await self.db.rollback()
            new_ids = list((await self.db.scalars(select(OrchestrationProcessRun.id).where(
                OrchestrationProcessRun.goal_id == self.goal_id,
                OrchestrationProcessRun.id.not_in(self.processes)))).all())
            if new_ids:
                await self.db.execute(delete(OrchestrationAuthorityDecision).where(
                    OrchestrationAuthorityDecision.source_process_run_id.in_(new_ids)))
                await self.db.execute(delete(OrchestrationProcessRun).where(
                    OrchestrationProcessRun.id.in_(new_ids)))
            for row_id, (status, superseded) in self.processes.items():
                await self.db.execute(update(OrchestrationProcessRun).where(
                    OrchestrationProcessRun.id == row_id).values(status=status, superseded_by_id=superseded))
            for row_id, reason in self.decisions.items():
                await self.db.execute(update(OrchestrationAuthorityDecision).where(
                    OrchestrationAuthorityDecision.id == row_id).values(status="pending", reason=reason))
            await self.db.commit()
