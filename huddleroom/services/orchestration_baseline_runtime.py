"""Interruptible provider boundary for local baseline operations."""
import asyncio
import uuid
from contextlib import asynccontextmanager
from contextvars import Context, ContextVar
from dataclasses import dataclass, field
from typing import Any
from fastapi import HTTPException
from sqlalchemy import select, update, delete
from huddleroom.models.orchestration import OrchestrationRun, OrchestrationGoal, OrchestrationAction
from huddleroom.models.orchestration_process import OrchestrationProcessRun, OrchestrationAuthorityDecision

class BaselineInterrupted(asyncio.CancelledError):
    """Reset cancellation bypasses analyzer error/repair handling."""

class BaselinePaused(asyncio.CancelledError):
    """A normal durable pause boundary bypasses analyzer error and repair handling."""


class BaselineTransactionConflict(asyncio.CancelledError):
    """Provider work cannot consume or publish a caller-owned transaction."""


async def require_control_admission(db, run_id, *, require_baseline_authorization=True):
    from huddleroom.services.orchestration_control import acquire_control_writer
    await acquire_control_writer(db, run_id)
    with db.no_autoflush:
        row = (await db.execute(select(
            OrchestrationRun.pause_requested, OrchestrationRun.status, OrchestrationRun.phase,
            OrchestrationRun.baseline_authorized, OrchestrationGoal.id,
            OrchestrationGoal.status, OrchestrationGoal.parent_goal_id,
        ).join(OrchestrationGoal, OrchestrationGoal.id == OrchestrationRun.goal_id).where(
            OrchestrationRun.id == run_id,
        ))).one_or_none()
        if row is None or row[0] or row[1] not in {"running", "blocked"} or row[5] not in {"active", "blocked"}:
            raise BaselinePaused()
        if require_baseline_authorization and row[2] == "baseline" and not row[3]:
            raise BaselinePaused()
        parent = row[6]
        seen = {row[4]}
        while parent is not None:
            if parent in seen:
                raise BaselinePaused()
            seen.add(parent)
            ancestor = (await db.execute(select(OrchestrationGoal.status, OrchestrationGoal.parent_goal_id).where(
                OrchestrationGoal.id == parent,
            ))).one_or_none()
            if ancestor is None or ancestor[0] not in {"active", "blocked"}:
                raise BaselinePaused()
            parent = ancestor[1]


@dataclass
class Operation:
    db: Any
    owns_transaction: bool
    run_id: Any = None
    tick: bool = False
    source_action_id: Any = None
    cancel_provider_on_cancel: bool = False
    savepoint: Any = None
    operation_id: Any = field(default_factory=uuid.uuid4)
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
async def baseline_operation(db, goal_id, *, owns_transaction, tick=False, register=True,
                             run_id=None, source_action_id=None, cancel_provider_on_cancel=False):
    require_baseline_available(goal_id)
    operation = Operation(db, owns_transaction, run_id=run_id, tick=tick, source_action_id=source_action_id,
                          cancel_provider_on_cancel=cancel_provider_on_cancel)
    token = _current.set(operation)
    if register:
        _active[goal_id] = operation
    try:
        if operation.run_id is None:
            operation.run_id = await db.scalar(
                select(OrchestrationRun.id).where(OrchestrationRun.goal_id == goal_id)
                .order_by(OrchestrationRun.created_at.desc()).limit(1)
            )
        yield
        from huddleroom.services.orchestration_control import apply_pause_status
        if operation.run_id is not None and not operation.interrupted.is_set():
            snapshot = await apply_pause_status(db, operation.run_id)
            if snapshot.pause_requested:
                await db.flush()
            if owns_transaction:
                await db.commit()
        if operation.interrupted.is_set():
            raise BaselineInterrupted()
    except BaselineTransactionConflict:
        raise HTTPException(409, detail={
            "conflict": "orchestration_transaction_shape",
            "message": "Commit the caller transaction before provider continuation",
        }) from None
    except BaselinePaused:
        from huddleroom.services.orchestration_control import apply_pause_status
        if operation.run_id is not None:
            await apply_pause_status(db, operation.run_id)
            await db.flush()
        if owns_transaction:
            await db.commit()
        raise HTTPException(status_code=409, detail="Orchestration admission paused") from None
    except BaselineInterrupted:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Baseline operation discarded by reset") from None
    finally:
        _current.reset(token)
        if _active.get(goal_id) is operation:
            _active.pop(goal_id, None)
        operation.finished.set()

async def publish_baseline_transaction(db):
    operation = _current.get()
    if operation is not None and operation.run_id is not None:
        from huddleroom.services.orchestration_control import apply_pause_status
        await apply_pause_status(db, operation.run_id)
    await db.flush()
    await db.commit()


def current_baseline_operation():
    return _current.get()


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

async def await_baseline_completion(completion_fn, request, *, observer=None):
    from huddleroom.services.agent_response_stream import supports_observer, mark_observation_truncated, _prefix, DisplayTextBuffer
    observable = observer is not None and supports_observer(completion_fn)
    operation = _current.get()
    if operation is None:
        if observable:
            return await completion_fn(**request, observer=observer)
        return await completion_fn(**request)
    if operation.interrupted.is_set():
        raise BaselineInterrupted()
    if not operation.owns_transaction:
        raise BaselineTransactionConflict()
    nested = operation.db.get_nested_transaction()
    registered = operation.savepoint
    if nested and (registered is None or nested is not registered.nested):
        raise BaselineTransactionConflict()
    await require_control_admission(
        operation.db, operation.run_id, require_baseline_authorization=operation.tick,
    )
    call_id = uuid.uuid4()
    source = await operation.db.scalar(select(OrchestrationProcessRun.id).where(
        OrchestrationProcessRun.run_id == operation.run_id,
        OrchestrationProcessRun.status == "running",
        OrchestrationProcessRun.superseded_by_id.is_(None),
    ).order_by(OrchestrationProcessRun.created_at.desc()).limit(1))
    operation.db.add(OrchestrationAction(
        id=call_id, run_id=operation.run_id, action_type="admit_control_call", status="completed",
        idempotency_key=f"run:{operation.run_id}:control_call:{call_id}",
        request={"operation_id": str(operation.operation_id), "call_id": str(call_id),
                 "source_process_id": str(source) if source else None,
                 "source_action_id": str(operation.source_action_id) if operation.source_action_id else None},
        target_type="process" if source else "orchestration_run", target_id=source or operation.run_id,
    ))
    if nested:
        await nested.commit()
        registered.published = True
    await operation.db.flush()
    await operation.db.commit()
    # Provider-only task: application writes and streaming output stay in parent.
    values = {key: value for key, value in request.items() if key != "stream"}
    # The provider closure retains only bounded transport state, never the call
    # or application context. Closing this conduit also releases queued records.
    queue = asyncio.Queue(maxsize=32)
    state = {"active": True, "truncated": False}
    display = DisplayTextBuffer()
    async def enqueue(stream, text, *, final=False):
        if not state["active"] or state["truncated"] or (not text and not final):
            return
        part, cutoff = _prefix(display.feed(stream, text, final=final), 16 * 1024)
        if not part:
            return
        try:
            queue.put_nowait((stream, part))
        except asyncio.QueueFull:
            state["truncated"] = True
        if cutoff:
            state["truncated"] = True
    async def drain():
        while True:
            stream, text = await queue.get()
            try:
                if state["active"] and not operation.interrupted.is_set():
                    if state["truncated"]:
                        mark_observation_truncated(observer)
                    await observer(stream, text)
            finally:
                queue.task_done()
    if observable:
        values["observer"] = enqueue
    observation_drain = asyncio.create_task(drain()) if observable else None
    async def invoke_provider():
        result = await completion_fn(**values)
        if observable:
            for channel in ("output", "reasoning", "stderr"):
                await enqueue(channel, "", final=True)
        return result
    provider = asyncio.create_task(invoke_provider(), context=Context())
    _detached.add(provider)
    provider.add_done_callback(_drain, context=Context())
    signal = asyncio.create_task(operation.interrupted.wait())
    try:
        waiting = {provider, signal}
        if observation_drain:
            waiting.add(observation_drain)
        await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        if operation.interrupted.is_set():
            raise BaselineInterrupted()
        if observation_drain and observation_drain.done():
            observation_drain.result()
        if operation.run_id is not None and await operation.db.scalar(
            select(OrchestrationRun.id).where(OrchestrationRun.id == operation.run_id)
        ) is None:
            raise BaselineInterrupted()
        if operation.interrupted.is_set():
            raise BaselineInterrupted()
        if observation_drain:
            joined = asyncio.create_task(queue.join())
            try:
                await asyncio.wait({joined, signal, observation_drain}, return_when=asyncio.FIRST_COMPLETED)
                if operation.interrupted.is_set():
                    raise BaselineInterrupted()
                if observation_drain.done():
                    observation_drain.result()
                if state["truncated"]:
                    mark_observation_truncated(observer)
            finally:
                joined.cancel()
                await asyncio.gather(joined, return_exceptions=True)
        response = provider.result()
        from huddleroom.services.orchestration_control import acquire_control_writer
        await acquire_control_writer(operation.db, operation.run_id)
        admission = await operation.db.get(OrchestrationAction, call_id)
        if admission is not None:
            usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
            safe_usage = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
                if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2 ** 63:
                    safe_usage[key] = value
            audit_call = getattr(getattr(observer, "__self__", None), "call_id", None)
            admission.dispatch_contract = {"provider_completed": True, "usage": safe_usage,
                                           "audit_call_id": str(audit_call) if audit_call else None}
            await operation.db.flush()
        return response
    except asyncio.CancelledError:
        if operation.cancel_provider_on_cancel and not operation.interrupted.is_set():
            provider.cancel()
            await asyncio.gather(provider, return_exceptions=True)
        raise
    finally:
        state["active"] = False
        if observation_drain:
            observation_drain.cancel()
            await asyncio.gather(observation_drain, return_exceptions=True)
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()
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
