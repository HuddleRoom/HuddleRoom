import asyncio
import logging
import uuid
from contextvars import Context

from fastapi import HTTPException

from huddleroom.workers.session_tasks import (
    execute_api_session,
    execute_cli_session,
    is_project_not_runnable_conflict,
)

logger = logging.getLogger(__name__)

_running_tasks: dict[str, asyncio.Task] = {}
_task_projects: dict[str, str] = {}
_deferred_session_dispatches: dict[tuple[str, asyncio.Task], tuple[str, str, str, str]] = {}
PROJECT_CANCELLATION_TIMEOUT_SECONDS = 5


def get_running_tasks() -> dict[str, asyncio.Task]:
    return _running_tasks


def is_task_live(task_id: str | None) -> bool:
    """Only an exact, live registry entry proves a local worker is alive."""
    return bool(task_id and (task := _running_tasks.get(task_id)) is not None and not task.done())


def publish_session_dispatch(
    session_id: str, adapter_type: str, project_id: str | uuid.UUID, task_id: str,
) -> None:
    """Publish one committed dispatch, retaining it until an exact closing worker exits."""
    from huddleroom.services.backend_restart import draining
    if draining:
        return
    worker = _running_tasks.get(task_id)
    if worker is None or worker.done():
        register_session(session_id, adapter_type, project_id, task_id=task_id)
        return
    key = (task_id, worker)
    if key in _deferred_session_dispatches:
        return
    _deferred_session_dispatches[key] = (session_id, adapter_type, str(project_id), task_id)
    worker.add_done_callback(
        lambda _done: _publish_deferred_session_dispatch(key), context=Context(),
    )


def _publish_deferred_session_dispatch(key: tuple[str, asyncio.Task]) -> None:
    """Consume the frozen intent once, after the old worker's finally cleanup."""
    intent = _deferred_session_dispatches.pop(key, None)
    from huddleroom.services.backend_restart import draining
    if intent is None or draining or key[1].cancelled():
        return
    current = _running_tasks.get(key[0])
    if current is not None and not current.done():
        return
    session_id, adapter_type, project_id, task_id = intent
    register_session(session_id, adapter_type, project_id, task_id=task_id)


def register_session(
    session_id: str,
    adapter_type: str,
    project_id: str | uuid.UUID,
    task_id: str | None = None,
) -> str:
    """Register a session task on the running event loop."""
    from huddleroom.services.backend_restart import ensure_accepting_work
    ensure_accepting_work()
    task_id = task_id or str(uuid.uuid4())

    if adapter_type == "api":
        coro = execute_api_session(session_id, task_id)
    elif adapter_type == "cli":
        coro = execute_cli_session(session_id, task_id)
    else:
        logger.warning("Unknown adapter type: %s", adapter_type)
        return task_id

    async def _wrapped():
        try:
            await coro
        except Exception as e:
            logger.error("Session task %s failed: %s", task_id, e)
            from huddleroom.services.backend_restart import draining
            if draining:
                raise
        finally:
            if _running_tasks.get(task_id) is asyncio.current_task():
                _running_tasks.pop(task_id, None)
                _task_projects.pop(task_id, None)

    task = asyncio.create_task(_wrapped())
    _running_tasks[task_id] = task
    _task_projects[task_id] = str(project_id)
    return task_id


async def dispatch_session(
    session_id: str,
    adapter_type: str,
    project_id: str | uuid.UUID,
    task_id: str | None = None,
) -> str:
    """Dispatch a session task. Returns a task ID for tracking."""
    return register_session(session_id, adapter_type, project_id, task_id)


async def cancel_task(task_id: str) -> bool:
    """Cancel a running task. Returns True if cancelled."""
    task = _running_tasks.get(task_id)
    if task is None:
        return False
    if not task.cancelling():
        task.cancel()
    try:
        # A second caller or cancellation of this waiter must not interrupt
        # the worker's process cleanup and terminal transaction commit.
        await asyncio.shield(task)
    except asyncio.CancelledError:
        if not task.done() or asyncio.current_task().cancelling():
            raise
    return True


async def cancel_project_sessions(project_id: str | uuid.UUID) -> int:
    """Cancel and await all in-process session tasks owned by a project."""
    project_key = str(project_id)
    selected = [
        (task_id, task)
        for task_id, task in _running_tasks.items()
        if _task_projects.get(task_id) == project_key
    ]
    for _, task in selected:
        task.cancel()
    if selected:
        _, pending = await asyncio.wait(
            (task for _, task in selected), timeout=PROJECT_CANCELLATION_TIMEOUT_SECONDS
        )
        if pending:
            raise TimeoutError(f"{len(pending)} project session task(s) did not stop before reset timeout")
    for task_id, selected_task in selected:
        if _running_tasks.get(task_id) is selected_task:
            _running_tasks.pop(task_id, None)
            _task_projects.pop(task_id, None)
    return len(selected)


async def retry_api_session(
    session_id: str,
    project_id: str | uuid.UUID,
    max_retries: int = 3,
) -> str:
    """Dispatch an API session with retry logic."""
    from huddleroom.services.backend_restart import ensure_accepting_work
    ensure_accepting_work()
    task_id = str(uuid.uuid4())

    async def _with_retry():
        for attempt in range(max_retries + 1):
            try:
                await execute_api_session(session_id, task_id)
                return
            except HTTPException as exc:
                if is_project_not_runnable_conflict(exc):
                    logger.warning("API session %s rejected: %s", session_id, exc.detail)
                    raise
                raise
            except Exception as e:
                if attempt == max_retries:
                    logger.error("API session %s failed after %d retries: %s", session_id, max_retries, e)
                    raise
                delay = [10, 30, 60][min(attempt, 2)]
                logger.warning("API session %s attempt %d failed, retrying in %ds: %s", session_id, attempt + 1, delay, e)
                await asyncio.sleep(delay)
    async def _wrapped():
        try:
            await _with_retry()
        except Exception:
            from huddleroom.services.backend_restart import draining
            if draining:
                raise
        finally:
            if _running_tasks.get(task_id) is asyncio.current_task():
                _running_tasks.pop(task_id, None)
                _task_projects.pop(task_id, None)

    task = asyncio.create_task(_wrapped())
    _running_tasks[task_id] = task
    _task_projects[task_id] = str(project_id)
    return task_id


async def cancel_all_sessions() -> int:
    """Bounded cancellation of registered local workers after producers stop."""
    from huddleroom.workers.meeting_tasks import get_running_meeting_tasks
    selected = list(_running_tasks.items()) + list(get_running_meeting_tasks().items())
    for _, task in selected:
        if not task.cancelling():
            task.cancel()
    if selected:
        done, pending = await asyncio.wait([task for _, task in selected], timeout=PROJECT_CANCELLATION_TIMEOUT_SECONDS)
        if pending:
            raise TimeoutError(f"{len(pending)} session task(s) did not stop before restart timeout")
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                raise RuntimeError("Session shutdown failed") from task.exception()
    for task_id, selected_task in selected:
        if _running_tasks.get(task_id) is selected_task:
            _running_tasks.pop(task_id, None)
            _task_projects.pop(task_id, None)
    return len(selected)
