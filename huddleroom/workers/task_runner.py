import asyncio
import logging
import uuid

from fastapi import HTTPException

from huddleroom.workers.session_tasks import (
    execute_api_session,
    execute_cli_session,
    is_project_not_runnable_conflict,
)

logger = logging.getLogger(__name__)

_running_tasks: dict[str, asyncio.Task] = {}
_task_projects: dict[str, str] = {}
PROJECT_CANCELLATION_TIMEOUT_SECONDS = 5


def get_running_tasks() -> dict[str, asyncio.Task]:
    return _running_tasks


def is_task_live(task_id: str | None) -> bool:
    """Only an exact, live registry entry proves a local worker is alive."""
    return bool(task_id and (task := _running_tasks.get(task_id)) is not None and not task.done())


def register_session(
    session_id: str,
    adapter_type: str,
    project_id: str | uuid.UUID,
    task_id: str | None = None,
) -> str:
    """Register a session task on the running event loop."""
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
        finally:
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
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        _running_tasks.pop(task_id, None)
        _task_projects.pop(task_id, None)
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
    for task_id, _ in selected:
        _running_tasks.pop(task_id, None)
        _task_projects.pop(task_id, None)
    return len(selected)


async def retry_api_session(
    session_id: str,
    project_id: str | uuid.UUID,
    max_retries: int = 3,
) -> str:
    """Dispatch an API session with retry logic."""
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
            pass
        finally:
            _running_tasks.pop(task_id, None)
            _task_projects.pop(task_id, None)

    task = asyncio.create_task(_wrapped())
    _running_tasks[task_id] = task
    _task_projects[task_id] = str(project_id)
    return task_id
