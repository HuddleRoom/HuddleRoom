"""Explicitly supervised, graceful backend restarts (never shell commands)."""
import os
import signal
import subprocess
import sys

from fastapi import Depends, Header, HTTPException, Request
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse

from huddleroom.config import settings
from huddleroom.dependencies import get_current_user

RESTART_EXIT_CODE = 75
draining = False


def ensure_accepting_work():
    if draining:
        raise HTTPException(503, "Backend is restarting")


class DrainMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if draining and scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1013})
            return
        await self.app(scope, receive, send)


class RestartController:
    def __init__(self, server, stop_producers=lambda: None, parent_pid=None):
        self.parent_pid = parent_pid
        self.server = server
        self.stop_producers = stop_producers
        self.requested = False
        self.shutdown_complete = False
        self.shutdown_error = None

    def stop(self):
        try:
            self.stop_producers()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Restart producer shutdown failed")
            self.shutdown_error = exc
        finally:
            self.server.should_exit = True

    def exit_code(self):
        if self.requested:
            return RESTART_EXIT_CODE if self.shutdown_complete and self.shutdown_error is None and not self.server.lifespan.shutdown_failed else 1
        return 0 if self.server.started else 1


async def request_restart(
    request: Request,
    confirmation: str | None = Header(default=None, alias="X-HuddleRoom-Confirm", examples=["restart"]),
    user=Depends(get_current_user),
):
    if not settings.allow_api_reset:
        raise HTTPException(404, "Backend restart is disabled")
    if request.headers.getlist("X-HuddleRoom-Confirm") != ["restart"]:
        raise HTTPException(400, "X-HuddleRoom-Confirm: restart is required")
    if len(request.headers.getlist("origin")) > 1:
        raise HTTPException(403, "Multiple origins are forbidden")
    origin = request.headers.get("origin")
    from urllib.parse import urlsplit
    def normalized(value):
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            return None
        return parsed.scheme.lower(), parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        same_origin = origin is None or normalized(origin) == normalized(str(request.base_url))
    except ValueError:
        same_origin = False
    if not same_origin:
        raise HTTPException(403, "Cross-origin restart is forbidden")
    controller = getattr(request.app.state, "backend_restart", None)
    if controller is None or not settings.is_sqlite or (controller.parent_pid is not None and os.getppid() != controller.parent_pid):
        raise HTTPException(503, "Restart requires huddleroom serve without --reload and local SQLite workers")
    if controller.requested:
        raise HTTPException(409, "Backend restart already requested")
    global draining
    controller.requested = True
    draining = True
    return JSONResponse({"status": "accepted", "detail": "Graceful restart requested; completion is not guaranteed"},
                        status_code=202, background=BackgroundTask(controller.stop))


def supervise(host, port):
    """Restart only a child that reports a completed API-requested shutdown."""
    child = None
    stopping = False
    stop_signal = signal.SIGTERM
    def forward(signum, _frame):
        nonlocal stopping, stop_signal
        stopping = True
        stop_signal = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while not stopping:
            environment = os.environ.copy()
            environment["_HUDDLEROOM_RESTART_PARENT_PID"] = str(os.getpid())
            child = subprocess.Popen([sys.executable, "-m", "huddleroom.cli", "serve", "--host", host,
                                      "--port", str(port), "--restart-child"], env=environment)
            if stopping and child.poll() is None:
                child.send_signal(stop_signal)
            result = child.wait()
            if stopping or result != RESTART_EXIT_CODE:
                return result if result >= 0 else 1
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


async def stop_tasks(tasks, *, restarting=False, timeout=5):
    """Do not interrupt a worker already performing cancellation cleanup."""
    import asyncio
    tasks = list(tasks)
    for task in tasks:
        if not task.cancelling():
            task.cancel()
    if not tasks:
        return
    if not restarting:
        await asyncio.gather(*tasks, return_exceptions=True)
        return
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        raise TimeoutError(f"{len(pending)} backend task(s) did not stop before restart timeout")
    for task in done:
        if not task.cancelled() and task.exception() is not None:
            raise RuntimeError("Backend task shutdown failed") from task.exception()


async def drain_scheduler(scheduler):
    """Join APScheduler async jobs before its synchronous shutdown discards them."""
    scheduler.pause()
    # APScheduler 3's AsyncIOExecutor.shutdown(wait=True) cancels without joining.
    # Capture its futures before shutdown clears the registry.
    futures = [future for executor in scheduler._executors.values()
               for future in executor._pending_futures]
    await stop_tasks(futures, restarting=True)
    import asyncio
    for future in futures:
        if not future.cancelled():
            for event in future.result() or []:
                error = getattr(event, "exception", None)
                if error is not None and not isinstance(error, asyncio.CancelledError):
                    raise RuntimeError("Scheduler job shutdown failed") from error
