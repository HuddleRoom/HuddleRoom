"""Disposable online snapshots of the configured SQLite database."""
import asyncio
from contextlib import closing
import logging
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from starlette.responses import FileResponse

from huddleroom.config import settings
from huddleroom.dependencies import get_current_user

BACKUP_TIMEOUT_SECONDS = 30
_logger = logging.getLogger(__name__)


def configured_source() -> Path:
    try:
        url = make_url(settings.database_url)
    except (ArgumentError, ValueError):
        raise HTTPException(503, "Configured database URL is unsupported") from None
    if (url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:"
            or url.database.startswith("file:") or url.query or url.host or url.username or url.password or url.port):
        raise HTTPException(503, "Export requires a configured file-backed SQLite database without URI options")
    source = Path(url.database).resolve()
    if not source.is_file():
        raise HTTPException(503, "Configured database file is unavailable")
    return source


def backup_database(source: Path, destination: Path, cancelled: threading.Event) -> None:
    deadline = time.monotonic() + BACKUP_TIMEOUT_SECONDS
    def progress(_status, _remaining, _total):
        if cancelled.is_set():
            raise InterruptedError("Database export cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError("Database export timed out")
    progress(0,0,0)
    # mode=ro forbids accidental creation or mutation of the configured database.
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=.1)) as origin:
        with closing(sqlite3.connect(destination)) as snapshot:
            origin.backup(snapshot, pages=128, progress=progress, sleep=.01)


async def wait_for_disconnect(receive) -> None:
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return
        if message["type"] != "http.request":
            raise RuntimeError("Unexpected request transport message")


async def create_snapshot(source: Path, receive=None) -> Path:
    descriptor, name = tempfile.mkstemp(prefix="huddleroom-export-", suffix=".sqlite3")
    os.close(descriptor)
    destination = Path(name)
    cancelled = threading.Event()
    worker = asyncio.create_task(asyncio.to_thread(backup_database, source, destination, cancelled))
    monitor = asyncio.create_task(wait_for_disconnect(receive)) if receive is not None else None
    try:
        if monitor is not None:
            done, _ = await asyncio.wait({worker, monitor}, return_when=asyncio.FIRST_COMPLETED)
            if monitor in done:
                await monitor
                raise InterruptedError("Database export client disconnected")
        await asyncio.shield(worker)
        return destination
    except BaseException:
        cancelled.set()
        # Cancelling to_thread does not stop the worker. Join before unlinking its file.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if worker.done() and not worker.cancelled():
            worker.exception()  # consume any worker exception while preserving caller failure
        destination.unlink(missing_ok=True)
        raise
    finally:
        try:
            if monitor is not None:
                if not monitor.done():
                    monitor.cancel()
                external_cancellation = None
                while not monitor.done():
                    try:
                        await asyncio.shield(monitor)
                    except asyncio.CancelledError as exc:
                        if asyncio.current_task().cancelling():
                            external_cancellation = exc
                if not monitor.cancelled():
                    monitor.exception()
                if external_cancellation is not None:
                    destination.unlink(missing_ok=True)
                    raise external_cancellation
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


class SnapshotResponse(FileResponse):
    async def __call__(self, scope, receive, send):
        # pathsend hands a path to the server; keep cleanup after actual file reading.
        scope = dict(scope)
        scope["extensions"] = {key: value for key,value in scope.get("extensions",{}).items()
                               if key != "http.response.pathsend"}
        transfer = asyncio.create_task(super().__call__(scope, receive, send))
        monitor = asyncio.create_task(wait_for_disconnect(receive))
        try:
            done, _ = await asyncio.wait({transfer, monitor}, return_when=asyncio.FIRST_COMPLETED)
            if transfer in done:
                await transfer
            else:
                await monitor  # propagate receive failures, too
        finally:
            for task in (transfer, monitor):
                if not task.done() and not task.cancelling():
                    task.cancel()
            joined = asyncio.gather(transfer, monitor, return_exceptions=True)
            cancelled = None
            while not joined.done():
                try:
                    await asyncio.shield(joined)
                except asyncio.CancelledError as exc:
                    cancelled = exc
            Path(self.path).unlink(missing_ok=True)
            if cancelled is not None:
                raise cancelled


def confirm_export(request: Request) -> None:
    if request.headers.getlist("X-HuddleRoom-Confirm") != ["export"]:
        raise HTTPException(400, "X-HuddleRoom-Confirm: export is required")
    origins = request.headers.getlist("origin")
    def normalized(value):
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http","https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.path not in {"","/"} or parsed.query or parsed.fragment):
            return None
        return parsed.scheme.lower(), parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        valid = not origins or (len(origins) == 1 and normalized(origins[0]) == normalized(str(request.base_url)))
    except ValueError:
        valid = False
    if not valid:
        raise HTTPException(403, "Cross-origin export is forbidden")


async def export_database(
    request: Request,
    confirmation: str | None = Header(default=None, alias="X-HuddleRoom-Confirm", examples=["export"]),
    user=Depends(get_current_user),
):
    if not settings.allow_db_export:
        raise HTTPException(404, "Database export is disabled")
    confirm_export(request)
    source = configured_source()
    try:
        snapshot = await create_snapshot(source, request.receive)
    except TimeoutError:
        raise HTTPException(503, "Database export timed out") from None
    except (sqlite3.Error, OSError):
        _logger.exception("Configured database export failed")
        raise HTTPException(503, "Database export failed") from None
    try:
        return SnapshotResponse(snapshot, filename="huddleroom.sqlite3", media_type="application/vnd.sqlite3",
                                headers={"Cache-Control":"no-store"})
    except BaseException:
        snapshot.unlink(missing_ok=True)
        raise
