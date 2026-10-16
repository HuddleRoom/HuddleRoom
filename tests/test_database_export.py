"""Database export tests use only disposable synthetic SQLite files."""
import asyncio
import sqlite3
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient
from huddleroom.config import Settings, settings
from huddleroom.main import create_app


async def connected_client():
    await asyncio.Event().wait()


def test_export_disabled_by_default():
    assert Settings(_env_file=None).allow_db_export is False


async def test_disabled_export_does_not_touch_database(monkeypatch):
    monkeypatch.setattr(settings,"allow_db_export",False)
    monkeypatch.setattr(sqlite3,"connect",lambda *a,**k: pytest.fail("database read when disabled"))
    async with AsyncClient(transport=ASGITransport(app=create_app()),base_url="http://test") as client:
        assert (await client.post("/api/system/database/export")).status_code == 404


async def test_export_includes_committed_wal_not_uncommitted(tmp_path,monkeypatch):
    source=tmp_path / "source.sqlite3"
    connection=sqlite3.connect(source)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE sample (value TEXT)")
    connection.execute("INSERT INTO sample VALUES ('committed')")
    connection.commit()
    connection.execute("INSERT INTO sample VALUES ('uncommitted')")
    monkeypatch.setattr(settings,"allow_db_export",True)
    monkeypatch.setattr(settings,"allow_api_reset",False)
    monkeypatch.setattr(settings,"database_url",f"sqlite+aiosqlite:///{source}")
    try:
        async with AsyncClient(transport=ASGITransport(app=create_app()),base_url="http://test") as client:
            response=await client.post("/api/system/database/export",headers={"X-HuddleRoom-Confirm":"export"})
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-type"] == "application/vnd.sqlite3"
        snapshot=tmp_path / "download.sqlite3"
        snapshot.write_bytes(response.content)
        with sqlite3.connect(snapshot) as copied:
            assert copied.execute("SELECT value FROM sample").fetchall() == [("committed",)]
    finally:
        connection.rollback()
        connection.close()


async def test_missing_source_is_not_created(tmp_path,monkeypatch):
    source=tmp_path / "missing.sqlite3"
    monkeypatch.setattr(settings,"allow_db_export",True)
    monkeypatch.setattr(settings,"database_url",f"sqlite+aiosqlite:///{source}")
    async with AsyncClient(transport=ASGITransport(app=create_app()),base_url="http://test") as client:
        response=await client.post("/api/system/database/export",headers={"X-HuddleRoom-Confirm":"export"})
    assert response.status_code == 503
    assert not source.exists()


def test_backup_closes_both_connections(tmp_path,monkeypatch):
    from huddleroom.services.database_export import backup_database
    import threading
    source=tmp_path / "source.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE sample(value TEXT)")
    closed=[]
    original=sqlite3.connect
    class Connection(sqlite3.Connection):
        def close(self):
            closed.append(self)
            super().close()
    monkeypatch.setattr(sqlite3,"connect",lambda *args,**kwargs: original(*args,**kwargs,factory=Connection))
    backup_database(source,tmp_path / "snapshot",threading.Event())
    assert len(closed) == 2


@pytest.mark.parametrize("url", ["sqlite+aiosqlite:///:memory:", "sqlite+aiosqlite://", "sqlite+aiosqlite:///file:test?uri=true", "postgresql://localhost/test", "not a URL"])
async def test_unsupported_sources_fail_closed(monkeypatch,url):
    monkeypatch.setattr(settings,"allow_db_export",True)
    monkeypatch.setattr(settings,"database_url",url)
    async with AsyncClient(transport=ASGITransport(app=create_app()),base_url="http://test") as client:
        response=await client.post("/api/system/database/export",headers={"X-HuddleRoom-Confirm":"export"})
    assert response.status_code == 503


async def test_response_cleanup_on_send_failure_and_success(tmp_path):
    from huddleroom.services.database_export import SnapshotResponse
    for fail in (False,True):
        snapshot=tmp_path / "snapshot"
        snapshot.write_bytes(b"database")
        async def send(message):
            assert message["type"] != "http.response.pathsend"
            if fail:
                raise OSError("client disconnected")
        response=SnapshotResponse(snapshot)
        scope={"type":"http","method":"POST","headers":[],"extensions":{"http.response.pathsend":{}}}
        if fail:
            with pytest.raises(OSError):
                await response(scope,connected_client,send)
        else:
            await response(scope,connected_client,send)
        assert not snapshot.exists()


async def test_cancelled_backup_joins_worker_before_deleting(tmp_path,monkeypatch):
    import threading
    from huddleroom.services import database_export as export
    entered,release,finished=threading.Event(),threading.Event(),threading.Event()
    files=[]
    def backup(source,destination,cancelled):
        files.append(destination)
        entered.set()
        assert release.wait(2)
        assert destination.exists()
        assert cancelled.is_set()
        finished.set()
    monkeypatch.setattr(export,"backup_database",backup)
    task=asyncio.create_task(export.create_snapshot(tmp_path / "unused"))
    while not entered.is_set():
        await asyncio.sleep(.001)
    task.cancel()
    await asyncio.sleep(.01)
    task.cancel()
    await asyncio.sleep(.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
    assert not files[0].exists()


async def test_backup_failure_cleans_snapshot(tmp_path,monkeypatch):
    from huddleroom.services import database_export as export
    files=[]
    def fail(source,destination,cancelled):
        files.append(destination)
        raise sqlite3.OperationalError("synthetic failure")
    monkeypatch.setattr(export,"backup_database",fail)
    with pytest.raises(sqlite3.OperationalError):
        await export.create_snapshot(tmp_path / "unused")
    assert not files[0].exists()


def test_backup_busy_timeout_closes_and_removes_temp(tmp_path,monkeypatch):
    from huddleroom.services import database_export as export
    import threading,time
    source=tmp_path / "locked.sqlite3"
    connection=sqlite3.connect(source)
    connection.execute("CREATE TABLE sample(value TEXT)")
    connection.commit()
    connection.execute("BEGIN EXCLUSIVE")
    monkeypatch.setattr(export,"BACKUP_TIMEOUT_SECONDS",.05)
    start=time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            export.backup_database(source,tmp_path / "snapshot",threading.Event())
        assert time.monotonic()-start < 1
    finally:
        connection.rollback()
        connection.close()


def test_export_config_precedence(tmp_path,monkeypatch):
    import huddleroom.config as config
    path=tmp_path / "config.toml"
    path.write_text("allow_db_export = true\n")
    monkeypatch.setattr(config,"DEFAULT_CONFIG_FILE",path)
    assert Settings(_env_file=None).allow_db_export
    monkeypatch.setenv("HUDDLEROOM_ALLOW_DB_EXPORT","false")
    assert not Settings(_env_file=None).allow_db_export
    assert Settings(_env_file=None,allow_db_export=True).allow_db_export


def test_swagger_exposes_binary_download_and_confirmation():
    operation=create_app().openapi()["paths"]["/api/system/database/export"]["post"]
    assert operation["responses"]["200"]["content"]["application/vnd.sqlite3"]["schema"]["format"] == "binary"
    assert any(item["in"] == "header" and item["name"] == "X-HuddleRoom-Confirm" for item in operation["parameters"])


@pytest.mark.parametrize("headers,status", [
    ({}, 400),
    ({"X-HuddleRoom-Confirm": "restart"}, 400),
    ([("X-HuddleRoom-Confirm", "export"), ("X-HuddleRoom-Confirm", "export")], 400),
    ({"X-HuddleRoom-Confirm": "export", "Origin": "null"}, 403),
    ({"X-HuddleRoom-Confirm": "export", "Origin": "https://other.test"}, 403),
    ([("X-HuddleRoom-Confirm", "export"), ("Origin", "http://test"), ("Origin", "http://evil")], 403),
])
async def test_export_rejections_never_start_snapshot(monkeypatch, headers, status):
    from huddleroom.services import database_export as export
    monkeypatch.setattr(settings, "allow_db_export", True)
    monkeypatch.setattr(export, "configured_source", lambda: pytest.fail("guard opened source"))
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as client:
        response = await client.post("/api/system/database/export", headers=headers)
    assert response.status_code == status


async def test_export_uses_only_configured_path_and_runs_off_event_loop(tmp_path, monkeypatch):
    import threading
    from huddleroom.services import database_export as export
    source = tmp_path / "configured # snapshot.sqlite3"
    other = tmp_path / "other.sqlite3"
    for path, value in ((source, "configured"), (other, "unrelated")):
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE sample(value TEXT)")
            db.execute("INSERT INTO sample VALUES (?)", (value,))
    before = source.read_bytes()
    thread_ids = []
    original = export.backup_database
    def record(*args):
        thread_ids.append(threading.get_ident())
        return original(*args)
    monkeypatch.setattr(export, "backup_database", record)
    monkeypatch.setattr(settings, "allow_db_export", True)
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{source}")
    main_thread = threading.get_ident()
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as client:
        assert (await client.get("/api/system/database/export")).status_code == 405
        response = await client.post("/api/system/database/export", params={"path": str(other)},
                                     json={"database_url": f"sqlite+aiosqlite:///{other}"},
                                     headers={"X-HuddleRoom-Confirm": "export", "Origin": "http://test:80"})
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="huddleroom.sqlite3"'
    assert thread_ids and all(item != main_thread for item in thread_ids)
    assert source.read_bytes() == before
    downloaded = tmp_path / "download.sqlite3"
    downloaded.write_bytes(response.content)
    with sqlite3.connect(downloaded) as db:
        assert db.execute("SELECT value FROM sample").fetchall() == [("configured",)]


async def test_cancelled_download_removes_snapshot(tmp_path):
    from huddleroom.services.database_export import SnapshotResponse
    snapshot = tmp_path / "snapshot.sqlite3"
    snapshot.write_bytes(b"synthetic")
    entered = asyncio.Event()
    async def send(message):
        if message["type"] == "http.response.body":
            entered.set()
            await asyncio.Event().wait()
    scope = {"type": "http", "method": "POST", "headers": []}
    task = asyncio.create_task(SnapshotResponse(snapshot)(scope, connected_client, send))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not snapshot.exists()


async def test_http_disconnect_stops_transfer_before_snapshot_cleanup(tmp_path):
    from huddleroom.services.database_export import SnapshotResponse
    snapshot = tmp_path / "snapshot.sqlite3"
    snapshot.write_bytes(b"synthetic")
    sending, cleanup = asyncio.Event(), asyncio.Event()
    async def send(message):
        if message["type"] == "http.response.body":
            sending.set()
            try:
                await asyncio.Event().wait()
            finally:
                assert snapshot.exists()
                cleanup.set()
    async def receive():
        await sending.wait()
        return {"type": "http.disconnect"}
    scope = {"type": "http", "method": "POST", "headers": []}
    await asyncio.wait_for(SnapshotResponse(snapshot)(scope, receive, send), 1)
    assert cleanup.is_set()
    assert not snapshot.exists()


async def test_http_disconnect_during_backup_joins_worker_before_unlink(tmp_path, monkeypatch):
    import threading
    from huddleroom.services import database_export as export
    entered, finished = threading.Event(), threading.Event()
    paths = []
    def backup(source, destination, cancelled):
        paths.append(destination)
        entered.set()
        assert cancelled.wait(2)
        assert destination.exists()
        finished.set()
    async def receive():
        while not entered.is_set():
            await asyncio.sleep(.001)
        return {"type": "http.disconnect"}
    monkeypatch.setattr(export, "backup_database", backup)
    with pytest.raises(InterruptedError, match="disconnected"):
        await export.create_snapshot(tmp_path / "synthetic-source", receive)
    assert finished.is_set()
    assert not paths[0].exists()


async def test_cancellation_during_successful_backup_monitor_cleanup_removes_snapshot(tmp_path, monkeypatch):
    from huddleroom.services import database_export as export
    entered, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    paths = []
    async def monitor(receive):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
    def backup(source, destination, cancelled):
        paths.append(destination)
    monkeypatch.setattr(export, "wait_for_disconnect", monitor)
    monkeypatch.setattr(export, "backup_database", backup)
    task = asyncio.create_task(export.create_snapshot(tmp_path / "synthetic-source", connected_client))
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.wait_for(cleaning.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not paths[0].exists()


async def test_monitor_cleanup_failure_after_backup_removes_snapshot(tmp_path, monkeypatch):
    from huddleroom.services import database_export as export
    paths = []
    async def monitor(receive):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError("monitor cleanup failed")
    def backup(source, destination, cancelled):
        paths.append(destination)
    monkeypatch.setattr(export, "wait_for_disconnect", monitor)
    monkeypatch.setattr(export, "backup_database", backup)
    with pytest.raises(RuntimeError, match="monitor cleanup failed"):
        await export.create_snapshot(tmp_path / "synthetic-source", connected_client)
    assert not paths[0].exists()
