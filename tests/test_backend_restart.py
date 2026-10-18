import asyncio
from types import SimpleNamespace
import pytest
from httpx import ASGITransport, AsyncClient
from huddleroom.config import Settings, settings
from huddleroom.main import create_app


@pytest.fixture(autouse=True)
def reset_drain():
    from huddleroom.services import backend_restart
    backend_restart.draining = False
    yield
    backend_restart.draining = False


def test_reset_disabled_by_default():
    assert Settings(_env_file=None).allow_api_reset is False


async def test_restart_acknowledges_then_drains(monkeypatch):
    from huddleroom.services.backend_restart import RestartController
    monkeypatch.setattr(settings, "allow_api_reset", True)
    server = SimpleNamespace(should_exit=False)
    app = create_app()
    controller = RestartController(server)
    app.state.backend_restart = controller
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/system/restart", headers={"X-HuddleRoom-Confirm": "restart"})
        assert response.status_code == 202
        assert server.should_exit
        assert (await client.post("/api/system/restart", headers={"X-HuddleRoom-Confirm": "restart"})).status_code == 409
        assert (await client.get("/api/v1/projects")).status_code == 503


@pytest.mark.parametrize("enabled,headers,expected", [
    (False, {}, 404), (True, {}, 400),
    (True, {"X-HuddleRoom-Confirm": "restart", "Origin": "http://evil"}, 403),
    (True, {"X-HuddleRoom-Confirm": "restart"}, 503),
])
async def test_restart_guards(monkeypatch, enabled, headers, expected):
    monkeypatch.setattr(settings, "allow_api_reset", enabled)
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://test") as client:
        assert (await client.post("/api/system/restart", headers=headers)).status_code == expected


async def test_cancel_all_registered_sessions():
    from huddleroom.workers.task_runner import get_running_tasks, cancel_all_sessions
    task = asyncio.create_task(asyncio.sleep(100))
    get_running_tasks()["restart-test"] = task
    assert await cancel_all_sessions() == 1
    assert task.cancelled()

async def test_worker_drain_timeout(monkeypatch):
    from huddleroom.workers import task_runner
    gate = asyncio.Event()
    async def resistant():
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            await gate.wait()
    task = asyncio.create_task(resistant())
    await asyncio.sleep(0)
    task_runner.get_running_tasks()["resistant"] = task
    monkeypatch.setattr(task_runner, "PROJECT_CANCELLATION_TIMEOUT_SECONDS", .01)
    with pytest.raises(TimeoutError):
        await task_runner.cancel_all_sessions()
    gate.set()
    await task
    task_runner.get_running_tasks().pop("resistant", None)


def test_controller_restart_requires_completed_shutdown():
    from huddleroom.services.backend_restart import RestartController, RESTART_EXIT_CODE
    controller = RestartController(SimpleNamespace(started=True, lifespan=SimpleNamespace(shutdown_failed=False)))
    controller.requested = True
    assert controller.exit_code() == 1
    controller.shutdown_complete = True
    assert controller.exit_code() == RESTART_EXIT_CODE
    controller.server.lifespan.shutdown_failed = True
    assert controller.exit_code() == 1


@pytest.mark.parametrize("codes,expected,spawns", [([75,0],0,2), ([1],1,1), ([-15],1,1)])
def test_supervisor_only_restarts_successful_requested_shutdown(monkeypatch, codes, expected, spawns):
    from huddleroom.services import backend_restart
    commands = []
    class Child:
        def __init__(self, command, **kwargs):
            commands.append(command)
        def wait(self):
            return codes.pop(0)
    monkeypatch.setattr(backend_restart.subprocess, "Popen", Child)
    assert backend_restart.supervise("127.0.0.1", 8123) == expected
    assert len(commands) == spawns
    assert commands[0] == [backend_restart.sys.executable, "-m", "huddleroom.cli", "serve", "--host", "127.0.0.1", "--port", "8123", "--restart-child"]


def test_config_file_and_environment_precedence(tmp_path, monkeypatch):
    import huddleroom.config as config
    path = tmp_path / "config.toml"
    path.write_text("allow_api_reset = true\n")
    monkeypatch.setattr(config, "DEFAULT_CONFIG_FILE", path)
    assert Settings(_env_file=None).allow_api_reset
    monkeypatch.setenv("HUDDLEROOM_ALLOW_API_RESET", "false")
    assert not Settings(_env_file=None).allow_api_reset
    assert Settings(allow_api_reset=True, _env_file=None).allow_api_reset


async def test_draining_refuses_new_registered_work():
    from huddleroom.services import backend_restart
    from huddleroom.workers.task_runner import register_session
    backend_restart.draining = True
    try:
        with pytest.raises(Exception, match="restarting"):
            register_session("unused", "api", "unused")
    finally:
        backend_restart.draining = False

async def test_ack_is_sent_before_shutdown(monkeypatch):
    from huddleroom.services.backend_restart import RestartController
    monkeypatch.setattr(settings, "allow_api_reset", True)
    app = create_app()
    server = SimpleNamespace(should_exit=False)
    app.state.backend_restart = RestartController(server)
    sent = []
    async def wrapped(scope, receive, send):
        async def record(message):
            if message["type"] in {"http.response.start", "http.response.body"}:
                assert not server.should_exit
                sent.append(message["type"])
            await send(message)
        await app(scope, receive, record)
    async with AsyncClient(transport=ASGITransport(app=wrapped), base_url="http://test") as client:
        assert (await client.post("/api/system/restart", headers={"X-HuddleRoom-Confirm": "restart"})).status_code == 202
    assert sent[-1] == "http.response.body"
    assert server.should_exit


@pytest.mark.parametrize("headers,expected", [
    ([("X-HuddleRoom-Confirm", "restart"), ("Origin", "null")],403),
    ([("X-HuddleRoom-Confirm", "restart"), ("Origin", "http://test"), ("Origin", "http://evil")],403),
    ([("X-HuddleRoom-Confirm", "restart"), ("X-HuddleRoom-Confirm", "restart")],400),
    ([("X-HuddleRoom-Confirm", "restart"), ("Origin", "http://test:80")],202),
])
async def test_origin_and_confirmation_headers(monkeypatch, headers, expected):
    from huddleroom.services.backend_restart import RestartController
    monkeypatch.setattr(settings, "allow_api_reset", True)
    app = create_app()
    server = SimpleNamespace(should_exit=False)
    app.state.backend_restart = RestartController(server)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/api/system/restart", headers=headers)).status_code == expected
    assert server.should_exit == (expected == 202)


async def test_websocket_admission_refused_while_draining():
    from huddleroom.services import backend_restart
    backend_restart.draining = True
    messages = []
    async def unreachable(*args):
        pytest.fail("Websocket admitted during restart")
    async def send(message):
        messages.append(message)
    await backend_restart.DrainMiddleware(unreachable)({"type":"websocket"}, None, send)
    assert messages == [{"type":"websocket.close", "code":1013}]


def test_supervisor_signal_racing_spawn_prevents_respawn(monkeypatch):
    from huddleroom.services import backend_restart
    handlers = {}
    signals = []
    def install(sig, handler):
        previous = handlers.get(sig)
        handlers[sig] = handler
        return previous
    class Child:
        def __init__(self, command, **kwargs):
            handlers[backend_restart.signal.SIGTERM](backend_restart.signal.SIGTERM, None)
        def poll(self):
            return None
        def send_signal(self, sig):
            signals.append(sig)
        def wait(self):
            return 75
    monkeypatch.setattr(backend_restart.signal, "signal", install)
    monkeypatch.setattr(backend_restart.subprocess, "Popen", Child)
    assert backend_restart.supervise("localhost",8000) == 75
    assert signals == [backend_restart.signal.SIGTERM]


async def test_worker_cleanup_failure_blocks_restart(monkeypatch):
    from huddleroom.services import backend_restart
    from huddleroom.workers import task_runner
    async def worker(*args):
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            raise RuntimeError("cleanup failed")
    monkeypatch.setattr(task_runner, "execute_api_session", worker)
    task_runner.register_session("unused","api","unused")
    await asyncio.sleep(0)
    backend_restart.draining = True
    with pytest.raises(RuntimeError, match="shutdown failed"):
        await task_runner.cancel_all_sessions()


def test_deprecated_reset_env_alias(monkeypatch):
    monkeypatch.setenv("RALLY_ALLOW_API_RESET", "true")
    assert Settings(_env_file=None).allow_api_reset
    monkeypatch.setenv("HUDDLEROOM_ALLOW_API_RESET", "false")
    assert not Settings(_env_file=None).allow_api_reset


async def test_scheduler_drain_waits_for_active_async_job():
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from huddleroom.services.backend_restart import drain_scheduler
    entered, cleaned = asyncio.Event(), asyncio.Event()
    async def job():
        entered.set()
        try:
            await asyncio.sleep(100)
        finally:
            await asyncio.sleep(.01)
            cleaned.set()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(job)
    scheduler.start()
    await asyncio.wait_for(entered.wait(),1)
    await drain_scheduler(scheduler)
    assert cleaned.is_set()
    scheduler.shutdown()


def test_producer_failure_never_authorizes_restart():
    from huddleroom.services.backend_restart import RestartController
    def fail():
        raise RuntimeError("producer failure")
    controller = RestartController(SimpleNamespace(should_exit=False, lifespan=SimpleNamespace(shutdown_failed=False)), fail)
    controller.requested = True
    controller.stop()
    controller.shutdown_complete = True
    assert controller.server.should_exit
    assert controller.exit_code() == 1


async def test_restart_cleanup_is_not_cancelled_twice():
    from huddleroom.services.backend_restart import stop_tasks
    entered, cleaned = asyncio.Event(), asyncio.Event()
    async def worker():
        try:
            await asyncio.sleep(100)
        finally:
            entered.set()
            await asyncio.sleep(.01)
            cleaned.set()
    task = asyncio.create_task(worker())
    await asyncio.sleep(0)
    task.cancel()
    await entered.wait()
    await stop_tasks([task], restarting=True)
    assert cleaned.is_set()

async def test_scheduler_cleanup_failure_blocks_restart():
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from huddleroom.services.backend_restart import drain_scheduler
    entered = asyncio.Event()
    async def job():
        entered.set()
        try:
            await asyncio.sleep(100)
        finally:
            raise RuntimeError("scheduler cleanup failed")
    scheduler = AsyncIOScheduler()
    scheduler.add_job(job)
    scheduler.start()
    await asyncio.wait_for(entered.wait(),1)
    try:
        with pytest.raises(RuntimeError, match="Scheduler job shutdown failed"):
            await drain_scheduler(scheduler)
    finally:
        scheduler.shutdown()


async def test_orphaned_supervised_child_refuses_shutdown(monkeypatch):
    from huddleroom.services.backend_restart import RestartController
    monkeypatch.setattr(settings, "allow_api_reset", True)
    app = create_app()
    server = SimpleNamespace(should_exit=False)
    app.state.backend_restart = RestartController(server, parent_pid=-1)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/system/restart", headers={"X-HuddleRoom-Confirm":"restart"})
    assert response.status_code == 503
    assert not server.should_exit


def test_supervisor_spawn_failure_does_not_retry(monkeypatch):
    from huddleroom.services import backend_restart
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise OSError("spawn failed")
    monkeypatch.setattr(backend_restart.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="spawn failed"):
        backend_restart.supervise("localhost",8000)
    assert len(calls) == 1


@pytest.mark.parametrize("marker", [None,"-1"])
def test_manual_restart_child_requires_parent_marker(monkeypatch, marker):
    from click.testing import CliRunner
    from huddleroom.cli import main
    if marker is None:
        monkeypatch.delenv("_HUDDLEROOM_RESTART_PARENT_PID", raising=False)
    else:
        monkeypatch.setenv("_HUDDLEROOM_RESTART_PARENT_PID", marker)
    result = CliRunner().invoke(main,["serve","--restart-child"])
    assert result.exit_code != 0
    assert "requires its supervising parent" in result.output


def test_supervised_child_can_disable_reset_in_new_config(monkeypatch):
    import os
    import huddleroom.cli as cli
    import uvicorn
    from click.testing import CliRunner
    monkeypatch.setenv("_HUDDLEROOM_RESTART_PARENT_PID",str(os.getppid()))
    monkeypatch.setattr(settings,"allow_api_reset",False)
    monkeypatch.setattr(cli,"_prepare_database",lambda value: value)
    calls=[]
    monkeypatch.setattr(uvicorn,"run",lambda *args,**kwargs: calls.append(args))
    result=CliRunner().invoke(cli.main,["serve","--restart-child"])
    assert result.exit_code == 0, result.output
    assert calls == [("huddleroom.main:app",)]


def test_swagger_exposes_confirmation_header():
    operation = create_app().openapi()["paths"]["/api/system/restart"]["post"]
    assert "202" in operation["responses"]
    header = next(item for item in operation["parameters"] if item["name"] == "X-HuddleRoom-Confirm")
    assert header["in"] == "header"
    assert "restart" in header["schema"]["examples"]


@pytest.mark.parametrize("failure", [None, "relay", "sessions"])
async def test_real_lifespan_authorizes_restart_only_after_all_cleanup(monkeypatch, failure):
    """Exercise the actual lifecycle glue without opening a database or provider."""
    from unittest.mock import AsyncMock
    from huddleroom import main
    from huddleroom.services import agent_response_relay, backend_restart
    from huddleroom.workers import scheduler, orchestration_tasks, task_runner
    from huddleroom.workers import consumers
    calls = []
    consumer_tasks = {}

    class Db:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        def begin(self):
            return self
        async def execute(self, *args, **kwargs):
            pass
        async def commit(self):
            pass
        async def rollback(self):
            pass

    async def idle(*args, **kwargs):
        await asyncio.Event().wait()
    async def event_supervisor(*args, ready, **kwargs):
        ready.set()
        await idle()
    async def response_hub(ready):
        ready.set()
        await idle()
    async def close_relay():
        calls.append("relay")
        if failure == "relay":
            raise RuntimeError("relay cleanup failed")
    async def cancel_sessions():
        calls.append("sessions")
        assert not controller.shutdown_complete
        if failure == "sessions":
            raise RuntimeError("sessions cleanup failed")

    monkeypatch.setattr(main, "init_db", AsyncMock())
    monkeypatch.setattr(main, "register_litellm_debug_logger", lambda: None)
    monkeypatch.setattr("sqlalchemy.ext.asyncio.AsyncSession", Db)
    monkeypatch.setattr("huddleroom.database.engine", SimpleNamespace(dialect=SimpleNamespace(name="sqlite")))
    monkeypatch.setattr("huddleroom.services.orchestration_service.OrchestrationService.pause_existing_goals_for_startup", AsyncMock())
    monkeypatch.setattr(main.orchestration_goals.conversation_service, "recover_all", AsyncMock())
    monkeypatch.setattr("huddleroom.services.session_service.SessionService.recover_orphaned_sessions", AsyncMock(return_value=0))
    monkeypatch.setattr("huddleroom.services.graph_service.GraphService.load_all_from_workspace", AsyncMock())
    monkeypatch.setattr("huddleroom.services.escalation_service.EscalationChainService.load_all_from_workspace", AsyncMock())
    monkeypatch.setattr(scheduler, "start_scheduler", lambda **kwargs: None)
    monkeypatch.setattr(scheduler, "bootstrap_orchestration_async", idle)
    monkeypatch.setattr(orchestration_tasks, "run_orchestration_event_supervisor", event_supervisor)
    monkeypatch.setattr(agent_response_relay, "run_agent_response_hub", response_hub)
    monkeypatch.setattr(agent_response_relay, "subscribe_agent_responses", lambda callback: lambda: calls.append("unsubscribe"))
    monkeypatch.setattr(agent_response_relay, "close_agent_response_relay", close_relay)
    monkeypatch.setattr(consumers, "get_consumer_tasks", lambda: consumer_tasks)
    monkeypatch.setattr(task_runner, "cancel_all_sessions", cancel_sessions)
    for module_name, function in (
        ("ws_hub", "run_ws_hub"), ("rule_engine", "run_rule_engine"),
        ("graph_engine", "run_graph_engine"), ("meeting_engine", "run_meeting_engine"),
        ("optimizer", "run_optimizer"),
    ):
        monkeypatch.setattr(f"huddleroom.workers.consumers.{module_name}.{function}", idle)
    app = create_app()
    controller = backend_restart.RestartController(SimpleNamespace(lifespan=SimpleNamespace(shutdown_failed=False)))
    app.state.backend_restart = controller

    async def exercise():
        async with main.lifespan(app):
            assert not controller.shutdown_complete
            controller.requested = True
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} cleanup failed"):
            await exercise()
        assert not controller.shutdown_complete
        assert controller.exit_code() == 1
    else:
        await exercise()
        assert controller.shutdown_complete
        assert controller.exit_code() == backend_restart.RESTART_EXIT_CODE
    assert calls == ["unsubscribe", "relay", "sessions"]


@pytest.mark.parametrize("reload", [False, True])
def test_enabled_cli_selects_supervisor_only_without_reload(monkeypatch, reload):
    import huddleroom.cli as cli
    import uvicorn
    from click.testing import CliRunner
    from huddleroom.services import backend_restart
    monkeypatch.setattr(settings, "allow_api_reset", True)
    monkeypatch.setattr(cli, "_prepare_database", lambda value: value)
    calls = []
    monkeypatch.setattr(backend_restart, "supervise", lambda host, port: calls.append("supervisor") or 0)
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append("reload"))
    result = CliRunner().invoke(cli.main, ["serve"] + (["--reload"] if reload else []))
    assert result.exit_code == 0, result.output
    assert calls == (["reload"] if reload else ["supervisor"])


def test_genuine_supervised_child_installs_controller(monkeypatch):
    import os
    import huddleroom.cli as cli
    import uvicorn
    from click.testing import CliRunner
    monkeypatch.setenv("_HUDDLEROOM_RESTART_PARENT_PID", str(os.getppid()))
    monkeypatch.setattr(settings, "allow_api_reset", True)
    monkeypatch.setattr(cli, "_prepare_database", lambda value: value)
    applications = []
    class Server:
        def __init__(self, config):
            applications.append(config.app)
            self.started = True
        def run(self):
            assert applications[-1].state.backend_restart.server is self
    monkeypatch.setattr(uvicorn, "Server", Server)
    result = CliRunner().invoke(cli.main, ["serve", "--restart-child"])
    assert result.exit_code == 0, result.output
    assert len(applications) == 1
    assert applications[0].state.backend_restart.parent_pid == os.getppid()


@pytest.fixture
def isolated_runner(monkeypatch):
    """Snapshot/restore task_runner module registries around a test."""
    from huddleroom.workers import meeting_tasks, task_runner
    for name in ("_running_tasks", "_task_projects", "_deferred_session_dispatches"):
        monkeypatch.setattr(task_runner, name, {})
    monkeypatch.setattr(meeting_tasks, "_running_meeting_tasks", {})
    return task_runner


async def test_stale_cleanup_keeps_replacement_real(isolated_runner, monkeypatch):
    tr = isolated_runner
    gates = [asyncio.Event(), asyncio.Event()]
    started = iter(gates)

    async def blocker(session_id, task_id):
        await next(started).wait()

    monkeypatch.setattr(tr, "execute_api_session", blocker)
    tr.register_session("s", "api", "p", task_id="T")
    old = tr._running_tasks["T"]
    await asyncio.sleep(0)
    tr.register_session("s", "api", "p", task_id="T")  # replacement while old is live
    new = tr._running_tasks["T"]
    assert new is not old
    await asyncio.sleep(0)
    gates[0].set()
    await old
    # Old worker's cleanup must not evict the replacement.
    assert tr._running_tasks.get("T") is new and tr._task_projects.get("T") == "p"
    gates[1].set()
    await new
    assert "T" not in tr._running_tasks


async def test_cancel_all_identity_keeps_replacement(isolated_runner):
    tr = isolated_runner
    replacement = asyncio.ensure_future(asyncio.sleep(10))

    async def stubborn():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            tr._running_tasks["T"] = replacement  # replaced during cancellation cleanup

    old = asyncio.ensure_future(stubborn())
    tr._running_tasks["T"] = old
    tr._task_projects["T"] = "p"
    await asyncio.sleep(0)
    try:
        assert await tr.cancel_all_sessions() == 1
        assert tr._running_tasks.get("T") is replacement
    finally:
        replacement.cancel()
        await asyncio.gather(replacement, return_exceptions=True)


async def test_cancel_project_identity_keeps_replacement(isolated_runner):
    tr = isolated_runner
    replacement = asyncio.ensure_future(asyncio.sleep(10))

    async def stubborn():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            tr._running_tasks["T"] = replacement

    old = asyncio.ensure_future(stubborn())
    tr._running_tasks["T"] = old
    tr._task_projects["T"] = "p"
    await asyncio.sleep(0)
    try:
        assert await tr.cancel_project_sessions("p") == 1
        assert tr._running_tasks.get("T") is replacement
    finally:
        replacement.cancel()
        await asyncio.gather(replacement, return_exceptions=True)


@pytest.fixture
def registered(isolated_runner, monkeypatch):
    calls = []
    monkeypatch.setattr(isolated_runner, "register_session", lambda *a, **k: calls.append((a, k)))
    return calls


async def test_publish_draining_drops(isolated_runner, registered, monkeypatch):
    from huddleroom.services import backend_restart
    tr = isolated_runner
    monkeypatch.setattr(backend_restart, "draining", True)  # restored by monkeypatch
    tr.publish_session_dispatch("s", "api", "p", "T")  # no worker
    worker = asyncio.ensure_future(asyncio.sleep(0.01))
    tr._running_tasks["T"] = worker
    tr.publish_session_dispatch("s", "api", "p", "T")  # live worker
    assert registered == [] and not tr._deferred_session_dispatches
    await worker
    await asyncio.sleep(0)
    assert registered == []


async def test_deferred_drain_after_publish_drops(isolated_runner, registered, monkeypatch):
    from huddleroom.services import backend_restart
    tr = isolated_runner
    worker = asyncio.ensure_future(asyncio.sleep(0.02))
    tr._running_tasks["T"] = worker
    tr.publish_session_dispatch("s", "api", "p", "T")
    assert len(tr._deferred_session_dispatches) == 1
    monkeypatch.setattr(backend_restart, "draining", True)
    await worker
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert registered == [] and not tr._deferred_session_dispatches


async def test_deferred_cancelled_worker_drops(isolated_runner, registered):
    tr = isolated_runner
    worker = asyncio.ensure_future(asyncio.sleep(10))
    tr._running_tasks["T"] = worker
    await asyncio.sleep(0)
    tr.publish_session_dispatch("s", "api", "p", "T")
    assert len(tr._deferred_session_dispatches) == 1
    worker.cancel()
    await asyncio.gather(worker, return_exceptions=True)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert registered == [] and not tr._deferred_session_dispatches
