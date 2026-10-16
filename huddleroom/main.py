import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from huddleroom.config import settings, validate_supported_settings

validate_supported_settings(settings)

from huddleroom.database import init_db
from huddleroom.schemas.common import ErrorResponse
from huddleroom.services.llm_debug_logging import register_litellm_debug_logger
from huddleroom.routers import (
    agent_self_service,
    agents,
    artifacts,
    auth,
    channels,
    escalations,
    events,
    hooks,
    knowledge,
    meetings,
    memory,
    messages,
    optimizations,
    orchestration,
    orchestration_advisor,
    orchestration_chat,
    orchestration_decisions,
    orchestration_goals,
    orchestration_memory,
    orchestration_processes,
    orchestration_warnings,
    projects,
    graphs,
    rules,
    sessions,
    tasks,
    users,
    websocket as ws_router,
)
_logger = logging.getLogger(__name__)
_RESPONSE_HUB_RESTART_DELAY = 1


@asynccontextmanager
async def lifespan(_app: FastAPI):
    app_state = getattr(_app, "state", None)
    from huddleroom.services import backend_restart
    backend_restart.draining = False
    validate_supported_settings(settings)
    import litellm
    litellm.drop_params = True
    register_litellm_debug_logger()
    await init_db()

    # Commit the restart pause before recovery or background workers can admit work.
    # Fail startup if this fence cannot be persisted.
    from sqlalchemy.ext.asyncio import AsyncSession
    from huddleroom.database import engine
    from huddleroom.services.orchestration_service import OrchestrationService
    async with AsyncSession(engine) as startup_db:
        async with startup_db.begin():
            await OrchestrationService().pause_existing_goals_for_startup(startup_db)

    try:
        await orchestration_goals.conversation_service.recover_all()
    except Exception as _exc:
        _logger.warning("Conversation recovery failed (non-fatal): %s", _exc)

    # Seed anon user for auth-disabled mode (idempotent)
    if not settings.auth_enabled:
        try:
            import uuid as _uuid_mod
            import sqlalchemy as _sa
            from sqlalchemy.ext.asyncio import AsyncSession as _AS
            from huddleroom.database import engine as _eng
            async with _AS(_eng) as _seed_s:
                async with _seed_s.begin():
                    _anon_id = _uuid_mod.UUID("00000000-0000-0000-0000-000000000000")
                    if _eng.dialect.name == "postgresql":
                        await _seed_s.execute(_sa.text(
                            "INSERT INTO users (id, email, hashed_password, display_name, role, is_active, created_at, updated_at) "
                            "VALUES (:id, 'anon@local', '$2b$12$roncFJiobp.SJUvZ5w/AI.d2sBUhQaQV/5C1Vjqsm7IqCPVIPQakS', 'Anonymous', 'admin', true, NOW(), NOW()) "
                            "ON CONFLICT (id) DO NOTHING"
                        ).bindparams(_sa.bindparam("id", value=_anon_id, type_=_sa.Uuid())))
                    else:
                        await _seed_s.execute(_sa.text(
                            "INSERT OR IGNORE INTO users (id, email, hashed_password, display_name, role, is_active, created_at, updated_at) "
                            "VALUES (:id, 'anon@local', '$2b$12$roncFJiobp.SJUvZ5w/AI.d2sBUhQaQV/5C1Vjqsm7IqCPVIPQakS', 'Anonymous', 'admin', 1, datetime('now'), datetime('now'))"
                        ).bindparams(_sa.bindparam("id", value=_anon_id, type_=_sa.Uuid())))
        except Exception as _exc:
            if "no such table" in str(_exc).lower() or "does not exist" in str(_exc).lower():
                _logger.debug("Anon user seed skipped — users table not yet created: %s", _exc)
            else:
                _logger.error("Anon user seed FAILED — FK errors likely in auth-disabled mode: %s", _exc)
                raise

    try:
        from sqlalchemy.ext.asyncio import AsyncSession
        from huddleroom.database import engine
        async with AsyncSession(engine) as _s:
            from sqlalchemy import text
            await _s.execute(text("SELECT 1 FROM agents LIMIT 1"))
            try:
                from pathlib import Path
                from huddleroom.services.escalation_service import EscalationChainService
                from huddleroom.services.graph_service import GraphService

                workspace_path = Path(settings.workspace_path)
                await GraphService().load_all_from_workspace(_s, workspace_path)
                await EscalationChainService().load_all_from_workspace(_s, workspace_path)
                await _s.commit()
            except Exception as _exc:
                await _s.rollback()
                _logger.warning("Workspace YAML load failed (non-fatal): %s", _exc)
    except Exception as _exc:
        _logger.warning("Database startup check failed after migrations: %s", _exc)

    # Postgres deployments use Celery which manages its own task lifecycle;
    # orphan recovery is only needed in SQLite/asyncio mode.
    if settings.is_sqlite:
        try:
            from sqlalchemy.ext.asyncio import AsyncSession as _AsyncSession
            from huddleroom.database import engine as _engine
            from huddleroom.services.session_service import SessionService as _SS
            async with _AsyncSession(_engine) as _s:
                async with _s.begin():
                    _count = await _SS().recover_orphaned_sessions(_s)
                    if _count:
                        _logger.warning("Recovered %d orphaned session(s) from previous process.", _count)
        except Exception as _exc:
            _logger.warning("Orphan session recovery failed (non-fatal): %s", _exc)

    if settings.jwt_secret == "change-me-in-production":
        _logger.warning("JWT_SECRET is set to the default insecure value. Set HUDDLEROOM_JWT_SECRET in your environment.")

    if settings.cors_origins == ["*"]:
        _logger.warning("CORS is open to all origins. Set HUDDLEROOM_CORS_ORIGINS for production.")

    if not settings.auth_enabled:
        _logger.warning("Authentication is DISABLED. Set HUDDLEROOM_AUTH_ENABLED=true for production.")

    scheduler = None
    bootstrap = None
    subscription_ready = asyncio.Event()
    runtime_ready = asyncio.Event()
    execution_ready = asyncio.Event()
    if settings.is_sqlite:
        from huddleroom.workers.scheduler import bootstrap_orchestration_async, start_scheduler
        scheduler = start_scheduler(defer_orchestration_startup=True)
        scheduler_ready_at = datetime.now(timezone.utc)
        bootstrap = asyncio.create_task(
            bootstrap_orchestration_async(
                scheduler, scheduler_ready_at, runtime_ready, execution_ready=execution_ready,
            ),
            name="startup:orchestration",
        )

    supervision_events = None
    if settings.is_sqlite:
        from huddleroom.workers.orchestration_tasks import run_orchestration_event_supervisor
        supervision_events = asyncio.create_task(
            run_orchestration_event_supervisor(
                ready=subscription_ready, execution_ready=execution_ready,
            ),
            name="consumer:orchestration_supervision",
        )

    from huddleroom.services.agent_response_relay import (
        close_agent_response_relay,
        run_agent_response_hub,
        subscribe_agent_responses,
    )
    from huddleroom.workers.consumers.ws_hub import get_registry
    unsubscribe_agent_responses = subscribe_agent_responses(get_registry().broadcast_response)
    response_hub_ready = asyncio.Event()

    async def supervise_response_hub() -> None:
        while True:
            try:
                await run_agent_response_hub(response_hub_ready)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not response_hub_ready.is_set():
                    raise RuntimeError(str(exc)) from exc
                _logger.exception("Agent response hub crashed; restarting")
            else:
                if not response_hub_ready.is_set():
                    raise RuntimeError("Agent response hub exited before subscribing")
                _logger.error("Agent response hub exited; restarting")
            await asyncio.sleep(_RESPONSE_HUB_RESTART_DELAY)

    response_hub = asyncio.create_task(supervise_response_hub(), name="consumer:agent_response_hub")
    ready_wait = asyncio.create_task(response_hub_ready.wait(), name="startup:response_hub_ready")
    subscription_wait = (
        asyncio.create_task(subscription_ready.wait(), name="startup:subscription_ready")
        if settings.is_sqlite else None
    )
    try:
        pending = {response_hub, ready_wait}
        if supervision_events is not None:
            pending.add(supervision_events)
        if subscription_wait is not None:
            pending.add(subscription_wait)
        while not ready_wait.done() or (subscription_wait is not None and not subscription_wait.done()):
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                pending.discard(task)
                if task is response_hub or task is supervision_events:
                    await task
                    raise RuntimeError(f"Required startup consumer {task.get_name()} exited before readiness")

        # Required consumers must still be running when the readiness barriers release.
        for task in (response_hub, supervision_events):
            if task is not None and task.done():
                await task
                raise RuntimeError(f"Required startup consumer {task.get_name()} exited before readiness")

        # Start consumer asyncio tasks (SQLite/in-process mode)
        if settings.is_sqlite:
            from huddleroom.workers.consumers import get_consumer_tasks
            from huddleroom.workers.consumers.ws_hub import run_ws_hub
            from huddleroom.workers.consumers.rule_engine import run_rule_engine
            from huddleroom.workers.consumers.graph_engine import run_graph_engine
            from huddleroom.workers.consumers.meeting_engine import run_meeting_engine
            from huddleroom.workers.consumers.optimizer import run_optimizer

            consumer_fns = {
                "ws_hub": run_ws_hub,
                "rule_engine": run_rule_engine,
                "graph_engine": run_graph_engine,
                "meeting_engine": run_meeting_engine,
                "optimizer": run_optimizer,
            }
            consumer_tasks = get_consumer_tasks()

            def _make_done_callback(cname, cfn):
                def _on_done(t: asyncio.Task) -> None:
                    from huddleroom.services.backend_restart import draining
                    if t.cancelled() or draining:
                        return
                    exc = t.exception()
                    if exc:
                        _logger.error("Consumer '%s' crashed: %s — restarting", cname, exc)
                        new_task = asyncio.create_task(cfn(), name=f"consumer:{cname}")
                        new_task.add_done_callback(_make_done_callback(cname, cfn))
                        consumer_tasks[cname] = new_task
                return _on_done

            for name, fn in consumer_fns.items():
                task = asyncio.create_task(fn(), name=f"consumer:{name}")
                task.add_done_callback(_make_done_callback(name, fn))
                consumer_tasks[name] = task
            _logger.info("Started %d consumer tasks", len(consumer_tasks))

        runtime_ready.set()
        def stop_restart_producers():
            if scheduler and scheduler.running:
                scheduler.pause()
            for task in (bootstrap, supervision_events):
                if task is not None and not task.cancelling():
                    task.cancel()
            if settings.is_sqlite:
                from huddleroom.workers.consumers import get_consumer_tasks
                for task in get_consumer_tasks().values():
                    if not task.cancelling():
                        task.cancel()
        if app_state is not None:
            app_state.stop_restart_producers = stop_restart_producers
        yield
    finally:
        from huddleroom.services.backend_restart import stop_tasks
        controller = getattr(app_state, "backend_restart", None)
        restarting = bool(controller and controller.requested)
        try:
            if scheduler:
                from huddleroom.workers.scheduler import stop_scheduler

                if restarting:
                    from huddleroom.services.backend_restart import drain_scheduler
                    await drain_scheduler(scheduler)
                stop_scheduler()
        finally:
            try:
                startup_tasks = [task for task in (
                    bootstrap, ready_wait, subscription_wait, supervision_events,
                ) if task is not None]
                await stop_tasks(startup_tasks, restarting=restarting)
            finally:
                try:
                    unsubscribe_agent_responses()
                finally:
                    try:
                        await stop_tasks([response_hub], restarting=restarting)
                    finally:
                        try:
                            if restarting:
                                await asyncio.wait_for(close_agent_response_relay(), timeout=5)
                            else:
                                await close_agent_response_relay()
                        finally:
                            if settings.is_sqlite:
                                from huddleroom.workers.consumers import get_consumer_tasks

                                consumer_tasks = get_consumer_tasks()
                                await stop_tasks(consumer_tasks.values(), restarting=restarting)
                                consumer_tasks.clear()
                                controller = getattr(app_state, "backend_restart", None)
                                if controller and controller.requested:
                                    from huddleroom.workers.task_runner import cancel_all_sessions
                                    await cancel_all_sessions()
        controller = getattr(app_state, "backend_restart", None)
        if controller and controller.requested:
            controller.shutdown_complete = True


def create_app() -> FastAPI:
    app = FastAPI(title="HuddleRoom", lifespan=lifespan)

    from huddleroom.services.database_export import SnapshotResponse, export_database
    app.add_api_route("/api/system/database/export", export_database, methods=["POST"], response_class=SnapshotResponse,
                      responses={200: {"content": {"application/vnd.sqlite3": {"schema": {"type": "string", "format": "binary"}}}}})

    from huddleroom.services.backend_restart import request_restart
    app.add_api_route("/api/system/restart", request_restart, methods=["POST"], status_code=202)
    from huddleroom.services.backend_restart import DrainMiddleware
    app.add_middleware(DrainMiddleware)

    @app.middleware("http")
    async def restart_drain(request, call_next):
        controller = getattr(app.state, "backend_restart", None)
        if controller and controller.requested and request.url.path != "/api/system/restart":
            return JSONResponse({"detail": "Backend is restarting"}, status_code=503)
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException):
        if not isinstance(exc.detail, str):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)
        detail = exc.detail
        payload = ErrorResponse(error="http_error", detail=detail).model_dump(exclude_none=True)
        return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
        payload = ErrorResponse(
            error="validation_error",
            detail="Validation failed",
            errors=exc.errors(),
        ).model_dump(exclude_none=True)
        status_code = 400 if request.url.path.endswith(
            "/decisions/agent-definition-review/batch-answer"
        ) else 422
        return JSONResponse(status_code=status_code, content=payload)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        _logger.exception("Unhandled API error on %s %s", request.method, request.url.path, exc_info=exc)
        payload = ErrorResponse(
            error="internal_error",
            detail="Internal server error",
        ).model_dump(exclude_none=True)
        return JSONResponse(status_code=500, content=payload)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # SPA dashboard + dev dashboard
    from starlette.responses import FileResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    _ASSET_EXTENSIONS = {
        ".js", ".css", ".map", ".png", ".jpg", ".jpeg", ".gif", ".svg",
        ".ico", ".woff", ".woff2", ".ttf", ".eot", ".json", ".webp",
    }

    class SPAStaticFiles(StaticFiles):
        async def get_response(self, path, scope):
            try:
                response = await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code != 404:
                    raise
                path_str = scope.get("path", "")
                _, _, last_segment = path_str.rpartition("/")
                _, ext = os.path.splitext(last_segment)
                if ext.lower() in _ASSET_EXTENSIONS:
                    raise
                index = os.path.join(str(self.directory), "index.html")
                if not os.path.isfile(index):
                    raise
                response = FileResponse(index, media_type="text/html")

            # Set no-cache on HTML — modify the response object before it sends headers
            ct = (getattr(response, "media_type", None) or "").lower()
            if "html" in ct:
                response.headers["cache-control"] = "no-cache, no-store, must-revalidate"
                response.headers["pragma"] = "no-cache"

            return response

    static_base = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_base, exist_ok=True)

    dashboard_dir = os.path.join(static_base, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    app.mount("/dashboard", SPAStaticFiles(directory=dashboard_dir, html=True), name="dashboard")

    dev_dashboard_dir = os.path.join(static_base, "dev-dashboard")
    if os.path.isdir(dev_dashboard_dir):
        app.mount("/dev-dashboard", StaticFiles(directory=dev_dashboard_dir, html=True), name="dev-dashboard")

    app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
    app.include_router(projects.router, prefix="/api/v1/projects", tags=["projects"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
    app.include_router(
        tasks.router,
        prefix="/api/v1/projects/{project_id}/tasks",
        tags=["tasks"],
    )
    app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])
    app.include_router(users.router, prefix="/api/v1", tags=["users"])
    app.include_router(knowledge.router, prefix="/api/v1", tags=["knowledge"])
    app.include_router(channels.router, prefix="/api/v1", tags=["channels"])
    app.include_router(messages.router, prefix="/api/v1", tags=["messages"])
    app.include_router(agent_self_service.router, prefix="/api/v1/agent", tags=["agent-self-service"])
    app.include_router(events.router, prefix="/api/v1/events", tags=["events"])
    app.include_router(orchestration.router, prefix="/api/v1/orchestration", tags=["orchestration"])
    app.include_router(
        orchestration_goals.router,
        prefix="/api/v1/projects/{project_id}/orchestration",
        tags=["orchestration"],
    )
    app.include_router(orchestration_chat.router, prefix="/api/v1/projects/{project_id}/orchestration", tags=["orchestration"])
    app.include_router(
        orchestration_advisor.router,
        prefix="/api/v1/projects/{project_id}/orchestration",
        tags=["orchestration"],
    )
    app.include_router(
        orchestration_memory.router,
        prefix="/api/v1/projects/{project_id}/orchestration",
        tags=["orchestration"],
    )
    app.include_router(
        orchestration_processes.router,
        prefix="/api/v1/projects/{project_id}/orchestration",
        tags=["orchestration"],
    )
    app.include_router(
        orchestration_decisions.router,
        prefix="/api/v1/projects/{project_id}/orchestration",
        tags=["orchestration"],
    )
    app.include_router(
        orchestration_warnings.router,
        prefix="/api/v1/projects/{project_id}/orchestration",
        tags=["orchestration"],
    )
    app.include_router(graphs.router, prefix="/api/v1", tags=["graphs"])
    app.include_router(artifacts.router, prefix="/api/v1", tags=["artifacts"])
    app.include_router(escalations.router, prefix="/api/v1", tags=["escalations"])
    app.include_router(meetings.router, prefix="/api/v1", tags=["meetings"])
    app.include_router(memory.router, prefix="/api/v1", tags=["memory"])
    app.include_router(rules.router, prefix="/api/v1", tags=["rules"])
    app.include_router(hooks.router, prefix="/api/v1", tags=["hooks"])
    app.include_router(optimizations.router, prefix="/api/v1", tags=["optimizations"])
    app.include_router(ws_router.router, tags=["websocket"])

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/v1/config")
    async def public_config():
        return {"auth_enabled": settings.auth_enabled}

    return app


app = create_app()
