"""Resume litellm/CLI calls from failure point.

Targeted coverage for the resumable-failure mechanics that shipped without
dedicated backend tests: API/CLI session resume (atomic claim + collision
guard), CLI provider_session_id capture, meeting-turn parking/resume, and
secret redaction landing in the persisted error fields.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.agent import Agent
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task

FAKE_API_KEY = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"


# ---------------------------------------------------------------------------
# API session (litellm) failure -> resumable + redacted error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_session_failure_sets_resumable_and_redacts_secret_in_event(
    db_session: AsyncSession, test_project, test_agent
):
    from huddleroom.adapters.api_adapter import ApiAdapter

    session = Session(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="running",
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()

    events: list[tuple[str, dict]] = []

    async def fake_emit_event(db, project_id, event_type, payload, *args, **kwargs):
        events.append((event_type, payload))

    async def boom(**kwargs):
        raise RuntimeError(f"provider rejected request authorization={FAKE_API_KEY}")

    with patch("huddleroom.adapters.api_adapter.run_tool_loop", new=boom):
        with patch("huddleroom.adapters.api_adapter.emit_event", new=fake_emit_event):
            with patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
                await ApiAdapter()._run_with_retry(
                    test_agent, session, [{"role": "user", "content": "hi"}], "", db_session
                )

    assert session.status == "failed"
    assert session.resumable is True
    assert FAKE_API_KEY not in (session.error or "")
    assert "[REDACTED]" in (session.error or "")

    failed_events = [payload for etype, payload in events if etype == "session.failed"]
    assert len(failed_events) == 1
    assert failed_events[0]["resumable"] is True


@pytest.mark.asyncio
async def test_api_budget_stop_persists_cumulative_authoritative_usage(
    db_session: AsyncSession, test_project, test_agent
):
    """Every provider call is charged, including calls made by a tool loop."""
    import litellm
    from huddleroom.adapters.api_adapter import ApiAdapter

    session = Session(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="running",
        metadata_={"_run_config": {"max_tokens": 150, "_roadmap_budget_enforced": True}},
    )
    db_session.add(session)
    await db_session.flush()

    async def two_calls(*, completion_fn, **_kwargs):
        await completion_fn(messages=[{"role": "user", "content": "first"}])
        await completion_fn(messages=[{"role": "user", "content": "second"}])

    async def completion(**_kwargs):
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=40, completion_tokens=40))

    with patch("huddleroom.adapters.api_adapter.run_tool_loop", new=two_calls), \
         patch.object(litellm, "acompletion", new=completion), \
         patch.object(litellm, "token_counter", return_value=40), \
         patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()), \
         patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
        await ApiAdapter()._run_with_retry(
            test_agent, session, [{"role": "user", "content": "hi"}], "", db_session
        )

    assert session.status == "failed"
    assert session.metadata_["token_count_in"] == 80
    assert session.metadata_["token_count_out"] == 80


@pytest.mark.asyncio
async def test_api_resume_keeps_prior_usage_but_enforces_only_new_grant(
    db_session: AsyncSession, test_project, test_agent
):
    """A resumed attempt records cumulative usage without shrinking its new grant."""
    import litellm
    from huddleroom.adapters.api_adapter import ApiAdapter

    session = Session(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="running",
        metadata_={
            "token_count_in": 20,
            "token_count_out": 30,
            "_run_config": {
                "max_tokens": 100,
                "_roadmap_budget_enforced": True,
                "_roadmap_prior_usage": {"max_tokens": "50", "max_turns": "1", "max_hours": "0"},
            },
        },
    )
    db_session.add(session)
    await db_session.flush()

    async def completion(**_kwargs):
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=40, completion_tokens=50))

    async def one_call(*, completion_fn, **_kwargs):
        await completion_fn(messages=[{"role": "user", "content": "resume"}])
        return "done"

    with patch("huddleroom.adapters.api_adapter.run_tool_loop", new=one_call), \
         patch.object(litellm, "acompletion", new=completion), \
         patch.object(litellm, "token_counter", return_value=10), \
         patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()), \
         patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
        await ApiAdapter()._run_with_retry(
            test_agent, session, [{"role": "user", "content": "hi"}], "", db_session
        )

    assert session.status == "completed"
    assert session.metadata_["token_count_in"] == 60
    assert session.metadata_["token_count_out"] == 80


@pytest.mark.asyncio
async def test_api_usage_counts_distinct_responses_when_mocked_ids_are_reused(
    db_session: AsyncSession, test_project, test_agent
):
    """A provider response is deduped by object identity, not its recyclable id."""
    import litellm
    from huddleroom.adapters.api_adapter import ApiAdapter

    session = Session(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="running",
        metadata_={"_run_config": {"max_tokens": 100, "_roadmap_budget_enforced": True}},
    )
    db_session.add(session)
    await db_session.flush()

    responses = (
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10)),
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10)),
    )
    provider_responses = iter(responses)

    async def two_calls(*, completion_fn, response_observer, **_kwargs):
        for response in responses:
            completed = await completion_fn(messages=[{"role": "user", "content": "hi"}])
            response_observer(completed)
        return "done"

    async def completion(**_kwargs):
        return next(provider_responses)

    with patch("huddleroom.adapters.api_adapter.run_tool_loop", new=two_calls), \
         patch.object(litellm, "acompletion", new=completion), \
         patch.object(litellm, "token_counter", return_value=1), \
         patch("huddleroom.adapters.api_adapter.id", new=lambda _response: 1, create=True), \
         patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()), \
         patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
        await ApiAdapter()._run_with_retry(
            test_agent, session, [{"role": "user", "content": "hi"}], "", db_session
        )

    assert session.status == "completed"
    assert session.metadata_["token_count_in"] == 20
    assert session.metadata_["token_count_out"] == 20


@pytest.mark.asyncio
async def test_api_budget_rejects_prompt_exhaustion_before_provider_call(
    db_session: AsyncSession, test_project, test_agent
):
    import litellm
    from huddleroom.adapters.api_adapter import ApiAdapter

    session = Session(agent_id=test_agent.id, project_id=test_project.id, adapter_type="api", status="running",
                      metadata_={"_run_config": {"max_tokens": 40, "_roadmap_budget_enforced": True}})
    db_session.add(session); await db_session.flush()
    provider = AsyncMock()
    with patch.object(litellm, "acompletion", new=provider), \
         patch.object(litellm, "token_counter", return_value=40), \
         patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()), \
         patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
        await ApiAdapter()._run_with_retry(test_agent, session, [{"role": "user", "content": "hi"}], "", db_session)
    provider.assert_not_awaited()
    assert session.metadata_["token_count_in"] == 0


@pytest.mark.asyncio
async def test_api_marks_missing_provider_usage_incomplete(
    db_session: AsyncSession, test_project, test_agent
):
    import litellm
    from huddleroom.adapters.api_adapter import ApiAdapter

    session = Session(agent_id=test_agent.id, project_id=test_project.id, adapter_type="api", status="running",
                      metadata_={"_run_config": {"max_tokens": 100, "_roadmap_budget_enforced": True}})
    db_session.add(session); await db_session.flush()

    async def completion(**_kwargs):
        return SimpleNamespace(usage=None)

    with patch.object(litellm, "acompletion", new=completion), \
         patch.object(litellm, "token_counter", return_value=1), \
         patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()), \
         patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
        await ApiAdapter()._run_with_retry(test_agent, session, [{"role": "user", "content": "hi"}], "", db_session)
    assert session.metadata_["token_usage_complete"] is False


@pytest.mark.asyncio
async def test_capped_api_provider_failure_marks_usage_incomplete(
    db_session: AsyncSession, test_project, test_agent
):
    """A failed capped provider call has no authoritative usage to settle."""
    import litellm
    from huddleroom.adapters.api_adapter import ApiAdapter

    session = Session(agent_id=test_agent.id, project_id=test_project.id, adapter_type="api", status="running",
                      metadata_={"_run_config": {"max_tokens": 100, "timeout": 1,
                                                 "_roadmap_budget_enforced": True}})
    db_session.add(session); await db_session.flush()

    async def unavailable(**_kwargs):
        raise litellm.ServiceUnavailableError("unavailable", "openai", "gpt-4o-mini")

    with patch("huddleroom.adapters.api_adapter.run_tool_loop", new=unavailable), \
         patch("huddleroom.adapters.api_adapter.asyncio.sleep", new=AsyncMock()), \
         patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()), \
         patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
        await ApiAdapter()._run_with_retry(test_agent, session, [{"role": "user", "content": "hi"}], "", db_session)

    assert session.status == "failed"
    assert session.metadata_["token_usage_complete"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("run_config", [
    {"timeout": 1, "_roadmap_budget_enforced": True},
    {"_roadmap_budget_enforced": True},
])
async def test_non_token_roadmap_caps_do_not_require_token_telemetry(
    db_session: AsyncSession, test_project, test_agent, run_config,
):
    import litellm
    from huddleroom.adapters.api_adapter import ApiAdapter

    session = Session(agent_id=test_agent.id, project_id=test_project.id, adapter_type="api", status="running",
                      metadata_={"_run_config": run_config})
    db_session.add(session); await db_session.flush()

    async def unavailable(**_kwargs):
        raise litellm.ServiceUnavailableError("unavailable", "openai", "gpt-4o-mini")

    with patch("huddleroom.adapters.api_adapter.run_tool_loop", new=unavailable), \
         patch("huddleroom.adapters.api_adapter.asyncio.sleep", new=AsyncMock()), \
         patch("huddleroom.adapters.api_adapter.emit_event", new=AsyncMock()), \
         patch("huddleroom.services.session_sync.sync_task_from_session", new=AsyncMock()):
        await ApiAdapter()._run_with_retry(test_agent, session, [{"role": "user", "content": "hi"}], "", db_session)

    assert "token_usage_complete" not in session.metadata_


# ---------------------------------------------------------------------------
# Session resume endpoint: atomic claim + task-collision guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_resume_atomic_claim_redispatches_and_second_call_409s(
    db_session: AsyncSession, test_project, test_agent
):
    from fastapi import HTTPException
    from huddleroom.services.session_service import SessionService

    session = Session(
        agent_id=test_agent.id,
        project_id=test_project.id,
        adapter_type="api",
        status="failed",
        error="rate_limit_exceeded",
        resumable=True,
        metadata_={},
    )
    db_session.add(session)
    await db_session.flush()
    old_runner_task_id = session.runner_task_id

    svc = SessionService()
    resumed = await svc.resume(db_session, session.id)

    assert resumed.status == "pending"
    assert resumed.resumable is False
    assert resumed.runner_task_id != old_runner_task_id
    # A fresh dispatch was queued for after-commit delivery (mock-healthy re-drive).
    pending = db_session.sync_session.info.get("pending_session_dispatches", [])
    assert any(item[0] == session.id for item in pending)

    # Second concurrent resume on the same (now-pending) session must 409 --
    # the atomic UPDATE finds 0 rows because status is no longer "failed".
    with pytest.raises(HTTPException) as exc_info:
        await svc.resume(db_session, session.id)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_session_resume_409_when_task_already_has_active_session(
    db_session: AsyncSession, test_project, test_agent
):
    from fastapi import HTTPException
    from huddleroom.services.session_service import SessionService

    task = Task(project_id=test_project.id, title="shared task")
    db_session.add(task)
    await db_session.flush()

    active_session = Session(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="pending",
        metadata_={},
    )
    failed_session = Session(
        agent_id=test_agent.id,
        project_id=test_project.id,
        task_id=task.id,
        adapter_type="api",
        status="failed",
        error="api_connection_error",
        resumable=True,
        metadata_={},
    )
    db_session.add_all([active_session])
    await db_session.flush()
    db_session.add(failed_session)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await SessionService().resume(db_session, failed_session.id)
    assert exc_info.value.status_code == 409
    assert "active session" in str(exc_info.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["max_tokens", "timeout"])
async def test_session_create_rejects_non_positive_requested_limits(
    db_session: AsyncSession, test_project, test_agent, field
):
    from fastapi import HTTPException
    from huddleroom.schemas.session import SessionCreate
    from huddleroom.services.session_service import SessionService

    kwargs = {field: 0}
    with pytest.raises(HTTPException, match="positive whole number"):
        await SessionService().create(db_session, SessionCreate(
            agent_id=test_agent.id, project_id=test_project.id, **kwargs,
        ))


@pytest.mark.asyncio
async def test_session_create_rejects_invalid_agent_limits_before_persisting(
    db_session: AsyncSession, test_project, test_agent
):
    from fastapi import HTTPException
    from huddleroom.schemas.session import SessionCreate
    from huddleroom.services.session_service import SessionService

    test_agent.config = {"max_tokens": "NaN", "session_timeout_seconds": 0}
    with pytest.raises(HTTPException, match="positive whole number"):
        await SessionService().create(db_session, SessionCreate(
            agent_id=test_agent.id, project_id=test_project.id,
        ))


# ---------------------------------------------------------------------------
# CLI task session: provider_session_id capture + resumable + resume argv
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _cli_task_fixture(test_engine, tmp_path, *, provider_session_id: str | None = None):
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db:
        project = Project(name=f"Project-{uuid.uuid4()}", workspace_path=str(tmp_path), config={})
        agent = Agent(
            name=f"agent-{uuid.uuid4()}",
            role="developer",
            provider="anthropic",
            model="claude",
            adapter_type="cli",
            capabilities=[],
            config={},
            cli_runtime="claude_code",
        )
        db.add_all([project, agent])
        await db.flush()
        session = Session(
            agent_id=agent.id,
            project_id=project.id,
            adapter_type="cli",
            status="pending",
            metadata_={},
            provider_session_id=provider_session_id,
        )
        db.add(session)
        await db.commit()
        yield db, session, agent, project


@pytest.mark.asyncio
async def test_cli_task_captures_provider_session_id_on_success(test_engine, tmp_path):
    from huddleroom.adapters.cli_adapter import CliAdapter

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b'{"result": "done", "session_id": "sess-success-1"}', b""

    async with _cli_task_fixture(test_engine, tmp_path) as (db, session, _agent, _project):
        adapter = CliAdapter()
        with patch.object(adapter, "_setup_sandbox", return_value=None):
            with patch.object(adapter, "_build_env", return_value={}):
                with patch.object(adapter, "_build_command", return_value=["claude", "--print"]):
                    with patch(
                        "huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                        new=AsyncMock(return_value=FakeProc()),
                    ):
                        await adapter.run(session.id, db)

        await db.refresh(session)
        assert session.status == "completed"
        assert session.provider_session_id == "sess-success-1"


@pytest.mark.asyncio
async def test_cli_task_nonzero_exit_captures_session_id_marks_resumable_and_redacts(test_engine, tmp_path):
    from huddleroom.adapters.cli_adapter import CliAdapter
    from huddleroom.models.event_log import EventLog

    class FakeProc:
        returncode = 1

        async def communicate(self):
            stdout = b'{"result": "partial", "session_id": "sess-fail-1"}'
            stderr = f"fatal: authorization={FAKE_API_KEY}".encode()
            return stdout, stderr

    async with _cli_task_fixture(test_engine, tmp_path) as (db, session, _agent, project):
        adapter = CliAdapter()
        with patch.object(adapter, "_setup_sandbox", return_value=None):
            with patch.object(adapter, "_build_env", return_value={}):
                with patch.object(adapter, "_build_command", return_value=["claude", "--print"]):
                    with patch(
                        "huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec",
                        new=AsyncMock(return_value=FakeProc()),
                    ):
                        await adapter.run(session.id, db)

        await db.refresh(session)
        assert session.status == "failed"
        assert session.provider_session_id == "sess-fail-1"
        assert session.resumable is True
        assert FAKE_API_KEY not in (session.error or "")
        assert "[REDACTED]" in (session.error or "")

        events = (
            await db.execute(
                select(EventLog).where(
                    EventLog.project_id == project.id, EventLog.event_type == "session.failed"
                )
            )
        ).scalars().all()
        assert len(events) == 1
        assert events[0].payload["resumable"] is True


@pytest.mark.asyncio
async def test_cli_task_resume_launch_uses_resume_flag_and_continue_prompt(test_engine, tmp_path):
    from huddleroom.adapters.cli_adapter import CliAdapter

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b'{"result": "resumed ok"}', b""

    captured: dict = {}

    async def fake_subprocess_exec(*args, **kwargs):
        captured["argv"] = args
        return FakeProc()

    async with _cli_task_fixture(
        test_engine, tmp_path, provider_session_id="abcDEF1234567890"
    ) as (db, session, _agent, _project):
        adapter = CliAdapter()
        with patch.object(adapter, "_setup_sandbox", return_value=None):
            with patch(
                "huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=fake_subprocess_exec
            ):
                await adapter.run(session.id, db)

    argv = list(captured["argv"])
    assert "--resume" in argv
    assert argv[argv.index("--resume") + 1] == "abcDEF1234567890"
    assert "continue" in argv


@pytest.mark.asyncio
async def test_onecli_resume_reuses_safe_child_environment(test_engine, tmp_path, monkeypatch):
    """Resuming cannot restore extras that bypass the selected OneCLI wrapper."""
    import huddleroom.adapters.cli_adapter as cli_adapter
    from huddleroom.adapters.cli_adapter import CliAdapter
    from huddleroom.config import Settings

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b'{"result": "resumed ok"}', b""

    captured: dict = {}

    async def fake_subprocess_exec(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return FakeProc()

    proxy = "http://agent:gateway-token@gateway.example:10255"
    for key, value in {
        "ONECLI_GATEWAY": "true",
        "HUDDLEROOM_ONECLI_AGENT": "gateway",
        "HUDDLEROOM_ONECLI_GATEWAY_URL": "http://gateway.example:10255",
        "HUDDLEROOM_ONECLI_MANAGEMENT_URL": "http://management.example:10256",
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "NO_PROXY": "api.openai.com,localhost",
        "SSL_CERT_FILE": "/trusted/gateway-ca.pem",
        "ONECLI_API_KEY": "management-secret",
    }.items():
        monkeypatch.setenv(key, value)
    for key in ("http_proxy", "https_proxy", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        cli_adapter,
        "settings",
        Settings(
            _env_file=None, credential_mode="onecli", onecli_agent="gateway",
            onecli_management_url="http://management.example:10256",
            onecli_gateway_url="http://gateway.example:10255",
        ),
    )

    async with _cli_task_fixture(test_engine, tmp_path, provider_session_id="abcDEF1234567890") as (db, session, agent, _project):
        agent.config = {
            "cli_env_extras": {
                "http_proxy": "http://attacker.invalid:8080",
                "OPENAI_API_KEY": "direct-secret",
                "ONECLI_API_KEY": "extra-management-secret",
            }
        }
        adapter = CliAdapter()
        with patch.object(adapter, "_setup_sandbox", return_value=None):
            with patch("huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=fake_subprocess_exec):
                await adapter.run(session.id, db)

    assert captured["env"]["http_proxy"] == proxy
    assert captured["env"]["OPENAI_API_KEY"] == "onecli-openai-placeholder"
    assert "ONECLI_API_KEY" not in captured["env"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "resume_args", "stdout"),
    [
        (
            "copilot",
            ("--resume=abcDEF1234567890",),
            b'{"type":"assistant.message","data":{"content":"resumed"}}\n'
            b'{"type":"result","sessionId":"new-session"}\n',
        ),
        (
            "opencode",
            ("--session", "abcDEF1234567890"),
            b'{"type":"text","sessionID":"new-session","part":{"type":"text","text":"resumed"}}\n',
        ),
        (
            "pi",
            ("--session", "abcDEF1234567890"),
            b'{"type":"session","id":"new-session"}\n'
            b'{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"resumed"}]}}\n',
        ),
    ],
)
async def test_step8_cli_task_resume_reuses_runtime_session(
    test_engine, tmp_path, runtime, resume_args, stdout,
):
    from huddleroom.adapters.cli_adapter import CliAdapter

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return stdout, b""

    captured: dict = {}

    async def fake_subprocess_exec(*args, **kwargs):
        captured["argv"] = args
        return FakeProc()

    async with _cli_task_fixture(
        test_engine, tmp_path, provider_session_id="abcDEF1234567890"
    ) as (db, session, agent, _project):
        agent.config = {"cli_runtime": runtime}
        with patch.object(CliAdapter(), "_setup_sandbox", return_value=None):
            with patch(
                "huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=fake_subprocess_exec
            ):
                await CliAdapter().run(session.id, db)
        await db.refresh(session)

    argv = list(captured["argv"])
    assert tuple(argv[argv.index(resume_args[0]):argv.index(resume_args[0]) + len(resume_args)]) == resume_args
    assert "continue" in argv
    assert session.output == "resumed"
    assert session.provider_session_id == "new-session"


@pytest.mark.asyncio
async def test_cli_resume_keeps_override_continuation_after_agent_model_change(test_engine, tmp_path):
    from huddleroom.adapters.cli_adapter import CliAdapter

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b'{"result": "resumed"}', b""

    captured: dict = {}

    async def fake_subprocess_exec(*args, **kwargs):
        captured["argv"] = args
        return FakeProc()

    async with _cli_task_fixture(
        test_engine, tmp_path, provider_session_id="abcDEF1234567890"
    ) as (db, session, agent, _project):
        session.metadata_ = {
            "_run_config": {"model_override": "fixed-override"},
            "_launch_config": {
                "provider": "anthropic", "model": "fixed-override", "cli_runtime": "claude_code",
            },
        }
        agent.model = "edited-agent-model"
        await db.commit()

        with patch.object(CliAdapter(), "_setup_sandbox", return_value=None):
            with patch(
                "huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=fake_subprocess_exec
            ):
                await CliAdapter().run(session.id, db)

        await db.refresh(session)

    argv = list(captured["argv"])
    assert argv[argv.index("--model") + 1] == "fixed-override"
    assert argv[argv.index("--resume") + 1] == "abcDEF1234567890"
    assert session.metadata_["_launch_config"]["model"] == "fixed-override"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["provider", "model", "runtime"])
async def test_cli_resume_with_changed_launch_config_starts_fresh(test_engine, tmp_path, change):
    from huddleroom.adapters.cli_adapter import CliAdapter

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return b'{"result": "fresh"}', b""

    captured: dict = {}

    async def fake_subprocess_exec(*args, **kwargs):
        captured["argv"] = args
        return FakeProc()

    async with _cli_task_fixture(
        test_engine, tmp_path, provider_session_id="abcDEF1234567890"
    ) as (db, session, agent, _project):
        session.metadata_ = {
            "_run_config": {"timeout": 5},
            "_launch_config": {"provider": "anthropic", "model": "claude", "cli_runtime": "claude_code"},
            "keep": "value",
        }
        if change == "provider":
            agent.provider = "openai"
        elif change == "model":
            agent.model = "changed-model"
        else:
            agent.cli_runtime = "codex"
        await db.commit()

        with patch.object(CliAdapter(), "_setup_sandbox", return_value=None):
            with patch(
                "huddleroom.adapters.cli_adapter.asyncio.create_subprocess_exec", new=fake_subprocess_exec
            ):
                await CliAdapter().run(session.id, db)

        await db.refresh(session)

    argv = list(captured["argv"])
    assert "--resume" not in argv
    assert argv[:2] == (["codex", "exec"] if change == "runtime" else ["claude", "--dangerously-skip-permissions"])
    assert argv[argv.index("--model") + 1] == ("changed-model" if change == "model" else "claude")
    assert session.provider_session_id is None
    assert session.metadata_["_run_config"]["timeout"] == 5
    assert session.metadata_["keep"] == "value"
    assert session.metadata_["model_used"] == ("changed-model" if change == "model" else "claude")


# ---------------------------------------------------------------------------
# Meeting turn failure: park, no phantom turn, halt loop; resume clears state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_turn_failure_parks_meeting_no_phantom_turn_and_halts_loop(
    db_session: AsyncSession, test_project, test_agent
):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingTurn
    from huddleroom.workers.meeting_tasks import run_meeting_turn_async

    meeting = Meeting(
        project_id=test_project.id,
        title="Parks on API failure",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(
        meeting_id=meeting.id, order=1, title="Q1", question="Proceed?", status="active",
    )
    db_session.add(item)
    await db_session.flush()

    async def boom(**kwargs):
        raise RuntimeError(f"provider unavailable authorization={FAKE_API_KEY}")

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("huddleroom.services.meeting_context.MeetingContextService.build_turn_prompt", new=AsyncMock(
            return_value=[{"role": "user", "content": "go"}]
        )):
            with patch("huddleroom.services.meeting_runner.run_tool_loop", new=boom):
                with patch(
                    "huddleroom.services.meeting_runner.MeetingRunner.evaluate_round_if_complete",
                    new_callable=AsyncMock,
                ) as mock_eval:
                    with patch("huddleroom.workers.meeting_tasks.dispatch_run_meeting_turn") as mock_dispatch:
                        await run_meeting_turn_async(str(meeting.id))

    mock_eval.assert_not_called()
    mock_dispatch.assert_not_called()

    await db_session.refresh(meeting)
    assert meeting.resume_state.get("failed") is True
    assert meeting.resume_state.get("adapter") == "api"
    assert FAKE_API_KEY not in (meeting.resume_state.get("error") or "")
    assert "[REDACTED]" in (meeting.resume_state.get("error") or "")

    turns = (await db_session.execute(select(MeetingTurn))).scalars().all()
    assert turns == []


@pytest.mark.asyncio
async def test_run_meeting_turn_async_noops_while_parked(
    db_session: AsyncSession, test_project, test_agent
):
    """The turn loop must not dispatch/run while a failure is parked."""
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem
    from huddleroom.workers.meeting_tasks import run_meeting_turn_async

    meeting = Meeting(
        project_id=test_project.id,
        title="Already parked",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()
    item = MeetingAgendaItem(meeting_id=meeting.id, order=1, title="Q1", question="?", status="active")
    db_session.add(item)
    await db_session.flush()

    meeting.resume_state = {
        "failed": True, "speaker_agent_id": str(test_agent.id), "adapter": "api", "error": "boom",
    }
    await db_session.flush()

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "huddleroom.services.meeting_runner.MeetingRunner.run_next_turn", new_callable=AsyncMock
        ) as mock_run_next:
            with patch("huddleroom.workers.meeting_tasks.dispatch_run_meeting_turn") as mock_dispatch:
                await run_meeting_turn_async(str(meeting.id))

    mock_run_next.assert_not_called()
    mock_dispatch.assert_not_called()


@pytest.mark.asyncio
async def test_resume_meeting_turn_async_double_dispatch_claim_blocks_concurrent_resume(
    db_session: AsyncSession, test_project, test_agent
):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem
    from huddleroom.workers.meeting_tasks import resume_meeting_turn_async

    meeting = Meeting(
        project_id=test_project.id,
        title="Double resume guard",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()
    item = MeetingAgendaItem(meeting_id=meeting.id, order=1, title="Q1", question="?", status="active")
    db_session.add(item)
    await db_session.flush()

    meeting.resume_state = {
        "failed": True,
        "speaker_agent_id": str(test_agent.id),
        "agenda_item_id": str(item.id),
        "adapter": "api",
        "error": "boom",
        "resuming": True,  # a resume is already in flight
    }
    await db_session.flush()

    db_session.commit = AsyncMock()
    db_session.rollback = AsyncMock()

    with patch("huddleroom.workers.meeting_tasks.AsyncSessionLocal") as mock_session_factory:
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=db_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "huddleroom.services.meeting_runner.MeetingRunner.resume_failed_turn", new_callable=AsyncMock
        ) as mock_resume:
            await resume_meeting_turn_async(str(meeting.id))

    mock_resume.assert_not_called()


@pytest.mark.asyncio
async def test_resume_failed_cli_turn_uses_resume_flag_and_grounded_turn_prompt(
    db_session: AsyncSession, test_project, test_agent
):
    from huddleroom.models.meeting import Meeting, MeetingAgendaItem
    from huddleroom.services.meeting_runner import MeetingRunner

    test_agent.adapter_type = "cli"
    test_agent.cli_runtime = "claude_code"
    await db_session.flush()

    meeting = Meeting(
        project_id=test_project.id,
        title="Resume CLI turn",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    item = MeetingAgendaItem(meeting_id=meeting.id, order=1, title="Q1", question="?", status="active")
    db_session.add(item)
    await db_session.flush()

    meeting.resume_state = {
        "failed": True,
        "speaker_agent_id": str(test_agent.id),
        "agenda_item_id": str(item.id),
        "adapter": "cli",
        "error": "boom",
        "cli_session_id": "cli-sess-1",
    }
    await db_session.flush()

    with patch(
        "huddleroom.adapters.cli_adapter.CliAdapter.run_meeting_turn",
        new_callable=AsyncMock,
        return_value=("resumed content", "cli-sess-1", 10),
    ) as run_mock:
        turn = await MeetingRunner(bus=None).resume_failed_turn(db=db_session, meeting=meeting)

    assert turn is not None
    prompt = run_mock.await_args.kwargs["prompt_text"]
    assert "Continue your current turn" in prompt
    assert "Never invent" in prompt
    assert "POSITION:" in prompt
    assert run_mock.await_args.kwargs["existing_session_id"] == "cli-sess-1"
    assert meeting.resume_state == {}


@pytest.mark.asyncio
async def test_meeting_resume_endpoint_requires_parked_state_and_dispatches_when_parked(
    client, auth_headers, db_session: AsyncSession, test_project, test_agent
):
    from huddleroom.models.meeting import Meeting

    meeting = Meeting(
        project_id=test_project.id,
        title="Resume endpoint",
        meeting_type="decision",
        participant_agent_ids=[str(test_agent.id)],
        status="active",
    )
    db_session.add(meeting)
    await db_session.flush()

    resp = await client.post(f"/api/v1/meetings/{meeting.id}/resume", headers=auth_headers)
    assert resp.status_code == 409

    meeting.resume_state = {
        "failed": True, "speaker_agent_id": str(test_agent.id), "adapter": "api", "error": "boom",
    }
    await db_session.flush()

    with patch("huddleroom.workers.meeting_tasks.dispatch_resume_meeting_turn") as mock_dispatch:
        resp = await client.post(f"/api/v1/meetings/{meeting.id}/resume", headers=auth_headers)

    assert resp.status_code == 200
    mock_dispatch.assert_called_once()
