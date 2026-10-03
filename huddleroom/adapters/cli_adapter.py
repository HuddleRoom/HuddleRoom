import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import time
import uuid
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from fastapi import HTTPException

from huddleroom.models.session import Session
from huddleroom.models.agent import Agent
from huddleroom.models.task import Task
from huddleroom.models.project import Project
from huddleroom.config import settings
from huddleroom.services.event_bus import emit_event
from huddleroom.services.llm_debug_logging import log_cli_exchange
from huddleroom.services.llm_structured_repair import cli_complete_with_repair
from huddleroom.services.project_service import ProjectService
from huddleroom.services.secret_redaction import redact_secrets
from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext
from huddleroom.services.cli_streaming import collect_cli_process

logger = logging.getLogger(__name__)


class CliTurnFailed(RuntimeError):
    """Raised when a CLI meeting turn fails. Optionally carries the provider session_id for resume."""
    def __init__(self, message: str, session_id: str | None = None):
        super().__init__(redact_secrets(message))
        self.session_id = session_id


class CliAdapter:
    _LEGACY_CONTEXT_FILENAME = "rally_context.json"
    _CONTEXT_FILENAME = "huddleroom_context.json"

    @staticmethod
    def _sandbox_dir(workspace: Path, *parts: str) -> Path:
        """Use the new sandbox root, retaining an existing one-release legacy sandbox."""
        legacy = workspace / ".rally" / Path(*parts)
        return legacy if legacy.exists() else workspace / ".huddleroom" / Path(*parts)

    @classmethod
    def _context_path(cls, sandbox_dir: Path) -> Path:
        filename = cls._LEGACY_CONTEXT_FILENAME if ".rally" in sandbox_dir.parts else cls._CONTEXT_FILENAME
        return sandbox_dir / filename

    @staticmethod
    def _claude_usage(stdout: str) -> tuple[int, int] | None:
        """Read the provider's terminal usage envelope; absence is explicitly incomplete."""
        for line in reversed(stdout.splitlines()):
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or record.get("type") != "result":
                    continue
                usage = record.get("usage", {})
                prompt, completion = usage.get("input_tokens"), usage.get("output_tokens")
                if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                       for value in (prompt, completion)):
                    return prompt, completion
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _store_roadmap_elapsed(session: Session) -> None:
        """Persist cumulative attempt time and turns for an enforced CLI roadmap budget."""
        config = (session.metadata_ or {}).get("_run_config", {})
        if not config.get("_roadmap_budget_enforced"):
            return
        prior = config.get("_roadmap_prior_usage", {})
        prior_seconds = Decimal(str(prior.get("max_hours", "0"))) * Decimal("3600")
        elapsed = Decimal("0")
        if session.started_at is not None and session.ended_at is not None:
            elapsed = max(Decimal("0"), Decimal(str((session.ended_at - session.started_at).total_seconds())))
        session.metadata_ = {
            **(session.metadata_ or {}),
            "_roadmap_elapsed_seconds": format((prior_seconds + elapsed).normalize(), "f"),
            "_roadmap_turn_count": int(Decimal(str(prior.get("max_turns", "0")))) + 1,
        }

    @staticmethod
    async def _fail_workspace_validation(
        db: AsyncSession, session: Session, exc: HTTPException, runner_task_id: str | None = None,
    ) -> bool:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        error = (
            f"project_not_runnable: {detail['reason']}"
            if detail.get("code") == "project_not_runnable" else str(exc.detail)
        )
        if isinstance((session.metadata_ or {}).get("attempt"), dict):
            if not runner_task_id:
                return False
            from huddleroom.workers.session_tasks import mark_attempt_project_not_runnable
            if not await mark_attempt_project_not_runnable(db, session.id, runner_task_id, error):
                return False
        session.status = "failed"
        session.error = error
        session.resumable = False
        session.ended_at = datetime.now(timezone.utc)
        CliAdapter._store_roadmap_elapsed(session)
        await db.flush()
        await emit_event(db, session.project_id, "session.failed", {
            "session_id": str(session.id), "task_id": str(session.task_id) if session.task_id else None,
            "error": session.error, "project_id": str(session.project_id), "resumable": False,
        })
        from huddleroom.services.session_sync import sync_task_from_session
        await sync_task_from_session(db, session)
        return True

    @staticmethod
    async def _terminate_process_group(proc, grace_seconds: float = 0.5) -> None:
        """Stop a CLI and every child it started, escalating after a bounded grace period."""
        process_group_id = proc.pid  # start_new_session=True makes the launch PID the process-group ID.

        async def group_exited() -> bool:
            deadline = asyncio.get_running_loop().time() + grace_seconds
            while True:
                try:
                    os.killpg(process_group_id, 0)
                except ProcessLookupError:
                    return True
                if asyncio.get_running_loop().time() >= deadline:
                    return False
                await asyncio.sleep(min(0.05, grace_seconds))

        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            if not await group_exited():
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if not await group_exited():
                    logger.warning("CLI process group %s did not exit after SIGKILL", process_group_id)

        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
            except asyncio.TimeoutError:
                logger.warning("CLI group leader %s did not reap after termination", proc.pid)

    @staticmethod
    async def _watch_project_status(
        db: AsyncSession, project_id: uuid.UUID, interval: float = 0.1,
    ) -> str:
        """Return the first non-active project status while a CLI is running."""
        async with async_sessionmaker(db.bind, expire_on_commit=False)() as watcher_db:
            while True:
                status = await watcher_db.scalar(select(Project.status).where(Project.id == project_id))
                if status != "active":
                    return status or "deleted"
                await watcher_db.rollback()
                await asyncio.sleep(interval)

    @staticmethod
    def _build_cli_request_display(task) -> dict | None:
        """Build source-owned request display for CLI invocation: task title/description only."""
        if not task:
            return None
        # ponytail: display only title + description, never identity/system prompt/context
        return {"title": task.title, "description": task.description or ""}

    @staticmethod
    def _build_meeting_cli_request_display(agenda_title: str, question: str) -> dict | None:
        """Build source-owned request display for meeting CLI turn: agenda title + question only."""
        # ponytail: display only agenda title + question, never transcript/context/system prompt
        if not agenda_title:
            return None
        return {"agenda": agenda_title, "question": question or ""}

    @staticmethod
    def _compose_task_content(task, agent, session_id: uuid.UUID) -> str:
        if not task:
            return ""
        identity = f"You are {agent.name}, the {agent.role}."
        if agent.system_prompt:
            identity = f"{identity}\n{agent.system_prompt}"

        task_content = f"{identity}\n\n# Task: {task.title}\n\n"
        if task.description:
            task_content += f"{task.description}\n\n"
        task_content += f"Task ID: {task.id}\nProject ID: {task.project_id}\n"
        return task_content

    @staticmethod
    def _build_command(
        cli_runtime: str, task_content: str, agent_config: dict, workspace: Path, task_path: Path,
        *, context_path: Path | None = None, existing_session_id: str | None = None, model: str | None = None,
    ) -> list[str] | None:
        if cli_runtime == "claude_code":
            cmd = ["claude", "--dangerously-skip-permissions", "--print", "--verbose"]
            if model:
                cmd += ["--model", model]
            if context_path:
                cmd += ["--output-format", "stream-json"]
            else:
                cmd += [task_content or "proceed", "--output-format", "stream-json"]
            if existing_session_id:
                if re.fullmatch(r"[a-zA-Z0-9_-]{8,128}", existing_session_id):
                    cmd += ["--resume", existing_session_id]
                else:
                    logger.warning("Ignoring invalid CLI session ID: %r", existing_session_id)
            return cmd + (["--file", str(context_path)] if context_path else [])
        elif cli_runtime == "codex":
            return [
                "codex", "exec", "--sandbox", "workspace-write", "--cd", str(workspace),
                "--ask-for-approval", "never", *( ["--model", model] if model else []), task_content or "proceed",
            ]
        elif cli_runtime == "aider":
            return ["aider", "--yes", "--no-pretty", *( ["--model", model] if model else []), "--message", task_content or "proceed"]
        elif cli_runtime == "copilot":
            cmd = [
                "copilot", "-p", task_content or "proceed", "-C", str(workspace), "--no-ask-user",
                "--allow-all-tools", "--output-format", "json", "--no-auto-update",
            ]
            if model:
                cmd += ["--model", model]
            if existing_session_id:
                cmd += [f"--resume={existing_session_id}"]
            return cmd
        elif cli_runtime == "opencode":
            cmd = ["opencode", "run", "--format", "json", "--auto", "--dir", str(workspace)]
            if model:
                cmd += ["--model", model]
            if existing_session_id:
                cmd += ["--session", existing_session_id]
            return cmd + [task_content or "proceed"]
        elif cli_runtime == "pi":
            cmd = ["pi", "--print", "--mode", "json", "--no-approve"]
            if model:
                cmd += ["--model", model]
            if existing_session_id:
                cmd += ["--session", existing_session_id]
            return cmd + [task_content or "proceed"]
        elif cli_runtime == "custom":
            script_path = agent_config.get("script_path")
            if not script_path:
                return None
            return [script_path, str(task_path)]
        raise RuntimeError(f"unsupported_cli_runtime: {cli_runtime}")

    @classmethod
    def launch_fingerprint(cls, session: Session, task, agent, project, workspace: Path) -> str:
        """Stable record of every persisted input to a CLI subprocess launch."""
        run_config = (session.metadata_ or {}).get("_run_config", {})
        runtime = agent.config.get("cli_runtime", agent.cli_runtime or "claude_code")
        model = run_config.get("model_override") or agent.model
        content = cls._compose_task_content(task, agent, session.id)
        try:
            command = cls._build_command(runtime, content, agent.config, workspace, Path("task.md"), model=model)
        except RuntimeError as exc:
            # Fingerprinting must not bypass the normal unsupported-runtime failure path.
            command = [str(exc)]
        launch = {
            "provider": agent.provider,
            "workspace": str(workspace),
            "cwd": str(workspace),
            "command": command,
            "timeout": run_config.get("timeout") or agent.config.get("session_timeout_seconds", 3600),
            "environment": hashlib.sha256(json.dumps(
                cls()._build_env(session.id, task, agent, project), sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(),
        }
        return hashlib.sha256(json.dumps(launch, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _setup_sandbox(self, sandbox_dir: Path, session_id: uuid.UUID, task, agent, project_id,
                       task_content: str = "") -> None:
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        if not task_content:
            task_content = self._compose_task_content(task, agent, session_id)
        (sandbox_dir / "task.md").write_text(task_content or "# No task context\n")
        self._context_path(sandbox_dir).write_text(json.dumps({
            "session_id": str(session_id),
            "task_id": str(task.id) if task else None,
            "agent_id": str(agent.id),
            "project_id": str(project_id),
            "api_base": settings.api_base_url,
            "api_key": "",  # TODO: inject per-session API key for agent self-service callbacks
        }, indent=2))

    _ALLOWED_ENV_KEYS = frozenset({
        "PATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL",
        "LC_CTYPE", "TERM", "USER", "LOGNAME", "SHELL",
    })
    _TOOL_API_KEYS = (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "GITLAB_TOKEN", "GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL",
    )
    _COPILOT_API_KEYS = ("GITHUB_TOKEN", "COPILOT_GITHUB_TOKEN", "GH_TOKEN")
    _OPENROUTER_API_KEYS = ("OPENROUTER_API_KEY",)
    _RUNTIME_API_KEYS = frozenset(_COPILOT_API_KEYS + _OPENROUTER_API_KEYS)
    _ONECLI_RUNTIME_ENV_KEYS = (
        "ONECLI_GATEWAY", "ONECLI_GATEWAY_SKILL_PATH", "HTTP_PROXY", "HTTPS_PROXY",
        "NO_PROXY", "NODE_USE_ENV_PROXY", "NODE_EXTRA_CA_CERTS",
    )
    _SECRET_KEY_PATTERNS = frozenset({
        "DATABASE_URL", "SECRET_KEY", "REDIS_URL", "POSTGRES_PASSWORD", "POSTGRES_URL",
        "RALLY_JWT_SECRET", "RALLY_DATABASE_URL", "RALLY_REDIS_URL", "RALLY_API_KEY",
        "HUDDLEROOM_JWT_SECRET", "HUDDLEROOM_DATABASE_URL", "HUDDLEROOM_REDIS_URL", "HUDDLEROOM_API_KEY",
    })

    @staticmethod
    def _context_env(**values: str) -> dict[str, str]:
        return {
            **{f"HUDDLEROOM_{key}": value for key, value in values.items()},
            **{f"RALLY_{key}": value for key, value in values.items()},
        }

    def _build_env(self, session_id: uuid.UUID, task, agent, project) -> dict:
        env = {k: v for k, v in os.environ.items() if k in self._ALLOWED_ENV_KEYS}
        env.update(self._context_env(
            SESSION_ID=str(session_id), TASK_ID=str(task.id) if task else "", AGENT_ID=str(agent.id),
            PROJECT_ID=str(project.id) if project else "", API_KEY="", API_BASE=settings.api_base_url,
        ))
        if agent.config.get("cli_env_extras"):
            safe_extras = {
                k: v for k, v in agent.config["cli_env_extras"].items()
                if k not in self._SECRET_KEY_PATTERNS and k not in self._RUNTIME_API_KEYS
            }
            env.update(safe_extras)
        # Apply tool API keys last so agent config cannot override them
        for key in self._TOOL_API_KEYS:
            if key in os.environ:
                env[key] = os.environ[key]
        runtime = agent.config.get("cli_runtime", agent.cli_runtime or "claude_code")
        if runtime in {"copilot", "opencode", "pi"}:
            for key in self._ONECLI_RUNTIME_ENV_KEYS:
                if key in os.environ:
                    env[key] = os.environ[key]
        if runtime == "copilot":
            for key in self._COPILOT_API_KEYS:
                if key in os.environ:
                    env[key] = os.environ[key]
        if runtime in {"opencode", "pi"}:
            for key in self._OPENROUTER_API_KEYS:
                if key in os.environ:
                    env[key] = os.environ[key]
        return env

    async def run(self, session_id: uuid.UUID, db: AsyncSession, runner_task_id: str | None = None) -> None:
        # 1. Load session (idempotency check)
        result = await db.execute(select(Session).where(Session.id == session_id))
        session = result.scalar_one_or_none()
        if not session or session.status not in {"pending", "running"}:
            return
        from huddleroom.workers.session_tasks import orchestration_lineage_state
        lineage = await orchestration_lineage_state(db, session)
        if lineage is False:
            return
        orchestration = lineage is True
        attempt = (session.metadata_ or {}).get("attempt")
        if orchestration and (session.status != "running" or not isinstance(attempt, dict)
                            or not runner_task_id or attempt.get("claimed_runner_task_id") != runner_task_id
                            or session.runner_task_id != runner_task_id):
            return

        async def finish_if_owned() -> bool:
            if not orchestration:
                return True
            from huddleroom.workers.session_tasks import _mark_attempt_result
            return await _mark_attempt_result(db, session.id, runner_task_id)

        # 2. Load agent
        result = await db.execute(select(Agent).where(Agent.id == session.agent_id))
        agent = result.scalar_one_or_none()
        if not agent:
            session.status = "failed"
            session.error = "agent_not_found"
            session.resumable = False
            if not await finish_if_owned():
                return
            await db.flush()
            await emit_event(db, session.project_id, "session.failed", {
                "session_id": str(session_id),
                "error": "agent_not_found",
                "project_id": str(session.project_id),
            })
            from huddleroom.services.session_sync import sync_task_from_session
            await sync_task_from_session(db, session)
            return

        # 2b. Determine cli_runtime early for resume-aware launch logic
        cli_runtime = agent.config.get("cli_runtime", agent.cli_runtime or "claude_code")
        run_config = session.metadata_.get("_run_config", {}) if session.metadata_ else {}
        effective_model = run_config.get("model_override") or agent.model
        # 3. Load task and project
        task = None
        project = None
        if session.task_id:
            result = await db.execute(select(Task).where(Task.id == session.task_id))
            task = result.scalar_one_or_none()
        result = await db.execute(select(Project).where(Project.id == session.project_id))
        project = result.scalar_one_or_none()

        # 4. Revalidate the workspace immediately before preparing the launch.
        roadmap = (session.input_context or {}).get("orchestrator_context", {}).get("roadmap", {})
        try:
            if isinstance(roadmap, dict) and roadmap.get("mutates_shared_state") and roadmap.get("staging_boundary"):
                workspace = await ProjectService().require_frozen_roadmap_workspace(
                    db, session.project_id, roadmap.get("staging_boundary"), (session.metadata_ or {}).get("_roadmap_workspace", "")
                )
            else:
                workspace = await ProjectService().require_runnable_project(db, session.project_id)
        except HTTPException as exc:
            await self._fail_workspace_validation(db, session, exc, runner_task_id)
            return
        launch_fingerprint = self.launch_fingerprint(session, task, agent, project, workspace)
        proven_launch = (session.metadata_ or {}).get("_recovery_exact_launch_fingerprint")
        if proven_launch is not None and proven_launch != launch_fingerprint:
            session.status = "failed"
            session.error = "exact_resume_launch_config_changed"
            session.resumable = False
            session.ended_at = datetime.now(timezone.utc)
            if not await finish_if_owned():
                return
            await db.flush()
            return
        metadata = session.metadata_ or {}
        prior_launch = metadata.get("_launch_config")
        current_launch = {"provider": agent.provider, "model": effective_model, "cli_runtime": cli_runtime}
        if (metadata.get("_launch_fingerprint") not in (None, launch_fingerprint)
                or (isinstance(prior_launch, dict) and prior_launch != current_launch)):
            session.provider_session_id = None
        sandbox_dir = self._sandbox_dir(workspace, "agents", str(agent.id), "sessions", str(session_id))

        # Build source-owned request display BEFORE composing task content
        request_display = self._build_cli_request_display(task)

        task_content = self._compose_task_content(task, agent, session_id)
        self._setup_sandbox(sandbox_dir, session_id, task, agent, session.project_id, task_content)

        # Orchestration recovery retains its established Claude-only exact resume.
        supports_resume = (
            cli_runtime == "claude_code"
            or not orchestration and cli_runtime in {"copilot", "opencode", "pi"}
        )
        if session.provider_session_id and supports_resume:
            task_content = "continue"

        # 5. Build command
        try:
            cmd = self._build_command(
                cli_runtime, task_content, agent.config, workspace, sandbox_dir / "task.md",
                existing_session_id=session.provider_session_id if supports_resume else None,
                model=effective_model,
            )
        except RuntimeError as exc:
            command_error = str(exc)
        else:
            command_error = "custom_runtime_missing_script_path" if cmd is None else None
        if command_error:
            session.status = "failed"
            session.error = command_error
            session.resumable = False
            session.ended_at = datetime.now(timezone.utc)
            self._store_roadmap_elapsed(session)
            if not await finish_if_owned():
                return
            await db.flush()
            await emit_event(db, session.project_id, "session.failed", {
                "session_id": str(session_id),
                "error": session.error,
                "project_id": str(session.project_id),
            })
            from huddleroom.services.session_sync import sync_task_from_session
            await sync_task_from_session(db, session)
            log_cli_exchange(
                prompt=task_content,
                session_id=session.provider_session_id,
                error=session.error,
                runtime=cli_runtime,
                rally_session_id=str(session_id),
            )
            return

        # 6. Mark session as running
        claimed = session.status == "running"
        session.status = "running"
        if not claimed:
            session.started_at = datetime.now(timezone.utc)
        session.sandbox_path = str(sandbox_dir)
        await db.flush()
        if not claimed:
            await emit_event(db, session.project_id, "session.started", {
                "session_id": str(session_id), "agent_id": str(session.agent_id),
                "task_id": str(session.task_id) if session.task_id else None,
                "project_id": str(session.project_id),
            })
        # Release SQLite's writer lock before waiting on the subprocess.
        await db.commit()

        # 7. Build env and launch subprocess
        env = self._build_env(session_id, task, agent, project)
        timeout = float(run_config.get("timeout") or agent.config.get("session_timeout_seconds", 3600))
        deadline = time.monotonic() + timeout
        try:
            if isinstance(roadmap, dict) and roadmap.get("mutates_shared_state") and roadmap.get("staging_boundary"):
                workspace = await ProjectService().require_frozen_roadmap_workspace(
                    db, session.project_id, roadmap.get("staging_boundary"), (session.metadata_ or {}).get("_roadmap_workspace", "")
                )
            else:
                workspace = await ProjectService().require_runnable_project(db, session.project_id)
        except HTTPException as exc:
            await self._fail_workspace_validation(db, session, exc, runner_task_id)
            return
        launch_fingerprint = self.launch_fingerprint(session, task, agent, project, workspace)
        if proven_launch is not None and proven_launch != launch_fingerprint:
            await self._fail_workspace_validation(
                db, session, HTTPException(status_code=409, detail="exact_resume_launch_config_changed"), runner_task_id,
            )
            return
        session.metadata_ = {
            **(session.metadata_ or {}), "_launch_fingerprint": launch_fingerprint,
            "_launch_config": {"provider": agent.provider, "model": effective_model, "cli_runtime": cli_runtime},
        }
        # Release SQLite's writer lock after metadata update before subprocess/watcher race.
        await db.commit()
        cmd = self._build_command(
            cli_runtime, task_content, agent.config, workspace, sandbox_dir / "task.md",
            existing_session_id=session.provider_session_id if supports_resume else None,
            model=effective_model,
        )

        try:
            if os.name != "posix":
                raise RuntimeError("CLI process-group isolation requires a POSIX platform")
            if orchestration:
                from huddleroom.workers.session_tasks import mark_attempt_effect_started
                if not await mark_attempt_effect_started(db, session.id, runner_task_id):
                    return
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(workspace),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError:
            session.status = "failed"
            session.error = f"command_not_found: {cmd[0]}"
            session.resumable = True
            session.ended_at = datetime.now(timezone.utc)
            self._store_roadmap_elapsed(session)
            if not await finish_if_owned():
                return
            await db.flush()
            await emit_event(db, session.project_id, "session.failed", {
                "session_id": str(session_id),
                "error": session.error,
                "project_id": str(session.project_id),
                "resumable": True,
            })
            from huddleroom.services.session_sync import sync_task_from_session
            await sync_task_from_session(db, session)
            log_cli_exchange(
                prompt=task_content,
                session_id=session.provider_session_id,
                error=session.error,
                runtime=cli_runtime,
                rally_session_id=str(session_id),
            )
            return
        except Exception as e:
            session.status = "failed"
            session.error = redact_secrets(str(e))
            session.resumable = True
            session.ended_at = datetime.now(timezone.utc)
            self._store_roadmap_elapsed(session)
            if not await finish_if_owned():
                return
            await db.flush()
            await emit_event(db, session.project_id, "session.failed", {
                "session_id": str(session_id),
                "error": session.error,
                "project_id": str(session.project_id),
                "resumable": True,
            })
            from huddleroom.services.session_sync import sync_task_from_session
            await sync_task_from_session(db, session)
            log_cli_exchange(
                prompt=task_content,
                session_id=session.provider_session_id,
                error=session.error,
                runtime=cli_runtime,
                rally_session_id=str(session_id),
            )
            return

        # 8. Create invocation context and call for agent response streaming
        invocation = AgentResponseInvocation(
            InvocationContext(
                session.project_id,
                "agent",
                str(agent.id),
                agent.name,
                "cli_main",
                "task",
                effective_model,
                request_display,
            )
        )

        # 9. Race process completion against the reset fence and timeout.
        async def collect_task():
            """Collect CLI process output concurrently."""
            collected = await collect_cli_process(proc, runtime=cli_runtime, call=call)
            return collected.stdout, collected.stderr, collected.returncode

        # Use the invocation call within async context manager
        async with invocation.call(messages=[]) as call:
            collect_task_obj = asyncio.create_task(collect_task())
            watcher_task = asyncio.create_task(self._watch_project_status(db, session.project_id))
            try:
                done, _ = await asyncio.wait(
                    (collect_task_obj, watcher_task), timeout=max(0, deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if collect_task_obj in done:
                    stdout_bytes, stderr_bytes, exit_code = collect_task_obj.result()
                else:
                    await self._terminate_process_group(proc)
                    collect_task_obj.cancel()
                    await asyncio.gather(collect_task_obj, return_exceptions=True)
                    if watcher_task in done:
                        try:
                            project_status = watcher_task.result()
                        except Exception as exc:
                            session.status = "failed"
                            session.error = f"project_status_watcher_failed: {exc}"
                            session.resumable = True
                            event_type = "session.failed"
                        else:
                            if project_status != "resetting":
                                session.status = "failed"
                                session.error = f"project_status_watcher_failed: unexpected status {project_status!r}"
                                session.resumable = True
                                event_type = "session.failed"
                            else:
                                session.status = "cancelled"
                                session.error = None
                                event_type = "session.cancelled"
                        session.ended_at = datetime.now(timezone.utc)
                        self._store_roadmap_elapsed(session)
                        if not await finish_if_owned():
                            return
                        await db.flush()
                        payload = {
                            "session_id": str(session_id),
                            "task_id": str(session.task_id) if session.task_id else None,
                            "project_id": str(session.project_id),
                        }
                        if session.error:
                            payload["error"] = session.error
                        if session.status == "failed":
                            payload["resumable"] = True
                        await emit_event(db, session.project_id, event_type, payload)
                        if session.status == "failed":
                            from huddleroom.services.session_sync import sync_task_from_session
                            await sync_task_from_session(db, session)
                            log_cli_exchange(
                                prompt=task_content,
                                session_id=session.provider_session_id,
                                error=redact_secrets(session.error),
                                runtime=cli_runtime,
                                rally_session_id=str(session_id),
                            )
                        return

                    session.status = "failed"
                    session.error = "timeout"
                    session.resumable = True
                    session.ended_at = datetime.now(timezone.utc)
                    self._store_roadmap_elapsed(session)
                    if not await finish_if_owned():
                        return
                    await db.flush()
                    await emit_event(db, session.project_id, "session.failed", {
                        "session_id": str(session_id),
                        "error": "timeout",
                        "project_id": str(session.project_id),
                        "resumable": True,
                    })
                    from huddleroom.services.session_sync import sync_task_from_session
                    await sync_task_from_session(db, session)
                    log_cli_exchange(
                        prompt=task_content,
                        session_id=session.provider_session_id,
                        error=session.error,
                        runtime=cli_runtime,
                        rally_session_id=str(session_id),
                    )
                    return
            except asyncio.CancelledError:
                await self._terminate_process_group(proc)
                collect_task_obj.cancel()
                await asyncio.gather(collect_task_obj, return_exceptions=True)
                session.status = "cancelled"
                session.error = None
                session.ended_at = datetime.now(timezone.utc)
                self._store_roadmap_elapsed(session)
                if not await finish_if_owned():
                    return
                await db.flush()
                await emit_event(db, session.project_id, "session.cancelled", {
                    "session_id": str(session_id),
                    "task_id": str(session.task_id) if session.task_id else None,
                    "project_id": str(session.project_id),
                })
                raise
            finally:
                watcher_task.cancel()
                await asyncio.gather(watcher_task, return_exceptions=True)

        # 9. Parse output and update session
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        provider_session_id = session.provider_session_id
        repair_error = None
        if cli_runtime == "claude_code":
            # Wire envelope parsing through auto-repair loop
            from huddleroom.services.cli_streaming import parse_claude_final

            async def run_fn_initial():
                """Return the already-captured stdout."""
                return stdout_text

            try:
                output, extracted_session_id = await cli_complete_with_repair(
                    run_fn=run_fn_initial,
                    resume_fn=lambda sid, fix_prompt: self._resume_raw(
                        sid, fix_prompt,
                        cli_runtime=cli_runtime,
                        agent_config=agent.config,
                        workspace=workspace,
                        task_path=sandbox_dir / "task.md",
                        model=effective_model,
                        cwd=str(workspace),
                        env=env,
                        timeout=timeout,
                        deadline=deadline,
                        invocation=invocation,
                    ),
                    parse=parse_claude_final,
                    session_id=provider_session_id,
                    max_attempts=3,
                )
                provider_session_id = extracted_session_id or provider_session_id
            except (ValueError, json.JSONDecodeError):
                # Repair exhausted; fallback to raw text (exact behavior as before)
                output = stdout_text
            except RuntimeError as exc:
                # A repair process cannot outlive this terminal session.
                output = stdout_text
                repair_error = redact_secrets(str(exc))
        elif cli_runtime in {"copilot", "opencode", "pi"}:
            from huddleroom.services.cli_streaming import parse_jsonl_final

            try:
                output, extracted_session_id = parse_jsonl_final(stdout_text, cli_runtime)
                provider_session_id = extracted_session_id or provider_session_id
            except ValueError:
                from huddleroom.services.cli_streaming import parse_jsonl_session_id
                try:
                    provider_session_id = parse_jsonl_session_id(stdout_text, cli_runtime) or provider_session_id
                except ValueError:
                    pass
                output = ""
                repair_error = "invalid_cli_output"
        else:
            output = stdout_text

        session.provider_session_id = provider_session_id
        session.output = output
        session.status = "failed" if repair_error or exit_code != 0 else "completed"
        if repair_error:
            session.error = repair_error
            session.resumable = True
        elif exit_code != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            session.error = f"exit_code_{exit_code}" + (f": {redact_secrets(stderr_text[:500])}" if stderr_text else "")
            session.resumable = True
        session.ended_at = datetime.now(timezone.utc)
        self._store_roadmap_elapsed(session)
        session.metadata_ = {
            **(session.metadata_ or {}),
            "exit_code": exit_code,
            "model_used": effective_model,
        }
        if (session.metadata_ or {}).get("_run_config", {}).get("_roadmap_budget_enforced"):
            usage = self._claude_usage(stdout_text) if cli_runtime == "claude_code" else None
            session.metadata_ = {
                **session.metadata_,
                "token_usage_complete": usage is not None,
                **({"token_count_in": usage[0], "token_count_out": usage[1]} if usage is not None else {}),
            }
        if not await finish_if_owned():
            return
        await db.flush()
        final_event = "session.completed" if session.status == "completed" else "session.failed"
        payload = {
            "session_id": str(session_id),
            "task_id": str(session.task_id) if session.task_id else None,
            "exit_code": exit_code,
            "project_id": str(session.project_id),
            **({"runner_task_id": runner_task_id} if orchestration and session.status == "completed" else {}),
        }
        if session.status != "completed":
            payload["error"] = session.error
            payload["resumable"] = True
        # INVARIANT: session.completed emitters MUST call sync_task_from_session
        # (task->done) in this same transaction before it commits; the
        # orchestrator's canonical-report consumption in
        # orchestration_service._ingest_session_evidence keys off
        # task.status == "done" and silently drops the report otherwise.
        await emit_event(db, session.project_id, final_event, payload)
        from huddleroom.services.session_sync import sync_task_from_session
        await sync_task_from_session(db, session)
        if exit_code == 0:
            log_cli_exchange(
                prompt=task_content,
                session_id=provider_session_id,
                response=output,
                runtime=cli_runtime,
                rally_session_id=str(session_id),
            )
        else:
            log_cli_exchange(
                prompt=task_content,
                session_id=provider_session_id,
                error=session.error,
                runtime=cli_runtime,
                rally_session_id=str(session_id),
            )

    def _build_meeting_env(self, meeting, agent, project) -> dict:
        """Build environment for a meeting turn subprocess."""
        env = {k: v for k, v in os.environ.items() if k in self._ALLOWED_ENV_KEYS}
        env.update(self._context_env(
            MEETING_ID=str(meeting.id), AGENT_ID=str(agent.id), PROJECT_ID=str(project.id) if project else "",
            API_KEY="", API_BASE=settings.api_base_url,
        ))
        if agent.config.get("cli_env_extras"):
            safe_extras = {
                k: v for k, v in agent.config["cli_env_extras"].items()
                if k not in self._SECRET_KEY_PATTERNS and k not in self._RUNTIME_API_KEYS
            }
            env.update(safe_extras)
        # Apply tool API keys last so agent config cannot override them
        for key in self._TOOL_API_KEYS:
            if key in os.environ:
                env[key] = os.environ[key]
        runtime = agent.config.get("cli_runtime", agent.cli_runtime or "claude_code")
        if runtime in {"copilot", "opencode", "pi"}:
            for key in self._ONECLI_RUNTIME_ENV_KEYS:
                if key in os.environ:
                    env[key] = os.environ[key]
        if runtime == "copilot":
            for key in self._COPILOT_API_KEYS:
                if key in os.environ:
                    env[key] = os.environ[key]
        if runtime in {"opencode", "pi"}:
            for key in self._OPENROUTER_API_KEYS:
                if key in os.environ:
                    env[key] = os.environ[key]
        return env

    async def _resume_raw(
        self, session_id: str, fix_prompt: str, *, cli_runtime: str, agent_config: dict,
        workspace: Path, task_path: Path, model: str | None, cwd: str, env: dict, timeout: float,
        deadline: float | None = None,
        invocation: AgentResponseInvocation | None = None,
    ) -> str:
        """Re-invoke CLI with fix prompt, resuming where the runtime supports it.

        Returns decoded stdout string.

        ponytail: codex/aider/custom re-run fresh from persisted context.
        """
        remaining_timeout = timeout if deadline is None else deadline - time.monotonic()
        if remaining_timeout <= 0:
            raise RuntimeError("resume deadline exhausted")
        cmd = self._build_command(
            cli_runtime, fix_prompt, agent_config, workspace, task_path,
            existing_session_id=session_id if cli_runtime in {"claude_code", "copilot", "opencode", "pi"} else None,
            model=model,
        )
        if cmd is None:
            raise RuntimeError("custom_runtime_missing_script_path")

        try:
            if deadline is not None:
                remaining_timeout = deadline - time.monotonic()
                if remaining_timeout <= 0:
                    raise RuntimeError("resume deadline exhausted")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"command_not_found: {cmd[0]}") from exc

        try:
            if deadline is not None:
                remaining_timeout = deadline - time.monotonic()
                if remaining_timeout <= 0:
                    await self._terminate_process_group(proc)
                    raise RuntimeError("resume deadline exhausted")
            if invocation:
                # Use agent response streaming for resume
                async def collect_task():
                    async with invocation.call(messages=[], invocation_kind="cli_resume", inherit_request=True) as call:
                        collected = await collect_cli_process(proc, runtime=cli_runtime, call=call)
                        # Return raw stdout for repair parser, not decoded content
                        return collected.stdout.decode("utf-8", errors="replace")

                collect_task_obj = asyncio.create_task(collect_task())
                try:
                    content = await asyncio.wait_for(collect_task_obj, timeout=remaining_timeout)
                except asyncio.TimeoutError:
                    await self._terminate_process_group(proc)
                    collect_task_obj.cancel()
                    await asyncio.gather(collect_task_obj, return_exceptions=True)
                    raise RuntimeError(f"resume subprocess timed out after {remaining_timeout}s")
                except asyncio.CancelledError:
                    await self._terminate_process_group(proc)
                    collect_task_obj.cancel()
                    await asyncio.gather(collect_task_obj, return_exceptions=True)
                    raise
                return content
            else:
                # Fallback: old-style communicate without streaming (for backwards compatibility)
                communicate_task = asyncio.create_task(proc.communicate())
                try:
                    stdout_bytes, _ = await asyncio.wait_for(communicate_task, timeout=remaining_timeout)
                except asyncio.TimeoutError:
                    await self._terminate_process_group(proc)
                    communicate_task.cancel()
                    await asyncio.gather(communicate_task, return_exceptions=True)
                    raise RuntimeError(f"resume subprocess timed out after {remaining_timeout}s")
                except asyncio.CancelledError:
                    await self._terminate_process_group(proc)
                    communicate_task.cancel()
                    await asyncio.gather(communicate_task, return_exceptions=True)
                    raise
                return stdout_bytes.decode("utf-8", errors="replace")
        except RuntimeError:
            raise
        except Exception as exc:
            await self._terminate_process_group(proc)
            raise RuntimeError(f"resume subprocess error: {exc}") from exc

    async def run_meeting_turn(
        self,
        db: AsyncSession,
        meeting,
        agent,
        project,
        prompt_text: str,
        existing_session_id: str | None,
        agenda_title: str | None = None,
        agenda_question: str | None = None,
        operation: str = "meeting_turn",
    ) -> tuple[str, str | None, int]:
        """Run a meeting turn via CLI agent. Returns (content, new_session_id, latency_ms)."""
        # 1. Setup sandbox
        workspace = await ProjectService().require_runnable_project(db, meeting.project_id)
        sandbox_dir = self._sandbox_dir(
            workspace, "agents", str(agent.id), "meetings", str(meeting.id), str(agent.id),
        )
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        # Build source-owned request display BEFORE writing context (agenda title + question only, NO transcript/context)
        request_display = self._build_meeting_cli_request_display(
            agenda_title or "Meeting discussion",
            agenda_question or ""
        )

        # 2. Write meeting context
        ctx_path = sandbox_dir / "meeting_context.md"
        await asyncio.to_thread(ctx_path.write_text, prompt_text)

        # 3. Write HuddleRoom context JSON
        huddleroom_context = json.dumps({
            "meeting_id": str(meeting.id),
            "agent_id": str(agent.id),
            "project_id": str(meeting.project_id),
            "api_base": settings.api_base_url,
        }, indent=2)
        await asyncio.to_thread(self._context_path(sandbox_dir).write_text, huddleroom_context)

        # 4. Resolve runtime and environment.
        cli_runtime = agent.config.get("cli_runtime", agent.cli_runtime or "claude_code")
        env = self._build_meeting_env(meeting, agent, project)

        # 6–8: run subprocess; clean sensitive context files on any failure
        _success = False
        start = time.monotonic()
        try:
            # Revalidate after writing context and immediately before launch.
            workspace = await ProjectService().require_runnable_project(db, meeting.project_id)
            cmd = self._build_command(
                cli_runtime,
                prompt_text,
                agent.config,
                workspace,
                ctx_path,
                context_path=ctx_path,
                existing_session_id=existing_session_id,
                model=agent.model,
            )
            if cmd is None:
                raise CliTurnFailed("custom_runtime_missing_script_path")
            # 6. Launch subprocess
            try:
                if os.name != "posix":
                    raise RuntimeError("CLI process-group isolation requires a POSIX platform")
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(workspace),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except FileNotFoundError:
                raise CliTurnFailed(f"CLI runtime not found: {cmd[0]}")

            # 7. Create invocation context and call for agent response streaming
            invocation = AgentResponseInvocation(
                InvocationContext(
                    meeting.project_id,
                    "agent",
                    str(agent.id),
                    agent.name,
                    "cli_meeting",
                    operation,
                    agent.model or "claude",
                    request_display,
                )
            )

            # 8. Race completion against timeout and a project reset.
            timeout = agent.config.get("meeting_turn_timeout_seconds", 300)

            async def collect_task():
                """Collect CLI process output concurrently."""
                collected = await collect_cli_process(proc, runtime=cli_runtime, call=call)
                return collected.stdout, collected.stderr

            async with invocation.call(messages=[]) as call:
                collect_task_obj = asyncio.create_task(collect_task())
                watcher_task = asyncio.create_task(self._watch_project_status(db, meeting.project_id))
                try:
                    done, _ = await asyncio.wait(
                        (collect_task_obj, watcher_task), timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if collect_task_obj in done:
                        stdout_bytes, stderr_bytes = collect_task_obj.result()
                    else:
                        await self._terminate_process_group(proc)
                        collect_task_obj.cancel()
                        await asyncio.gather(collect_task_obj, return_exceptions=True)
                        if watcher_task in done:
                            try:
                                watcher_task.result()
                            except Exception as exc:
                                raise CliTurnFailed(f"project status watcher failed: {exc}") from exc
                            raise HTTPException(
                                status_code=409,
                                detail={"code": "project_not_runnable", "reason": "project_inactive"},
                            )
                        raise CliTurnFailed(f"CLI agent timed out after {timeout}s")
                except asyncio.CancelledError:
                    await self._terminate_process_group(proc)
                    collect_task_obj.cancel()
                    await asyncio.gather(collect_task_obj, return_exceptions=True)
                    raise
                finally:
                    watcher_task.cancel()
                    await asyncio.gather(watcher_task, return_exceptions=True)

            # Check exit code
            if proc.returncode != 0:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                logger.warning(
                    "run_meeting_turn: CLI agent exited with code=%d agent=%s stderr=%r",
                    proc.returncode, agent.id, redact_secrets(stderr_text[:200]),
                )
                # Best-effort parse stdout JSON/JSONL for a resumable session ID.
                session_id_from_failure = None
                stdout_text_from_failure = stdout_bytes.decode("utf-8", errors="replace")
                if cli_runtime in {"copilot", "opencode", "pi"}:
                    from huddleroom.services.cli_streaming import parse_jsonl_session_id
                    try:
                        session_id_from_failure = parse_jsonl_session_id(stdout_text_from_failure, cli_runtime)
                    except ValueError:
                        pass
                else:
                    try:
                        parsed = json.loads(stdout_text_from_failure)
                        if isinstance(parsed, dict):
                            session_id_from_failure = parsed.get("session_id")
                    except (json.JSONDecodeError, ValueError):
                        pass
                raise CliTurnFailed(
                    f"CLI agent exit_code={proc.returncode}: {redact_secrets(stderr_text[:200])}",
                    session_id=session_id_from_failure
                )
            else:
                stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                if stderr_text:
                    logger.debug(
                        "run_meeting_turn: CLI agent stderr (exit 0) agent=%s: %r",
                        agent.id, redact_secrets(stderr_text[:500]),
                    )

            # 8. Parse output
            latency_ms = int((time.monotonic() - start) * 1000)
            stdout_text = stdout_bytes.decode("utf-8", errors="replace")

            if cli_runtime in {"copilot", "opencode", "pi"}:
                from huddleroom.services.cli_streaming import parse_jsonl_final
                try:
                    content, new_session_id = parse_jsonl_final(stdout_text, cli_runtime)
                except ValueError as exc:
                    raise CliTurnFailed("invalid_cli_output") from exc
                _success = True
                log_cli_exchange(
                    prompt=prompt_text,
                    session_id=new_session_id or existing_session_id,
                    response=content,
                    runtime=cli_runtime,
                    meeting_id=str(meeting.id),
                    agent_id=str(agent.id),
                )
                return (content, new_session_id, latency_ms)

            # Wire envelope parsing through auto-repair loop
            from huddleroom.services.cli_streaming import parse_claude_final

            def parse_envelope(raw: str) -> tuple[str, str | None]:
                """Parse envelope based on runtime."""
                if cli_runtime == "claude_code":
                    return parse_claude_final(raw)
                if cli_runtime in {"copilot", "opencode", "pi"}:
                    from huddleroom.services.cli_streaming import parse_jsonl_final
                    return parse_jsonl_final(raw, cli_runtime)
                else:
                    # Other runtimes: try to extract result field, fallback to raw text
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            result = parsed.get("result", "")
                            session_id = parsed.get("session_id")
                            if result or session_id:
                                return (result, session_id)
                    except json.JSONDecodeError:
                        pass
                    return (raw, None)

            async def run_fn_initial():
                """Return the already-captured stdout."""
                return stdout_text

            try:
                content, new_session_id = await cli_complete_with_repair(
                    run_fn=run_fn_initial,
                    resume_fn=lambda sid, fix_prompt: self._resume_raw(
                        sid, fix_prompt,
                        cli_runtime=cli_runtime,
                        agent_config=agent.config,
                        workspace=workspace,
                        task_path=ctx_path,
                        model=agent.model,
                        cwd=str(workspace),
                        env=env,
                        timeout=agent.config.get("meeting_turn_timeout_seconds", 300),
                        invocation=invocation,
                    ),
                    parse=parse_envelope,
                    session_id=existing_session_id,
                    max_attempts=3,
                )
            except (ValueError, json.JSONDecodeError):
                content = stdout_text
                new_session_id = None

            _success = True
            log_cli_exchange(
                prompt=prompt_text,
                session_id=new_session_id or existing_session_id,
                response=content,
                runtime=cli_runtime,
                meeting_id=str(meeting.id),
                agent_id=str(agent.id),
            )
            return (content, new_session_id, latency_ms)
        except CliTurnFailed as exc:
            log_cli_exchange(
                prompt=prompt_text,
                session_id=exc.session_id or existing_session_id,
                error=exc,
                runtime=cli_runtime,
                meeting_id=str(meeting.id),
                agent_id=str(agent.id),
            )
            raise
        except HTTPException:
            raise
        except Exception as exc:
            log_cli_exchange(
                prompt=prompt_text,
                session_id=existing_session_id,
                error=exc,
                runtime=cli_runtime,
                meeting_id=str(meeting.id),
                agent_id=str(agent.id),
            )
            raise
        finally:
            if not _success:
                for _fname in ("meeting_context.md", self._CONTEXT_FILENAME, self._LEGACY_CONTEXT_FILENAME):
                    try:
                        (sandbox_dir / _fname).unlink(missing_ok=True)
                    except OSError:
                        pass
