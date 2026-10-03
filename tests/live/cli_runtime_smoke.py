#!/usr/bin/env python3
"""Run a low-cost live smoke check for the three Step 8 CLI runtimes.

Run this script inside the configured onecli environment.  It uses the adapter's
actual argv, a disposable workspace, and a short no-tools prompt.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from huddleroom.adapters.cli_adapter import CliAdapter
from huddleroom.models.agent import Agent
from huddleroom.models.base import Base
from huddleroom.models.project import Project
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.cli_streaming import parse_jsonl_final
import huddleroom.models  # noqa: F401  # Register every table before creating the temporary database.


RUNTIMES = ("copilot", "opencode", "pi")


async def _run_adapter_task(runtime: str, workspace: Path, model: str, timeout: float) -> tuple[str, str | None]:
    """Exercise CliAdapter.run against a disposable SQLite database and workspace."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{workspace / 'adapter-smoke.db'}")
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        async with session_factory() as db:
            project = Project(name=f"cli-smoke-{runtime}-{uuid.uuid4()}", workspace_path=str(workspace), config={})
            agent = Agent(
                name=f"cli-smoke-{runtime}-{uuid.uuid4()}",
                role="smoke tester",
                provider="openrouter",
                model=model,
                adapter_type="cli",
                cli_runtime=runtime,
                capabilities=[],
                config={"cli_runtime": runtime, "session_timeout_seconds": timeout},
            )
            db.add_all([project, agent])
            await db.flush()
            task = Task(
                project_id=project.id,
                title="CLI runtime smoke",
                description="Reply with exactly RALLY_ADAPTER_SMOKE_OK. Do not use tools.",
                status="in_progress",
            )
            db.add(task)
            await db.flush()
            session = Session(
                task_id=task.id,
                agent_id=agent.id,
                project_id=project.id,
                adapter_type="cli",
                status="pending",
                metadata_={"_run_config": {"timeout": timeout}},
            )
            db.add(session)
            await db.commit()
            await CliAdapter().run(session.id, db)
            await db.refresh(session)
            if session.status != "completed" or "RALLY_ADAPTER_SMOKE_OK" not in (session.output or ""):
                raise RuntimeError(f"{runtime} adapter task failed: {session.status} {session.error or 'no error'}")
            return session.output or "", session.provider_session_id
    finally:
        await engine.dispose()


async def _cancel_runtime_process(runtime: str, workspace: Path, model: str | None) -> None:
    """Verify the shared process-group cancellation path against a live CLI process."""
    command = CliAdapter._build_command(
        runtime,
        "Run `sleep 60` now and do not answer until it finishes.",
        {},
        workspace,
        workspace / "task.md",
        model=model,
    )
    assert command is not None
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=workspace,
        env=os.environ.copy(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        await asyncio.sleep(0.25)
        if process.returncode is not None:
            raise RuntimeError(f"{runtime} exited before cancellation")
        await CliAdapter._terminate_process_group(process, grace_seconds=0.5)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        raise RuntimeError(f"{runtime} process group survived cancellation")
    finally:
        if process.returncode is None:
            await CliAdapter._terminate_process_group(process, grace_seconds=0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtimes", default=",".join(RUNTIMES))
    parser.add_argument("--model", help="Model accepted by each configured runtime")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--verify-resume", action="store_true", help="Run one resumed no-tools prompt per runtime")
    parser.add_argument("--adapter-task", action="store_true", help="Run the prompt through CliAdapter and a disposable SQLite DB")
    parser.add_argument("--verify-cancel", action="store_true", help="Start and kill a live runtime process group")
    args = parser.parse_args()
    runtimes = tuple(runtime.strip() for runtime in args.runtimes.split(",") if runtime.strip())
    unknown = set(runtimes) - set(RUNTIMES)
    if unknown:
        parser.error(f"unknown runtime(s): {', '.join(sorted(unknown))}")

    prompt = "Reply with exactly RALLY_CLI_SMOKE_OK. Do not use tools."
    with tempfile.TemporaryDirectory(prefix="rally-cli-smoke-") as temporary:
        workspace = Path(temporary).resolve()
        for runtime in runtimes:
            command = CliAdapter._build_command(
                runtime, prompt, {}, workspace, workspace / "task.md", model=args.model,
            )
            assert command is not None
            version = subprocess.run([command[0], "--version"], text=True, capture_output=True, check=True)
            completed = subprocess.run(
                command, cwd=workspace, text=True, capture_output=True, timeout=args.timeout,
            )
            if completed.returncode:
                raise RuntimeError(f"{runtime} exited {completed.returncode} (stderr captured)")
            content, session_id = parse_jsonl_final(completed.stdout, runtime)
            if "RALLY_CLI_SMOKE_OK" not in content:
                raise RuntimeError(f"{runtime} did not return the smoke marker")
            print(f"PASS {runtime} {version.stdout.strip()} session={session_id or 'unavailable'}")
            if args.verify_resume:
                if not session_id:
                    raise RuntimeError(f"{runtime} returned no resumable session ID")
                resume_command = CliAdapter._build_command(
                    runtime,
                    "Reply with exactly RALLY_CLI_RESUME_OK. Do not use tools.",
                    {},
                    workspace,
                    workspace / "task.md",
                    existing_session_id=session_id,
                    model=args.model,
                )
                assert resume_command is not None
                resumed = subprocess.run(
                    resume_command, cwd=workspace, text=True, capture_output=True, timeout=args.timeout,
                )
                if resumed.returncode:
                    raise RuntimeError(f"{runtime} resume exited {resumed.returncode} (stderr captured)")
                resumed_content, _ = parse_jsonl_final(resumed.stdout, runtime)
                if "RALLY_CLI_RESUME_OK" not in resumed_content:
                    raise RuntimeError(f"{runtime} did not return the resume marker")
                print(f"PASS {runtime} resume")
            if args.adapter_task:
                if not args.model:
                    raise RuntimeError("--adapter-task requires --model because Agent.model is required")
                adapter_output, adapter_session_id = asyncio.run(
                    _run_adapter_task(runtime, workspace, args.model, args.timeout)
                )
                print(f"PASS {runtime} adapter session={adapter_session_id or 'unavailable'}")
            if args.verify_cancel:
                asyncio.run(_cancel_runtime_process(runtime, workspace, args.model))
                print(f"PASS {runtime} cancellation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
