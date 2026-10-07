#!/usr/bin/env python3
"""Live create-and-assign e2e per CLI runtime, through the real HTTP API.

Per runtime: project (tmp workspace) -> cli agent -> task -> assign -> run -> poll session.
PASS criteria: session completed AND marker file contains token.
  - LLM runtimes (claude_code, codex, etc.): marker file ONLY (drop token-in-output).
  - custom: marker file OR output (can choose either).

Usage:
    .venv/bin/python tests/live/cli_adapter_e2e.py --runtimes claude_code,codex
    .venv/bin/python tests/live/cli_adapter_e2e.py --runtimes custom          # free, no LLM
    .venv/bin/python tests/live/cli_adapter_e2e.py --keep                     # skip cleanup
    .venv/bin/python tests/live/cli_adapter_e2e.py --runtimes claude_code,codex --resume  # + resume/poison rows
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_test import PROJECT_ROOT, check_server, start_server, stop_server  # noqa: E402

# Freeze litellm's bundled cost map to avoid remote fetch at server start.
# New models may lack pricing metadata; ensures offline/sandboxed startup succeeds.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
BINARIES = {"claude_code": "claude", "codex": "codex", "aider": "aider", "copilot": "copilot",
            "opencode": "opencode", "pi": "pi", "custom": None}
# ponytail: blank model => adapter omits --model (CLI native default); override with --model runtime=model.
DEFAULT_MODELS = {"claude_code": "claude-sonnet-5-5"}
NON_TERMINAL = {"pending", "queued", "starting", "running"}
RESUME_RUNTIMES = {"claude_code", "codex"}
LIVE_DB = PROJECT_ROOT / "tests" / "live" / "huddleroom_live_test.db"


def api(client: httpx.Client, method: str, path: str, **kw) -> dict:
    r = client.request(method, f"/api/v1{path}", **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
    return r.json()


def wait_session(client: httpx.Client, sid: str, timeout: int) -> dict:
    deadline = time.time() + timeout + 30
    session: dict = {}
    while time.time() < deadline:
        session = api(client, "GET", f"/sessions/{sid}")
        if session["status"] not in NON_TERMINAL:
            return session
        time.sleep(2)
    raise TimeoutError(f"timed out waiting; last status={session.get('status')}")


def resume_via_api(client: httpx.Client, sid: str, timeout: int, provider_id: str | None = None) -> dict:
    """Resume a COMPLETED session through POST /sessions/{id}/resume.

    The API only resumes failed+resumable sessions, and the product has no
    follow-up path for completed ones, so we flip the row in the isolated live DB
    (optionally poisoning provider_session_id) and then use the real endpoint.
    """
    con = sqlite3.connect(LIVE_DB, timeout=30)
    try:
        sets, args = "status='failed', resumable=1", []
        if provider_id:
            sets += ", provider_session_id=?"
            args.append(provider_id)
        n = con.execute(f"UPDATE sessions SET {sets} WHERE id IN (?, ?)", [*args, sid, sid.replace("-", "")]).rowcount
        con.commit()
    finally:
        con.close()
    if n != 1:
        raise RuntimeError(f"expected to update 1 session row in {LIVE_DB}, updated {n}")
    api(client, "POST", f"/sessions/{sid}/resume")
    return wait_session(client, sid, timeout)


def resume_checks(client: httpx.Client, runtime: str, sid: str, workspace: Path, token: str, timeout: int) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    first = api(client, "GET", f"/sessions/{sid}")
    pcs = first.get("provider_session_id")
    if not pcs:
        return [(f"{runtime}/resume", "FAIL", "provider_session_id empty after first session"),
                (f"{runtime}/poison", "SKIP", "no provider_session_id")]
    out = workspace / "e2e_resume.txt"
    # Resume: the adapter's prompt is just "continue" (token never in it); delete the marker
    # so the agent cannot re-read it and must recall the token from the resumed conversation.
    (workspace / "e2e_marker.txt").unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    try:
        s = resume_via_api(client, sid, timeout)
        outcome = (s.get("metadata") or {}).get("resume_outcome")
        ok = s["status"] == "completed" and outcome == "resumed" and out.exists() and token in out.read_text(errors="replace")
        rows.append((f"{runtime}/resume", "PASS" if ok else "FAIL",
                     f"status={s['status']} resume_outcome={outcome} resume_file={'ok' if out.exists() and token in out.read_text(errors='replace') else 'missing/wrong'}"))
    except Exception as exc:  # noqa: BLE001
        rows.append((f"{runtime}/resume", "FAIL", f"{type(exc).__name__}: {exc}"))
    try:
        s = resume_via_api(client, sid, timeout, provider_id=str(uuid.uuid4()))
        outcome = (s.get("metadata") or {}).get("resume_outcome")
        ok = s["status"] == "completed" and outcome == "fallback_fresh"
        rows.append((f"{runtime}/poison", "PASS" if ok else "FAIL", f"status={s['status']} resume_outcome={outcome} error={s.get('error')}"))
    except Exception as exc:  # noqa: BLE001
        rows.append((f"{runtime}/poison", "FAIL", f"{type(exc).__name__}: {exc}"))
    return rows


def run_runtime(client: httpx.Client, runtime: str, model: str, effort: str | None, timeout: int, keep: bool = False,
                resume: bool = False, extra_rows: list | None = None) -> tuple[str, str]:
    token = f"E2E_{uuid.uuid4().hex[:12]}"
    workspace = Path(tempfile.mkdtemp(prefix=f"cli-e2e-{runtime}-")).resolve()  # must equal resolved path
    script = None
    try:
        config: dict = {"cli_runtime": runtime, "session_timeout_seconds": timeout}
        if effort:
            config["reasoning_effort"] = effort
        if runtime == "custom":
            script = workspace.parent / f"{workspace.name}-script.sh"
            script.write_text(f'#!/bin/sh\nprintf %s {token} > "{workspace}/e2e_marker.txt"\necho {token}\n')
            script.chmod(0o755)
            config["script_path"] = str(script)
        suffix = uuid.uuid4().hex[:8]
        project = api(client, "POST", "/projects", json={"name": f"cli-e2e-{runtime}-{suffix}", "workspace_path": str(workspace)})
        agent = api(client, "POST", "/agents", json={
            "name": f"cli-e2e-{runtime}-{suffix}", "role": "e2e tester", "provider": runtime, "model": model,
            "adapter_type": "cli", "cli_runtime": runtime, "config": config})
        pid = project["id"]
        task = api(client, "POST", f"/projects/{pid}/tasks", json={
            "title": "Create e2e marker",
            "description": f"Create a file named e2e_marker.txt in the current workspace directory containing exactly "
                           f"the text {token} , then reply with {token}."
                           + (" If you later receive the message 'continue' instead of a task, then WITHOUT reading any file "
                              "write e2e_resume.txt containing exactly the token you wrote into e2e_marker.txt, and reply with it."
                              if resume else "")})
        api(client, "POST", f"/projects/{pid}/tasks/{task['id']}/assign", json={"agent_id": agent["id"]})
        run = api(client, "POST", f"/projects/{pid}/tasks/{task['id']}/run", json={"timeout": timeout})
        sid = run["session_id"]
        try:
            session = wait_session(client, sid, timeout)
        except TimeoutError as exc:
            return "FAIL", str(exc)
        out = api(client, "GET", f"/sessions/{sid}/output")
        output = out.get("output") or session.get("output") or ""
        marker = workspace / "e2e_marker.txt"
        marker_ok = marker.exists() and token in marker.read_text(errors="replace")
        if session["status"] == "completed":
            if runtime == "custom":
                if marker_ok or token in output:
                    return "PASS", f"marker={'yes' if marker_ok else 'no'} output={'yes' if token in output else 'no'}"
            else:
                if marker_ok:
                    if resume and runtime in RESUME_RUNTIMES and extra_rows is not None:
                        extra_rows.extend(resume_checks(client, runtime, sid, workspace, token, timeout))
                    return "PASS", "marker=yes"
        return "FAIL", (f"status={session['status']} error={session.get('error')}\n"
                        f"marker_exists={marker.exists()}\n--- output (last 2KB) ---\n{output[-2048:]}")
    finally:
        if not keep:
            shutil.rmtree(workspace, ignore_errors=True)
            if script:
                script.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runtimes", default="claude_code,codex,custom", help=f"comma list of {','.join(BINARIES)}")
    ap.add_argument("--model", action="append", default=[], metavar="RUNTIME=MODEL", help="per-runtime model override (repeatable)")
    ap.add_argument("--effort", default=None, help="reasoning_effort for runtimes that support it")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--no-start-server", action="store_true")
    ap.add_argument("--keep", action="store_true", help="skip cleanup of temp workspaces and scripts")
    ap.add_argument("--resume", action="store_true", help="also run resume + poisoned-id fallback rows (claude_code, codex)")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    runtimes = [r.strip() for r in args.runtimes.split(",") if r.strip()]
    bad = [r for r in runtimes if r not in BINARIES]
    if bad:
        ap.error(f"unknown runtimes: {bad}")
    models = dict(DEFAULT_MODELS)
    for item in args.model:
        rt, _, m = item.partition("=")
        models[rt] = m

    base_url = f"http://localhost:{args.port}"
    proc = log = None
    server_already_running = check_server(base_url)

    if server_already_running:
        if args.no_start_server:
            print(f"WARNING: Using existing server at {base_url} (data goes into its database)")
        else:
            print(f"ERROR: port {args.port} in use; refusing to write test data to an unknown server")
            return 1
    else:
        if args.no_start_server:
            print(f"ERROR: No server at {base_url}")
            return 1
        logs = Path(__file__).resolve().parent / "logs"
        logs.mkdir(exist_ok=True)
        try:
            proc, log = start_server(base_url, args.port, PROJECT_ROOT, logs / f"server_cli_e2e_{int(time.time())}.log")
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            return 1

    results: list[tuple[str, str, str]] = []
    try:
        with httpx.Client(base_url=base_url, timeout=60) as client:
            for rt in runtimes:
                binary = BINARIES[rt]
                if binary and not shutil.which(binary):
                    results.append((rt, "SKIP", f"{binary} not on PATH"))
                    if args.resume:
                        results += [(f"{rt}/resume", "SKIP", "runtime unavailable"), (f"{rt}/poison", "SKIP", "runtime unavailable")]
                    continue
                print(f"== {rt} ...", flush=True)
                results_extra: list[tuple[str, str, str]] = []
                try:
                    status, detail = run_runtime(client, rt, models.get(rt, ""), args.effort, args.timeout, args.keep, args.resume, results_extra)
                except Exception as exc:  # noqa: BLE001
                    status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
                if status == "FAIL":
                    print(detail)
                results.append((rt, status, detail.splitlines()[0]))
                if args.resume:
                    if results_extra:
                        results += [(n, s, d.splitlines()[0]) for n, s, d in results_extra]
                    else:
                        why = "unsupported runtime" if rt not in RESUME_RUNTIMES else "first session did not pass"
                        results += [(f"{rt}/resume", "SKIP", why), (f"{rt}/poison", "SKIP", why)]
    finally:
        if proc:
            stop_server(proc, log)

    print("\nRUNTIME                RESULT  DETAIL")
    for rt, status, detail in results:
        print(f"{rt:<22} {status:<7} {detail}")
    return 1 if any(s == "FAIL" for _, s, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
