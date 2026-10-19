"""Dry-run driver: one real baseline through the CLI shims, measured and capped.

Spends real subscription tokens. Run from HuddleRoom/:

    HUDDLEROOM_LIVE_ACCEPTANCE=1 HUDDLEROOM_LIVE_EVAL=1 .venv/bin/python -m tests.live_acceptance.dry_run \
        --evidence-root <abs dir> --budget 60000 --i-understand-this-spends-tokens --step all

Step 2a proves the shims intercept the baseline's codex call, then pauses at once.
Step 2b resumes and lets the baseline run to its first human decision (or a stop condition).
Exit codes: 0 pass, 1 step failed, 3 refused before start (flag/env/isolation), 4 CLI config changed.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import inspect
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

if not __package__:  # path run: make `tests.live_acceptance` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from tests.live_acceptance import auth_dir_watch, fixtures, oracle
from tests.live_acceptance.cli_shim import TOOLS, make_shim, resolve_real
from tests.live_acceptance.live_env import server_env_for_cli
from tests.live_acceptance.run_dir import (
    _REPO_ROOT, IsolationError, RunDir, assert_isolated, build_run_dir, init_workspace_repo, real_home,
)
from tests.live_acceptance.seed import create_goal, seed_minimal_roster
from tests.live_acceptance.server_supervisor import Supervisor, free_port
from tests.live_acceptance.spend_ledger import Ledger

SCENARIO = "dry_run"
MAX_CALL_TOKENS = 25_000
MAX_PROCESSES = 6
WARN_FRACTION = 0.8
EXIT_FAIL, EXIT_REFUSED, EXIT_CONFIG = 1, 3, 4
TERMINAL_RUN_STATUSES = ("failed", "blocked", "completed", "cancelled")
_SECRET_KEY = re.compile(r"^[A-Z0-9_]*(KEY|TOKEN|SECRET)[A-Z0-9_]*$")


class StepError(RuntimeError):
    pass


def _leaks(text: str) -> list[str]:
    """Names of secret-looking env vars whose value appears in text, or env-style secret keys in JSON text."""
    bad = [k for k, v in os.environ.items() if _SECRET_KEY.match(k) and len(v) >= 8 and v in text]
    bad += [m for m in re.findall(r'"([A-Z0-9_]+)"\s*:', text) if _SECRET_KEY.match(m)]
    return bad


def _scrub(text: str) -> str:
    for k, v in os.environ.items():
        if _SECRET_KEY.match(k) and len(v) >= 8:
            text = text.replace(v, "[REDACTED]")
    return text


class Timing:
    """Wait limits in seconds; tests shrink them."""
    wait_2a = 300.0
    wait_2b = 1500.0
    poll = 0.25
    grace = 20.0
    watchdog = 120.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cli_version(real: Path) -> str:
    try:
        r = subprocess.run([str(real), "--version"], capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"error: {e}"


def _repo_head() -> str:
    r = subprocess.run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or f"error: {r.stderr.strip()}"


def _auth_roots() -> list[Path]:
    h = real_home()
    return [h / ".codex", h / ".claude", h / ".claude.json"]


def _ledger_usage(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("k") == "usage":
            rows.append(r)
    return rows


def _log_lines(shim_dir: Path) -> list[str]:
    p = shim_dir / "shim.log"
    return p.read_text(errors="replace").splitlines() if p.exists() else []


def _tok(r: dict) -> int:
    return r["input_tokens"] - r["cached_tokens"] + r["output_tokens"]


class Api:
    def __init__(self, base_url: str, project_id: str, goal_id: str):
        self.base = f"{base_url}/api/v1/projects/{project_id}/orchestration/goals/{goal_id}"
        self.calls: list[dict] = []

    def _do(self, method: str, path: str, body: dict | None = None, timeout: float = 60) -> httpx.Response:
        r = httpx.request(method, self.base + path, json=body, timeout=timeout)
        self.calls.append({"t": _now(), "method": method, "route": path, "status": r.status_code})
        return r

    def detail(self) -> dict:
        r = self._do("GET", "")
        r.raise_for_status()
        return r.json()

    def start_baseline(self) -> dict:
        r = self._do("POST", "/baseline/authorize")
        if r.status_code != 200:
            raise StepError(f"baseline/authorize returned {r.status_code}: {r.text[:500]}")
        return r.json()

    def pause(self) -> dict:
        """Read control_version, POST pause-control; on 409 re-read and retry once."""
        result: dict = {"ok": False, "attempts": []}
        for _ in range(2):
            try:
                run = self.detail().get("run") or {}
            except Exception as e:  # noqa: BLE001
                result["error"] = f"detail failed: {e}"
                return result
            version = run.get("control_version")
            if not run.get("id") or version is None:
                result["error"] = "no active run or control_version missing"
                return result
            request_id = str(uuid.uuid4())
            r = self._do("POST", "/pause-control", {
                "run_id": run["id"], "expected_control_version": version, "request_id": request_id})
            ctype = r.headers.get("content-type", "")
            result["attempts"].append({"status": r.status_code, "request_id": request_id,
                                       "body": r.json() if ctype.startswith("application/json") else r.text[:300]})
            if r.status_code == 200:
                result["ok"] = True
                break
            if r.status_code != 409:
                break
        return result

    def resume(self) -> int:
        return self._do("POST", "/resume").status_code

    def retry_async(self, process_type: str, timeout: float = 60) -> dict:
        """POST baseline/retry in a thread: it blocks while the retried model call is held by the shim."""
        import threading
        result: dict = {"done": False, "status": None}

        def go():
            try:
                result["status"] = self._do("POST", "/baseline/retry", {"process_type": process_type}, timeout).status_code
            except Exception as e:  # noqa: BLE001
                result["error"] = str(e)
            result["done"] = True
        t = threading.Thread(target=go, daemon=True)
        t.start()
        result["thread"] = t
        return result


def _invocation_report(shims: dict, shim_dir: Path, ledger_path: Path, orc: oracle.Oracle, goal_id: str) -> list[dict]:
    usage = {r["label"]: _tok(r) for r in _ledger_usage(ledger_path)}
    try:
        procs = orc.process_runs(goal_id=goal_id)
    except Exception:  # noqa: BLE001 - evidence only
        procs = []
    out = []
    for tool, handle in shims.items():
        for inv in handle.invocations():
            nxt = None
            for p in procs:
                try:
                    ts = datetime.fromisoformat(str(p["created_at"]).replace("Z", "+00:00"))
                    ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if ts.timestamp() >= inv.started_at - 2:
                    nxt = {"id": p["id"], "process_type": p["process_type"], "status": p["status"]}
                    break
            out.append({"tool": tool, "ordinal": inv.ordinal, "started_at": inv.started_at,
                        # the shim does not log stdin; the head of args is the best prompt marker here
                        "prompt_head": inv.args[:200], "refused": inv.refused, "timed_out": inv.timed_out,
                        "settled_tokens": usage.get(f"{tool} {inv.ordinal}"), "next_process_run": nxt})
    return sorted(out, key=lambda i: (i["started_at"], i["tool"], i["ordinal"]))


def _stop_reason(shim_dir: Path, ledger: Ledger, budget: int, orc: oracle.Oracle, goal_id: str,
                 ledger_path: Path, *, baseline_spent: int = 0, max_processes: int = MAX_PROCESSES,
                 tolerate_blocked: bool = False) -> str | None:
    lines = _log_lines(shim_dir)
    if any(l.startswith(("refused ", "gate_timeout ")) for l in lines):
        return "budget_refused_or_gate_timeout"
    if any(p[0] == "gate" and p[1] == "claude" for p in (l.split(" ", 5) for l in lines) if len(p) >= 2):
        return "unexpected_claude_call"  # never released: the baseline must only use codex
    if any(int(p[4]) > HOLD_SPAN + 1 for p in (l.split(" ", 5) for l in lines)
           if len(p) >= 5 and p[0] == "start" and p[1] == "codex" and p[4].isdigit()):
        return "ordinal_beyond_hold_span"
    if any(_tok(r) > MAX_CALL_TOKENS for r in _ledger_usage(ledger_path)):
        return "single_call_over_25k"
    if ledger.totals()["global"]["non_cached"] - baseline_spent >= WARN_FRACTION * budget:
        return "budget_80_percent"
    decisions = orc.decisions(goal_id)
    if any(d["status"] == "pending" for d in decisions):
        return "pending_decision"
    for r in orc.runs(goal_id):
        if r.get("status") in TERMINAL_RUN_STATUSES and not (tolerate_blocked and r["status"] == "blocked"):
            return f"run_{r['status']}"
    if len(orc.process_runs(goal_id=goal_id)) > max_processes:
        return "over_6_processes_before_decision"
    return None


def _wait(timeout: float, check) -> object:
    deadline = time.monotonic() + timeout
    while True:
        got = check()
        if got:
            return got
        if time.monotonic() >= deadline:
            return None
        time.sleep(Timing.poll)


HOLD_SPAN = 20  # ordinals 2..HOLD_SPAN+1 are held upfront in step 2b
PASS_REASONS = ("pending_decision", "max_releases", "extra_cap")


def _gate_ordinals(lines: list[str]) -> set[int]:
    return {int(p[4]) for p in (l.split(" ", 5) for l in lines)
            if len(p) >= 5 and p[0] == "gate" and p[1] == "codex" and p[4].isdigit()}


def _settled_ordinals(lines: list[str]) -> set[int]:
    return {int(p[4]) for p in (l.split(" ", 5) for l in lines)
            if len(p) >= 5 and p[0] == "settled" and p[1] == "codex" and p[4].isdigit()}


def _start_args(lines: list[str], ordinal: int, tool: str = "codex") -> str:
    for l in lines:
        p = l.split(" ", 5)
        if len(p) >= 5 and p[0] == "start" and p[1] == tool and p[4] == str(ordinal):
            return p[5] if len(p) == 6 else ""
    return ""


def _schema_head(args: str) -> str | None:
    m = re.search(r"--output-schema (\S+)", args)
    try:
        return Path(m.group(1)).read_text(errors="replace")[:200] if m else None
    except OSError:
        return None


def _call_row(lines, orc, goal_id, n, phase, tool="codex") -> dict:
    procs = orc.process_runs(goal_id=goal_id)
    running = [p for p in procs if p.get("status") == "running"] or procs[-1:]
    args = _start_args(lines, n, tool)
    return {"tool": tool, "ordinal": n, "phase": phase, "args_head": args[:200], "schema_head": _schema_head(args),
            "process_type": running[-1].get("process_type") if running else None,
            "process_runs": [[p.get("process_type"), p.get("status")] for p in procs],
            "pending_decisions": sum(1 for d in orc.decisions(goal_id) if d["status"] == "pending"),
            "settled_tokens": None, "process_type_best_effort": True}


def _stepped_loop(shim_dir, release, ledger, orc, goal_id, spend, budget, findings, *,
                  max_releases, extra_releases=0, continue_to_decision=False, extra_cap=15000, wd=None) -> str:
    """Let goal_definition (ordinal 1) run, then release held codex calls one at a time.

    Returns the stop reason: pending_decision, max_releases, extra_cap, timeout, or a hard stop from
    _stop_reason. Never pauses; the caller does.
    """
    rows = findings.setdefault("model_calls", [])

    def stop():
        return _stop_reason(shim_dir, ledger, budget, orc, goal_id, spend) or (wd.check() if wd else None)

    def spent():
        return ledger.totals()["global"]["non_cached"]

    def tokens(n):
        return {r["label"]: _tok(r) for r in _ledger_usage(spend)}.get(f"codex {n}")

    got = _wait(Timing.wait_2b, lambda: "ok" if 1 in _settled_ordinals(_log_lines(shim_dir)) else stop())
    row = _call_row(_log_lines(shim_dir), orc, goal_id, 1, "goal_definition")
    row["settled_tokens"] = tokens(1)
    rows.append(row)
    if got != "ok":
        return got or "timeout"
    total = max_releases + (extra_releases if continue_to_decision else 0)
    extra_start = None
    for i in range(total):
        n = 2 + i
        extra = i >= max_releases
        if extra and extra_start is None:
            extra_start = spent()
        got = _wait(Timing.wait_2b, lambda: stop() or ("held" if n in _gate_ordinals(_log_lines(shim_dir)) else None))
        if got != "held":
            return got or "timeout"
        if extra and spent() - extra_start >= extra_cap:
            return "extra_cap"
        row = _call_row(_log_lines(shim_dir), orc, goal_id, n, "extra" if extra else "stepped")
        if any(r["process_type"] == row["process_type"] for r in rows[1:]):
            row["retry_detected"] = True
            findings["retry_detected"] = True
        rows.append(row)
        release(f"codex.nth.{n}")  # released right after the row is recorded
        got = _wait(Timing.wait_2b, lambda: "ok" if n in _settled_ordinals(_log_lines(shim_dir)) else stop())
        row["settled_tokens"] = tokens(n)
        if got != "ok":
            return got or "timeout"
    if extra_start is not None and spent() - extra_start >= extra_cap:
        return "extra_cap"
    n = 2 + total
    got = _wait(Timing.grace, lambda: stop() or ("held" if n in _gate_ordinals(_log_lines(shim_dir)) else None))
    return got if got not in (None, "held") else "max_releases"


def _decision_report(orc, goal_id) -> list[dict]:
    return [{"id": d["id"], "status": d["status"], "keys": sorted(d), "options": d.get("options"),
             "has_request_changes": "request_changes" in json.dumps(d.get("options"), default=str),
             "source_process_run_id": d.get("source_process_run_id")} for d in orc.decisions(goal_id)]


def _weight_info(api, orc, goal_id) -> dict:
    goal = api.detail().get("goal", {})
    info = {"goal_fields": {k: v for k, v in goal.items() if "weight" in k}, "classified": []}
    for p in orc.process_runs(goal_id=goal_id):
        if p.get("process_type") == "goal_definition":
            out = p.get("outputs") or {}
            info["classified"].append({"weight_keys": {k: v for k, v in out.items() if "weight" in k},
                                       "outputs_head": json.dumps(out, default=str)[:800]})
    return info


def _body_2b(args, rd, api, orc, ledger, codex, ids, goal_id, findings, budget) -> int:
    t0 = time.monotonic()
    findings["weight"] = _weight_info(api, orc, goal_id)
    for n in range(2, 2 + HOLD_SPAN):  # armed before authorize
        codex.hold_nth(n)
    ids["run_id"] = (api.start_baseline().get("run") or {}).get("id")
    wd = _make_watchdog(rd, orc, goal_id, ids["run_id"], findings)
    reason = _stepped_loop(rd.shim, codex.release, ledger, orc, goal_id, rd.spend, budget, findings,
                           max_releases=args.max_releases, extra_releases=args.extra_releases,
                           continue_to_decision=args.continue_to_decision, extra_cap=args.extra_cap, wd=wd)
    findings["pause_2b"] = api.pause()  # pause first, then snapshot (before any teardown deny/kill)
    findings["decisions"] = _decision_report(orc, goal_id)
    state = _cleanliness(orc, goal_id, ids["run_id"])
    findings["snapshot_state"] = state
    if reason in ("max_releases", "extra_cap"):  # a call is still held: capture it before teardown
        findings["snap_held_call"] = _snapshot_dir(rd, "snap-paused-at-held-call", state)
    snap = None
    if reason == "pending_decision":
        snap = oracle.snapshot(rd.db, rd.snapshots / "pending-decision.db")
        with tarfile.open(rd.snapshots / "pending-decision-workspace.tar.gz", "w:gz") as tf:
            tf.add(rd.workspace, arcname="workspace")
    ok = reason in PASS_REASONS and findings["pause_2b"]["ok"]
    findings["steps"]["2b"] = {"stop_reason": reason, "wall_s": round(time.monotonic() - t0, 1),
                               "snapshot_sha256": snap, "clean": state["clean"], "model_calls": len(findings.get("model_calls", [])),
                               "pass": ok}
    if not ok:
        findings["stop_reason"] = reason
    return 0 if ok else EXIT_FAIL


def _seed_and_goal(step, base, rd):
    if step == "2a":
        ids = seed_minimal_roster(base, rd)
        return ids, create_goal(base, ids["project_id"], fixtures.GOAL_TEXT, goal_type="outcome")
    for fn, name in ((seed_minimal_roster, "roster"), (create_goal, "weight")):
        if name not in inspect.signature(fn).parameters:
            raise StepError(f"{fn.__name__} has no {name!r} parameter (seed update missing)")
    ids = seed_minimal_roster(base, rd, roster="discovery", blank_definition_key="dev1")
    return ids, create_goal(base, ids["project_id"], fixtures.GOAL_TEXT, goal_type="outcome", weight="substantial")


ADR_KIND = "agent_definition_review_analyzer_error"
NO_PROGRESS = "no_progress"


def _blockers(orc, run_id) -> list:
    raw = (orc.run_state(run_id) or {}).get("active_blockers") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = [raw]
    return list(raw)


def _blocker_kinds(orc, run_id) -> list[str]:
    return [b.get("kind", "?") if isinstance(b, dict) else str(b) for b in _blockers(orc, run_id)]


def _lm_retry_processes(orc, goal_id) -> list[str]:
    return [p["process_type"] for p in orc.process_runs(goal_id=goal_id)
            if "_lm_retry" in json.dumps(p.get("outputs"), default=str)]


def _cleanliness(orc, goal_id, run_id) -> dict:
    kinds, retry = _blocker_kinds(orc, run_id), _lm_retry_processes(orc, goal_id)
    return {"clean": not kinds and not retry, "blocker_kinds": kinds, "lm_retry_processes": retry}


def _cleanliness_db(db_path, goal_id, run_id) -> dict:
    """Same as _cleanliness but straight from sqlite, before any server exists. Unknown counts as dirty."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            raw = con.execute("SELECT active_blockers FROM orchestration_runs WHERE id = ?",
                              (uuid.UUID(str(run_id)).hex,)).fetchone()[0]
            kinds = [b.get("kind", "?") if isinstance(b, dict) else str(b) for b in json.loads(raw or "[]")]
            retry = [t for t, o in con.execute("SELECT process_type, outputs FROM orchestration_process_runs "
                                               "WHERE goal_id = ?", (uuid.UUID(str(goal_id)).hex,))
                     if "_lm_retry" in str(o)]
        finally:
            con.close()
        return {"clean": not kinds and not retry, "blocker_kinds": kinds, "lm_retry_processes": retry}
    except Exception as e:  # noqa: BLE001
        return {"clean": None, "blocker_kinds": [], "unknown": str(e)}


def _active_warnings(db_path, goal_id) -> list[dict] | dict:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = con.execute("SELECT warning_type, severity, message FROM orchestration_warnings "
                              "WHERE goal_id = ? AND active = 1", (uuid.UUID(str(goal_id)).hex,))
            return [{"warning_type": a, "severity": b, "message": str(c)[:300]} for a, b, c in cur.fetchall()]
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001 - evidence only
        return {"error": str(e)}


class Watchdog:
    """Fails fast when nothing changes (process rows, shim model-call lines, non-noop actions)."""

    def __init__(self, state_fn, dump_fn, timeout=None):
        self.state_fn, self.dump_fn = state_fn, dump_fn
        self.timeout = Timing.watchdog if timeout is None else timeout
        self.last, self.since = None, time.monotonic()

    def check(self):
        state = self.state_fn()
        now = time.monotonic()
        if state != self.last:
            self.last, self.since = state, now
            return None
        if now - self.since >= self.timeout:
            self.dump_fn()
            return NO_PROGRESS
        return None


def _make_watchdog(rd, orc, goal_id, run_id, findings) -> Watchdog:
    def state():
        lines = sum(1 for l in _log_lines(rd.shim) if l.startswith(("start ", "gate ", "settled ")))
        acts = sum(1 for a in orc.actions(run_id) if a.get("action_type") != "noop")
        return (len(orc.process_runs(goal_id=goal_id)), lines, acts)

    def dump():
        findings["no_progress"] = {
            "active_blockers": _blockers(orc, run_id), "warnings": _active_warnings(rd.db, goal_id),
            "process_output_keys": {f"{i}:{p['process_type']}": sorted((p.get("outputs") or {}))
                                    for i, p in enumerate(orc.process_runs(goal_id=goal_id))}}
    return Watchdog(state, dump)


def _tool_events(lines: list[str], event: str, tool: str) -> list[int]:
    return [int(p[4]) for p in (l.split(" ", 5) for l in lines)
            if len(p) >= 5 and p[0] == event and p[1] == tool and p[4].isdigit()]


def _resume_loop(shim_dir, handles, ledger, orc, goal_id, spend, budget, findings, *, max_releases,
                 baseline_spent=0, max_processes=MAX_PROCESSES, wd=None, tolerate_blocked=False) -> str:
    """Release held model calls (any tool) strictly one at a time, in the order they blocked.

    Stops at a pending human decision, max_releases, or any _stop_reason. Never pauses.
    """
    rows = findings.setdefault("model_calls", [])
    released: set[tuple[str, int]] = set()

    def stop():
        return _stop_reason(shim_dir, ledger, budget, orc, goal_id, spend, baseline_spent=baseline_spent,
                            max_processes=max_processes, tolerate_blocked=tolerate_blocked) or (
            wd.check() if wd else None)

    def next_held():
        for l in _log_lines(shim_dir):
            p = l.split(" ", 5)
            if len(p) >= 5 and p[0] == "gate" and p[4].isdigit() and (p[1], int(p[4])) not in released:
                return (p[1], int(p[4]))
        return None

    def tokens(tool, n):
        return {r["label"]: _tok(r) for r in _ledger_usage(spend)}.get(f"{tool} {n}")

    for _ in range(max_releases):
        got = _wait(Timing.wait_2b, lambda: stop() or next_held())
        if not isinstance(got, tuple):
            return got or "timeout"
        tool, n = got
        row = _call_row(_log_lines(shim_dir), orc, goal_id, n, "resume", tool)
        if any(r["process_type"] == row["process_type"] for r in rows):
            row["retry_detected"] = True
            findings["retry_detected"] = True
        rows.append(row)
        released.add(got)
        handles[tool].release(f"{tool}.nth.{n}")  # only this call
        settled = _wait(Timing.wait_2b, lambda: "ok" if n in _tool_events(_log_lines(shim_dir), "settled", tool)
                        else stop())
        row["settled_tokens"] = tokens(tool, n)
        if settled != "ok":
            return settled or "timeout"
    return _wait(Timing.grace, stop) or "max_releases"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_dir(rd, name: str, state: dict | None = None) -> dict:
    """DB (sqlite backup) + workspace tar + sha256 manifest under <run>/snapshots/<name>/."""
    out = rd.snapshots / name
    out.mkdir(parents=True, exist_ok=False)
    canonical = oracle.snapshot(rd.db, out / "app.db")
    tar = out / "workspace.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(rd.workspace, arcname="workspace")
    manifest = {"created_at": _now(), "db_canonical_sha256": canonical, "db_file_sha256": _sha256(out / "app.db"),
                "workspace_tar_sha256": _sha256(tar),
                **(state or {"clean": None, "blocker_kinds": []})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class Refused(RuntimeError):
    pass


def _alive_shim_processes(rd) -> list[str]:
    try:
        ps = subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True).stdout
    except OSError as e:
        raise Refused(f"cannot inspect processes: {e}") from None
    return [l.strip() for l in ps.splitlines() if str(rd.root) in l and str(os.getpid()) != l.split()[0]]


def _open_run(args) -> tuple[RunDir, dict]:
    """Open an existing paused run dir for step 2c. Raises Refused; touches nothing."""
    evidence = Path(args.evidence_root).expanduser().resolve()
    root = Path(args.resume_run).expanduser().resolve()
    if not root.is_dir() or not root.is_relative_to(evidence) or root == evidence:
        raise Refused(f"run dir {root} is not inside evidence root {evidence}")
    rd = RunDir(root=root, db=root / "app.db", workspace=root / "workspace", snapshots=root / "snapshots",
                shim=root / "shim", proxy_log=root / "proxy.jsonl", spend=root / "spend.jsonl",
                config=root / "config.json")
    try:
        assert_isolated(rd, rd.db, rd.workspace)
    except IsolationError as e:
        raise Refused(str(e)) from None
    for d in (rd.db, rd.workspace, rd.shim):
        if not d.exists():
            raise Refused(f"{d} missing")
    rd.snapshots.mkdir(exist_ok=True)
    conn = sqlite3.connect(f"file:{rd.db}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT r.id, r.goal_id, r.status, r.pause_requested, g.project_id "
                           "FROM orchestration_runs r JOIN orchestration_goals g ON g.id = r.goal_id "
                           "ORDER BY r.rowid DESC LIMIT 1").fetchone()
    except sqlite3.Error as e:
        raise Refused(f"cannot read run state: {e}") from None
    finally:
        conn.close()
    if row is None or row[2] != "paused" or not row[3]:
        raise Refused(f"run is not paused with pause_requested=1: {row}")
    alive = _alive_shim_processes(rd)
    if alive:
        raise Refused(f"processes still using this run dir: {alive[:3]}")
    u = lambda h: str(uuid.UUID(h)) if len(str(h)) == 32 else str(h)
    return rd, {"run_id": u(row[0]), "goal_id": u(row[1]), "project_id": u(row[4])}


def _rewrite_ledger(path: Path, budget: int, scenario: str) -> dict:
    """Continue the existing ledger: settle stale holds, then reset the caps row to prior spend + budget."""
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    prior = 0
    if rows:
        old = Ledger(path, rows[0]["global"], rows[0]["scenarios"])
        old.reap()
        prior = old.totals()["global"]["non_cached"]
        shutil.copy(path, path.with_name(f"{path.name}.prev-{int(time.time())}"))
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()][1:]
    cap = prior + budget
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in
                           [{"k": "caps", "global": cap, "scenarios": {scenario: budget}}, *rows]))
    os.replace(tmp, path)
    return {"prior_spent": prior, "step_budget": budget, "global_cap": cap, "scenario": scenario}


def _body_2c(args, rd, api, orc, ledger, shims, ids, goal_id, findings, budget, lcfg) -> int:
    t0 = time.monotonic()
    phase = (orc.run_state(ids["run_id"]) or {}).get("phase")
    if phase != "baseline":
        raise StepError(f"run phase is {phase!r}, not 'baseline'; refusing to resume")
    prior_procs = len(orc.process_runs(goal_id=goal_id))
    for h in shims.values():  # every model call is held, armed before resume
        for n in range(1, 1 + HOLD_SPAN):
            h.hold_nth(n)
    status = api.resume()
    if status != 200:
        raise StepError(f"resume returned {status}")
    api.start_baseline()  # raises unless 200
    kinds, retry_procs = _blocker_kinds(orc, ids["run_id"]), _lm_retry_processes(orc, goal_id)
    findings["blockers_after_resume"] = {"kinds": kinds, "lm_retry_processes": retry_procs}
    retry = None
    if ADR_KIND in kinds or retry_procs:
        if not getattr(args, "allow_retry", True):
            raise StepError(f"run is blocked ({kinds or retry_procs}) and --no-allow-retry was given")
        ptype = "agent_definition_review" if ADR_KIND in kinds else retry_procs[0]
        retry = api.retry_async(ptype)  # blocks server-side while the retried call is held
        findings["retry"] = {"process_type": ptype}
    wd = _make_watchdog(rd, orc, goal_id, ids["run_id"], findings)
    reason = _resume_loop(rd.shim, shims, ledger, orc, goal_id, rd.spend, budget, findings,
                          max_releases=args.max_releases, baseline_spent=lcfg["prior_spent"],
                          max_processes=prior_procs + MAX_PROCESSES, wd=wd, tolerate_blocked=retry is not None)
    findings["pause_2c"] = api.pause()
    findings["decisions"] = _decision_report(orc, goal_id)
    state = _cleanliness(orc, goal_id, ids["run_id"])
    snap = _snapshot_dir(rd, "snap-B-first-pending-decision", state) if reason == "pending_decision" else None
    if retry is not None:
        retry["thread"].join(5)
        findings["retry"].update(done=retry["done"], status=retry["status"], error=retry.get("error"))
        if retry["done"] and not (retry["status"] and 200 <= retry["status"] < 300):
            raise StepError(f"baseline/retry returned {retry['status']} {retry.get('error', '')}")
    ok = reason in ("pending_decision", "max_releases") and findings["pause_2c"]["ok"]
    findings["steps"]["2c"] = {"stop_reason": reason, "wall_s": round(time.monotonic() - t0, 1),
                               "snap_b": snap, "snap_b_dirty": bool(snap) and not state["clean"],
                               "model_calls": len(findings.get("model_calls", [])), "pass": ok}
    if not ok:
        findings["stop_reason"] = reason
    return 0 if ok else EXIT_FAIL


def _skeleton_md(f: dict) -> str:
    return (
        "# Dry-run findings\n\n"
        f"Steps: {json.dumps(f['steps'], indent=2)}\n\n"
        "## To fill for the later dry-run steps\n"
        "- Goal shape that yields a planner call:\n"
        "- request_changes path (UI route or API-only gap):\n"
        "- Saved banner rendered while paused after restart:\n"
        "- Pause key / event strings (see pause_strings in dry-run-findings.json):\n"
        "- 120 s product timeout hits (see timeout_hints in dry-run-findings.json):\n"
    )


def run(args: argparse.Namespace, real_paths: dict[str, Path] | None = None) -> int:
    env = os.environ
    if not args.i_understand_this_spends_tokens:
        print("refused: --i-understand-this-spends-tokens is required", file=sys.stderr)
        return EXIT_REFUSED
    if env.get("HUDDLEROOM_LIVE_ACCEPTANCE") != "1" or env.get("HUDDLEROOM_LIVE_EVAL") != "1":
        print("refused: set HUDDLEROOM_LIVE_ACCEPTANCE=1 and HUDDLEROOM_LIVE_EVAL=1", file=sys.stderr)
        return EXIT_REFUSED
    if args.budget <= 0:
        print("refused: --budget must be positive", file=sys.stderr)
        return EXIT_REFUSED
    if args.max_releases is None:
        args.max_releases = 4 if args.step == "2c" else 2
    if args.step == "2c" and not args.resume_run:
        print("refused: --step 2c needs --resume-run", file=sys.stderr)
        return EXIT_REFUSED
    if args.step != "2c" and args.resume_run:
        print("refused: --resume-run only works with --step 2c", file=sys.stderr)
        return EXIT_REFUSED
    if args.max_releases < 0 or args.extra_releases < 0 or args.max_releases + args.extra_releases > HOLD_SPAN:
        print(f"refused: max-releases + extra-releases must be 0..{HOLD_SPAN}", file=sys.stderr)
        return EXIT_REFUSED
    budget = args.budget
    if args.step == "2c":
        return _run_step("2c", args, real_paths, budget)[0]
    if args.step in ("2a", "all"):
        code, spent = _run_step("2a", args, real_paths, budget)
        budget -= spent
        if code != 0 or args.step == "2a":
            return code
        if budget <= 0:
            print("refused: no budget left for 2b", file=sys.stderr)
            return EXIT_REFUSED
    return _run_step("2b", args, real_paths, budget)[0]


def _run_step(step: str, args: argparse.Namespace, real_paths, budget: int) -> tuple[int, int]:
    SCENARIO = {"2a": "dry_run", "2b": "dry_run_2b", "2c": "dry_run_2c"}[step]
    resumed_ids: dict = {}
    try:
        if step == "2c":
            rd, resumed_ids = _open_run(args)
        else:
            rd = build_run_dir(Path(args.evidence_root), "dry-run" if step == "2a" else "dry-run-2b")
    except (IsolationError, ValueError, Refused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED, 0

    findings: dict = {"step": step, "started_at": _now(), "budget": budget, "steps": {}, "stop_reason": None,
                      "errors": [], "api_calls": [],
                      "to_fill": {"goal_shape": None, "request_changes_path": None,
                                  "saved_render_while_paused": None}}
    code = 0
    sup = shims = ledger = auth_before = None
    api = None
    orc = None
    ids: dict = {}
    try:
        if step == "2c":  # first, before any shim or server: snapshot the paused state (no tokens)
            existing = rd.snapshots / "snap-A-paused-before-call4"
            if existing.exists():  # never overwrite; legacy snap-A has no clean flag, so it counts as dirty
                man = json.loads((existing / "manifest.json").read_text()) if (existing / "manifest.json").exists() else {}
                findings["snap_a"] = {"exists": True, "clean": bool(man.get("clean")), **man}
            else:
                findings["snap_a"] = _snapshot_dir(
                    rd, "snap-A-paused-before-call4", _cleanliness_db(rd.db, resumed_ids["goal_id"], resumed_ids["run_id"]))
        else:
            rd.workspace.rmdir()  # build_run_dir pre-creates it; init_workspace_repo wants to mkdir it
            init_workspace_repo(rd.workspace, rd)
    except IsolationError as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED, 0
    except FileExistsError as e:
        print(f"refused: snapshot already exists: {e}", file=sys.stderr)
        return EXIT_REFUSED, 0
    auth_before = auth_dir_watch.listing(_auth_roots())
    auth_dir_watch.write_listing(rd.root / "auth-before.json", auth_before)
    try:
        if step == "2c":
            lcfg = findings["ledger"] = _rewrite_ledger(rd.spend, budget, SCENARIO)
            ledger = Ledger(rd.spend, lcfg["global_cap"], {SCENARIO: budget})
            global_cap = lcfg["global_cap"]
            for sub in ("gates", "counters"):  # fresh gates and counters for the restart
                for f in (rd.shim / sub).glob("*") if (rd.shim / sub).is_dir() else ():
                    f.unlink(missing_ok=True)
            log = rd.shim / "shim.log"
            if log.exists():
                log.rename(rd.shim / f"shim.log.prev-{int(time.time())}")
        else:
            ledger = Ledger(rd.spend, budget, {SCENARIO: budget})
            global_cap = budget
        ledger.reap()
        shims = {}
        for tool in TOOLS:
            real = (real_paths or {}).get(tool) or resolve_real(tool, rd.shim)
            shims[tool] = make_shim(rd.shim, tool, Path(real), ledger_path=rd.spend, scenario=SCENARIO,
                                    global_cap=global_cap, scenario_cap=budget,
                                    output_ceiling=args.output_ceiling)
            shims[tool].real = Path(real)  # type: ignore[attr-defined]
        pre = {"repo_head": _repo_head(),
               "cli_versions": {t: _cli_version(h.real) for t, h in shims.items()},  # type: ignore[attr-defined]
               "orch_model": fixtures.ORCH_CLI_MODEL, "orch_effort": fixtures.ORCH_EFFORT,
               "agent_model": fixtures.AGENT_MODEL, "goal_text": fixtures.GOAL_TEXT,
               "shim_paths": {t: str(h.path) for t, h in shims.items()},
               "real_paths": {t: str(h.real) for t, h in shims.items()},  # type: ignore[attr-defined]
               "budget": budget, "scenario": SCENARIO, "output_ceiling": args.output_ceiling}
        sup = Supervisor(rd, args.port or free_port(), env_overrides=server_env_for_cli(), shim_dir=rd.shim,
                         extra_pid_dirs=sorted({h.shim_dir / "pids" for h in shims.values()}))
        sup.start()
        cfg = json.loads(rd.config.read_text()) if rd.config.exists() else {}
        rd.config.write_text(json.dumps({**cfg, **pre}, indent=2))
        findings["preflight"] = pre
        if step == "2c":
            ids = {"project_id": resumed_ids["project_id"], "run_id": resumed_ids["run_id"]}
            goal_id = resumed_ids["goal_id"]
        else:
            ids, goal = _seed_and_goal(step, sup.base_url, rd)
            oracle.snapshot(rd.db, rd.snapshots / "00-before.db")
            goal_id = (goal.get("goal") or goal)["id"]
        ids["goal_id"] = goal_id
        api = Api(sup.base_url, ids["project_id"], goal_id)
        orc = oracle.Oracle(rd.db)
        codex = shims["codex"]

        if step == "2a":
            # ---- step 2a: shim proof
            t0 = time.monotonic()
            run_id = (api.start_baseline().get("run") or {}).get("id")
            ids["run_id"] = run_id

            wd = _make_watchdog(rd, orc, goal_id, run_id, findings)

            def check_2a():
                reason = _stop_reason(rd.shim, ledger, budget, orc, goal_id, rd.spend) or wd.check()
                if reason:
                    return reason
                return "ok" if any(l.startswith("settled codex ") for l in _log_lines(rd.shim)) else None

            got = _wait(Timing.wait_2a, check_2a)
            lines = _log_lines(rd.shim)
            a = {"model_call_settled": any(l.startswith("settled codex ") for l in lines),
                 "login_status_passthrough": any(l.startswith("passthrough codex ") for l in lines),
                 "codex_invocations": len(codex.invocations()), "claude_invocations": len(shims["claude"].invocations()),
                 "wall_s": round(time.monotonic() - t0, 1), "outcome": got or "timeout"}
            findings["pause_2a"] = api.pause()
            a["pause_ok"] = findings["pause_2a"]["ok"]
            a["pass"] = bool(a["model_call_settled"] and a["login_status_passthrough"] and a["outcome"] == "ok"
                             and a["pause_ok"])
            findings["steps"]["2a"] = a
            if not a["pass"]:
                findings["stop_reason"] = f"2a_{a['outcome']}" if a["outcome"] != "ok" else "2a_failed"
                code = EXIT_FAIL

        elif step == "2c":
            code = _body_2c(args, rd, api, orc, ledger, shims, ids, goal_id, findings, budget, lcfg)
        else:
            code = _body_2b(args, rd, api, orc, ledger, codex, ids, goal_id, findings, budget)
    except BaseException as e:  # noqa: BLE001 - always write findings, then fail
        findings["errors"].append(f"{type(e).__name__}: {e}")
        code = EXIT_FAIL
        if api is not None and sup is not None:  # best effort: stop further spend while the server is still up
            try:
                findings["pause_error_path"] = api.pause()
            except BaseException as e2:  # noqa: BLE001
                findings["errors"].append(f"pause on error path: {type(e2).__name__}: {e2}")
    finally:
        def attempt(label, fn):
            try:
                fn()
            except BaseException as e:  # noqa: BLE001 - each cleanup step is independent
                findings["errors"].append(f"cleanup {label}: {type(e).__name__}: {e}")

        def collect():
            findings["api_calls"] = api.calls if api is not None else []
            if orc is not None and ids.get("goal_id"):
                findings["invocations"] = _invocation_report(shims, rd.shim, rd.spend, orc, ids["goal_id"])
                findings["process_runs"] = [{k: p[k] for k in ("id", "process_type", "status", "created_at")}
                                            for p in orc.process_runs(goal_id=ids["goal_id"])]
                if ids.get("run_id"):
                    findings["pause_strings"] = {
                        "idempotency_keys": [a["idempotency_key"] for a in orc.pause_control(ids["run_id"])],
                        "events": sorted({e["event_type"] for e in orc.event_log(ids["project_id"])
                                          if "pause" in e["event_type"]}),
                    }
                orc.dump("dry-run-end", rd.snapshots, run_id=ids.get("run_id"), goal_id=ids["goal_id"],
                         project_id=ids["project_id"])

        def hints():
            findings["timeout_hints"] = sum(
                1 for l in sup.log_path.read_text(errors="replace").splitlines()
                if re.search(r"timed out|timeout", l, re.I)) if sup.log_path.exists() else 0
            if findings["timeout_hints"]:
                findings["retry_detected"] = True  # a product timeout line implies a retry or failed call

        def release():
            for h in (shims or {}).values():
                h.release_all()

        def reap():
            if ledger is not None:
                findings["ledger_reaped"] = ledger.reap()
                findings["tokens"] = ledger.totals()

        def auth_after():
            nonlocal code
            after = auth_dir_watch.listing(_auth_roots())
            auth_dir_watch.write_listing(rd.root / "auth-after.json", after)
            cls = auth_dir_watch.classify(auth_dir_watch.diff(auth_before, after))
            findings["auth_diff"] = cls
            if cls["config_changed"]:
                print(f"FAIL: CLI config changed: {cls['config_changed']}", file=sys.stderr)
                code = EXIT_CONFIG

        # order matters: deny new shim calls, kill live ones, kill the server, and only then drop the gates
        for tool, h in (shims or {}).items():
            attempt(f"{tool} shim deny", h.deny)
        for tool, h in (shims or {}).items():
            attempt(f"{tool} shim kill_live", h.kill_live)
        if sup is not None:
            attempt("server kill", sup.kill)
            attempt("timeout hints", hints)
        attempt("shim release", release)

        def no_leftovers():
            left = {t: h.live_pids() for t, h in (shims or {}).items() if h.live_pids()}
            if left:
                raise RuntimeError(f"shim/CLI processes still alive after teardown: {left}")
        attempt("no leftover processes", no_leftovers)
        attempt("oracle collect", collect)
        if orc is not None:
            attempt("oracle close", orc.close)
        attempt("ledger reap", reap)
        attempt("auth after", auth_after)

        def write_findings():
            nonlocal code
            if findings["errors"] and code == 0:
                code = EXIT_FAIL
            findings["exit_code"] = code
            findings["finished_at"] = _now()
            texts = {"dry-run-findings.json": json.dumps(findings, indent=2, default=str),
                     "findings.md": _skeleton_md(findings),
                     "config.json": rd.config.read_text() if rd.config.exists() else ""}
            leaks = {n: _leaks(t) for n, t in texts.items() if _leaks(t)}
            if leaks:
                findings["errors"].append(f"secret-looking content scrubbed: {leaks}")
                findings["exit_code"] = code = code or EXIT_FAIL
                texts["dry-run-findings.json"] = json.dumps(findings, indent=2, default=str)
            for n, t in texts.items():
                if n == "config.json" and not t:
                    continue
                (rd.root / n).write_text(_scrub(t))

        attempt("write findings", write_findings)
    print(f"run dir: {rd.root}\nexit {code}; steps: {json.dumps(findings['steps'])}")
    return code, findings.get("tokens", {}).get("global", {}).get("non_cached", 0)


def main(argv: list[str] | None = None, real_paths: dict[str, Path] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--evidence-root", required=True)
    p.add_argument("--budget", type=int, required=True, help="hard non-cached token cap")
    p.add_argument("--i-understand-this-spends-tokens", action="store_true")
    p.add_argument("--step", choices=("2a", "2b", "2c", "all"), default="2a")
    p.add_argument("--resume-run", help="existing paused run dir (with --step 2c)")
    p.add_argument("--allow-retry", action=argparse.BooleanOptionalAction, default=True,
                   help="2c: POST baseline/retry when resume finds an analyzer-error blocker")
    p.add_argument("--output-ceiling", type=int, default=3000)
    p.add_argument("--max-releases", type=int, default=None, help="default 2 (2b) or 4 (2c)")
    p.add_argument("--continue-to-decision", action="store_true")
    p.add_argument("--extra-releases", type=int, default=1)
    p.add_argument("--extra-cap", type=int, default=15000)
    p.add_argument("--port", type=int, default=0)
    return run(p.parse_args(argv), real_paths)


if __name__ == "__main__":
    sys.exit(main())
