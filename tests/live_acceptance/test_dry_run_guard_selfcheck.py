"""Zero-token guards for the dry-run driver, using fake codex/claude binaries."""

import ast
import json
from pathlib import Path
import stat
import sys

import pytest

from tests.live_acceptance import dry_run

pytestmark = pytest.mark.live

FAKE = """#!/bin/sh
case "$*" in
  *--version*) echo "fake 1.0" ;;
  *login*) echo "Logged in using ChatGPT" >&2 ;;
  *) cat >/dev/null
     echo '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}'
     echo '{"type":"result","usage":{"input_tokens":10,"output_tokens":5}}' ;;
esac
"""


@pytest.fixture
def fakes(tmp_path):
    out = {}
    for tool in ("codex", "claude"):
        p = tmp_path / "fake" / tool
        p.parent.mkdir(exist_ok=True)
        p.write_text(FAKE)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        out[tool] = p
    return out


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("HUDDLEROOM_LIVE_ACCEPTANCE", "1")
    monkeypatch.setenv("HUDDLEROOM_LIVE_EVAL", "1")
    monkeypatch.setattr(dry_run.Timing, "wait_2a", 45.0)
    monkeypatch.setattr(dry_run.Timing, "wait_2b", 30.0)
    monkeypatch.setattr(dry_run.Timing, "grace", 0.0)
    monkeypatch.setattr(dry_run.Timing, "poll", 0.5)


def _argv(root, *extra, budget="100"):
    return ["--evidence-root", str(root), "--budget", budget, "--i-understand-this-spends-tokens", *extra]


def test_refuses_without_flag(tmp_path, fakes):
    assert dry_run.main(["--evidence-root", str(tmp_path), "--budget", "100"], fakes) == dry_run.EXIT_REFUSED
    assert not list(tmp_path.glob("run-*"))


def test_refuses_without_budget(tmp_path, fakes):
    with pytest.raises(SystemExit) as e:
        dry_run.main(["--evidence-root", str(tmp_path), "--i-understand-this-spends-tokens"], fakes)
    assert e.value.code != 0
    assert dry_run.main(_argv(tmp_path, budget="0"), fakes) == dry_run.EXIT_REFUSED


@pytest.mark.parametrize("missing", ["HUDDLEROOM_LIVE_ACCEPTANCE", "HUDDLEROOM_LIVE_EVAL"])
def test_refuses_without_env(tmp_path, fakes, monkeypatch, missing):
    monkeypatch.delenv(missing)
    assert dry_run.main(_argv(tmp_path), fakes) == dry_run.EXIT_REFUSED
    assert not list(tmp_path.glob("run-*"))


def test_isolation_failure_refuses_before_start(fakes, monkeypatch):
    started = []
    monkeypatch.setattr(dry_run, "Supervisor", lambda *a, **k: started.append(1))
    inside_repo = dry_run._REPO_ROOT / "tests" / "_dry_run_evidence_never_created"
    assert dry_run.main(_argv(inside_repo), fakes) == dry_run.EXIT_REFUSED
    assert not started and not inside_repo.exists()


def test_tiny_budget_stops_cleanly_with_findings(tmp_path, fakes):
    code = dry_run.main(_argv(tmp_path / "ev", "--step", "all"), fakes)
    assert code != 0
    (run,) = (tmp_path / "ev").glob("run-*")
    findings = json.loads((run / "dry-run-findings.json").read_text())
    assert findings["exit_code"] == code and findings["steps"]["2a"]["pass"] is False
    assert "2b" not in findings["steps"]
    assert (run / "findings.md").exists()
    assert findings["tokens"]["global"]["non_cached"] <= 100


# ---- cleanup, pause and stop-reason checks (zero tokens)

def _findings(root):
    (run,) = root.glob("run-*")
    return json.loads((run / "dry-run-findings.json").read_text()), run


def test_no_start_route_ever_requested(tmp_path, fakes, monkeypatch):
    import httpx
    seen = []
    real = httpx.request
    monkeypatch.setattr(httpx, "request", lambda m, u, **k: (seen.append((m, u)), real(m, u, **k))[1])
    dry_run.main(_argv(tmp_path / "ev"), fakes)
    assert any(u.endswith("/baseline/authorize") for _, u in seen)
    assert not any(u.endswith("/start") for _, u in seen)


def test_exception_mid_2a_kills_server_and_writes_findings(tmp_path, fakes, monkeypatch):
    from tests.live_acceptance.cli_shim import ShimHandle
    from tests.live_acceptance.server_supervisor import Supervisor
    calls = []
    kill, rel = Supervisor.kill, ShimHandle.release_all
    monkeypatch.setattr(Supervisor, "kill", lambda self, *a: (calls.append("kill"), kill(self, *a))[1])
    monkeypatch.setattr(ShimHandle, "release_all", lambda self: (calls.append("release"), rel(self))[1])

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(dry_run, "_wait", boom)
    assert dry_run.main(_argv(tmp_path / "ev"), fakes) != 0
    f, _ = _findings(tmp_path / "ev")
    assert "kill" in calls and "release" in calls and calls.index("kill") < calls.index("release")
    assert any("boom" in e for e in f["errors"]) and "auth_diff" in f


def test_failing_cleanup_step_does_not_skip_later_steps(tmp_path, fakes, monkeypatch):
    from tests.live_acceptance.spend_ledger import Ledger
    n = []
    real = Ledger.reap

    def reap(self, *a, **k):
        n.append(1)
        if len(n) > 1:  # the first call is the start-of-run reap
            raise OSError("reap failed")
        return real(self, *a, **k)
    monkeypatch.setattr(Ledger, "reap", reap)
    assert dry_run.main(_argv(tmp_path / "ev"), fakes) != 0
    f, run = _findings(tmp_path / "ev")
    assert any("ledger reap" in e for e in f["errors"])
    assert "auth_diff" in f and (run / "auth-after.json").exists() and (run / "findings.md").exists()


def test_pause_double_failure_fails_and_skips_2b(tmp_path, fakes, monkeypatch):
    resumed = []
    monkeypatch.setattr(dry_run, "_wait", lambda *a, **k: "ok")
    monkeypatch.setattr(dry_run, "_stop_reason", lambda *a, **k: None)
    monkeypatch.setattr(dry_run, "_log_lines", lambda d: ["settled codex 1 1 1 usage", "passthrough codex 1 1"])
    monkeypatch.setattr(dry_run.Api, "pause", lambda self: {"ok": False, "attempts": [{"status": 409}, {"status": 409}]})
    monkeypatch.setattr(dry_run.Api, "resume", lambda self: resumed.append(1) or 200)
    assert dry_run.main(_argv(tmp_path / "ev", "--step", "all"), fakes) != 0
    f, _ = _findings(tmp_path / "ev")
    assert f["steps"]["2a"]["pass"] is False and "2b" not in f["steps"] and not resumed


def _api(monkeypatch, statuses, version=3):
    api = dry_run.Api("http://x", "p", "g")
    posts = []

    class R:
        def __init__(self, st):
            self.status_code, self.headers, self.text = st, {"content-type": "application/json"}, ""

        def json(self):
            return {}
    monkeypatch.setattr(api, "detail", lambda: {"run": {"id": "r", "control_version": version}})
    it = iter(statuses)
    monkeypatch.setattr(api, "_do", lambda m, p, body=None: (posts.append(body), R(next(it)))[1])
    return api, posts


def test_pause_409_then_success_retries_once(monkeypatch):
    api, posts = _api(monkeypatch, [409, 200])
    assert api.pause()["ok"] and len(posts) == 2 and posts[0]["request_id"] != posts[1]["request_id"]


def test_pause_double_409_and_missing_version(monkeypatch):
    api, posts = _api(monkeypatch, [409, 409, 409])
    assert not api.pause()["ok"] and len(posts) == 2
    api, posts = _api(monkeypatch, [200], version=None)
    assert not api.pause()["ok"] and not posts


class _Led:
    def __init__(self, n):
        self.n = n

    def totals(self):
        return {"global": {"non_cached": self.n}}


class _Orc:
    def __init__(self, decisions=(), procs=0, runs=()):
        self.d, self.p, self.r = list(decisions), procs, list(runs)

    def decisions(self, g):
        return self.d

    def process_runs(self, goal_id=None):
        return [{}] * self.p

    def runs(self, g):
        return self.r


def _reason(tmp_path, *, log="", usage=(), ledger=0, orc=None, budget=1000):
    import json as j
    (tmp_path / "shim.log").write_text(log)
    (tmp_path / "spend.jsonl").write_text("".join(j.dumps({"k": "usage", "input_tokens": u, "output_tokens": 0,
                                                           "cached_tokens": 0}) + "\n" for u in usage))
    return dry_run._stop_reason(tmp_path, _Led(ledger), budget, orc or _Orc(), "g", tmp_path / "spend.jsonl")


@pytest.mark.parametrize("kw,want", [
    ({}, None),
    ({"log": "refused codex 1 1 1 x\n"}, "budget_refused_or_gate_timeout"),
    ({"log": "gate_timeout codex 1 1\n"}, "budget_refused_or_gate_timeout"),
    ({"usage": [25001]}, "single_call_over_25k"),
    ({"ledger": 800}, "budget_80_percent"),
    ({"orc": _Orc(decisions=[{"status": "pending"}])}, "pending_decision"),
    ({"orc": _Orc(procs=7)}, "over_6_processes_before_decision"),
    ({"orc": _Orc(runs=[{"status": "failed"}])}, "run_failed"),
    ({"orc": _Orc(runs=[{"status": "blocked"}])}, "run_blocked"),
    ({"orc": _Orc(runs=[{"status": "completed"}])}, "run_completed"),
    ({"orc": _Orc(runs=[{"status": "cancelled"}])}, "run_cancelled"),
    ({"orc": _Orc(procs=6, runs=[{"status": "running"}])}, None),
])
def test_stop_reasons(tmp_path, kw, want):
    assert _reason(tmp_path, **kw) == want


def test_secret_scan(monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "supersecretvalue123")
    assert dry_run._leaks("x supersecretvalue123 y") == ["SOME_API_KEY"]
    assert dry_run._leaks('{"MY_TOKEN": 1}') == ["MY_TOKEN"]
    assert not dry_run._leaks('{"settled_tokens": 1, "idempotency_keys": []}')
    assert "supersecretvalue123" not in dry_run._scrub("a supersecretvalue123")


# ---- stepped discovery (step 2b), stub shim/ledger/oracle

class _SteppedWorld:
    """Fake shim.log + ledger. release(n) appends settled/usage rows and the next gate line."""

    def __init__(self, tmp_path, tokens=100, decision_after=None, call1=100):
        self.dir, self.spend = tmp_path, tmp_path / "spend.jsonl"
        self.tokens, self.decision_after, self.released = tokens, decision_after, []
        self.decisions, self.procs = [], [{"process_type": "goal_definition", "status": "running"}]
        self.spend.write_text("")
        self.log = tmp_path / "shim.log"
        self.log.write_text("")
        self._settle(1, call1)
        self._gate(2)

    def _w(self, path, text):
        with open(path, "a") as f:
            f.write(text)

    def _settle(self, n, tokens):
        self._w(self.log, f"start codex 1 1 {n} exec --output-schema /nonexistent\nsettled codex 1 1 {n} usage\n")
        self._w(self.spend, json.dumps({"k": "usage", "label": f"codex {n}", "input_tokens": tokens,
                                        "output_tokens": 0, "cached_tokens": 0}) + "\n")

    def _gate(self, n):
        self._w(self.log, f"gate codex 1 1 {n} codex.nth.{n}\n")

    def release(self, token):
        n = int(token.rsplit(".", 1)[1])
        self.released.append(n)
        self._settle(n, self.tokens)
        self.procs.append({"process_type": f"p{n}", "status": "running"})
        if n == self.decision_after:
            self.decisions.append({"id": "d", "status": "pending", "options": []})
        else:
            self._gate(n + 1)

    # oracle / ledger stubs
    def decisions_(self, g):
        return self.decisions

    def process_runs(self, goal_id=None):
        return self.procs

    def runs(self, g):
        return []

    def totals(self):
        return {"global": {"non_cached": sum(dry_run._tok(r) for r in dry_run._ledger_usage(self.spend))}}


def _loop(w, **kw):
    orc = type("O", (), {"decisions": lambda self, g: w.decisions, "process_runs": lambda self, goal_id=None: w.procs,
                         "runs": lambda self, g: []})()
    findings = {}
    reason = dry_run._stepped_loop(w.dir, w.release, w, orc, "g", w.spend, kw.pop("budget", 100000), findings, **kw)
    return reason, findings


def test_stepped_release_order_and_max_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    w = _SteppedWorld(tmp_path)
    reason, f = _loop(w, max_releases=2)
    assert reason == "max_releases" and w.released == [2, 3]
    assert [r["ordinal"] for r in f["model_calls"]] == [1, 2, 3]
    assert [r["settled_tokens"] for r in f["model_calls"]] == [100, 100, 100]
    assert f["model_calls"][1]["process_type"] == "goal_definition" and f["model_calls"][2]["process_type"] == "p2"


def test_stepped_stops_on_pending_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    w = _SteppedWorld(tmp_path, decision_after=2)
    reason, _ = _loop(w, max_releases=5)
    assert reason == "pending_decision" and w.released == [2]


def test_stepped_extra_releases_need_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    w = _SteppedWorld(tmp_path)
    assert _loop(w, max_releases=1, extra_releases=3)[0] == "max_releases" and w.released == [2]


def test_stepped_extra_cap_respected(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    w = _SteppedWorld(tmp_path, tokens=100)
    reason, _ = _loop(w, max_releases=1, extra_releases=3, continue_to_decision=True, extra_cap=50)
    assert reason == "extra_cap" and w.released == [2, 3]  # one extra call ran, the next was not released


def test_stepped_per_call_and_budget_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    assert _loop(_SteppedWorld(tmp_path, call1=26000),
                 max_releases=2)[0] == "single_call_over_25k"
    (tmp_path / "b").mkdir()
    w = _SteppedWorld(tmp_path / "b", tokens=900)
    assert _loop(w, max_releases=3, budget=1000)[0] == "budget_80_percent"


def test_step_2b_end_to_end_fake_arms_holds_and_kills_first(tmp_path, fakes, monkeypatch):
    from tests.live_acceptance.cli_shim import ShimHandle
    from tests.live_acceptance.server_supervisor import Supervisor
    calls = []
    kill, rel, hold = Supervisor.kill, ShimHandle.release_all, ShimHandle.hold_nth
    monkeypatch.setattr(Supervisor, "kill", lambda self, *a: (calls.append("kill"), kill(self, *a))[1])
    monkeypatch.setattr(ShimHandle, "release_all", lambda self: (calls.append("release"), rel(self))[1])
    monkeypatch.setattr(ShimHandle, "hold_nth", lambda self, n: (calls.append(f"hold{n}"), hold(self, n))[1])
    code = dry_run.main(_argv(tmp_path / "ev", "--step", "2b"), fakes)
    assert code != 0
    f, run = _findings(tmp_path / "ev")
    assert calls.count("kill") >= 1 and calls.index("kill") < calls.index("release")
    assert sum(c.startswith("hold") for c in calls) == dry_run.HOLD_SPAN
    assert calls.index("hold2") < calls.index("kill")
    assert f["model_calls"] and f["model_calls"][0]["ordinal"] == 1 and "weight" in f
    assert run.name.endswith(tuple("0123456789abcdef")) and "dry-run-2b" in run.name


# ---- teardown order, hold span, output ceiling, retry detection

def test_teardown_order_deny_kill_live_server_then_release(tmp_path, fakes, monkeypatch):
    from tests.live_acceptance.cli_shim import ShimHandle
    from tests.live_acceptance.server_supervisor import Supervisor
    calls = []

    def spy(cls, name):
        orig = getattr(cls, name)
        monkeypatch.setattr(cls, name, lambda self, *a: (calls.append(f"{name}:{getattr(self, 'tool', '')}"),
                                                         orig(self, *a))[1], raising=False)
    for n in ("deny", "kill_live", "release_all"):
        spy(ShimHandle, n)
    spy(Supervisor, "kill")
    dry_run.main(_argv(tmp_path / "ev"), fakes)
    first = lambda prefix: next(i for i, c in enumerate(calls) if c.startswith(prefix))
    last = lambda prefix: max(i for i, c in enumerate(calls) if c.startswith(prefix))
    assert last("deny") < first("kill_live") <= last("kill_live") < first("kill:") < first("release_all")
    assert {c for c in calls if c.startswith(("deny", "kill_live"))} == {
        "deny:codex", "deny:claude", "kill_live:codex", "kill_live:claude"}


def test_leftover_shim_process_fails_run(tmp_path, fakes, monkeypatch):
    from tests.live_acceptance.cli_shim import ShimHandle
    monkeypatch.setattr(ShimHandle, "live_pids", lambda self: [4242], raising=False)
    assert dry_run.main(_argv(tmp_path / "ev"), fakes) != 0
    f, _ = _findings(tmp_path / "ev")
    assert any("still alive" in e for e in f["errors"])


def test_ordinal_beyond_hold_span_stops(tmp_path):
    span = dry_run.HOLD_SPAN + 1
    assert _reason(tmp_path, log=f"start codex 1 1 {span} exec x\n") is None
    assert _reason(tmp_path, log=f"start codex 1 1 {span + 1} exec x\n") == "ordinal_beyond_hold_span"


def test_release_budget_over_span_refused(tmp_path, fakes):
    assert dry_run.main(_argv(tmp_path, "--max-releases", "15", "--extra-releases", "6"), fakes) == dry_run.EXIT_REFUSED
    assert not list(tmp_path.glob("run-*"))


def test_output_ceiling_passed_to_both_shims(tmp_path, fakes, monkeypatch):
    seen = {}
    real = dry_run.make_shim
    monkeypatch.setattr(dry_run, "make_shim", lambda d, tool, p, **k: (seen.update({tool: k["output_ceiling"]}),
                                                                         real(d, tool, p, **k))[1])
    dry_run.main(_argv(tmp_path / "ev", "--output-ceiling", "1234"), fakes)
    assert seen == {"codex": 1234, "claude": 1234}
    f, _ = _findings(tmp_path / "ev")
    assert f["preflight"]["output_ceiling"] == 1234
    assert dry_run.argparse.ArgumentParser  # default checked below
    seen.clear()
    dry_run.main(_argv(tmp_path / "ev2"), fakes)
    assert seen == {"codex": 3000, "claude": 3000}


def test_retry_detected_when_process_type_reappears(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    w = _SteppedWorld(tmp_path)
    orig = w.release

    def release(token):
        orig(token)
        w.procs[-1]["process_type"] = "same"
    w.release = release
    _, f = _loop(w, max_releases=3)
    assert f["retry_detected"] is True and f["model_calls"][-1]["retry_detected"] is True
    assert all(r["process_type_best_effort"] for r in f["model_calls"])


def test_real_teardown_kills_held_shim_outside_server_group(tmp_path, fakes, monkeypatch):
    import subprocess
    import time
    marker = tmp_path / "exec-marker"
    fakes["codex"].write_text(FAKE.replace("  *) cat", f"  *) touch {marker}; cat", 1))
    handles, procs = [], []
    real_make = dry_run.make_shim
    monkeypatch.setattr(dry_run, "make_shim", lambda d, t, p, **k: handles.append(real_make(d, t, p, **k)) or handles[-1])

    def hold_then_fail(*a, **k):
        codex = next(h for h in handles if h.tool == "codex")
        codex.hold("codex")
        procs.append(subprocess.Popen([str(codex.path), "exec", "x"], stdin=subprocess.DEVNULL,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True))
        deadline = time.monotonic() + 15
        while not codex.live_pids() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert codex.live_pids(), "held shim never became live"
        raise RuntimeError("stop here")
    monkeypatch.setattr(dry_run, "_wait", hold_then_fail)
    assert dry_run.main(_argv(tmp_path / "ev"), fakes) != 0
    f, run = _findings(tmp_path / "ev")
    rc = procs[0].wait(timeout=10)
    assert rc != 0
    assert not marker.exists(), "real CLI was exec'd"
    assert all(h.live_pids() == [] for h in handles)
    ps = subprocess.run(["ps", "-axww", "-o", "command="], capture_output=True, text=True).stdout
    assert str(run / "shim") not in ps
    assert not any("still alive" in e for e in f["errors"])


# ---- step 2c: resume a paused run in place

def _paused_run(root, status="paused", pause_requested=1, spent=(5000, 2000)):
    import sqlite3
    from tests.live_acceptance.run_dir import build_run_dir
    rd = build_run_dir(root, "resume")
    con = sqlite3.connect(rd.db)
    con.executescript("create table orchestration_runs(id, goal_id, status, pause_requested);"
                      "create table orchestration_goals(id, project_id);")
    gid, pid, rid = "a" * 32, "b" * 32, "c" * 32
    con.execute("insert into orchestration_runs values (?,?,?,?)", (rid, gid, status, pause_requested))
    con.execute("insert into orchestration_goals values (?,?)", (gid, pid))
    con.commit()
    con.close()
    (rd.workspace / "README.md").write_text("x")
    rows = [{"k": "caps", "global": 800000, "scenarios": {"dry_run_2b": 75000}}]
    rows += [{"k": "usage", "rid": None, "scenario": "dry_run_2b", "label": f"codex {i}", "input_tokens": t,
              "output_tokens": 0, "cached_tokens": 0} for i, t in enumerate(spent, 1)]
    rd.spend.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rd


def _argv2c(evidence, run, budget="1000", *extra):
    return ["--evidence-root", str(evidence), "--budget", budget, "--i-understand-this-spends-tokens",
            "--step", "2c", "--resume-run", str(run), *extra]


def test_resume_refuses_non_paused_run(tmp_path, fakes):
    rd = _paused_run(tmp_path / "ev", status="running")
    assert dry_run.main(_argv2c(tmp_path / "ev", rd.root), fakes) == dry_run.EXIT_REFUSED
    assert not list(rd.snapshots.iterdir())
    rd2 = _paused_run(tmp_path / "ev2", pause_requested=0)
    assert dry_run.main(_argv2c(tmp_path / "ev2", rd2.root), fakes) == dry_run.EXIT_REFUSED


def test_resume_refuses_run_outside_evidence_root(tmp_path, fakes):
    rd = _paused_run(tmp_path / "other")
    assert dry_run.main(_argv2c(tmp_path / "ev", rd.root), fakes) == dry_run.EXIT_REFUSED
    assert not list(rd.snapshots.iterdir())
    assert dry_run.main(["--evidence-root", str(tmp_path / "ev"), "--budget", "1", "--step", "2c",
                         "--i-understand-this-spends-tokens"], fakes) == dry_run.EXIT_REFUSED  # no --resume-run


def test_resume_snap_a_before_shims_and_server_and_ledger_continues(tmp_path, fakes, monkeypatch):
    from tests.live_acceptance.spend_ledger import Ledger
    order = []
    rd = _paused_run(tmp_path / "ev")
    (rd.shim / "gates").mkdir(parents=True, exist_ok=True)
    (rd.shim / "gates" / "stale").touch()
    real_snap, real_make = dry_run._snapshot_dir, dry_run.make_shim
    monkeypatch.setattr(dry_run, "_snapshot_dir", lambda r, n, *a: (order.append(f"snap:{n}"), real_snap(r, n, *a))[1])
    monkeypatch.setattr(dry_run, "make_shim", lambda *a, **k: (order.append("make_shim"), real_make(*a, **k))[1])

    def no_server(*a, **k):
        order.append("supervisor")
        raise RuntimeError("sentinel")
    monkeypatch.setattr(dry_run, "Supervisor", no_server)
    assert dry_run.main(_argv2c(tmp_path / "ev", rd.root, "1000"), fakes) != 0
    assert order[0] == "snap:snap-A-paused-before-call4" and order.index("make_shim") > 0
    assert order.index("supervisor") > order.index("make_shim")
    snap = rd.snapshots / "snap-A-paused-before-call4"
    man = json.loads((snap / "manifest.json").read_text())
    assert (snap / "app.db").exists() and (snap / "workspace.tar.gz").exists() and man["db_canonical_sha256"]
    f = json.loads((rd.root / "dry-run-findings.json").read_text())
    assert f["ledger"] == {"prior_spent": 7000, "step_budget": 1000, "global_cap": 8000, "scenario": "dry_run_2c"}
    led = Ledger(rd.spend, 8000, {"dry_run_2c": 1000})
    assert led.totals()["global"]["non_cached"] == 7000
    assert not (rd.shim / "gates" / "stale").exists()
    assert not any("still alive" in e for e in f["errors"])
    from tests.live_acceptance.cli_shim import ShimHandle
    assert all(ShimHandle(rd.shim, t, rd.shim / t).live_pids() == [] for t in ("codex", "claude"))


class _Handle:
    def __init__(self, tool, world):
        self.tool, self.w, self.holds, self.released = tool, world, [], []

    def hold_nth(self, n):
        self.holds.append(n)

    def release(self, token):
        n = int(token.rsplit(".", 1)[1])
        self.released.append(n)
        self.w._w(self.w.log, f"start {self.tool} 1 1 {n} x\nsettled {self.tool} 1 1 {n} usage\n")
        self.w._w(self.w.spend, json.dumps({"k": "usage", "label": f"{self.tool} {n}", "input_tokens": 10,
                                            "output_tokens": 0, "cached_tokens": 0}) + "\n")
        self.w.procs.append({"process_type": f"p{self.tool}{n}", "status": "running"})
        if self.w.decision_after == (self.tool, n):
            self.w.decisions.append({"id": "d", "status": "pending", "options": [{"value": "request_changes"}]})


def _orc(w):
    return type("O", (), {"decisions": lambda self, g: w.decisions, "process_runs": lambda self, goal_id=None: w.procs,
                          "runs": lambda self, g: [], "actions": lambda self, r: getattr(w, "actions", []),
                          "run_state": lambda self, r: {"phase": getattr(w, "phase", "baseline"),
                                                        "active_blockers": getattr(w, "blockers", [])}})()


def test_resume_loop_releases_one_call_at_a_time_across_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    w = _SteppedWorld(tmp_path)
    w.log.write_text("gate codex 1 1 1 codex.nth.1\ngate codex 1 1 2 codex.nth.2\n")
    w.spend.write_text("")
    handles = {"codex": _Handle("codex", w), "claude": _Handle("claude", w)}
    f = {}
    reason = dry_run._resume_loop(w.dir, handles, w, _orc(w), "g", w.spend, 100000, f, max_releases=1)
    assert reason == "max_releases"
    assert handles["codex"].released == [1]  # first-blocked only
    assert [(r["tool"], r["ordinal"]) for r in f["model_calls"]] == [("codex", 1)]


def test_held_claude_call_stops_and_is_never_released(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    w = _SteppedWorld(tmp_path)
    w.log.write_text("gate claude 1 1 1 claude.nth.1\ngate codex 1 1 1 codex.nth.1\n")
    w.spend.write_text("")
    handles = {"codex": _Handle("codex", w), "claude": _Handle("claude", w)}
    assert dry_run._resume_loop(w.dir, handles, w, _orc(w), "g", w.spend, 100000, {}, max_releases=3) \
        == "unexpected_claude_call"
    assert handles["claude"].released == [] and handles["codex"].released == []
    assert _reason(tmp_path, log="gate claude 1 1 1 claude.nth.1\n") == "unexpected_claude_call"


def test_ledger_rewrite_is_atomic(tmp_path, monkeypatch):
    import os
    path = tmp_path / "spend.jsonl"
    path.write_text(json.dumps({"k": "caps", "global": 10, "scenarios": {}}) + "\n" + json.dumps(
        {"k": "usage", "rid": None, "scenario": "s", "label": "codex 1", "input_tokens": 4, "output_tokens": 0,
         "cached_tokens": 0}) + "\n")
    before = path.read_text()
    seen = []
    real = os.replace

    def spy(src, dst):
        seen.append((os.path.dirname(src) == os.path.dirname(dst), path.read_text() == before))
        raise OSError("replace failed")
    monkeypatch.setattr(dry_run.os, "replace", spy)
    with pytest.raises(OSError):
        dry_run._rewrite_ledger(path, 100, "dry_run_2c")
    assert seen == [(True, True)] and path.read_text() == before  # temp in same dir; original untouched
    monkeypatch.setattr(dry_run.os, "replace", real)
    assert dry_run._rewrite_ledger(path, 100, "dry_run_2c")["global_cap"] == 104
    assert json.loads(path.read_text().splitlines()[0])["global"] == 104


def test_body_2c_refuses_when_phase_is_not_baseline(tmp_path):
    from types import SimpleNamespace
    (tmp_path / "w").mkdir()
    w = _SteppedWorld(tmp_path / "w")
    w.phase = "authorized"
    resumed = []
    api = SimpleNamespace(resume=lambda: resumed.append(1) or 200)
    with pytest.raises(dry_run.StepError, match="authorized"):
        dry_run._body_2c(SimpleNamespace(max_releases=1), None, api, _orc(w), w, {}, {"run_id": "r"}, "g", {},
                         1000, {"prior_spent": 0})
    assert not resumed


def test_body_2c_stops_on_pending_decision_and_writes_snap_b(tmp_path, monkeypatch):
    import sqlite3
    from types import SimpleNamespace
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    (tmp_path / "w").mkdir()
    w = _SteppedWorld(tmp_path / "w")
    w.decision_after = 3  # the _SteppedWorld release already handles decisions on codex
    ws, snaps = tmp_path / "ws", tmp_path / "snaps"
    ws.mkdir()
    snaps.mkdir()
    (ws / "f.txt").write_text("x")
    sqlite3.connect(tmp_path / "app.db").executescript("create table t(a); insert into t values (1);")
    rd = SimpleNamespace(shim=w.dir, spend=w.spend, db=tmp_path / "app.db", workspace=ws, snapshots=snaps)

    class Api:
        resumed = started = 0

        def resume(self):
            self.resumed += 1
            return 200

        def start_baseline(self):
            self.started += 1
            return {}

        def pause(self):
            return {"ok": True}
    api, codex = Api(), type("H", (), {"holds": [], "hold_nth": lambda s, n: s.holds.append(n),
                                       "release": lambda s, t: w.release(t)})()
    f = {"steps": {}}
    args = SimpleNamespace(max_releases=4)
    w.phase = "baseline"
    code = dry_run._body_2c(args, rd, api, _orc(w), w, {"codex": codex}, {"run_id": "r"}, "g", f, 100000,
                            {"prior_spent": 0})
    assert code == 0 and w.released == [2, 3]
    assert (api.resumed, api.started) == (1, 1) and codex.holds == list(range(1, dry_run.HOLD_SPAN + 1))
    assert f["steps"]["2c"]["stop_reason"] == "pending_decision" and f["steps"]["2c"]["snap_b"]["db_canonical_sha256"]
    assert (snaps / "snap-B-first-pending-decision" / "manifest.json").exists()
    assert f["decisions"][0]["id"] == "d"


# ---- no-progress watchdog, retry flow, snapshot cleanliness

def test_watchdog_fires_on_no_progress_and_dumps(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    monkeypatch.setattr(dry_run.Timing, "wait_2b", 5.0)
    w = _SteppedWorld(tmp_path)
    w.log.write_text("")  # nothing ever gets held
    w.blockers = [{"kind": dry_run.ADR_KIND}]
    w.procs = [{"process_type": "agent_definition_review", "status": "running", "outputs": {"_lm_retry": {}, "error": "x"}}]
    f = {}
    rd = SimpleNamespace(shim=w.dir, db=tmp_path / "missing.db")
    wd = dry_run.Watchdog(dry_run._make_watchdog(rd, _orc(w), "a" * 32, "r", f).state_fn,
                          dry_run._make_watchdog(rd, _orc(w), "a" * 32, "r", f).dump_fn, timeout=0.2)
    reason = dry_run._resume_loop(w.dir, {"codex": _Handle("codex", w)}, w, _orc(w), "g", w.spend, 100000, f,
                                  max_releases=1, wd=wd)
    assert reason == "no_progress"
    npg = f["no_progress"]
    assert npg["active_blockers"][0]["kind"] == dry_run.ADR_KIND and "warnings" in npg
    assert npg["process_output_keys"] == {"0:agent_definition_review": ["_lm_retry", "error"]}


def test_watchdog_resets_on_progress():
    state = [0]
    wd = dry_run.Watchdog(lambda: state[0], lambda: None, timeout=0.05)
    import time
    for i in range(5):
        state[0] = i
        time.sleep(0.03)
        assert wd.check() is None
    time.sleep(0.08)
    assert wd.check() == "no_progress"


def test_tolerate_blocked_only_ignores_blocked(tmp_path):
    (tmp_path / "shim.log").write_text("")
    (tmp_path / "spend.jsonl").write_text("")
    orc = _Orc(runs=[{"status": "blocked"}])
    args = (tmp_path, _Led(0), 1000, orc, "g", tmp_path / "spend.jsonl")
    assert dry_run._stop_reason(*args) == "run_blocked"
    assert dry_run._stop_reason(*args, tolerate_blocked=True) is None
    orc.r = [{"status": "failed"}]
    assert dry_run._stop_reason(*args, tolerate_blocked=True) == "run_failed"


def _body_2c_world(tmp_path, blockers=(), retry_status=200, allow_retry=True):
    import sqlite3
    from types import SimpleNamespace
    (tmp_path / "w").mkdir()
    w = _SteppedWorld(tmp_path / "w", decision_after=3)
    w.phase, w.blockers = "baseline", list(blockers)
    ws, snaps = tmp_path / "ws", tmp_path / "snaps"
    ws.mkdir()
    snaps.mkdir()
    sqlite3.connect(tmp_path / "app.db").executescript("create table t(a); insert into t values (1);")
    rd = SimpleNamespace(shim=w.dir, spend=w.spend, db=tmp_path / "app.db", workspace=ws, snapshots=snaps)
    retried = []

    class Thread:
        def join(self, t=None):
            pass

    class Api:
        def resume(self): return 200
        def start_baseline(self): return {}
        def pause(self): return {"ok": True}

        def retry_async(self, ptype):
            retried.append(ptype)
            return {"done": True, "status": retry_status, "thread": Thread()}
    codex = type("H", (), {"hold_nth": lambda s, n: None, "release": lambda s, t: w.release(t)})()
    f = {"steps": {}}
    args = SimpleNamespace(max_releases=4, allow_retry=allow_retry)
    return w, rd, Api(), codex, f, args, retried


def test_2c_retries_analyzer_blocker_then_continues_and_labels_dirty_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    w, rd, api, codex, f, args, retried = _body_2c_world(tmp_path, blockers=[{"kind": dry_run.ADR_KIND}])
    code = dry_run._body_2c(args, rd, api, _orc(w), w, {"codex": codex}, {"run_id": "r"}, "g", f, 100000,
                            {"prior_spent": 0})
    assert code == 0 and retried == ["agent_definition_review"] and w.released == [2, 3]
    assert f["blockers_after_resume"]["kinds"] == [dry_run.ADR_KIND] and f["retry"]["status"] == 200
    man = json.loads((rd.snapshots / "snap-B-first-pending-decision" / "manifest.json").read_text())
    assert man["clean"] is False and man["blocker_kinds"] == [dry_run.ADR_KIND]
    assert f["steps"]["2c"]["snap_b_dirty"] is True


def test_2c_clean_run_does_not_retry_and_snap_b_is_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    w, rd, api, codex, f, args, retried = _body_2c_world(tmp_path)
    assert dry_run._body_2c(args, rd, api, _orc(w), w, {"codex": codex}, {"run_id": "r"}, "g", f, 100000,
                            {"prior_spent": 0}) == 0
    assert not retried and "retry" not in f and f["steps"]["2c"]["snap_b_dirty"] is False
    man = json.loads((rd.snapshots / "snap-B-first-pending-decision" / "manifest.json").read_text())
    assert man["clean"] is True


def test_2c_retry_failure_and_disabled_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    blockers = [{"kind": dry_run.ADR_KIND}]
    (tmp_path / "a").mkdir()
    w, rd, api, codex, f, args, _ = _body_2c_world(tmp_path / "a", blockers=blockers, retry_status=500)
    with pytest.raises(dry_run.StepError, match="baseline/retry returned 500"):
        dry_run._body_2c(args, rd, api, _orc(w), w, {"codex": codex}, {"run_id": "r"}, "g", f, 100000,
                         {"prior_spent": 0})
    (tmp_path / "b").mkdir()
    w, rd, api, codex, f, args, retried = _body_2c_world(tmp_path / "b", blockers=blockers, allow_retry=False)
    with pytest.raises(dry_run.StepError, match="allow-retry"):
        dry_run._body_2c(args, rd, api, _orc(w), w, {"codex": codex}, {"run_id": "r"}, "g", f, 100000,
                         {"prior_spent": 0})
    assert not retried


def test_retry_route_and_body(monkeypatch):
    api = dry_run.Api("http://x", "p", "g")
    seen = []

    class R:
        status_code = 200
    monkeypatch.setattr(api, "_do", lambda m, path, body=None, timeout=60: (seen.append((m, path, body, timeout)), R())[1])
    res = api.retry_async("agent_definition_review")
    res["thread"].join(5)
    assert seen == [("POST", "/baseline/retry", {"process_type": "agent_definition_review"}, 60)] and res["status"] == 200


def test_api_request_timeout_defaults_and_accepts_override(monkeypatch):
    api = dry_run.Api("http://x", "p", "g")
    seen = []

    class R:
        status_code = 200

    monkeypatch.setattr(dry_run.httpx, "request", lambda *args, **kwargs: (seen.append((args, kwargs)), R())[1])
    api._do("GET", "")
    api._do("POST", "/baseline/retry", {"process_type": "team_hierarchy"}, timeout=300)
    assert [kwargs["timeout"] for _, kwargs in seen] == [60, 300]


def test_existing_snap_a_is_not_overwritten_and_counts_dirty(tmp_path, fakes, monkeypatch):
    rd = _paused_run(tmp_path / "ev")
    old = rd.snapshots / "snap-A-paused-before-call4"
    old.mkdir()
    (old / "manifest.json").write_text(json.dumps({"db_canonical_sha256": "legacy"}))
    monkeypatch.setattr(dry_run, "Supervisor", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sentinel")))
    code = dry_run.main(_argv2c(tmp_path / "ev", rd.root, "1000"), fakes)
    assert code not in (0, dry_run.EXIT_REFUSED)
    assert json.loads((old / "manifest.json").read_text()) == {"db_canonical_sha256": "legacy"}
    assert not (old / "app.db").exists()
    f = json.loads((rd.root / "dry-run-findings.json").read_text())
    assert f["snap_a"]["exists"] is True and f["snap_a"]["clean"] is False


def test_2b_snapshots_at_held_call_before_teardown(tmp_path, monkeypatch):
    import sqlite3
    from types import SimpleNamespace
    monkeypatch.setattr(dry_run.Timing, "poll", 0.01)
    (tmp_path / "w").mkdir()
    w = _SteppedWorld(tmp_path / "w")
    ws, snaps = tmp_path / "ws", tmp_path / "snaps"
    ws.mkdir()
    snaps.mkdir()
    sqlite3.connect(tmp_path / "app.db").executescript("create table t(a);")
    rd = SimpleNamespace(shim=w.dir, spend=w.spend, db=tmp_path / "app.db", workspace=ws, snapshots=snaps)

    class Api:
        def detail(self): return {"goal": {"weight": "substantial"}}
        def start_baseline(self): return {"run": {"id": "r"}}
        def pause(self): return {"ok": True}
    codex = type("H", (), {"hold_nth": lambda s, n: None, "release": lambda s, t: w.release(t)})()
    f = {"steps": {}}
    args = SimpleNamespace(max_releases=1, extra_releases=0, continue_to_decision=False, extra_cap=1000)
    ids = {}
    assert dry_run._body_2b(args, rd, Api(), _orc(w), w, codex, ids, "g", f, 100000) == 0
    man = json.loads((snaps / "snap-paused-at-held-call" / "manifest.json").read_text())
    assert man["clean"] is True and f["steps"]["2b"]["clean"] is True and ids["run_id"] == "r"
