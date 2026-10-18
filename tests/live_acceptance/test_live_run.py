"""Live run L1: one staged run from the first-pending-decision snapshot.

Steps (see STEPS): answer the agent-definition batch through the UI under injected faults, retry it over
HTTP, snapshot, kill the server with nothing held, restart, Resume, Continue baseline, hold the team
design call (100 s clock: the product CLI times out at 120 s), pause during it, pause again with a dropped
response while the same call is held, release it while paused, pause with a reload, answer the team
approval with request_changes and a reload mid-request. Each step runs in a try/except; the first failure stops the run and
the remaining checks stay "not_run". verdict.json is rewritten after every step.

Run (spends subscription tokens; scenario cap 60000):
  HUDDLEROOM_LIVE_ACCEPTANCE=1 HUDDLEROOM_LIVE_EVAL=1 LIVE_ACCEPTANCE_SNAPSHOT_B=<snapshot dir> \
  LIVE_ACCEPTANCE_EVIDENCE_ROOT=<evidence root> .venv/bin/python -m pytest -m live -s \
  tests/live_acceptance/test_live_run.py
Print the plan without running anything: .venv/bin/python -m tests.live_acceptance.test_live_run --dry-plan
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

from tests.live_acceptance.run_dir import real_home
from tests.live_acceptance.scenario import LiveScenario, _latest_ids

# The team design call is the only baseline call whose prompt (codex stdin, matched by the shim) holds
# this phrase (orchestration_team_hierarchy_analyzer.py build_request). Dry-run args/schema heads are
# identical for all calls, so they cannot tell calls apart.
TH_MARKER = r"Act as the team architect"
FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
CAP = 60_000
MAX_NON_TH_CALLS = 0  # any held non-team-design call fails the run
HOLD_LIMIT_S = 100
KEYS = ("pause_usable", "accept_before_successor", "pause_uncertain", "inflight_completes_paused",
        "saved_matches_receipt", "answer_uncertain", "retry_idempotent", "paused_recovery",
        "risk_answer_while_paused", "risk_chain_continues")
# (name, checks it can set, description)
STEPS = (
    ("1-approve-batch", ("saved_matches_receipt", "answer_uncertain", "risk_answer_while_paused"),
     "approve the agent-definition batch in the UI: refetch held then 500, response dropped after commit"),
    ("2-retry", ("retry_idempotent",), "httpx identical retry (no duplicate), conflicting retry (409)"),
    ("3-pre-kill", (), "answer still pending application; snapshot L1-pre-kill"),
    ("4-restart-resume", ("paused_recovery", "risk_chain_continues"),
     "SIGKILL (nothing held), restart, Resume, Continue baseline, applied once, TH held, start 100 s clock"),
    ("5-pause", ("pause_usable", "accept_before_successor"), "Pause while TH held, 15 s poll"),
    ("6-release-th", ("inflight_completes_paused",), "release TH while paused (<100 s), snapshot snap-D"),
    ("7-pause-drop", ("pause_uncertain",), "Resume + pause with dropped response (no call in flight)"),
    ("8-pause-reload", ("pause_uncertain",), "Resume + pause with reload (no call in flight)"),
    ("9-request-changes", ("saved_matches_receipt", "answer_uncertain"),
     "request_changes on the team approval, reload mid-request (not_run if only agent proposals)"),
)

SNAP = os.environ.get("LIVE_ACCEPTANCE_SNAPSHOT_B")
ROOT = os.environ.get("LIVE_ACCEPTANCE_EVIDENCE_ROOT")
_hex = lambda v: uuid.UUID(str(v)).hex


class L1:
    SPEC = "tests-live/live-run.spec.ts"

    def __init__(self, sc: LiveScenario):
        self.sc = sc
        self.codex, self.claude = sc.shims["codex"], sc.shims["claude"]
        self.ids = sc.ids = {**sc.ids, **_latest_ids(sc.run_dir.db)}
        self.run_id, self.goal_id, self.project_id = (self.ids[k] for k in ("run_id", "goal_id", "project_id"))
        self.base = f"{sc.base_url}/api/v1/projects/{self.project_id}/orchestration/goals/{self.goal_id}"
        self.checks = {k: {"status": "not_run", "evidence": []} for k in KEYS}
        self.steps: dict[str, str] = {n: "not_run" for n, _, _ in STEPS}
        self.released: list[int] = []
        self.th = None
        self.th_decision = None
        self.adr_ids = [d["id"] for d in sc.oracle.decisions(self.goal_id)
                        if d["status"] == "pending" and str(d["decision_key"]).startswith("agent_definition_review")]
        self.batch_post: dict = {}
        self.notes: dict = {"adr_decision_ids": self.adr_ids}

    # -- verdict
    def write_verdict(self) -> None:
        (self.sc.run_dir.root / "verdict.json").write_text(json.dumps(
            {"passed": None, "checks": self.checks, "steps": self.steps, "notes": self.notes}, indent=2, default=str))

    def verify(self, key: str, conds: dict[str, bool], **evidence) -> None:
        """Record conds under check `key` (worst status wins) and raise if any failed."""
        bad = sorted(k for k, v in conds.items() if not v)
        c = self.checks[key]
        c["status"] = "fail" if bad or c["status"] == "fail" else "pass"
        c["evidence"].append({"conds": conds, **evidence})
        self.write_verdict()
        if bad:
            raise AssertionError(f"{key}: failed {bad}")

    # -- helpers
    def pw(self, name: str, **env) -> dict:
        out = self.sc.run_dir.root / "playwright" / name
        out.mkdir(parents=True, exist_ok=True)
        e = {**os.environ, "LIVE_ACCEPTANCE_BASE_URL": self.sc.base_url, "LIVE_ACCEPTANCE_OUTPUT": str(out),
             "LIVE_ACCEPTANCE_PROJECT_ID": self.project_id, "LIVE_ACCEPTANCE_GOAL_ID": self.goal_id, **env}
        # pytest fakes HOME; the installed browsers live under the real one (HOME itself stays as is)
        e.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(real_home() / "Library/Caches/ms-playwright"))
        r = subprocess.run(["npx", "playwright", "test", "-c", "playwright.live.config.ts", self.SPEC,
                            "--grep", rf"(^|\s){name}$"], cwd=FRONTEND, env=e, capture_output=True, text=True, timeout=500)
        (out.parent / f"{name}.log").write_text(r.stdout + "\n" + r.stderr)
        self.sc.mark(f"playwright {name} rc={r.returncode}")
        f = out / f"{name}.json"
        data = json.loads(f.read_text()) if f.exists() else {}
        data["_rc"], data["_dir"] = r.returncode, str(out)
        return data

    def pw_ok(self, name: str, **env) -> dict:
        d = self.pw(name, **env)
        if d["_rc"] != 0:
            raise AssertionError(f"playwright {name} failed (see {d['_dir']}/../{name}.log)")
        return d

    def guard(self) -> None:
        """Stop on any claude call or any executed codex call nobody released on purpose."""
        stray = [i.ordinal for i in self.codex.invocations() if i.exec_at is not None and i.ordinal not in self.released]
        if self.claude.invocations():
            raise AssertionError("unexpected claude model call")
        if stray:
            raise AssertionError(f"unexpected executed codex calls: {stray}")

    def invocation_keys(self) -> set:
        return {(t, i.ordinal) for t, h in self.sc.shims.items() for i in h.invocations()}

    def id_sets(self) -> dict:
        """Everything a successor dispatch would add: actions (id -> type), process runs, sessions, CLI calls."""
        o = self.sc.oracle
        return {"actions": {a["id"]: a["action_type"] for a in o.actions(self.run_id)},
                "process_runs": {p["id"] for p in o.process_runs(goal_id=self.goal_id)},
                "sessions": {s["id"] for s in o.sessions()}, "invocations": self.invocation_keys()}

    def diff(self, before: dict) -> dict:
        now = self.id_sets()
        return {"actions": sorted(t for i, t in now["actions"].items() if i not in before["actions"]),
                "process_runs": sorted(now["process_runs"] - before["process_runs"]),
                "sessions": sorted(now["sessions"] - before["sessions"]),
                "invocations": sorted(now["invocations"] - before["invocations"])}

    @staticmethod
    def nothing_new(d: dict, allow_actions: tuple = ("pause_control",)) -> bool:
        return not d["process_runs"] and not d["sessions"] and not d["invocations"] \
            and all(t in allow_actions for t in d["actions"])

    def decision(self, did: str) -> dict:
        return next(d for d in self.sc.oracle.decisions(self.goal_id) if d["id"] == _hex(did))

    def row_hash(self, dids: list[str]) -> str:
        rows = [self.decision(d) for d in sorted(dids)]
        return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()

    def answer_actions(self) -> int:
        return len(self.sc.oracle.actions(self.run_id, "apply_authority_answer"))

    def db_receipts(self, dids: list[str]) -> list[dict]:
        return self.sc.oracle.answer_receipt(self.run_id, dids)["receipts"]

    def answered_conds(self, dids: list[str], option: str) -> dict:
        ds = [self.decision(d) for d in dids]
        return {"decisions_answered": all(d["status"] != "pending" and d["selected_option"] == option for d in ds),
                "one_receipt": len(self.db_receipts(dids)) == 1}

    @staticmethod
    def nested(resp) -> dict:
        """The receipt of an answer or decision response: body["answer_receipt"] {action_id, decision_ids, application_state}."""
        return (resp or {}).get("answer_receipt") or {}

    def ui_receipt(self, d: dict) -> dict:
        """First receipt the UI saw: from the answer POST response, else from a decision readback."""
        for r in [*(d.get("answer_posts") or []), *(d.get("decision_gets") or [])]:
            rc = self.nested(r.get("response"))
            if rc.get("action_id"):
                return rc
        rc = self.nested(d.get("readback"))
        return rc

    def receipt_conds(self, rc: dict, dids: list[str], paused: bool = True) -> dict:
        db = self.db_receipts(dids)
        return {"ui_receipt_seen": bool(rc.get("action_id")),
                "action_id_matches": bool(rc.get("action_id")) and bool(db) and _hex(rc["action_id"]) == db[0]["id"],
                "decision_ids_match": sorted(_hex(x) for x in rc.get("decision_ids", [])) == sorted(_hex(x) for x in dids),
                "application_state_pending_while_paused": (rc.get("application_state") == "pending") or not paused}

    def apply_state(self, dids: list[str], completed: bool) -> dict:
        """The answer's application action and the decisions' continuation_applied_at, before or after Continue baseline."""
        acts = self.db_receipts(dids)
        decs = [self.decision(d) for d in dids]
        a = acts[0] if len(acts) == 1 else {}
        contract = a.get("dispatch_contract") or {}
        if completed:
            return {"one_action": len(acts) == 1, "action_completed": a.get("status") == "completed",
                    "continuation_applied": all(d["continuation_applied_at"] for d in decs)}
        return {"one_action": len(acts) == 1, "action_reserved": a.get("status") == "reserved",
                "application_state_pending": contract.get("application_state") == "pending",
                "continuation_not_applied": all(d["continuation_applied_at"] is None for d in decs)}

    def th_held(self):
        """Return the held team design call; any held non-TH call fails the run (args dumped)."""
        held = [i for i in self.codex.invocations() if i.exec_at is None and i.gates and not i.refused and not i.timed_out]
        other = [i for i in held if not any(".match." in g for g in i.gates)]
        if len(other) > MAX_NON_TH_CALLS:
            self.notes["unexpected_held_calls"] = [{"ordinal": i.ordinal, "args": i.args} for i in other]
            raise AssertionError(f"held non-team-design codex call(s): {[i.ordinal for i in other]}")
        th = [i for i in held if any(".match." in g for g in i.gates)]
        return th[0] if th else None

    def wait(self, timeout: float, check, every: float = 1.0):
        end = time.time() + timeout
        while time.time() < end:
            self.guard()
            got = check()
            if got:
                return got
            time.sleep(every)
        return None

    def not_authorized(self) -> bool:
        run = self.sc.oracle.run_state(self.run_id)
        return run["phase"] != "baseline" or run["baseline_authorized"] == 0

    def elapsed(self) -> float:
        return time.time() - self.th.started_at

    def check_clock(self) -> None:
        if self.elapsed() >= HOLD_LIMIT_S:
            self.notes["timing_exceeded"] = round(self.elapsed(), 1)
            raise AssertionError(f"timing_exceeded: held call is {self.elapsed():.0f}s old (limit {HOLD_LIMIT_S}s)")

    # -- steps
    def s1(self):
        d = self.pw_ok("approve-batch")
        posts = d["answer_posts"]
        self.batch_post = posts[0]
        rc = self.ui_receipt(d)
        self.notes["approve_batch_posts"] = len(posts)
        self.verify("risk_answer_while_paused", {"answer_while_paused_accepted": d["drop_hits"] >= 1 and self.answered_conds(self.adr_ids, "approve")["decisions_answered"]})
        self.verify("answer_uncertain", {**self.answered_conds(self.adr_ids, "approve"), "dropped": d["drop_hits"] >= 1,
                    "one_post": len(posts) == 1, "checking_observed": d.get("checking_observed") is True, "no_checking_with_saved": not d["checking_with_saved"]}, ui=d["_dir"])
        self.verify("saved_matches_receipt", {**self.receipt_conds(rc, self.adr_ids), "saved_stayed": d["saved_stayed"],
                    "refetch_hit": d["refetch_hits"] >= 1}, ui=d["_dir"], ui_receipt=rc)
        self.guard()

    def s2(self):
        post, ids = self.batch_post, self.adr_ids
        before = {"hash": self.row_hash(ids), "actions": self.answer_actions(), "receipt": self.db_receipts(ids)}
        same = httpx.post(post["url"], json=post["request"], timeout=60)
        mid = {"hash": self.row_hash(ids), "actions": self.answer_actions()}
        other = httpx.post(post["url"], json=json.loads(json.dumps(post["request"]).replace('"approve"', '"reject"')), timeout=60)
        after = {"hash": self.row_hash(ids), "actions": self.answer_actions(), "receipts": len(self.db_receipts(ids))}
        self.verify("retry_idempotent", {
            "identical_retry_ok": same.status_code == 200, "no_duplicate_action": mid["actions"] == before["actions"],
            "row_unchanged": mid["hash"] == before["hash"],
            "same_action_id": same.status_code == 200 and _hex(self.nested(same.json()).get("action_id", uuid.UUID(int=0))) == before["receipt"][0]["id"],
            "conflicting_retry_409": other.status_code == 409,
            "conflict_detail": "conflicts with saved answer" in other.text.lower(),
            "conflict_changed_nothing": after["hash"] == before["hash"] and after["actions"] == before["actions"],
            "one_receipt": after["receipts"] == 1},
            statuses=[same.status_code, other.status_code], before=before["hash"], after=after["hash"])
        self.guard()

    def s3(self):
        self.verify("paused_recovery", {f"pre_kill_{k}": v for k, v in self.apply_state(self.adr_ids, False).items()})
        self.sc.oracle_dump("pre-kill")
        self.sc.take_snapshot("L1-pre-kill", require_clean=True)

    def s4(self):
        # kill only with nothing held and nothing running: held/live shim processes would die with it
        live = {t: h.live_pids() for t, h in self.sc.shims.items() if h.live_pids()}
        if live or self.codex.invocations() or self.claude.invocations():
            raise AssertionError(f"model calls exist before kill; refusing to kill: {live}")
        self.sc.mark("SIGKILL server (nothing held)")
        self.sc.sup.kill(signal.SIGKILL)
        self.sc.sup.start()
        self.sc.mark("server restarted on the same DB")
        run = self.sc.oracle.run_state(self.run_id)
        restart = {"paused_after_restart": run["status"] == "paused" and run["pause_requested"] == 1,
                   "phase_baseline": run["phase"] == "baseline", "baseline_not_authorized": run["baseline_authorized"] == 0,
                   **{f"restart_{k}": v for k, v in self.apply_state(self.adr_ids, False).items()}}
        # The Saved banner is client state; restart proof is the DB, so no banner assertion here.
        self.pw_ok("resume")
        self.pw_ok("continue-baseline")
        self.guard()
        th = self.wait(300, self.th_held)
        if th is None:
            self.verify("risk_chain_continues", {"th_dispatched_and_held": False})
        self.th = th
        self.notes["th_started_at"] = th.started_at
        self.notes["th_gates"] = th.gates
        # short wait only: the held call's age is on the 100 s clock; applied state is re-asserted after step 7
        applied = self.wait(5, lambda: self.apply_state(self.adr_ids, True)["action_completed"]) is not None
        self.verify("paused_recovery", restart, applied_within_5s=applied, after_continue=self.apply_state(self.adr_ids, True))
        self.verify("risk_chain_continues", {"th_dispatched_and_held": True}, adr_completed_within_5s=applied)
        self.sc.oracle_dump("th-held")

    def pause_conds(self, before: dict, v_before: int, req_id) -> tuple[dict, dict]:
        pc = self.sc.oracle.pause_control(self.run_id, req_id) if req_id else []
        run = self.sc.oracle.run_state(self.run_id)
        got = httpx.get(f"{self.base}/pause-control/{req_id}", timeout=30) if req_id else None
        d = self.diff(before)
        return {"pause_control_completed": len(pc) == 1 and pc[0]["status"] == "completed",
                "pause_requested_1": run["pause_requested"] == 1, "control_version_plus_1": run["control_version"] == v_before + 1,
                "run_paused": run["status"] == "paused",
                "pause_get_accepted": got is not None and got.status_code == 200 and got.json().get("state") in ("accepted", "applied"),
                "nothing_after_acceptance": self.nothing_new(d), "baseline_not_authorized_after_pause": self.not_authorized(),
                "th_still_held": self.th_held() is not None}, d

    def s5(self):
        before = self.id_sets()
        v_before = self.sc.oracle.run_state(self.run_id)["control_version"]
        d = self.pw("pause")
        req_id = ((d.get("pause_post") or {}).get("request") or {}).get("request_id")
        self.sc.oracle_dump("th-held-paused")
        self.verify("pause_usable", {
            "ui_step_passed": d["_rc"] == 0, "accepted_under_5s": d.get("click_to_accepted_ms", 1e9) < 5000,
            "never_active_or_running": d.get("never_active_or_running") is True,
            "pause_goal_never_enabled": d.get("pause_goal_never_enabled") is True}, ui=d["_dir"],
            click_to_accepted_ms=d.get("click_to_accepted_ms"))
        conds, diff = self.pause_conds(before, v_before, req_id)
        self.verify("accept_before_successor", conds, new_since_before_click=diff)
        self.guard()

    def pause_variant(self, name: str) -> None:
        self.pw_ok("resume")
        before = self.id_sets()
        base_pc = len(self.sc.oracle.pause_control(self.run_id))
        d = self.pw(name)
        self.sc.oracle_dump(f"after-{name}")
        run = self.sc.oracle.run_state(self.run_id)
        diff = self.diff(before)
        self.verify("pause_uncertain", {
            f"{name}_ui_passed": d["_rc"] == 0, f"{name}_run_paused": run["status"] == "paused" and run["pause_requested"] == 1,
            f"{name}_new_pause_control": len(self.sc.oracle.pause_control(self.run_id)) > base_pc,
            f"{name}_nothing_after_acceptance": self.nothing_new(diff),
            f"{name}_baseline_not_authorized": self.not_authorized()}, ui=d["_dir"], new_since_resume=diff)
        self.guard()

    def s7(self):
        self.notes["pause_uncertain_note"] = ("pause_uncertain variants ran without an in-flight call "
                                              "due to the product 120 s CLI timeout")
        self.pause_variant("pause-drop")

    def s6(self):
        self.check_clock()
        self.notes["elapsed_at_release_s"] = round(self.elapsed(), 1)
        before = self.id_sets()
        for g in self.th.gates:
            self.codex.release(g)
        self.released.append(self.th.ordinal)
        self.sc.mark(f"TH call released while paused after {self.elapsed():.0f}s")
        goal = self.goal_id
        proc = self.wait(180, lambda: next((p for p in reversed(self.sc.oracle.process_runs(goal_id=goal))
                                            if p["process_type"] == "team_hierarchy" and p["status"] == "waiting_decision"), None))
        # snapshot right after the call lands, before any assertion can fail the step
        if proc is not None:
            snap = self.sc.take_snapshot("snap-D", require_clean=False)
            self.notes["snap_D_clean"] = snap.get("clean")
        time.sleep(10)  # time for a forbidden successor dispatch to show up
        self.sc.oracle_dump("th-landed-paused")
        pending = [d for d in self.sc.oracle.decisions(goal) if d["status"] == "pending"
                   and not str(d["decision_key"]).startswith("agent_definition_review")]
        self.th_decision = next((d for d in pending if d["decision_key"] == "team_hierarchy:approval"
                                 and str(d.get("authority")).lower() == "human"), None)
        self.notes["th_pending"] = [{"decision_key": d["decision_key"], "authority": d.get("authority"), "options": d["options"]}
                                    for d in pending]
        run = self.sc.oracle.run_state(self.run_id)
        diff = self.diff(before)
        self.notes["actions_after_th_release"] = diff["actions"]  # any action rows are allowed here, only recorded
        self.verify("paused_recovery", {f"applied_after_release_{k}": v for k, v in self.apply_state(self.adr_ids, True).items()})
        self.verify("inflight_completes_paused", {
            "th_landed_waiting_decision": proc is not None, "a_decision_pending": bool(pending),
            "still_paused": run["status"] == "paused" and run["pause_requested"] == 1,
            "no_successor_or_admission": self.nothing_new({**diff, "actions": []}), "no_claude_call": not self.claude.invocations(),
            "baseline_not_authorized": self.not_authorized()}, new_since_release=diff, snap_D_clean=self.notes.get("snap_D_clean"))

    def s8(self):
        self.pause_variant("pause-reload")

    def s9(self):
        d = self.th_decision
        if d is None:
            self.notes["request_changes_skipped"] = "only agent proposals pending (or no team approval); see th_pending"
            raise _NotRun(self.notes["request_changes_skipped"])
        if "request_changes" not in json.dumps(d["options"]):
            self.notes["request_changes_gap"] = "request_changes not offered in the team approval options"
            raise AssertionError(self.notes["request_changes_gap"])
        before = self.id_sets()
        r = self.pw("request-changes")
        post = (r.get("answer_posts") or [{}])[0]
        did = d["id"]
        self.verify("answer_uncertain", {"ui_step_passed": r["_rc"] == 0, "reloaded_once": r.get("hits", 0) >= 1,
                    **self.answered_conds([did], "request_changes"), "readback_ok": r.get("readback_status") == 200},
                    ui=r["_dir"], post_url=post.get("url"))
        self.verify("saved_matches_receipt", self.receipt_conds(self.ui_receipt(r), [did]), ui=r["_dir"])
        self.verify("saved_matches_receipt", {"no_new_work": self.nothing_new(self.diff(before), allow_actions=("apply_authority_answer",)),
                    "still_paused": self.sc.oracle.run_state(self.run_id)["status"] == "paused"})
        self.guard()


class _NotRun(Exception):
    """A step that cannot apply (kept as not_run, does not stop or fail the run)."""


def run_l1(sc: LiveScenario) -> None:
    l1 = L1(sc)
    l1.write_verdict()
    stopped = False
    for (name, _, _), fn in zip(STEPS, (l1.s1, l1.s2, l1.s3, l1.s4, l1.s5, l1.s6, l1.s7, l1.s8, l1.s9)):
        if stopped:
            continue
        try:
            fn()
            l1.steps[name] = "pass"
        except _NotRun as e:
            l1.steps[name] = f"not_run: {e}"[:400]
        except Exception as e:  # noqa: BLE001 - first failure stops the run, rest stay not_run
            l1.steps[name] = f"fail: {type(e).__name__}: {e}"[:400]
            stopped = True
            sc.mark(f"STEP FAILED {name}: {l1.steps[name]}")
        l1.write_verdict()
    skipped_ok = {"saved_matches_receipt", "answer_uncertain"} if l1.steps[STEPS[-1][0]].startswith("not_run") else set()
    ok = not stopped and all(c["status"] == "pass" for k, c in l1.checks.items() if k not in skipped_ok or c["status"] != "not_run")
    sc.set_verdict(ok, checks=l1.checks, steps=l1.steps, **l1.notes)
    assert ok, l1.steps


@pytest.mark.live
@pytest.mark.skipif(not (SNAP and ROOT), reason="needs LIVE_ACCEPTANCE_SNAPSHOT_B and LIVE_ACCEPTANCE_EVIDENCE_ROOT")
def test_live_run_l1():
    # hold every model call up front; the team design call is also held by prompt marker
    spec = {t: list(range(1, 21)) for t in ("codex", "claude")} | {"codex_match": [TH_MARKER]}
    with LiveScenario("live_run_l1", Path(SNAP), Path(ROOT), CAP, hold_spec=spec) as sc:
        run_l1(sc)


if __name__ == "__main__":
    if "--dry-plan" in sys.argv:
        for n, keys, text in STEPS:
            print(f"{n}: {text}  [{', '.join(keys) or '-'}]")
        print(f"checks: {', '.join(KEYS)}; cap {CAP}; TH marker {TH_MARKER!r}")
