"""Live run L2: a planner session that finishes while the run is paused is consumed exactly once, and only
after Resume, including across a server restart.

Product facts this run relies on (read from huddleroom/services, recorded 2026-10-18):

(a) After team_hierarchy the baseline has no further processes. effectiveness_review and goal_closeout run
    only after work starts (closeout when the run is ready to finish; the review on triggers such as
    pre_completion), so they are not on this path. Once team_hierarchy and agent_definition_review are
    terminal the tick moves the run to phase "ready" without a model call (orchestration_service.py, the
    "baseline_ready" block). Answering the team approval may still dispatch a model call; any such call is
    released one at a time and recorded.
(b) phase "ready" waits for an explicit Start (UI button "Start work", POST /goals/{id}/start,
    start_run: reserves authorize_execution and sets phase "authorized"). The next tick
    (_advance_authorized_execution_once) finds no accepted plan and calls request_llm_decision: ONE Sol call
    through the codex shim that chooses request_plan.
(c) request_plan creates a planning task assigned to the plan author (work function "planning"; the roster
    agent the approved team gives it), which runs as a claude_code agent session through cli_adapter: one
    CLI call through the claude shim. That is the planner; it is held by ordinal. Agent sessions time out
    after 3600 s by default, so a held planner does not need the 100 s clock the codex calls need.
(d) When the session completes, task.status becomes done and the next evidence pass records
    run:<run>:kind:report_consumed:task:<task_id> (consume_canonical_report). Results landing while the
    baseline is unauthorized pend as pending_result_continuation on an action and are later applied or
    settled (pending_result_continuation_settled, _settle_pending_result_continuations). Whether a result
    landing while PAUSED is consumed is what this run checks.

Cost estimate: remaining Sol calls ~1 (start decision) + 0-2 (team approval follow-ups) at ~19k each, one
claude planner session ~30k, later decisions stay held and cost 0: about 50-100k of the 150k cap.

Steps: restore snap-D (paused, team approval pending), Resume, Continue baseline, accept the team approval,
release Sol calls one at a time (max 3) until phase ready, Start work, release the start decision, wait for
the held planner, Pause, snapshot A, release the planner (it completes while paused), snapshot B, SIGTERM
restart, snapshot C, Resume, wait for exactly one consumption, recheck counts, snapshot D. Any model call
after that stays held and teardown kills the server first.

Run (spends subscription tokens; cap 150000):
  HUDDLEROOM_LIVE_ACCEPTANCE=1 HUDDLEROOM_LIVE_EVAL=1 LIVE_ACCEPTANCE_SNAPSHOT_D=<snap-D dir> \
  LIVE_ACCEPTANCE_EVIDENCE_ROOT=<evidence root> .venv/bin/python -m pytest -m live -s \
  tests/live_acceptance/test_live_run_l2.py
Print the plan: .venv/bin/python -m tests.live_acceptance.test_live_run_l2 --dry-plan
"""

from __future__ import annotations

import os
import signal
import sqlite3
import sys
import time
import uuid
from pathlib import Path

import pytest

from tests.live_acceptance.scenario import LiveScenario
from tests.live_acceptance.test_live_run import L1  # reuse the L1 step machinery

CAP = 150_000
MAX_SOL_RELEASES = 3
KEYS = ("planner_completes_paused", "survives_reload", "consumed_once_after_resume", "no_consumption_before_resume")
STEPS = (
    ("1-start-work", (), "Resume, Continue baseline, accept team approval, release Sol calls to phase ready, Start work"),
    ("2-planner-held", (), "release the start decision; wait for the held planner (claude) call"),
    ("3-pause-A", ("no_consumption_before_resume",), "Pause (accepted), snapshot A: planner held, nothing consumed"),
    ("4-planner-done-B", ("planner_completes_paused", "no_consumption_before_resume"),
     "release the planner; it completes while paused; snapshot B"),
    ("5-restart-C", ("survives_reload", "no_consumption_before_resume"), "SIGTERM, restart, snapshot C"),
    ("6-resume-consume-D", ("consumed_once_after_resume",), "Resume; exactly one consumption; counts stable; snapshot D"),
)
TERMINAL_SESSION = ("completed", "failed", "cancelled", "error", "timeout")

SNAP = os.environ.get("LIVE_ACCEPTANCE_SNAPSHOT_D")
ROOT = os.environ.get("LIVE_ACCEPTANCE_EVIDENCE_ROOT")


class L2(L1):
    SPEC = "tests-live/live-run-l2.spec.ts"

    def __init__(self, sc: LiveScenario):  # does not call L1.__init__: different keys and no batch
        self.sc = sc
        self.codex, self.claude = sc.shims["codex"], sc.shims["claude"]
        from tests.live_acceptance.scenario import _latest_ids
        self.ids = sc.ids = {**sc.ids, **_latest_ids(sc.run_dir.db)}
        self.run_id, self.goal_id, self.project_id = (self.ids[k] for k in ("run_id", "goal_id", "project_id"))
        self.checks = {k: {"status": "not_run", "evidence": []} for k in KEYS}
        self.steps: dict[str, str] = {n: "not_run" for n, _, _ in STEPS}
        self.released: list[int] = []  # codex ordinals released on purpose
        self.claude_released: list[int] = []
        self.release_log: list[dict] = []
        self.notes: dict = {}
        self.planner_task = None
        self.planner_inv = None
        self.pend_seen = False

    # -- db helpers
    def q(self, sql: str, *args):
        conn = sqlite3.connect(f"file:{self.sc.run_dir.db}?mode=ro", uri=True)
        try:
            return conn.execute(sql, args).fetchall()
        finally:
            conn.close()

    def run(self) -> dict:
        return self.sc.oracle.run_state(self.run_id)

    def plan_tasks(self) -> list[dict]:
        return [t for t in self.sc.oracle.tasks(self.goal_id)
                if ((t.get("metadata") or {}).get("orchestration") or {}).get("work_function") == "planning"]

    def consumed(self) -> list[dict]:
        """report_consumed markers for the planner task (empty before it is known)."""
        if self.planner_task is None:
            return []
        tail = f":task:{uuid.UUID(self.planner_task)}"
        return [a for a in self.sc.oracle.report_consumed(self.run_id) if a["idempotency_key"].endswith(tail)]

    def marker_counts(self) -> dict:
        n = lambda key: self.q("SELECT count(*) FROM orchestration_actions WHERE run_id = ? AND "
                               f"json_extract(request, '$.{key}') IS NOT NULL", self.run_id.replace("-", ""))[0][0]
        return {"pending": n("pending_result_continuation"), "settled": n("pending_result_continuation_settled")}

    def see_pend(self) -> None:
        """Remember whether a pend existed at a snapshot point (pends only occur under an unauthorized baseline)."""
        m = self.marker_counts()
        self.pend_seen = self.pend_seen or m["pending"] > 0
        self.notes["pend_seen"] = self.pend_seen

    def planner_session(self) -> list[dict]:
        return self.sc.oracle.sessions(self.planner_task) if self.planner_task else []

    def tasks_state(self) -> dict:
        ts = self.sc.oracle.tasks(self.goal_id)
        return {"ids": sorted(t["id"] for t in ts), "titles": sorted(str(t.get("title")) for t in ts)}

    # -- model call control
    def held(self, handle) -> list:
        return [i for i in handle.invocations() if i.exec_at is None and i.gates and not i.refused and not i.timed_out]

    def guard(self) -> None:
        """Claude calls are expected only for the planner; nothing may execute unless released on purpose."""
        stray = [i.ordinal for i in self.codex.invocations() if i.exec_at is not None and i.ordinal not in self.released]
        stray_c = [i.ordinal for i in self.claude.invocations() if i.exec_at is not None and i.ordinal not in self.claude_released]
        if stray or stray_c:
            raise AssertionError(f"unexpected executed calls: codex {stray}, claude {stray_c}")

    def release_codex(self, why: str) -> None:
        for inv in self.held(self.codex):
            if inv.ordinal in self.released:
                continue
            if len(self.released) >= MAX_SOL_RELEASES:
                self.notes["unexpected_held_codex"] = [{"ordinal": i.ordinal, "args": i.args[:300]} for i in self.held(self.codex)]
                raise AssertionError(f"more than {MAX_SOL_RELEASES} Sol calls needed (estimate exceeded)")
            self.released.append(inv.ordinal)
            run = self.run()
            self.release_log.append({"ordinal": inv.ordinal, "why": why, "phase": run["phase"], "args": inv.args[:200],
                                     "process_runs": [(p["process_type"], p["status"])
                                                      for p in self.sc.oracle.process_runs(goal_id=self.goal_id)]})
            self.notes["codex_releases"] = self.release_log
            self.sc.mark(f"releasing codex call {inv.ordinal} ({why})")
            for g in inv.gates:
                self.codex.release(g)
            self.write_verdict()
            break  # one call per pass, so each release is logged with the state that led to it

    def drive(self, timeout: float, done, why: str):
        """Release held Sol calls one at a time until done() is truthy."""
        end = time.time() + timeout
        while time.time() < end:
            self.guard()
            got = done()
            if got:
                return got
            self.release_codex(why)
            time.sleep(1)
        return None

    def no_new_work(self, before: dict) -> dict:
        d = self.diff(before)
        d["tasks"] = sorted(set(self.tasks_state()["ids"]) - set(before["tasks"]))
        return d

    def ids_plus(self) -> dict:
        return {**self.id_sets(), "tasks": set(self.tasks_state()["ids"])}

    # -- steps
    def s1(self):
        self.pw_ok("resume")
        if self.run()["phase"] == "baseline":
            self.pw_ok("continue-baseline")
        self.notes["answer_th"] = self.pw_ok("answer-th")["_dir"]
        ready = self.drive(300, lambda: self.run()["phase"] == "ready", "baseline step before ready")
        if not ready:
            raise AssertionError(f"phase never became ready (phase {self.run()['phase']})")
        self.notes["sol_calls_before_ready"] = len(self.released)
        self.notes["start_work"] = self.pw_ok("start-work")["_dir"]
        self.sc.oracle_dump("after-start")

    def planner_held(self):
        held = self.held(self.claude)
        if len(self.claude.invocations()) > 1:
            raise AssertionError("more than one claude call before the planner finished")
        return held[0] if held else None

    def s2(self):
        inv = self.drive(300, self.planner_held, "start decision")
        if inv is None:
            raise AssertionError("planner (claude) call never held")
        self.planner_inv = inv
        tasks = self.plan_tasks()
        if len(tasks) != 1:
            raise AssertionError(f"expected one planning task, found {len(tasks)}")
        self.planner_task = tasks[0]["id"]
        self.notes["planner"] = {"claude_ordinal": inv.ordinal, "gates": inv.gates, "task_id": self.planner_task,
                                 "agent_id": tasks[0].get("assigned_to"), "sol_calls_used": len(self.released)}
        self.sc.oracle_dump("planner-held")

    def pause_checks(self, label: str) -> dict:
        run = self.run()
        return {f"{label}_run_paused": run["status"] == "paused" and run["pause_requested"] == 1,
                f"{label}_nothing_consumed": not self.consumed()}

    def s3(self):
        before = self.ids_plus()
        d = self.pw("pause")
        self.sc.oracle_dump("planner-held-paused")
        diff = self.no_new_work(before)
        self.verify("no_consumption_before_resume", {
            "pause_ui_passed": d["_rc"] == 0, "accepted_under_5s": d.get("click_to_accepted_ms", 1e9) < 5000,
            **self.pause_checks("A"), "planner_still_held": self.planner_held() is not None,
            "no_new_work": self.nothing_new(diff) and not diff["tasks"]}, ui=d["_dir"], new_since_before_click=diff)
        self.see_pend()
        self.sc.take_snapshot("L2-A-planner-held-paused", require_clean=True)
        self.guard()

    def s4(self):
        before = self.ids_plus()
        for g in self.planner_inv.gates:
            self.claude.release(g)
        self.claude_released.append(self.planner_inv.ordinal)
        self.sc.mark("planner released while paused")

        def done():
            ss = self.planner_session()
            return ss if ss and ss[-1]["status"] in TERMINAL_SESSION else None
        ss = self.wait(300, done)
        time.sleep(10)  # time for a forbidden consumption or successor dispatch
        self.sc.oracle_dump("planner-done-paused")
        sessions = self.planner_session()
        s = sessions[-1] if sessions else {}
        tasks = {t["id"]: t for t in self.sc.oracle.tasks(self.goal_id)}
        diff = self.no_new_work(before)
        diff["sessions"] = [x for x in diff["sessions"] if x not in {r["id"] for r in sessions}]
        self.notes["planner_session"] = {"status": s.get("status"), "output_chars": len(str(s.get("output") or "")),
                                         "task_status": (tasks.get(self.planner_task) or {}).get("status")}
        self.verify("planner_completes_paused", {
            "session_terminal": ss is not None, "session_completed": s.get("status") == "completed",
            "output_stored": bool(str(s.get("output") or "").strip()), "one_planner_session": len(sessions) == 1,
            **self.pause_checks("B"), "no_new_work_after_pause": self.nothing_new(diff, allow_actions=()) and not diff["tasks"]},
            new_since_release=diff)
        self.verify("no_consumption_before_resume", {"B_nothing_consumed": not self.consumed()}, markers=self.marker_counts())
        self.see_pend()
        self.sc.take_snapshot("L2-B-planner-done-paused", require_clean=True)
        self.guard()

    def s5(self):
        live = {t: h.live_pids() for t, h in self.sc.shims.items() if h.live_pids()}
        if live:
            raise AssertionError(f"model CLI processes alive before restart; refusing to kill: {live}")
        before = self.ids_plus()
        sessions_before = [(x["id"], x["status"]) for x in self.planner_session()]
        self.sc.mark("SIGTERM server (nothing held)")
        self.sc.sup.kill(signal.SIGTERM)
        self.sc.sup.start()
        self.sc.mark("server restarted on the same DB")
        time.sleep(15)  # time for startup recovery to act (it must not consume while paused)
        self.sc.oracle_dump("after-restart")
        run = self.run()
        diff = self.no_new_work(before)
        snap = self.sc.take_snapshot("L2-C-after-restart", require_clean=False)
        self.notes["snap_C_clean"] = snap.get("clean")
        self.verify("survives_reload", {
            "run_paused": run["status"] == "paused" and run["pause_requested"] == 1,
            "session_unchanged": [(x["id"], x["status"]) for x in self.planner_session()] == sessions_before,
            "no_new_work": self.nothing_new(diff, allow_actions=()) and not diff["tasks"],
            "baseline_not_authorized": run["phase"] != "baseline" or run["baseline_authorized"] == 0})
        self.see_pend()
        self.verify("no_consumption_before_resume", {"C_nothing_consumed": not self.consumed()}, markers=self.marker_counts())
        self.guard()

    def s6(self):
        self.pw_ok("resume")
        if self.run()["phase"] == "baseline":
            self.pw_ok("continue-baseline")
        phase_at_resume = self.run()["phase"]
        got = self.wait(60, lambda: len(self.consumed()) >= 1)
        c1 = {"consumed": len(self.consumed()), "markers": self.marker_counts(), "tasks": self.tasks_state(),
              "sessions": len(self.planner_session()), "release": self.notes.get("codex_releases")}
        self.sc.oracle_dump("after-consume")
        snap = self.sc.take_snapshot("L2-D-consumed", require_clean=False)
        self.notes["snap_D_clean"] = snap.get("clean")
        time.sleep(30)  # two more ticks: nothing may change
        c2 = {"consumed": len(self.consumed()), "markers": self.marker_counts(), "tasks": self.tasks_state(),
              "sessions": len(self.planner_session())}
        # Task counts are evidence only: a result-correction task and a new claude call after consumption are
        # legitimate; any such call stays held (never released) and is recorded here.
        self.notes["after_consume"] = {"tasks": c2["tasks"], "claude_calls": [
            {"ordinal": i.ordinal, "held": i.exec_at is None} for i in self.claude.invocations()]}
        if phase_at_resume in ("baseline", "ready"):
            self.notes["spr241_deferred_note"] = f"consumed after Resume with phase {phase_at_resume}, no pend recorded"
        self.notes["pend_seen"] = self.pend_seen
        self.verify("consumed_once_after_resume", {
            "consumed_after_resume": got is not None, "exactly_one_marker": c1["consumed"] == 1 and c2["consumed"] == 1,
            "pending_absent": c2["markers"]["pending"] == 0, "settled_present_if_pend_seen": c2["markers"]["settled"] >= 1 or not self.pend_seen,
            "one_planner_session": c2["sessions"] == 1, "counts_stable": {k: c1[k] for k in ("consumed", "sessions")} == {k: c2[k] for k in ("consumed", "sessions")}},
            first=c1, second=c2)
        self.guard()


def run_l2(sc: LiveScenario) -> None:
    l2 = L2(sc)
    l2.write_verdict()
    stopped = False
    for (name, _, _), fn in zip(STEPS, (l2.s1, l2.s2, l2.s3, l2.s4, l2.s5, l2.s6)):
        if stopped:
            continue
        try:
            fn()
            l2.steps[name] = "pass"
        except Exception as e:  # noqa: BLE001 - first failure stops the run, rest stay not_run
            l2.steps[name] = f"fail: {type(e).__name__}: {e}"[:400]
            stopped = True
            sc.mark(f"STEP FAILED {name}: {l2.steps[name]}")
        l2.write_verdict()
    ok = not stopped and all(c["status"] == "pass" for c in l2.checks.values())
    sc.set_verdict(ok, checks=l2.checks, steps=l2.steps, **l2.notes)
    assert ok, l2.steps


@pytest.mark.live
@pytest.mark.skipif(not (SNAP and ROOT), reason="needs LIVE_ACCEPTANCE_SNAPSHOT_D and LIVE_ACCEPTANCE_EVIDENCE_ROOT")
def test_live_run_l2():
    spec = {t: list(range(1, 21)) for t in ("codex", "claude")}  # hold every model call up front
    with LiveScenario("live_run_l2", Path(SNAP), Path(ROOT), CAP, hold_spec=spec) as sc:
        run_l2(sc)


if __name__ == "__main__":
    if "--dry-plan" in sys.argv:
        for n, keys, text in STEPS:
            print(f"{n}: {text}  [{', '.join(keys) or '-'}]")
        print(f"checks: {', '.join(KEYS)}; cap {CAP}; max Sol releases {MAX_SOL_RELEASES}")
