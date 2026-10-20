"""Opt-in G5--G6 live acceptance: Run S5 (seed) and Run L5 (live), plus a zero-spend served smoke.

Reuses the G2--G4 flow and adds read-only UI/API/oracle observation points. Flow steps are hard
(the first failure stops the run); G5/G6 checks are soft (recorded, flow continues) except C6.
Importing this module and normal collection never create a run or spend tokens.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from tests.live_acceptance.cli_shim import seed_campaign
from tests.live_acceptance.dry_run import Api
from huddleroom.services.orchestration_manager_selection import DECISION_KEY_PREFIX, HUMAN_AS_MANAGER_OPTION
from tests.live_acceptance.fixtures import G2G4_GOAL_CONSTRAINTS, G5G6_GOAL_TEXT
from tests.live_acceptance.run_dir import real_home
from tests.live_acceptance.scenario import _latest_ids
from tests.live_acceptance.seed import create_goal
from tests.live_acceptance.test_live_run import FRONTEND, L1, TH_MARKER
from tests.live_acceptance.test_live_run_g2g4 import (
    PLAN_MARKER, G2G4, G2G4Scenario, _complete_s, _json_bytes, _ledger_totals_readonly,
    _live_agent_ids, _pending, _same_uuid,
)

SPEC5 = "tests-live/live-run-g5g6.spec.ts"
ROOT = os.environ.get("LIVE_ACCEPTANCE_EVIDENCE_ROOT")
SNAP = os.environ.get("LIVE_ACCEPTANCE_SNAPSHOT_S")
LIVE_ENABLED = (os.environ.get("HUDDLEROOM_LIVE_ACCEPTANCE") == "1"
                and os.environ.get("HUDDLEROOM_LIVE_EVAL") == "1")
HUMAN_MANAGER = os.environ.get("LIVE_ACCEPTANCE_G5G6_MANAGER") == "human"  # roster without a manager agent
PROBE = os.environ.get("LIVE_ACCEPTANCE_G5G6_PROBE") == "1"  # stop right after manager_selection completes
IGNORED_LEAVES = {"occurred_at", "fetched_at", "generated_at"}  # time-of-fetch fields
S_KEYS = ("C0", "C1", "C2", "C3", "C4", "C5", "C7", "D1", "D2", "D3", "D4")
L_KEYS = ("C8", "C9", "C10", "C11", "C12", "C13", "D1", "D2", "D4", "D5", "E1", "E2", "E3")
EXPECT = {  # receipt fields each step must write (a missing one is listed in results.md)
    "authorize-baseline": ("workflow", "facts", "controls", "authorize_button_name", "post_urls", "button_gone",
                           "active_element_after"),
    "status-baseline": ("workflow", "facts", "controls", "layout"), "layout": ("layout",),
    "status-pending": ("workflow", "facts", "controls", "question", "title", "question_visible",
                       "title_visible", "debug_tab_present"),
    "status-paused-pending": ("workflow", "facts", "controls", "question", "question_visible"),
    "status-saved": ("workflow", "facts", "after_reload"),
    "status-authorize": ("workflow", "facts", "controls", "authorize_button_name"),
    "status-th-pending": ("workflow", "facts", "controls", "question", "question_visible",
                          "th_summary_text", "th_step_text"),
    "status-paused-admitted": ("workflow", "facts", "controls", "live_claim_matches", "layout"),
    "calls-live": ("calls_tab_selected", "ask_aria_selected", "entry_appeared_at_ms", "entry_text"),
    "reload-calls-empty": ("facts", "empty_text_present", "live_note_present", "forbidden_matches"),
    "stale-refresh": ("stale_text", "controls_while_stale", "fresh_text_after_release", "refetch_trigger"),
    "calls-disconnected": ("disconnected_text_present", "connectivity_text", "shows_connected"),
    "status-plan-review": ("workflow", "facts", "controls", "question", "question_visible",
                           "start_implementation_offered", "layout"),
}
TRACE = (("SPR188", ("C1", "C3", "C6", "C9", "C10", "C11")),
         ("SPR232", ("C0", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C12", "C13")),
         ("SPR235", ("D1", "D2", "D3", "D4", "D5")), ("SPR190", ("E1", "E2", "E3")))
FIXTURE_ONLY = (
    "Other-run decisions never current (232): needs a second run; presenter unit tests",
    "Stale G3 review cycle not current (232): needs a revision cycle; unit-covered",
    "Blocked/Failed application copy (232): cannot be forced live; mocked Playwright",
    "Terminal goal inert; legacy null workflow: unit and mocked tests",
    "Zero-DML GETs (authoritative): unit test; C13 is supporting evidence only",
    "Mismatched cached revisions, goal/project switch: vitest/mocked",
    "Multi-feedback sticky-bar focus; 200% zoom: known unverified (not asserted even in fixtures)",
    "Codex resume argv read-only: observation only, not_exercised unless it occurs",
)


# -- small helpers ------------------------------------------------------------------------------
def _g(value, *path):
    """Defensive nested get: None when any hop is missing or not a dict."""
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _norm(text) -> str:
    return " ".join(str(text or "").split())


def _has(text, needle: str) -> bool:
    return needle.lower() in _norm(text).lower()


def _url(item) -> str:
    return str((item.get("url") if isinstance(item, dict) else item) or "").split("?")[0]


def _buttons(r) -> list[str]:
    return [str(b) for b in (_g(r, "controls", "visible_buttons") or [])]


def _diff(a, b, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b)) for p in _diff(a.get(k), b.get(k), f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _diff(x, y, f"{path}/{i}")]
    return [] if a == b else [path]


def _g5_cap() -> int:
    try:
        cap = int(os.environ.get("LIVE_ACCEPTANCE_G5G6_CAP", ""))
    except ValueError:
        cap = 0
    if cap <= 0:
        pytest.fail("LIVE_ACCEPTANCE_G5G6_CAP must be a positive integer")
    return cap


def _relabel(sc, mapping: dict[str, str]) -> None:
    """Reuse G2--G4 steps that hard-code snapshot labels."""
    orig = sc.take_snapshot
    sc.take_snapshot = lambda label, **kw: orig(mapping.get(label, label), **kw)


# -- isolation ledger ---------------------------------------------------------------------------
def _isolation_snapshot() -> dict:
    repo = Path(__file__).resolve().parents[2]
    paths = {str(repo / ".env")}
    for pattern in ("rally.db*", "rally_test.db*"):
        paths.update(str(p) for p in repo.glob(pattern))
    home = real_home() / ".huddleroom"
    if home.is_dir():
        paths.update(str(p) for p in home.rglob("*"))
    for pattern in filter(None, os.environ.get("LIVE_ACCEPTANCE_PROTECTED_GLOBS", "").split(os.pathsep)):
        paths.update(glob.glob(os.path.expanduser(pattern), recursive=True))
    out = {}
    for name in sorted(paths):
        p = Path(name)
        try:
            if p.is_dir():
                continue
            st = p.stat()
            digest = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            out[name] = {"sha256": digest.hexdigest(), "size": st.st_size, "mtime": st.st_mtime_ns}
        except FileNotFoundError:
            out[name] = None
        except OSError as exc:
            out[name] = {"error": str(exc)}
    return out


def _isolation_diff(before: dict, after: dict) -> dict:
    return {p: {"before": before.get(p), "after": after.get(p)}
            for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)}


def _execute(make, body) -> None:
    """Run body(sc) in a scenario with protected-file isolation evidence before and after."""
    before = _isolation_snapshot()
    sc, diff = None, {}
    try:
        with make() as sc:
            (sc.artifact_root / "isolation-before.json").write_text(json.dumps(before, indent=2))
            try:
                body(sc)
            finally:
                diff = _isolation_diff(before, _isolation_snapshot())
                sc.verdict = {**(sc.verdict or {"passed": False}), "isolation_diff": diff}
                if diff:
                    sc.verdict["passed"] = False
    finally:
        after = _isolation_snapshot()
        diff = _isolation_diff(before, after)
        dest = sc.artifact_root if sc is not None and hasattr(sc, "artifact_root") else Path(ROOT)
        (dest / "isolation-after.json").write_text(json.dumps(after, indent=2))
        verdict_file = dest / "verdict.json"
        if diff and verdict_file.exists():  # a diff that appeared during teardown
            data = json.loads(verdict_file.read_text())
            verdict_file.write_text(json.dumps({**data, "passed": False, "isolation_diff": diff}, indent=2))
    assert not diff, f"protected files changed: {sorted(diff)}"


# -- shared observation/recording behaviour -----------------------------------------------------
class G5Mixin:
    title = "G5G6"
    smoke = False

    def _g5_init(self, keys) -> None:
        self.soft_checks = {k: {"status": "not_run", "evidence": []} for k in keys}
        self.obs: dict = {}
        self.step_log: dict[str, list] = {}
        self.flow: dict[str, str] = {}
        self.objective = None
        self.calls_proc = None

    # -- playwright
    def pw5(self, name: str, **env) -> dict:
        self.SPEC = SPEC5  # instance attribute shadows the class SPEC for this call only
        try:
            r = self.pw(name, **env)
        finally:
            del self.SPEC
        self._log_step(name, r)
        return r

    def _log_step(self, name: str, r: dict) -> None:
        r["_step"] = name
        if name == "authorize-baseline" and "after_settle" not in r:  # only written when a settle wait is requested
            self.notes.setdefault("after_settle_absent", []).append(len(self.step_log.get(name, [])) + 1)
        self.step_log.setdefault(name, []).append({
            "rc": r.get("_rc"), "ui_errors": r.get("ui_errors"), "dir": r.get("_dir"),
            "missing_fields": [f for f in EXPECT.get(name, ()) if f not in r]})

    def pw_spec(self, spec: str, name: str) -> dict:
        self.SPEC = spec
        try:
            return self.pw_ok(name)
        finally:
            del self.SPEC

    # -- api
    def api_detail(self) -> dict:
        r = httpx.get(self.base, timeout=60)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def workflow_of(detail: dict) -> dict:
        return _g(detail, "supervision", "status_line", "workflow") or {}

    def checkpoint(self, label: str) -> dict:
        detail = self.api_detail()
        self.objective = _g(detail, "goal", "objective") or self.objective
        data = {"status_line": _g(detail, "supervision", "status_line"),
                "goal_status": _g(detail, "goal", "status"), "run": detail.get("run")}
        (self.sc.artifact_root / f"api-{label}.json").write_text(json.dumps(data, indent=2, default=str))
        self.sc.oracle_dump(label)
        data["workflow"] = (data["status_line"] or {}).get("workflow") or {}
        return data

    # -- checks
    def soft(self, key: str, conds: dict, step: str | None = None, label: str | None = None, **evidence) -> None:
        bad = sorted(k for k, v in conds.items() if not v)
        c = self.soft_checks.setdefault(key, {"status": "not_run", "evidence": []})
        c["status"] = "fail" if bad or c["status"] == "fail" else "pass"
        files = ([f"api-{label}.json"] if label else []) + ([f"playwright/{step}/{step}.json"] if step else [])
        c["evidence"].append({"conds": conds, "failed": bad, "files": files, **evidence})
        self.write_verdict()

    @staticmethod
    def rc_ok(r: dict) -> dict:
        return {f"pw_rc0_{r.get('_step')}": r.get("_rc") == 0}

    def d_checks(self, moment: str, r: dict, keys: tuple, label: str | None = None) -> None:
        lay = r.get("layout") or {}
        step = r.get("_step")
        for key in keys:
            conds: dict = {}
            for w in ("1440", "375"):
                d1, d3 = _g(lay, w, "d1"), _g(lay, w, "d3")
                if key == "D1":
                    els = _g(d1, "boxes") or {}
                    conds[f"{moment}_{w}_in_viewport"] = bool(els) and all(
                        _g(e, "inViewport") is True for e in els.values())
                    conds[f"{moment}_{w}_no_overflow"] = isinstance(d1, dict) and not d1.get("doc_overflow") \
                        and not d1.get("shell_overflow")
                if key == "D3":
                    conds[f"{moment}_{w}_found"] = _g(d3, "found") is True
                    conds[f"{moment}_{w}_text_is_objective"] = bool(self.objective) and \
                        _norm(_g(d3, "expanded_text")) == _norm(self.objective)
                    conds[f"{moment}_{w}_toggle"] = _g(d3, "toggle_text_after_open") == "Hide full objective"
                    conds[f"{moment}_{w}_collapses"] = _g(d3, "collapsed_after_second_enter") is True
            if key == "D2":
                conds[f"{moment}_1440_bar_in_viewport"] = _g(lay, "1440", "d2", "bar_in_viewport_after_scroll") is True
            if key == "D4":
                conds[f"{moment}_1440_no_violations"] = _g(lay, "1440", "d4", "violations") == []
            extra = {"raw_violations": _g(lay, "1440", "d4", "violations")} if key == "D4" else {}
            self.soft(key, conds, step=step, label=label, **extra)

    def eval_c10_e2(self, before: dict, after: dict, r: dict, sfx: str = "") -> None:
        changed = _diff(before["workflow"], after["workflow"])
        real = [p for p in changed if p.rsplit("/", 1)[-1] not in IGNORED_LEAVES]
        self.soft("C10" + sfx, {
            "workflow_equal_ignoring_fetch_time": not real,
            "badge_paused": _has(_g(r, "facts", "goal_badge_text"), "Paused"),
            "paused_copy": _g(r, "facts", "paused_copy_present") is True, **self.rc_ok(r)},
            step="reload-calls-empty", workflow_diff=changed)
        self.soft("E2" + sfx, {"empty_text": r.get("empty_text_present") is True,
                               "live_note": r.get("live_note_present") is True,
                               "no_forbidden": r.get("forbidden_matches") == []}, step="reload-calls-empty")

    def eval_d5(self, r: dict, sfx: str = "") -> None:
        buttons = [str(b) for b in (_g(r, "controls_while_stale", "visible_buttons") or [])]
        self.soft("D5" + sfx, {
            "stale_copy": _has(r.get("stale_text"), "this status may be out of date"),
            "pause_resume_still_rendered": any(re.match(r"(?i)\s*(pause|resume)", b) for b in buttons),
            "refreshed_after_release": _has(r.get("fresh_text_after_release"), "Status refreshed"),
            **self.rc_ok(r)}, step="stale-refresh", refetch_trigger=r.get("refetch_trigger"))

    def eval_e3(self, r: dict, sfx: str = "") -> None:
        self.soft("E3" + sfx, {"disconnected_text": r.get("disconnected_text_present") is True,
                               "connectivity_text_recorded": bool(r.get("connectivity_text")),
                               "not_shown_connected": r.get("shows_connected") is False, **self.rc_ok(r)},
                  step="calls-disconnected", connectivity=r.get("connectivity_text"))

    # -- human-manager variant
    def check_human_manager(self, answer_override: bool = False) -> None:
        """Drive until manager_selection completes, then require the human as manager (fail fast otherwise).

        answer_override (full run only): a single pending review-override decision whose deterministic
        recommendation is the human is answered once through the product API the UI uses."""
        override_key = f"{DECISION_KEY_PREFIX}review_override"
        answered: list[str] = []

        def done():
            pending = [d for d in self.sc.oracle.decisions(self.goal_id)
                       if d["status"] == "pending" and not str(d["decision_key"]).startswith("agent_definition_review")]
            if (answer_override and len(pending) == 1 and pending[0]["decision_key"] == override_key
                    and not answered):
                self.answer_review_override(pending[0])
                answered.append(pending[0]["id"])
                self.notes.setdefault("manager_selection", {})["override_answered"] = True
                return None
            if pending:  # a manager choice/override needing the human is not "selection picks the human"
                raise AssertionError("manager selection needs a human answer instead of completing: "
                                     f"{[str(d['decision_key']) for d in pending]}")
            return next((p for p in self.sc.oracle.process_runs(goal_id=self.goal_id)
                         if p["process_type"] == "manager_selection" and p["status"] == "completed"
                         and not p.get("superseded_by_id")), None)

        process = self.drive_codex(300, done, "manager selection")
        if process is None:
            raise AssertionError("manager_selection did not complete within 300s")
        auth = self.sc.oracle.goal_authority(self.goal_id) or {}
        outputs = self.sc.oracle.process_outputs(process["id"]) or {}
        self.notes["manager_selection"] = {**self.notes.get("manager_selection", {}), "authority_model": auth.get("authority_model"),
                                           "manager_user_id": auth.get("manager_user_id"),
                                           "manager_agent_id": auth.get("manager_agent_id"),
                                           "manager_review": outputs.get("manager_review")}
        self.obs["manager_selection"] = self.notes["manager_selection"]
        self.write_verdict()
        if auth.get("authority_model") != "human_manager" or not auth.get("manager_user_id"):
            raise AssertionError(f"expected authority_model human_manager with manager_user_id set: {auth}")

    def answer_review_override(self, decision: dict) -> None:
        """Answer the LLM-override review so the outcome is the human, via the same endpoint the UI posts to."""
        context = str(decision.get("context") or "")
        det = re.search(r"Deterministic:\s*(\S+)", context)
        llm = re.search(r"LLM:\s*(\S+)", context)
        if not det or det.group(1) != HUMAN_AS_MANAGER_OPTION:
            raise AssertionError(f"override review: deterministic recommendation is not the human: {context!r}")
        # approve keeps the LLM choice (only if it is the human); reject falls back to the deterministic human
        option = "approve" if llm and llm.group(1) == HUMAN_AS_MANAGER_OPTION else "reject"
        keys = {o.get("key") for o in decision.get("options") or [] if isinstance(o, dict)}
        if option not in keys:
            raise AssertionError(f"override review lacks option {option!r}: {keys}")
        self.budget_ok()
        r = httpx.post(f"{self.base}/decisions/{decision['id']}/answer", timeout=60, json={
            "selected_option": option, "reason": "live acceptance: the human acts as manager"})
        r.raise_for_status()
        self.notes.setdefault("manager_selection", {})["override_answer"] = option
        self.sc.mark(f"answered manager review override with {option}")

    # -- verdict / results
    def write_verdict(self) -> None:
        try:  # spend.used must match the ledger at write time (read-only)
            totals = _ledger_totals_readonly(self.sc.run_dir.spend)
            self.notes["spend"] = {"used": totals["global"]["non_cached"], "totals": totals}
        except (OSError, ValueError, KeyError):
            pass
        (self.sc.artifact_root / "verdict.json").write_text(json.dumps(
            {"passed": None, "flow": self.flow, "checks": self.checks, "soft_checks": self.soft_checks,
             "observations": self.obs, "step_log": self.step_log, "notes": self.notes}, indent=2, default=str))

    def _status(self, key: str) -> tuple[str, list]:
        c = self.soft_checks.get(key) or self.checks.get(key) or {"status": "not_run", "evidence": []}
        files = sorted({f for e in c["evidence"] if isinstance(e, dict) for f in e.get("files", [])})
        return c["status"], files

    def write_results(self) -> None:
        sc = self.sc
        lines = [f"# G5--G6 live acceptance: {self.title}", "", "## Flow steps", "", "| step | status |", "|---|---|"]
        lines += [f"| {n} | {s} |" for n, s in self.flow.items()]
        lines += ["", "## Playwright steps (rc, ui_errors, fields missing from the receipt)", "",
                  "| step | rc | ui_errors | missing fields |", "|---|---|---|---|"]
        for name, rows in self.step_log.items():
            for row in rows:
                lines.append(f"| {name} | {row['rc']} | {json.dumps(row['ui_errors'])[:200]} | {row['missing_fields']} |")
        keys = [*self.soft_checks, *([k for k in ("C6",) if k in self.checks and self.title.startswith("S")])]
        lines += ["", "## Checks", "", "| id | status | evidence |", "|---|---|---|"]
        for key in sorted(keys):
            status, files = self._status(key)
            lines.append(f"| {key} | {status} | {', '.join(files)} |")
        lines += ["", "## SPR traceability", ""]
        for spr, ids in TRACE:
            lines += [f"### {spr}", "", "| id | status | evidence |", "|---|---|---|"]
            for key in ids:
                status, files = self._status(key)
                lines.append(f"| {key} | {status} | {', '.join(files)} |")
            lines.append("")
        lines += ["## Fixture-only (not run live)", "", *[f"- fixture-only: {item}" for item in FIXTURE_ONLY]]
        lines += ["", "## Observations (not pass/fail)", ""]
        lines += [f"- {k}: {json.dumps(v, default=str)[:600]}" for k, v in self.obs.items()]
        lines += [f"- codex resume: {self.notes.get('codex_resume', 'not_exercised')}"]
        try:
            run = sc.oracle.run_state(self.run_id) if self.run_id else None
            ledger = _ledger_totals_readonly(sc.run_dir.spend)["global"]
            campaign = _ledger_totals_readonly(sc.campaign_ledger_path)["global"]
        except Exception as exc:  # noqa: BLE001 - results must still be written
            run, ledger, campaign = None, {"error": str(exc)}, {}
        plan_decision = getattr(self, "plan_decision", None)
        lines += ["", "## Spend (non-cached / cached)", "", f"- scenario ledger: {ledger}", f"- campaign ledger: {campaign}",
                  "", "## Snapshots", "", *[f"- {k}: {v}" for k, v in sc.snapshots.items()],
                  "", "## Final state", "",
                  f"- run status/phase/plan_state.status: {(run or {}).get('status')} / {(run or {}).get('phase')} / "
                  f"{((run or {}).get('plan_state') or {}).get('status')}",
                  f"- project_id={getattr(self, 'project_id', None)} goal_id={getattr(self, 'goal_id', None)} "
                  f"run_id={self.run_id} plan_decision_id={(plan_decision or {}).get('id')}"]
        (sc.artifact_root / "results.md").write_text("\n".join(lines) + "\n")

    def cleanup_calls_proc(self) -> None:
        """Kill only the Playwright process group started for calls-live; never a model CLI."""
        proc = getattr(self, "calls_proc", None)
        if proc is not None:  # pgid == proc.pid (own session); also reaps stray children of a finished parent
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.notes["calls_live_unreaped"] = proc.pid

    def run_flow(self, steps, always=()) -> bool:
        self.flow = {name: "not_run" for name, _ in steps}
        failed = False
        self.write_verdict()
        try:
            for name, fn in steps:
                if failed and name not in always:
                    continue
                try:
                    fn()
                    self.flow[name] = "pass"
                except Exception as exc:  # noqa: BLE001 - first failure stops the flow
                    self.flow[name] = f"fail: {type(exc).__name__}: {exc}"[:400]
                    self.sc.mark(f"STEP FAILED {name}: {exc}")
                    failed = True
                self.write_verdict()
                self.write_results()
        finally:
            self.cleanup_calls_proc()
        return not failed

    def finish(self, flow_ok: bool, hard=()) -> None:
        ok = flow_ok and all(self.checks[k]["status"] == "pass" for k in hard)
        failures = sorted(k for k, c in self.soft_checks.items() if c["status"] == "fail")
        self.sc.set_verdict(ok, flow=self.flow, checks=self.checks, soft_checks=self.soft_checks,
                            soft_failures=failures, observations=self.obs, step_log=self.step_log, notes=self.notes)
        self.write_results()
        assert ok, self.flow


# -- Run S5 -------------------------------------------------------------------------------------
class G5S(G5Mixin, G2G4):
    title = "S5"

    @classmethod
    def prepare(cls, sc, keys=S_KEYS) -> "G5S":
        goal = create_goal(sc.base_url, sc.ids["project_id"], G5G6_GOAL_TEXT, goal_type="roadmap",
                           constraints=G2G4_GOAL_CONSTRAINTS)
        sc.ids["goal_id"] = goal["goal"]["id"]
        d = cls.__new__(cls)
        d.sc, d.codex, d.claude, d.released, d.claude_released = sc, sc.shims["codex"], sc.shims["claude"], [], []
        d.ids = sc.ids = {**sc.ids, **_live_agent_ids(sc)}
        d.project_id, d.goal_id, d.run_id = d.ids["project_id"], d.ids["goal_id"], None
        d.base = f"{sc.base_url}/api/v1/projects/{d.project_id}/orchestration/goals/{d.goal_id}"
        d.notes, d.steps = {"release_timeline": []}, {}
        d.checks = {"C6": {"status": "not_run", "evidence": []}}
        d.p2 = d.planner_task = d.planner_inv = None
        d.feedback, d.adr_ids = {}, []
        d._g5_init(keys)
        _relabel(sc, {"S-th-approval-pending-paused": "S5-th-pending-paused"})
        return d

    def bind_run(self) -> None:
        end = time.monotonic() + 30
        while time.monotonic() < end and not _latest_ids(self.sc.run_dir.db).get("run_id"):
            time.sleep(1)
        self.ids.update(_latest_ids(self.sc.run_dir.db))
        self.run_id = self.ids.get("run_id")

    def authorize_posts(self, r: dict) -> list:
        return [u for u in (r.get("post_urls") or []) if _url(u).endswith("/baseline/authorize")]

    # S0
    def s0(self) -> None:
        pre = self.checkpoint("S0-pre-authorize")
        r = self.pw5("authorize-baseline", LIVE_ACCEPTANCE_CLICK="1")
        posts = self.authorize_posts(r)
        self.notes["s0_post_urls"] = r.get("post_urls")
        self.bind_run()
        if not posts:  # the UI click never posted the authorize request
            self.notes["s0_fallback"] = "Api.start_baseline (UI click did not post exactly one authorize)"
            self.sc.mark("S0 fallback to Api.start_baseline")
            Api(self.sc.base_url, self.project_id, self.goal_id).start_baseline()
            self.bind_run()
        if not self.run_id:
            raise AssertionError("no run exists after authorizing the baseline")
        self.eval_c0(pre, r, posts, "S0-pre-authorize")

    def eval_c0(self, pre: dict, r: dict, posts: list, label: str, key: str = "C0") -> None:
        guidance = _g(pre, "workflow", "guidance") or {}
        text = guidance.get("label")
        self.soft(key, {
            "api_kind_authorize_baseline": guidance.get("kind") == "authorize_baseline",
            "api_label": text == "Authorize baseline",
            "ui_button_equals_label": bool(text) and r.get("authorize_button_name") == text,
            "no_continue_or_start": not any(re.match(r"(?i)\s*(continue|start)", b) for b in _buttons(r)),
            "one_authorize_post": len(posts) == 1, **self.rc_ok(r)}, step="authorize-baseline", label=label)

    # S1
    def s1(self) -> None:
        if HUMAN_MANAGER:
            self.check_human_manager(answer_override=True)

        def review_held():
            running = any(p["process_type"] == "agent_definition_review" and p["status"] == "running"
                          for p in self.sc.oracle.process_runs(goal_id=self.goal_id))
            return running and bool(self._held(self.codex)) and not self._executing()

        if not self.drive_codex(300, review_held, "baseline to first review"):
            raise AssertionError("no agent_definition_review running with a held codex call")
        r = self.pw5("status-baseline")
        api = self.checkpoint("S1-baseline")
        text = _g(r, "facts", "current_operation_text") or ""
        self.soft("C1", {
            "api_phase_baseline": api["workflow"].get("phase") == "baseline",
            "operation_not_only_idle": bool(_norm(text)) and not re.fullmatch(
                r"(?i)\s*(no active worker|unknown|idle)\.?\s*", text),
            "operation_names_review": _has(text, "review"),
            "operation_has_time": bool(re.search(r"\d", text)),
            "no_live_activity_claim": not re.search(r"(?i)\b(now|currently) (running|working)\b|\bis (running|working)\b", text),
            **self.rc_ok(r)}, step="status-baseline", label="S1-baseline", operation_text=text)
        self.d_checks("C1", r, ("D1", "D2", "D3", "D4"), label="S1-baseline")
        if not self.drive_codex(300, self.adr_ready, "baseline review"):
            raise AssertionError("agent-definition reviews did not all reach pending")

    # S2
    def s2(self) -> None:
        pending = [d for d in self.sc.oracle.decisions(self.goal_id)
                   if d["status"] == "pending" and str(d["decision_key"]).startswith("agent_definition_review")]
        self.adr_ids = [d["id"] for d in pending]
        r = self.pw5("status-pending")
        api = self.checkpoint("S2-pending")
        wf, g = api["workflow"], api["workflow"].get("guidance") or {}
        current = next((d for d in pending if _same_uuid(d["id"], wf.get("pending_decision_id"))), None)
        question = _norm(r.get("question"))
        label = g.get("label")
        buttons = _buttons(r)
        next_texts = _g(r, "facts", "next_action_text") or []
        debug = {}
        if r.get("debug_tab_present") is True:
            debug = {"debug_separate": r.get("debug_separate") is True,
                     "debug_pending_label": r.get("debug_pending_text") is True,
                     "debug_runtime_waits_label": r.get("debug_runtime_waits_text") is True}
        else:
            self.obs["C2_debug"] = "not_exercised: debug tab disabled"
        self.soft("C2", {
            "pending_is_adr": current is not None, "additional_pending_is_adr_count_minus_1": wf.get("additional_pending_count") == self.notes["adr_count"] - 1,
            "question_visible": r.get("question_visible") is True and bool(question),
            "pending_id_matches_receipt": _same_uuid(r.get("pending_decision_id"), wf.get("pending_decision_id")),
            "answer_in_needs_you": _g(r, "facts", "answer_in_needs_you_present") is True,
            "next_action_is_guidance_label": bool(label) and any(_has(t, label) for t in next_texts),
            "guidance_decision_id": _same_uuid(g.get("decision_id"), wf.get("pending_decision_id")),
            "no_continue_or_start": not any(re.match(r"(?i)\s*(continue|start)", b) for b in buttons),
            **debug, **self.rc_ok(r)}, step="status-pending", label="S2-pending", question=question)

    # S3
    def s3(self) -> None:
        self.pw_ok("pause")
        r = self.pw5("status-paused-pending")
        self.checkpoint("S3-paused-pending")
        self.soft("C3", {
            "badge_paused": _has(_g(r, "facts", "goal_badge_text"), "Paused"),
            "paused_copy": _g(r, "facts", "paused_copy_present") is True,
            "question_visible": r.get("question_visible") is True,
            "next_action_resume": _has(_g(r, "facts", "next_action_text"), "Resume"),
            "run_paused": self.sc.oracle.run_state(self.run_id)["status"] == "paused",
            **self.rc_ok(r)}, step="status-paused-pending", label="S3-paused-pending")

    # S4
    def s4(self) -> None:
        self.pw_spec(L1.SPEC, "approve-batch")
        r = self.pw5("status-saved")
        self.checkpoint("S4-saved")
        state = self.apply_state(self.adr_ids, completed=False)
        self.soft("C4", {
            "saved_not_applied": _g(r, "facts", "saved_not_applied_present") is True,
            "saved_after_reload": _g(r, "after_reload", "saved_not_applied_present") is True,
            **{f"oracle_{k}": v for k, v in state.items()}, **self.rc_ok(r)}, step="status-saved", label="S4-saved")

    # S5
    def s5(self) -> None:
        self.pw_ok("resume")
        r = self.pw5("status-authorize")
        api = self.checkpoint("S5-authorize")
        g = api["workflow"].get("guidance") or {}
        self.soft("C5", {
            "api_kind_authorize_baseline": g.get("kind") == "authorize_baseline",
            "api_label": g.get("label") == "Authorize remaining baseline",
            "button_equals_label": r.get("authorize_button_name") == g.get("label") == "Authorize remaining baseline",
            "no_generic_continue": not any(re.match(r"(?i)\s*continue", b) for b in _buttons(r)),
            **self.rc_ok(r)}, step="status-authorize", label="S5-authorize")

    # S6
    def s6(self) -> None:
        r = self.pw5("authorize-baseline", LIVE_ACCEPTANCE_CLICK="1", LIVE_ACCEPTANCE_SETTLE_MS="30000")
        self.obs["G-2_active_element_after"] = r.get("active_element_after")
        posts = self.authorize_posts(r)
        self.checkpoint("S6-applied")
        conds = {**{f"apply_{k}": v for k, v in self.apply_state(self.adr_ids, completed=True).items()},
                 "one_apply_action": self.answer_actions() == 1,
                 "one_receipt": len(self.db_receipts(self.adr_ids)) == 1,
                 "ui_saved_gone_after_settle": _g(r, "after_settle", "saved_not_applied_present") is False,
                 "one_authorize_post": len(posts) == 1, **self.rc_ok(r)}
        time.sleep(2)
        conds["still_one_action"] = self.answer_actions() == 1
        conds["still_one_receipt"] = len(self.db_receipts(self.adr_ids)) == 1
        self.verify("C6", conds, step="authorize-baseline")

    # S7
    def s7(self) -> None:
        self.p1 = self.drive_codex(300, lambda: next((p for p in self.sc.oracle.process_runs(goal_id=self.goal_id)
            if p["process_type"] == "team_hierarchy" and p["status"] == "waiting_decision"), None), "baseline hierarchy")
        if self.p1 is None:
            raise AssertionError("P1 did not reach waiting_decision")
        live = [p for p in self.sc.oracle.process_runs(goal_id=self.goal_id)
                if p["process_type"] == "team_hierarchy" and not p.get("superseded_by_id")]
        self.verify("C6", {"one_unsuperseded_team_hierarchy": len(live) == 1}, step="authorize-baseline")
        r = self.pw5("status-th-pending")
        api = self.checkpoint("S7-th-pending")
        decision = _pending(self.sc.oracle, self.goal_id, "team_hierarchy:approval")
        self.obs["G-1_th_screenshots"] = sorted(glob.glob(str(Path(r.get("_dir", "")) / "*.png")))
        self.obs["G-1_th_summary_text"] = r.get("th_summary_text")
        self.obs["G-1_th_step_text"] = r.get("th_step_text")
        self.soft("C7", {
            "oracle_th_decision_pending": decision is not None,
            "api_pending_is_th": decision is not None and _same_uuid(api["workflow"].get("pending_decision_id"), decision["id"]),
            "question_visible": r.get("question_visible") is True,
            "question_present": bool(_norm(r.get("question"))), **self.rc_ok(r)},
            step="status-th-pending", label="S7-th-pending")

    # S8
    def s8(self) -> None:
        _complete_s(self.sc, self, self.p1)

    def steps_s(self):
        return [("S0-authorize", self.s0), ("S1-baseline-review", self.s1), ("S2-pending", self.s2),
                ("S3-pause", self.s3), ("S4-answer-saved", self.s4), ("S5-resume", self.s5),
                ("S6-authorize-remaining", self.s6), ("S7-th-pending", self.s7), ("S8-pause-snapshot", self.s8)]

    def steps_probe(self):
        """Paid but minimal: authorize, let manager selection run, assert the human was chosen, stop."""
        return [("S0-authorize", self.s0), ("manager-selection", self.check_human_manager)]

    # -- smoke (zero spend: nothing is ever released)
    def steps_smoke(self):
        def authorize():
            # Prior campaign spend is allowed; the smoke must add nothing to it.
            self.notes["smoke_zero_spend"] = {
                "campaign_before": _ledger_totals_readonly(self.sc.campaign_ledger_path)["global"]["non_cached"]}
            pre = self.checkpoint("smoke-pre-authorize")
            r = self.pw5("authorize-baseline", LIVE_ACCEPTANCE_CLICK="1")
            posts = self.authorize_posts(r)
            self.notes["smoke_post_urls"] = r.get("post_urls")
            self.bind_run()
            self.eval_c0(pre, r, posts, "smoke-pre-authorize")
            if not (r["_rc"] == 0 and len(posts) == 1):
                raise AssertionError(f"expected exactly one POST ending /baseline/authorize: {r.get('post_urls')}")

        def held():
            if not self.passive(180, lambda: bool(self._held(self.codex))):
                raise AssertionError("no held codex call within 180s")

        def observe():
            # Pause first so the reload check (C10) sees a paused goal; resume/pause exercised at the end.
            self._log_step("g2g4-pause", self.pw("pause"))
            self.pw5("status-baseline")
            self.pw5("layout")
            before = self.checkpoint("smoke-before-reload")
            r = self.pw5("reload-calls-empty")
            self.eval_c10_e2(before, self.checkpoint("smoke-after-reload"), r, sfx="(dry)")
            self.eval_d5(self.pw5("stale-refresh"), sfx="(dry)")
            self.eval_e3(self.pw5("calls-disconnected"), sfx="(dry)")
            for name in ("resume", "pause"):
                self._log_step(f"g2g4-{name}", self.pw(name))
            bad = {n: rows for n, rows in self.step_log.items() if any(row["rc"] != 0 for row in rows)}
            if bad:
                raise AssertionError(f"playwright steps with non-zero rc: {sorted(bad)}")

        def ledger():
            scenario = self.sc.ledger.totals()["global"]["non_cached"]
            campaign = _ledger_totals_readonly(self.sc.campaign_ledger_path)["global"]["non_cached"]
            zs = self.notes["smoke_zero_spend"]
            before = zs.get("campaign_before")
            delta = None if before is None else campaign - before
            executed = [(t, i.ordinal) for t, h in self.sc.shims.items() for i in h.invocations() if i.exec_at is not None]
            zs.update({"scenario": scenario, "campaign_after": campaign, "campaign_delta": delta,
                       "executed": executed, "executing": self._executing()})
            if scenario or before is None or campaign != before or executed or self._executing() or self.released or any(self.sc.released_ordinals.values()):
                raise AssertionError(f"smoke spent tokens, executed or released a call: {self.notes['smoke_zero_spend']}")

        return [("authorize", authorize), ("held-codex-call", held), ("observe", observe), ("zero-spend", ledger)]


# -- Run L5 -------------------------------------------------------------------------------------
class G5L(G5Mixin, G2G4):
    title = "L5"

    def __init__(self, sc):
        super().__init__(sc)
        self._g5_init(L_KEYS)
        self.base = f"{sc.base_url}/api/v1/projects/{self.project_id}/orchestration/goals/{self.goal_id}"
        self.release_ms = None
        _relabel(sc, {"L-th-approved": "L5-th-approved"})

    def resume_and_continue(self) -> None:
        """Resume only: with P1 approval pending the guidance is review_decision, not authorize_baseline."""
        initial = self.sc.oracle.run_state(self.run_id)
        if initial["phase"] != "baseline":
            raise AssertionError(f"restored P1 must resume from baseline, got {initial['phase']!r}")
        self.pw_ok("resume")
        self.steps["resume"] = "pass"
        api = self.checkpoint("L1-resumed")
        r = self.pw5("status-th-pending")  # P1 question view (C8 "before")
        decision = _pending(self.sc.oracle, self.goal_id, "team_hierarchy:approval")
        guidance = api["workflow"].get("guidance") or {}
        self.obs["L1_guidance_after_resume"] = {
            "guidance": guidance, "p1_decision_id": (decision or {}).get("id"),
            "kind_is_review_decision": guidance.get("kind") == "review_decision",
            "decision_matches": decision is not None and _same_uuid(guidance.get("decision_id"), decision["id"]),
            "visible_buttons": _buttons(r), "p1_question": _norm(r.get("question"))}

    def authorize_after_answer(self) -> None:
        last: dict = {}
        end = time.monotonic() + 90
        while time.monotonic() < end:
            self.guard()
            last = self.workflow_of(self.api_detail())
            if (last.get("guidance") or {}).get("kind") == "authorize_baseline":
                break
            time.sleep(2)
        else:
            raise AssertionError(f"guidance never became authorize_baseline (product-defect candidate): {last}")
        self.checkpoint("L1-authorize-guidance")
        r = self.pw5("authorize-baseline", LIVE_ACCEPTANCE_CLICK="1")
        self.obs["L1_authorize_button_name"] = r.get("authorize_button_name")
        self.obs["L1_active_element_after"] = r.get("active_element_after")
        run = self.sc.oracle.run_state(self.run_id)
        if r["_rc"] != 0 or run["phase"] != "baseline" or not run["baseline_authorized"]:
            raise AssertionError(f"authorize-after-answer did not authorize the baseline: rc={r['_rc']} run={run}")
        self.steps["continue-baseline"] = "pass"

    def request_changes(self) -> None:
        self.checkpoint("L1-before-request-changes")
        super().request_changes()

    def revision(self) -> None:
        super().revision()
        self.checkpoint("L2-after-revision")

    def c8(self) -> None:
        r = self.pw5("status-th-pending")
        api = self.checkpoint("L2-th-p2-pending")
        decision = _pending(self.sc.oracle, self.goal_id, "team_hierarchy:approval")
        current = api["workflow"].get("pending_decision_id")
        history = (self.notes.get("revision_ui") or {}).get("p1_history_controls")
        self.soft("C8", {
            "p2_decision_pending": decision is not None,
            "api_pending_is_p2": decision is not None and _same_uuid(current, decision["id"]),
            "not_p1_decision": not _same_uuid(current, self.notes.get("p1_approval_decision_id")),
            "question_visible": r.get("question_visible") is True,
            "p1_history_inert": history == "pass" or str(history).startswith("not_exercised:"),
            **self.rc_ok(r)}, step="status-th-pending", label="L2-th-p2-pending",
            question=_norm(r.get("question")), p1_decision=self.notes.get("p1_approval_decision_id"))

    # -- L4: planner hooks
    def before_planner_release(self, inv) -> None:
        try:
            r = self.pw5("status-paused-admitted")
            self.checkpoint("L4-paused-admitted")
            self.soft("C9", {
                "badge_paused": _has(_g(r, "facts", "goal_badge_text"), "Paused"),
                "paused_copy": _g(r, "facts", "paused_copy_present") is True,
                "operation_refers_to_planning": _has(_g(r, "facts", "current_operation_text"), "plan"),
                "no_unqualified_live_claim": _g(r, "live_claim_matches") == [], **self.rc_ok(r)},
                step="status-paused-admitted", label="L4-paused-admitted")
            self.d_checks("C9", r, ("D1", "D2", "D4"), label="L4-paused-admitted")
            self.start_calls_live()
        except Exception as exc:  # noqa: BLE001 - observation must never block the release
            self.notes["before_planner_release_error"] = f"{type(exc).__name__}: {exc}"
            self.soft("C9", {"hook_ran": False})
        self.release_ms = int(time.time() * 1000)

    def start_calls_live(self) -> None:
        name = "calls-live"
        out = self.sc.artifact_root / "playwright" / name
        out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "LIVE_ACCEPTANCE_BASE_URL": self.sc.base_url, "LIVE_ACCEPTANCE_OUTPUT": str(out),
               "LIVE_ACCEPTANCE_PROJECT_ID": self.project_id, "LIVE_ACCEPTANCE_GOAL_ID": self.goal_id,
               "LIVE_ACCEPTANCE_CALLS_WAIT_MS": "300000"}
        env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(real_home() / "Library/Caches/ms-playwright"))
        self.calls_log = open(out.parent / f"{name}.log", "w")
        self.calls_out = out
        self.calls_proc = subprocess.Popen(
            ["npx", "playwright", "test", "-c", "playwright.live.config.ts", SPEC5, "--grep", rf"(^|\s){name}$"],
            cwd=FRONTEND, env=env, stdout=self.calls_log, stderr=subprocess.STDOUT,
            start_new_session=True)  # own process group so cleanup never touches anything else
        end = time.monotonic() + 120
        while time.monotonic() < end and self.calls_proc.poll() is None and not (out / "calls-live.ready").exists():
            time.sleep(1)
        self.notes["calls_live_ready"] = (out / "calls-live.ready").exists()
        self.sc.mark(f"calls-live ready={self.notes['calls_live_ready']}")

    def after_planner_complete(self) -> None:
        r: dict = {"_step": "calls-live"}
        proc = self.calls_proc
        if proc is not None:
            try:
                proc.wait(timeout=400)
            except subprocess.TimeoutExpired:
                self.cleanup_calls_proc()
                self.notes["calls_live_killed"] = True
            self.calls_log.close()
            self.calls_proc = None
            f = self.calls_out / "calls-live.json"
            try:
                r = json.loads(f.read_text()) if f.exists() else {}
            except (OSError, ValueError):
                r = {}
            r["_rc"], r["_dir"] = proc.returncode, str(self.calls_out)
            self._log_step("calls-live", r)
        appeared = r.get("entry_appeared_at_ms")
        self.obs["G-3_ask_aria_selected"] = r.get("ask_aria_selected")
        self.soft("E1", {
            "calls_tab_selected": r.get("calls_tab_selected") is True,
            "entry_within_60s": isinstance(appeared, (int, float)) and self.release_ms is not None
            and -2000 <= appeared - self.release_ms <= 60_000, **self.rc_ok(r)},
            step="calls-live", release_ms=self.release_ms, appeared_ms=appeared, entry_text=r.get("entry_text"))
        try:
            self.guard()
            guard_ok = True
        except AssertionError as exc:
            guard_ok = False
            self.notes["c9_guard"] = str(exc)
        self.checkpoint("L4-planner-done-paused")
        self.soft("C9", {
            "report_not_consumed": self.sc.oracle.report_consumed_count(self.run_id, self.planner_task) == 0,
            "still_paused": self.sc.oracle.run_state(self.run_id)["status"] == "paused",
            "guard_ok": guard_ok}, label="L4-planner-done-paused")
        if not guard_ok:
            raise AssertionError(f"guard failed after planner release: {self.notes['c9_guard']}")

    # -- L5 block
    def l5_block(self) -> None:
        executing = self._executing()
        if executing:
            self.notes["l5_block_skipped"] = f"CLI executing: {executing}"
            for key in ("C10", "C13", "E2", "D5", "E3"):
                self.soft(key, {"block_ran": False})
            return
        before = self.checkpoint("L5-before-reload")
        r = self.pw5("reload-calls-empty")
        after = self.checkpoint("L5-after-reload")
        self.eval_c10_e2(before, after, r)
        self.eval_d5(self.pw5("stale-refresh"))
        self.eval_e3(self.pw5("calls-disconnected"))
        self.c13()

    def c13(self) -> None:
        o = self.sc.oracle
        ids_before, run_before = self.id_sets(), o.run_state(self.run_id)

        def plan_hash(run):
            return hashlib.sha256(_json_bytes(run.get("plan_state"))).hexdigest()

        for _ in range(5):  # ponytail: detail only; add other status endpoints if the UI uses them
            self.api_detail()
        delta, run_after = self.diff(ids_before), o.run_state(self.run_id)
        self.soft("C13", {"nothing_new": self.nothing_new(delta, allow_actions=()),
                          "run_row_unchanged": run_before == run_after,
                          "plan_state_unchanged": plan_hash(run_before) == plan_hash(run_after)}, delta=delta)

    def restart_resume(self) -> None:
        try:
            super().restart_resume()
        except Exception as exc:
            self.soft("C11", {"consumed_once_stable": False}, error=str(exc))
            raise
        self.soft("C11", {"consumed_once_stable": True})

    def c12(self) -> None:
        authority = (self.notes.get("plan_approval_authority") or {}).get("authority")
        auth = self.sc.oracle.goal_authority(self.goal_id) or {}
        self.notes["c12_variant"] = {"plan_approval_authority": authority, "goal_authority": auth}
        human = authority == "human"
        if HUMAN_MANAGER and not human:
            raise AssertionError(f"human-manager variant: plan approval authority must be human, got {authority!r}")
        r = self.pw5("status-plan-review", LIVE_ACCEPTANCE_PLAN_AUTHORITY=authority or "")
        api = self.checkpoint("L7-plan-review")
        wf, g = api["workflow"], api["workflow"].get("guidance") or {}
        decision_id = self.plan_decision["id"]
        conds = {
            "pending_is_plan_decision": _same_uuid(wf.get("pending_decision_id"), decision_id),
            "no_start_implementation": r.get("start_implementation_offered") is False
            and not any(_has(b, "start implementation") for b in _buttons(r)),
            "no_start_buttons_visible": r.get("start_buttons_visible") == 0,
            "question_visible": r.get("question_visible") is True, **self.rc_ok(r)}
        if human:  # literal criterion: guidance names the plan decision itself
            controls = r.get("guidance_control") or {}
            label = g.get("label")
            conds |= {
                "guidance_kind_review_decision": g.get("kind") == "review_decision",
                "guidance_points_to_plan_decision": _same_uuid(g.get("decision_id"), decision_id),
                # the real "Next:" cue must carry the API label (not a page-wide text search)
                "ui_next_cue_carries_api_label": bool(label) and r.get("api_guidance_label") == label and all(
                    _norm(_g(controls, w, "text")).startswith("Next:") and _has(_g(controls, w, "text"), label)
                    for w in ("1440", "375")),
                "guidance_cue_visible_in_viewport_1440_375": all(
                    _g(controls, w, "visible") is True and _g(controls, w, "in_viewport") is True
                    for w in ("1440", "375"))}
        else:  # agent-manager authority: historical lenient assertion, recorded as the agent path
            conds["guidance_consistent"] = (_same_uuid(g.get("decision_id"), decision_id)
                                            or g.get("kind") in ("review_decision", "resume"))
        self.soft("C12", conds, step="status-plan-review", label="L7-plan-review", guidance=g,
                  plan_approval_authority=authority, variant="human" if human else "agent")
        self.d_checks("C12", r, ("D1",), label="L7-plan-review")

    def final_pause(self) -> None:
        self.pw_ok("pause")
        paused = self.sc.oracle.run_state(self.run_id)["status"] == "paused"
        self.soft("C12", {"final_pause_succeeds": paused})
        if not paused:
            raise AssertionError("final pause did not leave the run paused")
        self.guard()
        self.checkpoint("L8-final-paused")
        self.sc.take_snapshot("L5-plan-review-pending-paused", require_clean=True)

    def steps_l(self):
        return [("resume", self.resume_and_continue), ("request-changes", self.request_changes),
                ("authorize-after-answer", self.authorize_after_answer), ("revision", self.revision), ("inspect-revision", self.inspect_revision), ("C8", self.c8),
                ("approve-hierarchy", self.approve_hierarchy), ("start-planning", self.start_planning),
                ("planner", self.planner), ("L5-block", self.l5_block), ("restart-resume", self.restart_resume),
                ("inspect-plan", self.inspect_plan), ("C12", self.c12), ("final-pause", self.final_pause)]


# -- tests --------------------------------------------------------------------------------------
def _make(name: str, snap, spec: dict, cap: int):
    root = Path(ROOT)
    campaign = root / "campaign-spend-g5g6.jsonl"
    seed_campaign(campaign, cap, {})

    def make():
        sc = G2G4Scenario(name, snap, root, cap, hold_spec=spec, roster="g2g4_human" if HUMAN_MANAGER else "g2g4",
                          campaign_ledger_path=campaign, campaign_cap=cap)
        sc.continue_through_warning = True  # the warning at cap-150k is informational only
        sc.warning_recorded = set()
        return sc
    return make


@pytest.mark.live
@pytest.mark.skipif(not LIVE_ENABLED, reason="requires HUDDLEROOM_LIVE_ACCEPTANCE=1 and HUDDLEROOM_LIVE_EVAL=1")
def test_live_run_g5g6_s():
    if PROBE and not HUMAN_MANAGER:
        pytest.fail("LIVE_ACCEPTANCE_G5G6_PROBE=1 needs LIVE_ACCEPTANCE_G5G6_MANAGER=human")
    if not ROOT:
        pytest.skip("needs LIVE_ACCEPTANCE_EVIDENCE_ROOT")
    cap = _g5_cap()
    smoke = os.environ.get("LIVE_ACCEPTANCE_G5G6_SMOKE") == "1"
    spec = {"codex_each": True, "claude_each": True, "codex_match": [TH_MARKER]}

    def body(sc):
        probe = PROBE and not smoke
        run = G5S.prepare(sc, keys=() if smoke else ("C0",) if probe else S_KEYS)
        run.smoke = smoke
        run.title = "SMOKE" if smoke else "PROBE" if probe else "S5"
        run.notes["manager_variant"] = "human" if HUMAN_MANAGER else "agent"
        steps = run.steps_smoke() if smoke else run.steps_probe() if probe else run.steps_s()
        ok = run.run_flow(steps, always=("zero-spend",) if smoke else ())
        run.finish(ok, hard=() if smoke or probe else ("C6",))

    _execute(_make("g5g6-SMOKE" if smoke else "g5g6-S", None, spec, cap), body)


@pytest.mark.live
@pytest.mark.skipif(not LIVE_ENABLED, reason="requires HUDDLEROOM_LIVE_ACCEPTANCE=1 and HUDDLEROOM_LIVE_EVAL=1")
@pytest.mark.skipif(not SNAP, reason="needs LIVE_ACCEPTANCE_SNAPSHOT_S from Run S5")
def test_live_run_g5g6_l():
    if not ROOT:
        pytest.skip("needs LIVE_ACCEPTANCE_EVIDENCE_ROOT")
    cap = _g5_cap()
    spec = {"codex_each": True, "claude_each": True, "codex_match": [PLAN_MARKER]}

    def body(sc):
        run = G5L(sc)
        run.notes["manager_variant"] = "human" if HUMAN_MANAGER else "agent"
        ok = run.run_flow(run.steps_l())
        run.finish(ok)

    _execute(_make("g5g6-L", Path(SNAP), spec, cap), body)
