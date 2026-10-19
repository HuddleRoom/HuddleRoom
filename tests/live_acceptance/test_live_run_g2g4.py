"""Opt-in G2--G4 live acceptance campaign.

Run S stops at a paused P1 hierarchy decision. Run L restores it and stops
immediately after accepting the read-only planning candidate. Importing this
module and normal collection never create a run or spend tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import shlex
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

from huddleroom.services.orchestration_team_hierarchy_analyzer import TeamHierarchyAnalyzer
from tests.live_acceptance import auth_dir_watch, oracle
from tests.live_acceptance.cli_shim import ShimHandle, TOOLS, make_shim, resolve_real, seed_campaign
from tests.live_acceptance.dry_run import Api, _auth_roots
from tests.live_acceptance.fixtures import G2G4_GOAL_CONSTRAINTS, G2G4_GOAL_TEXT
from tests.live_acceptance.live_env import server_env_for_cli
from tests.live_acceptance.run_dir import RunDir, assert_isolated
from tests.live_acceptance.scenario import LiveScenario, _latest_ids
from tests.live_acceptance.seed import create_goal
from tests.live_acceptance.server_supervisor import Supervisor, free_port
from tests.live_acceptance.spend_ledger import Ledger
from tests.live_acceptance.test_live_run import FRONTEND, L1, TH_MARKER, real_home

CAP = 400_000
CAMPAIGN_CAP, CAMPAIGN_PRIOR = 800_000, {"g1": 131_000}
ROOT = os.environ.get("LIVE_ACCEPTANCE_EVIDENCE_ROOT")
SNAP = os.environ.get("LIVE_ACCEPTANCE_SNAPSHOT_S")
RESUME_S = os.environ.get("LIVE_ACCEPTANCE_G2G4_RESUME_S") == "1"
RESUME_S_2 = os.environ.get("LIVE_ACCEPTANCE_G2G4_RESUME_S") == "2"
RESUME_S_3 = os.environ.get("LIVE_ACCEPTANCE_G2G4_RESUME_S") == "3"
RESUME_ROOT_RAW = os.environ.get("LIVE_ACCEPTANCE_G2G4_RESUME_ROOT", "")
RESUME_ROOT = Path(RESUME_ROOT_RAW) if RESUME_ROOT_RAW else None
EXPECTED_RESUME_LEAF = "run-20261019T140732Z-g2g4-S-405489"
LIVE_ENABLED = (os.environ.get("HUDDLEROOM_LIVE_ACCEPTANCE") == "1"
                and os.environ.get("HUDDLEROOM_LIVE_EVAL") == "1")
PLAN_MARKER = r"# Task: Planning:"
TERMINAL_SESSION = {"completed", "failed", "cancelled", "error", "timeout"}
CHECKS = ("feedback_lineage", "documented_gap", "model_drift", "planning_only",
          "read_only_planner", "plan_review_binding", "plan_ownership", "no_implementation")


def _s_cap() -> int:
    """Return the explicitly authorized fresh Run S cap, or the legacy cap."""
    raw = os.environ.get("LIVE_ACCEPTANCE_G2G4_S_CAP")
    if raw is None:
        return CAP
    try:
        cap = int(raw)
    except ValueError:
        pytest.fail("LIVE_ACCEPTANCE_G2G4_S_CAP must be a positive integer")
    if cap <= 0:
        pytest.fail("LIVE_ACCEPTANCE_G2G4_S_CAP must be a positive integer")
    return cap


def _expected_resume_root() -> Path:
    if not ROOT:
        raise AssertionError("LIVE_ACCEPTANCE_EVIDENCE_ROOT is required for the authorized resume")
    evidence_root = Path(ROOT)
    if not evidence_root.is_absolute():
        raise AssertionError("LIVE_ACCEPTANCE_EVIDENCE_ROOT must be absolute for the authorized resume")
    return evidence_root.resolve() / EXPECTED_RESUME_LEAF


def _ledger_totals_readonly(path: Path) -> dict[str, dict[str, int]]:
    """Read an existing ledger without creating or repairing it."""
    if not path.exists() or not path.stat().st_size:
        return {"global": {"non_cached": 0, "cached": 0}}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    used, _ = Ledger._sum(rows)
    totals = {scenario: {"non_cached": value[0], "cached": value[1]} for scenario, value in used.items()}
    totals["global"] = {"non_cached": sum(value[0] for value in used.values()),
                        "cached": sum(value[1] for value in used.values())}
    return totals


def _pending(oracle, goal_id, key: str | None = None):
    rows = [row for row in oracle.decisions(goal_id) if row["status"] == "pending"]
    return next((row for row in rows if key is None or row["decision_key"] == key), None)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str).encode("utf-8")


def _plan_review_hash(identity: dict) -> str:
    return hashlib.sha256(_json_bytes(identity)).hexdigest()


def _same_uuid(left: object, right: object) -> bool:
    try:
        return uuid.UUID(str(left)) == uuid.UUID(str(right))
    except (TypeError, ValueError):
        return False


def _live_agent_ids(sc: LiveScenario) -> dict[str, str]:
    response = httpx.get(f"{sc.base_url}/api/v1/agents", timeout=60)
    response.raise_for_status()
    names = {"live-manager": "manager_id", "live-writer": "writer_id",
             "live-analyst": "analyst_id", "live-reviewer": "reviewer_id"}
    items = response.json().get("items") or []
    ids = {}
    for name, key in names.items():
        matches = [item.get("id") for item in items if item.get("name") == name]
        if len(matches) != 1 or not matches[0]:
            raise AssertionError(f"expected exactly one {name} agent, got {len(matches)}")
        ids[key] = matches[0]
    return ids


class G2G4Scenario(LiveScenario):
    """A teardown must never kill a model child still executing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.released_ordinals: dict[str, set[int]] = {"codex": set(), "claude": set()}

    def _teardown(self, failed: bool = False) -> None:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            pending, executing = {}, {}
            for tool, shim in self.shims.items():
                invocations = {inv.ordinal: inv for inv in shim.invocations()}
                settled = shim.settled_ordinals()
                released = []
                for ordinal in self.released_ordinals[tool]:
                    inv = invocations.get(ordinal)
                    if ordinal not in settled and (inv is None or not inv.refused and not inv.timed_out):
                        released.append(ordinal)
                active = [inv.ordinal for inv in invocations.values()
                          if inv.exec_at is not None and inv.ordinal not in settled]
                if released:
                    pending[tool] = released
                if active:
                    executing[tool] = active
            if not pending and not executing:
                return super()._teardown(failed)
            time.sleep(0.2)
        fatal = {"released_pending": pending, "executing_unsettled": executing}
        self.errors.append(f"CLI calls unsettled at teardown; server intentionally left running: {fatal}")
        try:
            verdict_path = self.artifact_root / "verdict.json"
            verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}
            verdict.update({"passed": False, "fatal_teardown": fatal})
            verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
        except OSError as exc:
            self.errors.append(f"fatal verdict write: {exc}")


class G2G4ResumeScenario(G2G4Scenario):
    """Reopen one paused S run without recreating its baseline or artifacts."""

    artifact_name = "resume-S-1m"

    def __init__(self, original_root: Path, prior_spend: int, *args, **kwargs):
        self.original_root = Path(original_root).resolve()
        self.prior_spend = prior_spend
        self.continue_through_warning = True
        self.warning_cap = 800_000
        self.warning_recorded: set[str] = set()
        super().__init__(*args, **kwargs)

    def _historical_codex(self, root: Path) -> tuple[ShimHandle, int]:
        historical = ShimHandle(root / "shim", "codex", root / "shim" / "codex")
        old_invocations, old_settled = historical.invocations(), historical.settled_ordinals()
        if max((item.ordinal for item in old_invocations), default=0) != 14:
            raise AssertionError("expected historical codex ordinal 14")
        ordinal_14 = [item for item in old_invocations if item.ordinal == 14]
        if (len(ordinal_14) != 1 or ordinal_14[0].exec_at is not None or ordinal_14[0].refused
                or ordinal_14[0].timed_out or 14 in old_settled):
            raise AssertionError("historical codex ordinal 14 must be held and never executed")
        unsettled = [item.ordinal for item in old_invocations if item.exec_at is not None and item.ordinal not in old_settled]
        if unsettled:
            raise AssertionError(f"historical executed codex calls are unsettled: {unsettled}")
        return historical, 14

    def _start(self) -> None:
        root = self.original_root
        artifact_root = root / self.artifact_name
        if not (root / "app.db").is_file() or not (root / "workspace").is_dir():
            raise AssertionError(f"resume root is incomplete: {root}")
        if artifact_root.exists():
            raise AssertionError(f"resume artifacts already exist: {artifact_root}")
        artifact_root.mkdir()
        for name in ("snapshots", "shim"):
            (artifact_root / name).mkdir()
        seed_campaign(artifact_root / "campaign-spend.jsonl", self.campaign_cap,
                      {"g2g4-S-prior": self.prior_spend})
        _, self.historical_codex_last = self._historical_codex(root)
        (artifact_root / "shim" / "counters").mkdir()
        (artifact_root / "shim" / "counters" / "codex").write_text(str(self.historical_codex_last))
        self.run_dir = RunDir(
            root=root, db=root / "app.db", workspace=root / "workspace",
            snapshots=artifact_root / "snapshots", shim=artifact_root / "shim",
            proxy_log=artifact_root / "proxy.jsonl", spend=artifact_root / "spend.jsonl",
            config=artifact_root / "config.json",
        )
        assert_isolated(self.run_dir, self.run_dir.db, self.run_dir.workspace)
        self._artifact_root = artifact_root
        self._set_artifact_root()
        self.ids = _latest_ids(self.run_dir.db)
        self.mark(f"scenario {self.name} resume start; root={root}")
        self.auth_before = auth_dir_watch.listing(_auth_roots())
        auth_dir_watch.write_listing(self.artifact_root / "auth-before.json", self.auth_before)
        self.ledger = Ledger(self.run_dir.spend, self.budget, {self.name: self.budget})
        self.ledger.record(self.name, "prior_S_attempts", self.prior_spend, 0)
        extra = {"campaign_ledger": {"path": str(self.campaign_ledger_path.absolute()),
                                      "cap": self.campaign_cap, "scenario": self.name}}
        self.findings["campaign_ledger"] = {"active": True, **extra["campaign_ledger"]}
        for tool in TOOLS:
            real = self.real_paths.get(tool) or resolve_real(tool, self.run_dir.shim)
            handle = self.shims[tool] = make_shim(
                self.run_dir.shim, tool, Path(real), ledger_path=self.run_dir.spend, scenario=self.name,
                global_cap=self.budget, scenario_cap=self.budget, output_ceiling=self.output_ceiling,
                hold_each=bool(self.hold_spec.get(f"{tool}_each")), **extra)
            for rx in self.hold_spec.get(f"{tool}_match", ()):
                handle.hold_matching(rx)
        self.sup = Supervisor(self.run_dir, self.port or free_port(), env_overrides=server_env_for_cli(),
                              shim_dir=self.run_dir.shim,
                              extra_pid_dirs=sorted({h.shim_dir / "pids" for h in self.shims.values()}))
        prior_logs = [int(path.stem.split("-")[1]) for path in root.glob("server-*.log")
                      if path.stem.split("-")[1].isdigit()]
        self.sup.incarnation = max(prior_logs, default=0)
        self.sup.start()
        self.mark(f"server up at {self.sup.base_url}")
        self.oracle = oracle.Oracle(self.run_dir.db)
        oracle.snapshot(self.run_dir.db, self.run_dir.snapshots / "00-before.db")


class G2G4SecondResumeScenario(G2G4ResumeScenario):
    """Continue only the saved 1M resume; do not recreate either earlier run."""

    artifact_name = "resume-S-1m-2"

    def _historical_codex(self, root: Path) -> tuple[ShimHandle, int]:
        super()._historical_codex(root)
        prior = root / "resume-S-1m" / "shim"
        historical = ShimHandle(prior, "codex", prior / "codex")
        invocations, settled = historical.invocations(), historical.settled_ordinals()
        ordinals = [item.ordinal for item in invocations]
        if ordinals != [15, 16, 17]:
            raise AssertionError(f"expected first-resume codex ordinals 15--17, got {ordinals}")
        if any(item.exec_at is None or item.refused or item.timed_out for item in invocations):
            raise AssertionError("first-resume codex calls 15--17 must all execute")
        if settled != {15, 16, 17}:
            raise AssertionError(f"first-resume codex calls must settle: {sorted(settled)}")
        return historical, 17


class G2G4ThirdResumeScenario(G2G4SecondResumeScenario):
    """Continue only the saved 1M retry checkpoint after ordinals 18--20."""

    artifact_name = "resume-S-1m-3"

    def _historical_codex(self, root: Path) -> tuple[ShimHandle, int]:
        super()._historical_codex(root)
        prior = root / "resume-S-1m-2" / "shim"
        historical = ShimHandle(prior, "codex", prior / "codex")
        invocations, settled = historical.invocations(), historical.settled_ordinals()
        ordinals = [item.ordinal for item in invocations]
        if ordinals != [18, 19, 20]:
            raise AssertionError(f"expected second-resume codex ordinals 18--20, got {ordinals}")
        if any(item.exec_at is None or item.refused or item.timed_out for item in invocations):
            raise AssertionError("second-resume codex calls 18--20 must all execute")
        if settled != {18, 19, 20}:
            raise AssertionError(f"second-resume codex calls must settle: {sorted(settled)}")
        return historical, 20


class G2G4(L1):
    SPEC = "tests-live/live-run-g2g4.spec.ts"

    def __init__(self, sc: G2G4Scenario):
        super().__init__(sc)
        self.claude_released: list[int] = []
        agent_ids = _live_agent_ids(sc)
        self.ids = sc.ids = {**sc.ids, **agent_ids}
        self.checks = {key: {"status": "not_run", "evidence": []} for key in CHECKS}
        self.steps = {name: "not_run" for name in (
            "resume", "continue-baseline", "request-changes", "revision", "inspect-revision",
            "approve-hierarchy", "pause", "model-drift", "start-planning", "planner-held",
            "restart-resume", "inspect-plan", "plan-owners", "approve-plan")}
        self.p1 = next((p for p in sc.oracle.process_runs(goal_id=self.goal_id)
                        if p["process_type"] == "team_hierarchy" and p["status"] == "waiting_decision"), None)
        if self.p1 is None:
            raise AssertionError("restored snapshot has no waiting P1 hierarchy proposal")
        self.p1_evidence = sc.oracle.process_input_evidence(self.p1["id"])
        self.p1_evidence_bytes = _json_bytes(self.p1_evidence)
        self.workspace_head = self._git("rev-parse", "HEAD").strip()
        self.p2 = self.planner_task = self.planner_inv = None
        self.feedback: dict = {}
        self.notes.update({"cap": sc.budget, "warn": sc.budget * 0.8, "campaign_cap": sc.campaign_cap,
                           "campaign_prior": CAMPAIGN_PRIOR,
                           "retry_policy": {"automatic_retries": 0, "maximum": 1,
                                            "status": "not_exercised"}})

    def pw(self, name: str, **env) -> dict:
        """Keep continuation UI evidence under its exclusive artifact directory."""
        out = self.sc.artifact_root / "playwright" / name
        out.mkdir(parents=True, exist_ok=True)
        values = {**os.environ, "LIVE_ACCEPTANCE_BASE_URL": self.sc.base_url,
                  "LIVE_ACCEPTANCE_OUTPUT": str(out), "LIVE_ACCEPTANCE_PROJECT_ID": self.project_id,
                  "LIVE_ACCEPTANCE_GOAL_ID": self.goal_id, **env}
        values.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(real_home() / "Library/Caches/ms-playwright"))
        result = subprocess.run(["npx", "playwright", "test", "-c", "playwright.live.config.ts", self.SPEC,
                                 "--grep", rf"(^|\s){name}$"], cwd=FRONTEND, env=values,
                                capture_output=True, text=True, timeout=500)
        (out.parent / f"{name}.log").write_text(result.stdout + "\n" + result.stderr)
        self.sc.mark(f"playwright {name} rc={result.returncode}")
        receipt = json.loads((out / f"{name}.json").read_text()) if (out / f"{name}.json").exists() else {}
        receipt["_rc"], receipt["_dir"] = result.returncode, str(out)
        return receipt

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.sc.run_dir.workspace, check=True,
                              capture_output=True, text=True).stdout

    def _held(self, handle):
        return [inv for inv in handle.invocations()
                if inv.exec_at is None and inv.gates and inv.ordinal not in self.sc.released_ordinals[handle.tool]
                and not inv.refused and not inv.timed_out]

    def _executing(self) -> dict:
        executing = {}
        for tool, shim in self.sc.shims.items():
            ordinals = [inv.ordinal for inv in shim.invocations()
                        if inv.exec_at is not None and inv.ordinal not in shim.settled_ordinals()]
            if ordinals:
                executing[tool] = ordinals
        return executing

    def budget_ok(self) -> None:
        totals = self.sc.ledger.totals()
        used = totals["global"]["non_cached"]
        self.notes["spend"] = {"used": used, "totals": totals}
        cap = self.sc.budget
        warn = getattr(self.sc, "warning_cap", cap * 0.8)
        if used >= cap:
            raise AssertionError("budget cap reached; no further call may be released")
        if used >= warn:
            self.notes["operator_confirmation_required"] = {"used": used, "warn": warn, "cap": cap}
            if not getattr(self.sc, "continue_through_warning", False):
                raise AssertionError("budget warning threshold reached; no further call may be released")
            if not self.sc.warning_recorded:
                self.sc.warning_recorded.add("acknowledged")
                self.sc.mark(f"budget warning acknowledged; continuing used={used} warn={warn} cap={cap}")
        campaign = Ledger(self.sc.campaign_ledger_path, self.sc.campaign_cap).totals()
        campaign_used = campaign["global"]["non_cached"]
        if campaign_used >= self.sc.campaign_cap:
            raise AssertionError("campaign budget cap reached; no further call may be released")
        campaign_warn = getattr(self.sc, "warning_cap", self.sc.campaign_cap * 0.8)
        if campaign_used >= campaign_warn:
            self.notes["operator_confirmation_required"] = {
                "used": campaign_used, "warn": campaign_warn, "cap": self.sc.campaign_cap,
                "ledger": "campaign",
            }
            if not getattr(self.sc, "continue_through_warning", False):
                raise AssertionError("campaign budget warning threshold reached; no further call may be released")

    def release_one(self, handle, why: str, ordinal: int | None = None):
        """Record one ordinal before it can execute; release no future private gates."""
        self.budget_ok()
        if self._executing():
            return None
        held = self._held(handle)
        if not held:
            if ordinal is not None:
                raise AssertionError(f"requested held {handle.tool} ordinal={ordinal} is unavailable")
            return None
        inv = held[0] if ordinal is None else next((item for item in held if item.ordinal == ordinal), None)
        if inv is None:
            raise AssertionError(f"requested held {handle.tool} ordinal={ordinal} is unavailable")
        if handle.tool == "codex" and not self.sc.released_ordinals["codex"] and hasattr(self.sc, "historical_codex_last"):
            if inv.ordinal != self.sc.historical_codex_last + 1:
                raise AssertionError(f"resumed first codex ordinal is {inv.ordinal}, not {self.sc.historical_codex_last + 1}")
        self.sc.released_ordinals[handle.tool].add(inv.ordinal)
        (self.released if handle is self.codex else self.claude_released).append(inv.ordinal)
        self.notes.setdefault("release_timeline", []).append(
            {"tool": handle.tool, "ordinal": inv.ordinal, "why": why, "before_execution": True})
        self.sc.mark(f"released {handle.tool} ordinal={inv.ordinal} reason={why}")
        handle.release_invocation(inv.ordinal)
        for gate in inv.gates:
            if ".match." in gate:
                handle.release(gate)
        return inv

    def guard(self) -> None:
        released = self.sc.released_ordinals
        stray = {tool: [inv.ordinal for inv in shim.invocations()
                        if inv.exec_at is not None and inv.ordinal not in released[tool]]
                 for tool, shim in self.sc.shims.items()}
        implementation = [task["id"] for task in self.sc.oracle.tasks(self.goal_id)
                          if ((task.get("metadata") or {}).get("orchestration") or {}).get("work_function") == "implementation"]
        if any(stray.values()) or implementation:
            raise AssertionError(f"unexpected execution={stray}; implementation_tasks={implementation}")

    def drive_codex(self, timeout: float, done, why: str):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.guard()
            value = done()
            if value:
                return value
            if not self._executing():
                self.release_one(self.codex, why)
            time.sleep(1)
        return None

    def passive(self, timeout: float, done):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.guard()
            value = done()
            if value:
                return value
            time.sleep(1)
        return None

    def eligibility(self) -> dict[str, list[str]]:
        outputs = next((self.sc.oracle.process_outputs(p["id"]) or {}
                        for p in self.sc.oracle.process_runs(goal_id=self.goal_id)
                        if p["process_type"] == "agent_definition_review"), {})
        return outputs.get("eligibility") or {}

    def proposal(self, process_id: str) -> dict:
        return self.sc.oracle.process_outputs(process_id) or {}

    def feedback_message(self) -> str:
        proposal = self.proposal(self.p1["id"]).get("proposal") or {}
        eligibility = self.eligibility()
        assignments = {item.get("work_function"): item.get("agent_ref") for item in proposal.get("assignments", [])
                       if isinstance(item, dict)}
        selected = next(((function, owner, sorted(agent for agent, functions in eligibility.items() if function in functions))
                         for function, owner in assignments.items()
                         if any(_same_uuid(owner, agent) and function in functions
                                for agent, functions in eligibility.items())
                         and any(function in functions and not _same_uuid(agent, owner)
                                 for agent, functions in eligibility.items())), None)
        if selected is None:
            raise AssertionError("no P1 assignment has an eligible alternate owner")
        function, current, owners = selected
        target = next(owner for owner in owners if not _same_uuid(owner, current))
        message = f"Move {function} to {target}; assign validation to live-reviewer."
        self.feedback = {"message": message, "function": function, "current": current, "target": target,
                         "eligibility": eligibility}
        self.notes["feedback"] = self.feedback
        return message

    def resume_and_continue(self) -> None:
        initial = self.sc.oracle.run_state(self.run_id)
        if initial["phase"] != "baseline":
            raise AssertionError(f"restored P1 must resume from baseline, got {initial['phase']!r}")
        self.pw_ok("resume")
        self.steps["resume"] = "pass"
        self.pw_ok("continue-baseline")
        resumed = self.sc.oracle.run_state(self.run_id)
        if resumed["phase"] != "baseline" or not resumed["baseline_authorized"]:
            raise AssertionError(f"resume/continue did not authorize the baseline: {resumed}")
        self.steps["continue-baseline"] = "pass"

    def request_changes(self) -> None:
        decision = _pending(self.sc.oracle, self.goal_id, "team_hierarchy:approval")
        if decision is None:
            raise AssertionError("P1 approval decision missing")
        self.notes["p1_approval_decision_id"] = decision["id"]
        self.pw_ok("request-changes", LIVE_ACCEPTANCE_FEEDBACK_MESSAGE=self.feedback_message())

    def revision(self) -> None:
        self.p2 = self.drive_codex(300, lambda: next((p for p in self.sc.oracle.process_runs(goal_id=self.goal_id)
            if p["process_type"] == "team_hierarchy" and p["id"] != self.p1["id"] and p["status"] == "waiting_decision"), None),
            "team hierarchy revision")
        if self.p2 is None:
            raise AssertionError("P2 did not reach waiting_decision")
        outputs = self.proposal(self.p2["id"])
        analysis, proposal = outputs.get("analysis") or {}, outputs.get("proposal") or {}
        revision = outputs.get("revision_context") or {}
        feedback = analysis.get("feedback_results") or []
        source_items = revision.get("feedback_items") or []
        p2_assignments = {item.get("work_function"): item.get("agent_ref") for item in proposal.get("assignments", []) if isinstance(item, dict)}
        lineage = self.sc.oracle.process_input_evidence(self.p1["id"])
        p1_row = next(p for p in self.sc.oracle.process_runs(goal_id=self.goal_id) if p["id"] == self.p1["id"])
        source_ids = [item.get("item_id") for item in source_items if isinstance(item, dict) and item.get("item_id")]
        result_ids = [item.get("item_id") for item in feedback if isinstance(item, dict) and item.get("item_id")]
        source_by_id = {item["item_id"]: item for item in source_items if isinstance(item, dict) and item.get("item_id")}
        eligibility = self.eligibility()
        selected = [item for item in feedback if item.get("disposition") == "applied" and
                    f"assignment:{self.feedback['function']}" in item.get("change_refs", [])]
        validation_ids = [item_id for item_id, item in source_by_id.items()
                          if "validation" in str(item.get("source_excerpt", "")).lower()]
        self.verify("feedback_lineage", {
            "feedback_ids_exact": bool(source_ids) and len(source_ids) == len(set(source_ids)) and set(source_ids) == set(result_ids) and len(result_ids) == len(set(result_ids)),
            "feedback_dispositions_reasoned": bool(feedback) and all(item.get("disposition") in {"applied", "declined", "impossible"} and bool(item.get("reason")) for item in feedback),
            "requested_move_applied": bool(selected) and not _same_uuid(self.feedback["current"], self.feedback["target"]) and _same_uuid(p2_assignments.get(self.feedback["function"]), self.feedback["target"]),
            "validation_not_applied": bool(validation_ids) and all(any(item.get("item_id") == item_id and item.get("disposition") != "applied" for item in feedback) for item_id in validation_ids),
            "all_assignment_functions_eligible": all(owner and any(_same_uuid(owner, agent) and function in functions
                                                                      for agent, functions in eligibility.items())
                                                     for function, owner in p2_assignments.items()),
            "p1_superseded": _same_uuid(p1_row.get("superseded_by_id"), self.p2["id"]),
            "p1_input_bytes_unchanged": _json_bytes({key: lineage.get(key) for key in ("fingerprint", "capture")}) == _json_bytes({key: self.p1_evidence.get(key) for key in ("fingerprint", "capture")}),
            "rerun_reason": (lineage.get("rerun_intent") or {}).get("reason") == "hierarchy_changes_requested",
            "rerun_source_decision": _same_uuid((lineage.get("rerun_intent") or {}).get("source_decision_id"), self.notes["p1_approval_decision_id"]),
            "rerun_consumed_by_p2": _same_uuid((lineage.get("rerun_intent") or {}).get("consumed_by_process_id"), self.p2["id"]),
        }, outputs=outputs, lineage=lineage)
        coverage = proposal.get("responsibility_coverage") or []
        validation_gap = next((row for row in coverage if isinstance(row, dict)
                               and row.get("work_function") == "validation"
                               and row.get("status") == "gap" and not row.get("agent_ref")), None)
        self.validation_gap_id = validation_gap.get("responsibility_id") if validation_gap else None
        self.verify("documented_gap", {"validation_unowned": bool(self.validation_gap_id)}, coverage=coverage)
        self.sc.oracle_dump("P2-waiting-decision")

    def approve_hierarchy(self) -> None:
        ui = self.pw_ok("approve-hierarchy")
        ready = self.drive_codex(300, lambda: self.sc.oracle.run_state(self.run_id)
                                 if self.sc.oracle.run_state(self.run_id)["phase"] == "ready" else None,
                                 "hierarchy approval follow-up")
        outputs = self.proposal(self.p2["id"])
        self.verify("documented_gap", {"documented_gap_option": ui.get("documented_gap_option") is True,
                                        "plain_approve_absent": ui.get("plain_approve_absent") is True,
                                        "approval_persisted": outputs.get("approval") == "approve_with_documented_gaps",
                                        "accepted_gaps": bool(outputs.get("accepted_responsibility_gap_ids")),
                                        "validation_gap_accepted": self.validation_gap_id in (outputs.get("accepted_responsibility_gap_ids") or []),
                                        "phase_ready": ready is not None}, outputs=outputs, run=ready)
        self.sc.oracle_dump("P2-approved-ready")
        self.pw_ok("pause")
        self.steps["pause"] = "pass"
        self.sc.take_snapshot("L-th-approved", require_clean=True)
        self.pw_ok("resume")
        resumed = self.sc.oracle.run_state(self.run_id)
        if resumed["status"] != "running" or resumed["phase"] != "ready":
            raise AssertionError(f"approved hierarchy did not resume ready: {resumed}")

    def inspect_revision(self) -> None:
        ui = self.pw_ok("inspect-revision")
        history = ui.get("p1_history_controls")
        self.notes["revision_ui"] = ui
        if ui.get("no_horizontal_overflow_1440") is not True or ui.get("no_horizontal_overflow_390") is not True:
            raise AssertionError(f"revision UI overflow check failed: {ui}")
        if history != "pass" and not str(history).startswith("not_exercised:"):
            raise AssertionError(f"P1 history controls are not inert: {history!r}")

    def model_drift(self) -> None:
        before_runs = self.sc.oracle.process_runs(goal_id=self.goal_id)
        before_hierarchy_ids = {row["id"] for row in before_runs if row["process_type"] == "team_hierarchy"}
        before_calls = {(tool, inv.ordinal) for tool, shim in self.sc.shims.items() for inv in shim.invocations()}
        before = self.sc.oracle.process_input_snapshot(self.p2["id"]) or {}
        before_outputs = self.proposal(self.p2["id"])
        reviews = httpx.get(f"{self.sc.base_url}/api/v1/projects/{self.project_id}/orchestration/goals/{self.goal_id}/agent-reviews", timeout=60)
        reviews.raise_for_status()
        writer_review = next((review for review in reviews.json() if _same_uuid(review.get("agent_id"), self.ids["writer_id"])), None)
        response = httpx.put(f"{self.sc.base_url}/api/v1/agents/{self.ids['writer_id']}", json={"model": "claude-sonnet-5-5"}, timeout=60)
        response.raise_for_status()
        current = httpx.get(f"{self.sc.base_url}/api/v1/agents/{self.ids['writer_id']}", timeout=60)
        current.raise_for_status()
        reviews_after = httpx.get(f"{self.sc.base_url}/api/v1/projects/{self.project_id}/orchestration/goals/{self.goal_id}/agent-reviews", timeout=60)
        reviews_after.raise_for_status()
        writer_review_after = next((review for review in reviews_after.json()
                                    if _same_uuid(review.get("agent_id"), self.ids["writer_id"])), None)
        after = self.sc.oracle.process_input_snapshot(self.p2["id"]) or {}
        after_outputs = self.proposal(self.p2["id"])
        result = response.json()
        reviewed_model = ((writer_review or {}).get("definition_snapshot") or {}).get("model")
        self.verify("model_drift", {"served_model_sonnet": result.get("model") == "claude-sonnet-5-5" and current.json().get("model") == "claude-sonnet-5-5",
                                    "reviewed_model_haiku": reviewed_model == "claude-haiku-5-5",
                                    "review_snapshot_unchanged": _json_bytes((writer_review or {}).get("definition_snapshot")) == _json_bytes((writer_review_after or {}).get("definition_snapshot")),
                                    "outputs_unchanged": _json_bytes(before_outputs) == _json_bytes(after_outputs),
                                    "effective_fingerprint_unchanged": (before.get("effective_comparison") == after.get("effective_comparison")),
                                    "no_hierarchy_rerun": {row["id"] for row in self.sc.oracle.process_runs(goal_id=self.goal_id) if row["process_type"] == "team_hierarchy"} == before_hierarchy_ids,
                                    "no_new_shim_call": before_calls == {(tool, inv.ordinal) for tool, shim in self.sc.shims.items() for inv in shim.invocations()}}, response=result)

    def start_planning(self) -> None:
        if self.sc.oracle.run_state(self.run_id)["phase"] != "ready":
            raise AssertionError("planning can start only from ready phase")
        self.planning_started_at = time.time()
        ui = self.pw_ok("start-planning")
        run = self.sc.oracle.run_state(self.run_id)
        actions = [action for action in self.sc.oracle.actions(self.run_id) if action["action_type"] == "authorize_execution"]
        action = next((item for item in actions if item["status"] == "completed"
                       and _same_uuid(item.get("run_id"), self.run_id)
                       and (item.get("request") or {}).get("scope") == "planning_only"), None)
        request = action.get("request") if action else {}
        self.verify("planning_only", {"ui_planning_only": ui.get("planning_only") is True,
                                      "phase_advanced": run["phase"] == "authorized",
                                      "authorize_execution_completed": action is not None,
                                      "request_scope_planning_only": request.get("scope") == "planning_only",
                                      "scope_contract_version": (run.get("plan_state") or {}).get("scope_contract_version") == 1}, run=run, request=request)
        self.sc.oracle_dump("planning-started")

    def planner(self) -> None:
        def planning_tasks():
            tasks = [task for task in self.sc.oracle.tasks(self.goal_id)
                     if ((task.get("metadata") or {}).get("orchestration") or {}).get("work_function") == "planning"]
            return tasks if len(tasks) == 1 else None

        tasks = None
        end = time.monotonic() + 300
        while time.monotonic() < end:
            self.guard()
            tasks = planning_tasks()
            if tasks:
                break
            if not self._executing():
                control = [item for item in self._held(self.codex) if not any(".match." in gate for gate in item.gates)]
                if control:
                    self.release_one(self.codex, "orchestration start decision", ordinal=control[0].ordinal)
            time.sleep(1)
        if tasks is None:
            raise AssertionError("expected exactly one planning task before selecting the planner invocation")
        task = tasks[0]
        metadata = (task.get("metadata") or {}).get("orchestration") or {}
        if (not _same_uuid(task.get("project_id"), self.project_id)
                or not _same_uuid(metadata.get("goal_id"), self.goal_id)
                or not _same_uuid(metadata.get("run_id"), self.run_id)):
            raise AssertionError("planning task does not belong to the current project, goal, and run")
        candidates = self.passive(300, lambda: [item for item in self._held(self.codex)
            if len([gate for gate in item.gates if ".match." in gate]) == 1
            and item.started_at >= self.planning_started_at - 1]
            if len([item for item in self._held(self.codex)
                    if len([gate for gate in item.gates if ".match." in gate]) == 1
                    and item.started_at >= self.planning_started_at - 1]) == 1 else None)
        if candidates is None:
            raise AssertionError("planning task did not produce one held matcher-gated codex invocation")
        inv = candidates[0]
        argv = shlex.split(inv.args)
        sandbox = any(argv[index:index + 2] == ["--sandbox", "read-only"] for index in range(len(argv) - 1))
        self.planner_task, self.planner_inv = task["id"], inv
        self.verify("planning_only", {"planner_owned_by_manager": _same_uuid(task.get("assigned_to"), self.ids["manager_id"]),
                                      "read_only_sandbox": sandbox,
                                      "planner_started_after_start": inv.started_at >= self.planning_started_at - 1}, task=task, args=argv,
                    matched_gates=[gate for gate in inv.gates if ".match." in gate])
        self.pw_ok("pause")
        paused = self.sc.oracle.run_state(self.run_id)
        if paused["status"] != "paused" or self.sc.oracle.report_consumed_count(self.run_id, self.planner_task) != 0:
            raise AssertionError("planner must remain unconsumed while paused before release")
        self.sc.oracle_dump("planner-held-paused")
        self.release_one(self.codex, "planner only", ordinal=inv.ordinal)
        sessions = self.passive(300, lambda: self.sc.oracle.sessions(self.planner_task)
            if len(self.sc.oracle.sessions(self.planner_task)) == 1
            and self.sc.oracle.sessions(self.planner_task)[0]["status"] in TERMINAL_SESSION else None)
        if sessions is None or sessions[0]["status"] != "completed":
            raise AssertionError(f"planner did not complete exactly once: {sessions}")
        self.planner_session = sessions[0]

    def restart_resume(self) -> None:
        run = self.sc.oracle.run_state(self.run_id)
        if run["status"] != "paused":
            self.pw_ok("pause")
            run = self.sc.oracle.run_state(self.run_id)
        if run["status"] != "paused" or self.sc.oracle.report_consumed_count(self.run_id, self.planner_task) != 0:
            raise AssertionError("restart requires a paused run with the planner report still unconsumed")
        if self._executing():
            raise AssertionError(f"refusing SIGTERM while CLI executes: {self._executing()}")
        live = {tool: shim.live_pids() for tool, shim in self.sc.shims.items() if shim.live_pids()}
        if live:
            raise AssertionError(f"refusing SIGTERM while model CLI processes are alive: {live}")
        pre_restart_ordinals = {inv.ordinal for inv in self.codex.invocations()}
        self.sc.oracle_dump("planner-done-paused-pre-restart")
        self.sc.sup.kill(signal.SIGTERM)
        for shim in self.sc.shims.values():
            shim.undeny()
        self.sc.sup.start()
        self.sc.oracle_dump("planner-done-paused-post-restart")
        self.pw_ok("resume")
        count = self.passive(90, lambda: self.sc.oracle.report_consumed_count(self.run_id, self.planner_task) == 1)
        if count is None or self.sc.oracle.report_consumed_count(self.run_id, self.planner_task) != 1:
            raise AssertionError("planner report was not consumed exactly once after restart/resume")
        time.sleep(2)
        if self.sc.oracle.report_consumed_count(self.run_id, self.planner_task) != 1:
            raise AssertionError("planner report consumption was not stable after resume")
        resumes = [inv for inv in self.codex.invocations()
                   if inv.ordinal not in pre_restart_ordinals
                   and shlex.split(inv.args)[:2] == ["exec", "resume"]]
        self.notes["codex_resume"] = "not_exercised" if not resumes else [
            {"ordinal": inv.ordinal,
             "read_only": any(shlex.split(inv.args)[index:index + 2] == ["-c", "sandbox_mode=read-only"]
                              for index in range(len(shlex.split(inv.args)) - 1))}
            for inv in resumes]
        if resumes and not all(item["read_only"] for item in self.notes["codex_resume"]):
            raise AssertionError("codex resume omitted read-only sandbox mode")

    def workspace_violations(self) -> list[str]:
        """Workspace changes other than the server-written designated plan projection."""
        goal = str(uuid.UUID(str(self.goal_id)))
        run = str(uuid.UUID(str(self.run_id)))
        allowed = re.compile(rf"^plans/goal-{re.escape(goal)}/run-{re.escape(run)}/plan-v\d+-review-\d+\.md$")
        lines = self._git("status", "--porcelain", "--untracked-files=all").splitlines()
        projection = [line[3:] for line in lines if line.startswith("?? ") and allowed.match(line[3:])]
        self.notes["plan_projection_files"] = projection
        bad = [line for line in lines if not (line.startswith("?? ") and allowed.match(line[3:]))]
        if self._git("rev-parse", "HEAD").strip() != self.workspace_head:
            bad.append("HEAD moved")
        if (self.sc.run_dir.workspace / "PLAN.md").exists():
            bad.append("PLAN.md exists")
        return bad

    def inspect_plan(self) -> None:
        violations = self.workspace_violations()
        if violations:
            raise AssertionError(f"planner changed the workspace: {violations}")
        def plan_decisions():
            found = [d for d in self.sc.oracle.decisions(self.goal_id)
                     if d["status"] == "pending" and str(d["decision_key"]).startswith("roadmap_plan:")]
            return found or None

        def work_function(task: dict):
            return ((task.get("metadata") or {}).get("orchestration") or {}).get("work_function")

        def verifier_only() -> bool:
            tasks = self.sc.oracle.tasks(self.goal_id)
            if any(work_function(task) == "implementation" for task in tasks):
                return False
            active = [task for task in tasks if task.get("status") in ("ready", "in_progress")]
            return bool(active) and all(work_function(task) in ("planning", "review", "validation") for task in active)

        decisions = None
        end = time.monotonic() + 600
        while time.monotonic() < end:
            self.guard()
            decisions = plan_decisions()
            if decisions:
                break
            if not self._executing():
                control = [item for item in self._held(self.codex) if not any(".match." in gate for gate in item.gates)]
                if control:
                    self.release_one(self.codex, "plan review work", ordinal=control[0].ordinal)
                elif self._held(self.claude) and verifier_only():
                    held = self._held(self.claude)[0]
                    inv = self.release_one(self.claude, "plan review verifier", ordinal=held.ordinal)
                    if inv is not None:
                        active = [task for task in self.sc.oracle.tasks(self.goal_id)
                                  if task.get("status") in ("ready", "in_progress")]
                        self.notes.setdefault("claude_releases", []).append({
                            "ordinal": inv.ordinal,
                            "active_tasks": [{"id": task["id"], "work_function": work_function(task)} for task in active],
                        })
            time.sleep(1)
        decisions = decisions or []
        violations = self.workspace_violations()
        if violations:
            raise AssertionError(f"planner or verifier changed the workspace: {violations}")
        if len(decisions) != 1:
            raise AssertionError(f"expected one plan review decision, got {[d['decision_key'] for d in decisions]}")
        self.plan_decision = decisions[0]
        context = json.loads(self.plan_decision.get("context") or "{}")
        identity, digest = context.get("plan_review_identity"), context.get("plan_review_identity_hash")
        required = {"project_id", "goal_id", "run_id", "artifact_id", "planning_task_id", "planning_version", "winning_session_id", "provenance_hash", "canonical_content_hash", "renderer_version", "rendered_content_hash", "review_cycle"}
        if not isinstance(identity, dict) or set(identity) != required or digest != _plan_review_hash(identity):
            raise AssertionError("plan review decision does not bind the full canonical PlanReviewIdentity")
        if (not _same_uuid(identity["project_id"], self.project_id)
                or not _same_uuid(identity["goal_id"], self.goal_id)
                or not _same_uuid(identity["run_id"], self.run_id)
                or not _same_uuid(identity["planning_task_id"], self.planner_task)
                or not _same_uuid(identity["winning_session_id"], self.planner_session["id"])):
            raise AssertionError("plan review identity does not bind the current run and completed planner lineage")
        bindings = self.sc.oracle.plan_state_bindings(self.run_id, identity["artifact_id"]) or {}
        stored = (bindings.get("plan_review_identities") or {}).get(str(identity["artifact_id"]))
        if stored != identity:
            raise AssertionError("plan_state plan review identity differs from decision binding")
        artifact = httpx.get(f"{self.sc.base_url}/api/v1/projects/{self.project_id}/artifacts/{identity['artifact_id']}", timeout=60)
        artifact.raise_for_status()
        artifact_body = artifact.json()
        metadata = artifact_body.get("metadata") or {}
        if (artifact_body.get("content_hash") != identity["canonical_content_hash"]
                or metadata.get("planning_task_id") != identity["planning_task_id"]
                or metadata.get("planning_version") != identity["planning_version"]
                or not _same_uuid(metadata.get("winning_session_id"), self.planner_session["id"])):
            raise AssertionError("artifact metadata does not match the reviewed plan identity")
        self.plan_identity, self.plan_hash = identity, digest
        self.plan_items = metadata.get("plan_items") or []
        ui = self.pw_ok("inspect-plan", LIVE_ACCEPTANCE_PLAN_DECISION_TITLE=str(self.plan_decision.get("title") or ""))
        self.verify("read_only_planner", {"head_unchanged": self._git("rev-parse", "HEAD").strip() == self.workspace_head,
                                          "workspace_clean": not self.workspace_violations(),
                                          "plan_canary_absent": not (self.sc.run_dir.workspace / "PLAN.md").exists(),
                                          "planner_consumed_once": self.sc.oracle.report_consumed_count(self.run_id, self.planner_task) == 1})
        self.verify("plan_review_binding", {"decision_readable": ui.get("decision_readable") is True,
                                             "pending_approval_decision": self.plan_decision["status"] == "pending",
                                             "canonical_hash_matches": digest == _plan_review_hash(identity),
                                             "stored_identity_matches": stored == identity,
                                             "artifact_hash_matches": artifact_body.get("content_hash") == identity["canonical_content_hash"],
                                             "artifact_task_and_version_bound": (metadata.get("planning_task_id") == identity["planning_task_id"]
                                                                                  and metadata.get("planning_version") == identity["planning_version"]),
                                             "artifact_winning_session_bound": _same_uuid(metadata.get("winning_session_id"), self.planner_session["id"])}, context=context, bindings=bindings)
        self.sc.oracle_dump("plan-review-pending")

    def plan_owners(self) -> None:
        eligibility = self.eligibility()
        agents = [item for item in self.plan_items if isinstance(item, dict) and item.get("owner_type") == "agent"]
        task_agents = [item for item in agents if item.get("unit_type") != "goal"]
        goal_agents = [item for item in agents if item.get("unit_type") == "goal"]
        humans = [item for item in self.plan_items if isinstance(item, dict) and item.get("owner_type") == "human"]
        assignments = {item.get("work_function"): item.get("agent_ref") for item in (self.proposal(self.p2["id"]).get("proposal") or {}).get("assignments", [])
                       if isinstance(item, dict)}
        expected, agent_resolved = {}, []
        for item in goal_agents:
            expected[str(item.get("id") or item.get("item_key"))] = {"unit_type": "goal"}
        for item in task_agents:
            resolved = item.get("agent_id") or assignments.get(item.get("work_function"))
            try:
                resolved = str(uuid.UUID(str(resolved)))
            except (TypeError, ValueError):
                agent_resolved.append(False)
                continue
            agent_resolved.append(
                _same_uuid(resolved, assignments.get(item.get("work_function")))
                and item.get("work_function") in eligibility.get(resolved, [])
            )
            expected[str(item.get("id") or item.get("item_key"))] = {
                "agent_id": resolved, "work_function": item.get("work_function"),
            }
        self.expected_plan_ownership = expected
        self.notes["human_plan_rows"] = (
            "not_exercised: generated plan has no human-owned rows" if not humans
            else "blocked: product ownership validation rejects human executable work; no dispatch is permitted"
        )
        self.verify("plan_ownership", {"items_present": bool(self.plan_items),
                                        "all_owner_types_explicit": all(isinstance(item, dict) and item.get("owner_type") in {"agent", "human"} for item in self.plan_items),
                                        "agent_functions_eligible": all(agent_resolved),
                                        "goal_rows_need_no_agent_or_function": all(not item.get("agent_id") and not item.get("work_function") for item in goal_agents),
                                        "human_rows_not_exercised": not humans}, items=self.plan_items, eligibility=eligibility,
                    accepted_assignments=assignments)

    def approve_plan(self) -> None:
        before_tasks = {task["id"] for task in self.sc.oracle.tasks(self.goal_id)}
        before_sessions = {session["id"] for session in self.sc.oracle.sessions()}
        ui = self.pw_ok("approve-plan", LIVE_ACCEPTANCE_PLAN_DECISION_TITLE=str(self.plan_decision.get("title") or ""))
        decision_key = str(self.plan_decision["decision_key"])
        prefix = "roadmap_plan:"
        review_marker = ":review:"
        if not decision_key.startswith(prefix) or review_marker not in decision_key:
            raise AssertionError("plan approval decision key is not a roadmap review binding")
        fingerprint, review_hash = decision_key[len(prefix):].split(review_marker, 1)
        if review_hash != self.plan_hash or not fingerprint:
            raise AssertionError("plan approval decision key does not bind the reviewed hash")
        settled = self.passive(60, lambda: next(((decision, run) for decision in self.sc.oracle.decisions(self.goal_id)
            for run in [self.sc.oracle.run_state(self.run_id)]
            if decision["id"] == self.plan_decision["id"] and decision["status"] == "answered"
            and decision.get("selected_option") == "approve"
            and (run.get("plan_state") or {}).get("status") == "accepted"), None))
        if settled is None:
            raise AssertionError("plan approval did not reach answered accepted state")
        answered, run = settled
        context = json.loads(answered.get("context") or "{}")
        state = run.get("plan_state") or {}
        time.sleep(2)
        stable_decision = next((decision for decision in self.sc.oracle.decisions(self.goal_id)
                                if decision["id"] == self.plan_decision["id"]), None)
        stable_run = self.sc.oracle.run_state(self.run_id)
        if (stable_decision is None or stable_decision.get("status") != "answered"
                or stable_decision.get("selected_option") != "approve"
                or (stable_run.get("plan_state") or {}).get("status") != "accepted"):
            raise AssertionError("plan acceptance was not stable after the bounded guard")
        run = self.sc.oracle.run_state(self.run_id)
        state = run.get("plan_state") or {}
        answered = stable_decision
        context = json.loads(answered.get("context") or "{}")
        bindings = self.sc.oracle.plan_state_bindings(self.run_id, self.plan_identity["artifact_id"]) or {}
        accepted = bindings.get("accepted_plan_ownership") or {}
        impl = [t for t in self.sc.oracle.tasks(self.goal_id) if ((t.get("metadata") or {}).get("orchestration") or {}).get("work_function") == "implementation"]
        new_tasks = {task["id"] for task in self.sc.oracle.tasks(self.goal_id)} - before_tasks
        new_sessions = {session["id"] for session in self.sc.oracle.sessions()} - before_sessions
        self.guard()
        self.verify("no_implementation", {"approved": ui.get("approved") is True,
                                            "answer_posted": bool(ui.get("answer_posts")),
                                            "answered_approve": answered.get("selected_option") == "approve",
                                            "answer_bound_to_hash": context.get("plan_review_identity_hash") == self.plan_hash,
                                            "accepted_state": state.get("status") == "accepted",
                                            "accepted_artifact_bound": _same_uuid(state.get("accepted_artifact_id"), self.plan_identity["artifact_id"]),
                                            "accepted_fingerprint_bound": state.get("accepted_plan_fingerprint") == fingerprint,
                                            "accepted_ownership_bound": accepted.get("review_identity") == self.plan_identity,
                                            "complete_ownership_bound": accepted.get("items") == self.expected_plan_ownership,
                                            "no_implementation_task": not impl,
                                            "no_new_task_or_session": not new_tasks and not new_sessions,
                                            "no_claude_invocation": not self.claude.invocations()}, answered=answered, bindings=bindings)
        self.sc.oracle_dump("plan-approved")

    def write_results(self) -> None:
        rows = {
            "SPR228": ("feedback_lineage",), "SPR229": ("documented_gap",), "SPR233": ("feedback_lineage",),
            "SPR234": ("model_drift",), "SPR236": ("plan_review_binding",), "SPR237": ("plan_ownership",),
            "SPR238": ("planning_only", "read_only_planner", "no_implementation"), "SPR197(4)": ("read_only_planner",),
        }
        lines = ["# G2--G4 live acceptance", "", "## Steps", ""]
        lines.extend(f"- {name}: {value}" for name, value in self.steps.items())
        lines.extend(["", "## Design checks", ""])
        lines.extend(f"- {name}: {check['status']}" for name, check in self.checks.items())
        lines.extend(["", "## SPR traceability", ""])
        for spr, checks in rows.items():
            statuses = ", ".join(f"{key}={self.checks[key]['status']}" for key in checks)
            lines.append(f"- {spr}: {statuses}")
        design_steps = (
            (1, "request-changes"), (2, "revision"), (3, "revision"), (4, "revision"),
            (5, "inspect-revision"), (6, "approve-hierarchy"), (7, "model-drift"),
            (8, "start-planning"), (9, "planner-held"), (10, "restart-resume"),
            (11, "inspect-plan"), (12, "inspect-plan"), (13, "plan-owners"), (14, "approve-plan"),
        )
        lines.extend(["", "## Design rows", ""])
        lines.extend(f"- {number}: {self.steps.get(step, 'not_run')}" for number, step in design_steps)
        lines.extend(["", "## Evidence", "", "- oracle dumps: planning-started, planner-held-paused, planner-done-paused-pre-restart, planner-done-paused-post-restart, plan-review-pending, plan-approved", "- retry: 0 (not exercised)", f"- codex resume: {self.notes.get('codex_resume', 'not_exercised')}", f"- human plan rows: {self.notes.get('human_plan_rows', 'not_exercised')}", "- P1 UI history: not exercised (accepted risk)"])
        (self.sc.run_dir.root / "results.md").write_text("\n".join(lines) + "\n")


def _run_l(sc: G2G4Scenario) -> None:
    run = G2G4(sc)
    run.write_verdict()
    failed = False
    for name, step in (("resume", run.resume_and_continue), ("request-changes", run.request_changes),
                       ("revision", run.revision), ("inspect-revision", run.inspect_revision),
                       ("approve-hierarchy", run.approve_hierarchy), ("model-drift", run.model_drift),
                       ("start-planning", run.start_planning), ("planner-held", run.planner),
                       ("restart-resume", run.restart_resume), ("inspect-plan", run.inspect_plan),
                       ("plan-owners", run.plan_owners), ("approve-plan", run.approve_plan)):
        if failed:
            continue
        try:
            step()
            run.steps[name] = "pass"
        except Exception as exc:
            run.steps[name] = f"fail: {type(exc).__name__}: {exc}"[:400]
            sc.mark(f"STEP FAILED {name}: {exc}")
            failed = True
        run.write_verdict()
        run.write_results()
    ok = not failed and all(check["status"] == "pass" for check in run.checks.values())
    sc.set_verdict(ok, checks=run.checks, steps=run.steps, **run.notes)
    assert ok, run.steps


def _complete_s(sc: G2G4Scenario, driver: G2G4, p1: dict) -> None:
    """Assert the shared Run S P1 stop condition and write its snapshot."""
    outputs = sc.oracle.process_outputs(p1["id"]) or {}
    proposal = outputs.get("proposal") or {}
    coverage, eligibility = proposal.get("responsibility_coverage") or [], driver.eligibility()
    assignments = [item for item in proposal.get("assignments", []) if isinstance(item, dict)]
    goal_detail = httpx.get(f"{sc.base_url}/api/v1/projects/{driver.project_id}/orchestration/goals/{driver.goal_id}", timeout=60)
    goal_detail.raise_for_status()
    persisted_goal = goal_detail.json().get("goal") or goal_detail.json()
    conditions = {
        "validation_gap": any(row.get("work_function") == "validation" and row.get("status") == "gap" and not row.get("agent_ref") for row in coverage if isinstance(row, dict)),
        "manager_owns_planning_or_management": any(item.get("work_function") in {"planning", "management"} and _same_uuid(item.get("agent_ref"), driver.ids["manager_id"]) for item in assignments),
        "assignment_owners_eligible": all(item.get("work_function") in eligibility.get(str(item.get("agent_ref")), []) for item in assignments),
        "alternate_eligible_owner": any(
            item.get("work_function") in eligibility.get(str(item.get("agent_ref")), []) and any(
                not _same_uuid(agent_id, item.get("agent_ref")) and item.get("work_function") in functions
                for agent_id, functions in eligibility.items()) for item in assignments),
        "selected_manager": _same_uuid(persisted_goal.get("manager_agent_id"), driver.ids["manager_id"]),
        "planning_mode_roadmap": persisted_goal.get("planning_mode") == "roadmap",
    }
    if not all(conditions.values()):
        raise AssertionError(f"Run S prerequisites failed: {conditions}")
    sc.findings["planning_mode"] = persisted_goal.get("planning_mode")
    p1_input = sc.oracle.process_input_evidence(p1["id"])
    sc.findings["p1_input_evidence"] = p1_input
    sc.findings["p1_input_evidence_bytes_sha256"] = hashlib.sha256(_json_bytes(p1_input)).hexdigest()
    sc.oracle_dump("P1-waiting-decision")
    driver.pw_ok("pause")
    driver.guard()
    sc.take_snapshot("S-th-approval-pending-paused", require_clean=True)
    (sc.artifact_root / "results.md").write_text(
        "# G2--G4 Run S\n\n- P1 hierarchy proposal: pass\n"
        f"- Prerequisites: {json.dumps(conditions, sort_keys=True)}\n")
    sc.set_verdict(True, p1=p1["id"], prerequisites=conditions, findings=sc.findings)


def _run_s(sc: G2G4Scenario) -> None:
    goal = create_goal(sc.base_url, sc.ids["project_id"], G2G4_GOAL_TEXT, goal_type="roadmap",
                       constraints=G2G4_GOAL_CONSTRAINTS)
    sc.ids["goal_id"] = goal["goal"]["id"]
    api = Api(sc.base_url, sc.ids["project_id"], sc.ids["goal_id"])
    sc.ids["run_id"] = api.start_baseline()["run"]["id"]
    driver = G2G4.__new__(G2G4)
    driver.sc, driver.codex, driver.claude, driver.released, driver.claude_released = sc, sc.shims["codex"], sc.shims["claude"], [], []
    driver.ids = sc.ids = {**sc.ids, **_latest_ids(sc.run_dir.db), **_live_agent_ids(sc)}
    driver.run_id, driver.goal_id, driver.project_id = (driver.ids[k] for k in ("run_id", "goal_id", "project_id"))
    driver.notes, driver.checks, driver.steps = {"release_timeline": []}, {key: {"status": "not_run", "evidence": []} for key in CHECKS}, {}
    driver.p2 = driver.planner_task = driver.planner_inv = None
    driver.feedback = {}
    def adr_ready() -> bool:
        pending_decisions = [d for d in sc.oracle.decisions(driver.goal_id) if d["status"] == "pending"]
        unexpected_keys = [str(d["decision_key"]) for d in pending_decisions
                           if not str(d["decision_key"]).startswith("agent_definition_review")]
        if unexpected_keys:
            raise AssertionError(f"Baseline needs user input before agent review batch: {', '.join(unexpected_keys)}")
        pending_adr = [d for d in pending_decisions if str(d["decision_key"]).startswith("agent_definition_review")]
        return len(pending_adr) == 4

    adr_ready = driver.drive_codex(300, adr_ready, "baseline review")
    if not adr_ready:
        raise AssertionError("agent-definition reviews did not all reach pending")
    l1 = L1(sc)
    l1.pw_ok("pause")
    l1.pw_ok("approve-batch")
    l1.pw_ok("resume")
    l1.pw_ok("continue-baseline")
    p1 = driver.drive_codex(300, lambda: next((p for p in sc.oracle.process_runs(goal_id=driver.goal_id)
        if p["process_type"] == "team_hierarchy" and p["status"] == "waiting_decision"), None), "baseline hierarchy")
    if p1 is None:
        raise AssertionError("P1 did not reach waiting_decision")
    _complete_s(sc, driver, p1)


def _refresh_retry_system_message(db: Path, process_id: str) -> tuple[str, str]:
    """Replace only the saved retry request's architect instructions with HEAD wording.

    The preamble and user payload stay byte-identical; input_snapshot is never touched.
    """
    pid = uuid.UUID(str(process_id)).hex
    marker = "Act as the team architect."
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute("SELECT outputs FROM orchestration_process_runs WHERE id = ?", (pid,)).fetchone()
        if row is None:
            raise AssertionError(f"no orchestration process run {process_id} in {db}")
        outputs = json.loads(row[0])
        request = outputs["_lm_retry"]["request"]
        old_system = request["messages"][0]["content"]
        user_payload = json.loads(request["messages"][1]["content"])
        new_tail = TeamHierarchyAnalyzer.build_request(user_payload, project=None)["messages"][0]["content"]
        if old_system.count(marker) != 1 or new_tail.count(marker) != 1:
            raise AssertionError("architect marker must occur exactly once in saved and HEAD system messages")
        new_system = old_system[:old_system.index(marker)] + new_tail[new_tail.index(marker):]
        if "criterion:<1-based success-criteria index>" not in new_system:
            raise AssertionError("refreshed system message must carry the HEAD criterion rule")
        old_sha = hashlib.sha256(_json_bytes(request)).hexdigest()
        if new_system == old_system:
            # ponytail: idempotent re-run; already refreshed, nothing to write.
            return old_sha, old_sha
        request["messages"][0]["content"] = new_system
        new_sha = hashlib.sha256(_json_bytes(request)).hexdigest()
        cursor = conn.execute("UPDATE orchestration_process_runs SET outputs = ? WHERE id = ?",
                              (json.dumps(outputs), pid))
        if cursor.rowcount != 1:
            raise AssertionError(f"expected one orchestration process row, updated {cursor.rowcount}")
        conn.commit()
    finally:
        conn.close()
    return old_sha, new_sha


def _resume_preflight(root: Path, require_first_resume: bool = False, previous: str | None = None,
                      artifact: str | None = None) -> tuple[dict[str, str], dict]:
    """Refuse to alter any state unless this is the exact paused retry checkpoint."""
    if (previous is None) != (artifact is None):
        raise ValueError("previous and artifact must be given together")
    rd = RunDir(root, root / "app.db", root / "workspace", root / "snapshots", root / "shim",
                root / "proxy.jsonl", root / "spend.jsonl", root / "config.json")
    assert_isolated(rd, rd.db, rd.workspace)
    with oracle.Oracle(rd.db) as before:
        ids = _latest_ids(rd.db)
        run = before.run_state(ids.get("run_id"))
        processes = before.process_runs(run_id=ids.get("run_id"))
        current = [row for row in processes if row["process_type"] == "team_hierarchy" and not row.get("superseded_by_id")]
        pending = [row["decision_key"] for row in before.decisions(ids.get("goal_id")) if row["status"] == "pending"]
        retry = before.process_outputs(current[-1]["id"]) if len(current) == 1 else None
    checkpoint = retry.get("_lm_retry") if isinstance(retry, dict) else None
    request = checkpoint.get("request") if isinstance(checkpoint, dict) else None
    lsof = subprocess.run(["lsof", "-t", str(rd.db)], capture_output=True, text=True, check=False)
    if lsof.returncode not in (0, 1):
        raise AssertionError(f"lsof failed for saved app.db: rc={lsof.returncode} {lsof.stderr.strip()[:200]}")
    clean = subprocess.run(["git", "status", "--porcelain"], cwd=rd.workspace,
                           capture_output=True, text=True, check=True).stdout
    resident = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True, check=True).stdout
    owned_processes = [line for line in resident.splitlines() if str(root) in line and
                       ("-m huddleroom.cli serve" in line or "cli_shim_runner.py" in line)]
    conditions = {
        "ids": all(ids.get(key) for key in ("project_id", "goal_id", "run_id")),
        "paused": bool(run and run["status"] == "paused" and run["pause_requested"] == 1),
        "baseline": bool(run and run["phase"] == "baseline"),
        "one_current_running_hierarchy": len(current) == 1 and current[0]["status"] == "running",
        "lm_retry": isinstance(checkpoint, dict) and checkpoint.get("kind") == "team_hierarchy",
        "retry_request": isinstance(request, dict),
        "no_pending_decisions": not pending,
        "clean_workspace": not clean.strip() and not (rd.workspace / "PLAN.md").exists(),
        "no_run_processes": not owned_processes and lsof.returncode == 1,
        "resume_artifacts": (((root / previous).is_dir() and not (root / artifact).exists())
                             if previous and artifact else
                             (root / "resume-S-1m").is_dir() and not (root / "resume-S-1m-2").exists()
                             if require_first_resume else not (root / "resume-S-1m").exists()),
    }
    if not all(conditions.values()):
        raise AssertionError(f"resume preflight failed: {conditions}; pending={pending}")
    return ids, request


def _run_resume_s(sc: G2G4ResumeScenario, frozen_input: dict, retry_request: dict) -> None:
    driver = G2G4.__new__(G2G4)
    driver.sc, driver.codex, driver.claude, driver.released, driver.claude_released = (
        sc, sc.shims["codex"], sc.shims["claude"], [], [])
    driver.ids = sc.ids = {**sc.ids, **_latest_ids(sc.run_dir.db), **_live_agent_ids(sc)}
    driver.run_id, driver.goal_id, driver.project_id = (driver.ids[key] for key in ("run_id", "goal_id", "project_id"))
    driver.notes, driver.checks, driver.steps = {"release_timeline": []}, {key: {"status": "not_run", "evidence": []} for key in CHECKS}, {}
    driver.p2 = driver.planner_task = driver.planner_inv = None
    driver.feedback = {}
    api = Api(sc.base_url, driver.project_id, driver.goal_id)
    if api.resume() != 200:
        raise AssertionError(f"resume endpoint failed: {api.calls[-1]}")
    retry = api.retry_async("team_hierarchy", timeout=300)
    p1 = driver.drive_codex(360, lambda: next((row for row in sc.oracle.process_runs(goal_id=driver.goal_id)
        if row["process_type"] == "team_hierarchy" and row["status"] == "waiting_decision"), None), "team hierarchy retry")
    retry["thread"].join(10)
    if not retry["done"] or retry["status"] != 200:
        raise AssertionError(f"baseline/retry did not complete successfully: {retry}")
    if p1 is None:
        raise AssertionError("resumed P1 did not reach waiting_decision")
    reused = sc.oracle.process_input_snapshot(p1["id"])
    if _json_bytes(reused) != _json_bytes(frozen_input):
        raise AssertionError("resumed hierarchy did not reuse the frozen request")
    new_ordinals = [item.ordinal for item in driver.codex.invocations()]
    first = min(new_ordinals, default=0)
    if first != sc.historical_codex_last + 1:
        raise AssertionError(f"resumed codex first ordinal {first} does not follow {sc.historical_codex_last}")
    sc.findings.update({"resume_calls": api.calls, "frozen_request_reused": True,
                        "retry_request_sha256": hashlib.sha256(_json_bytes(retry_request)).hexdigest(),
                        "retry_request_reuse": "TeamHierarchyProcess.retry_failed -> analyzer.review_request(request)",
                        "historical_ordinals": {"codex_last": sc.historical_codex_last, "new_first": first}})
    _complete_s(sc, driver, p1)


@pytest.mark.live
@pytest.mark.skipif(not LIVE_ENABLED, reason="requires HUDDLEROOM_LIVE_ACCEPTANCE=1 and HUDDLEROOM_LIVE_EVAL=1")
def test_live_run_g2g4_s():
    if not ROOT:
        pytest.skip("needs LIVE_ACCEPTANCE_EVIDENCE_ROOT")
    if os.environ.get("LIVE_ACCEPTANCE_RETRY_FROM"):
        pytest.fail("automatic retry is not implemented; preserve evidence and start no replacement run")
    root = Path(ROOT)
    cap = _s_cap()
    historical_path = root / "campaign-spend.jsonl"
    fresh = os.environ.get("LIVE_ACCEPTANCE_G2G4_S_CAP") is not None
    campaign_path = root / "campaign-spend-g2g4-S-fresh.jsonl" if fresh else historical_path
    campaign_cap = cap if fresh else CAMPAIGN_CAP
    if fresh:
        historical_totals = _ledger_totals_readonly(historical_path)
        seed_campaign(campaign_path, campaign_cap, {})
    else:
        historical_totals = seed_campaign(historical_path, CAMPAIGN_CAP, CAMPAIGN_PRIOR).totals()
    prior_s = historical_totals.get("g2g4-S", {}).get("non_cached", 0)
    spec = {"codex_each": True, "claude_each": True, "codex_match": [TH_MARKER]}
    with G2G4Scenario("g2g4-S", None, root, cap, hold_spec=spec, roster="g2g4",
                      campaign_ledger_path=campaign_path, campaign_cap=campaign_cap) as sc:
        sc.findings["historical_campaign_audit"] = {
            "path": str(historical_path), "totals": historical_totals,
        }
        if not fresh:
            # Carry prior failed S usage into this run's local budget, not as new spend.
            sc.ledger.record(sc.name, "prior_S_attempts", prior_s, 0)
        _run_s(sc)


@pytest.mark.live
@pytest.mark.skipif(not (LIVE_ENABLED and RESUME_S),
                    reason="requires both live flags and LIVE_ACCEPTANCE_G2G4_RESUME_S=1")
def test_live_run_g2g4_resume_s_1m():
    """One user-authorized continuation; never seeds or restarts baseline work."""
    if not RESUME_ROOT_RAW:
        pytest.skip("needs LIVE_ACCEPTANCE_G2G4_RESUME_ROOT")
    if not Path(RESUME_ROOT_RAW).is_absolute():
        raise AssertionError("LIVE_ACCEPTANCE_G2G4_RESUME_ROOT must be absolute")
    assert RESUME_ROOT is not None
    root = RESUME_ROOT.resolve()
    expected_root = _expected_resume_root()
    if root != expected_root:
        raise AssertionError(f"resume is authorized only for {expected_root}")
    ids, retry_request = _resume_preflight(root)
    historical = _ledger_totals_readonly(root / "spend.jsonl")
    old_fresh = historical["global"]["non_cached"]
    if old_fresh != 213_983:
        raise AssertionError(f"unexpected saved-run spend: {old_fresh}")
    campaign = _ledger_totals_readonly(root.parent / "campaign-spend-g2g4-S-fresh.jsonl")
    prior = campaign["global"]["non_cached"]
    if prior != 487_444:
        raise AssertionError(f"unexpected fresh campaign spend: {prior}")
    with oracle.Oracle(root / "app.db") as before:
        current = [row for row in before.process_runs(run_id=ids["run_id"])
                   if row["process_type"] == "team_hierarchy" and not row.get("superseded_by_id")]
        frozen = before.process_input_snapshot(current[0]["id"])
    if not isinstance(frozen, dict):
        raise AssertionError("current hierarchy has no frozen input snapshot")
    artifact = root / "resume-S-1m"
    with G2G4ResumeScenario(
        root, prior, "g2g4-S-resume", None, root.parent, 1_000_000,
        hold_spec={"codex_each": True, "claude_each": True, "codex_match": [TH_MARKER]}, roster="g2g4",
        campaign_ledger_path=artifact / "campaign-spend.jsonl", campaign_cap=1_000_000,
    ) as sc:
        sc.findings["historical_campaign_audit"] = {
            "saved_run_spend": old_fresh, "prior_total": prior, "cap": 1_000_000, "warn": 800_000,
        }
        _run_resume_s(sc, frozen, retry_request)


@pytest.mark.live
@pytest.mark.skipif(not (LIVE_ENABLED and RESUME_S_2),
                    reason="requires both live flags and LIVE_ACCEPTANCE_G2G4_RESUME_S=2")
def test_live_run_g2g4_resume_s_1m_2():
    """Continue only the paused 1M retry checkpoint after ordinals 15--17."""
    if not RESUME_ROOT_RAW:
        pytest.skip("needs LIVE_ACCEPTANCE_G2G4_RESUME_ROOT")
    if not Path(RESUME_ROOT_RAW).is_absolute():
        raise AssertionError("LIVE_ACCEPTANCE_G2G4_RESUME_ROOT must be absolute")
    assert RESUME_ROOT is not None
    root = RESUME_ROOT.resolve()
    expected_root = _expected_resume_root()
    if root != expected_root:
        raise AssertionError(f"resume is authorized only for {expected_root}")
    ids, retry_request = _resume_preflight(root, require_first_resume=True)
    first = root / "resume-S-1m"
    campaign = _ledger_totals_readonly(first / "campaign-spend.jsonl")
    prior = campaign["global"]["non_cached"]
    if prior != 562_663:
        raise AssertionError(f"unexpected first-resume campaign spend: {prior}")
    with oracle.Oracle(root / "app.db") as current:
        runs = [row for row in current.process_runs(run_id=ids["run_id"])
                if row["process_type"] == "team_hierarchy" and not row.get("superseded_by_id")]
        frozen = current.process_input_snapshot(runs[0]["id"])
    with oracle.Oracle(first / "snapshots" / "00-before.db") as saved:
        saved_ids = _latest_ids(first / "snapshots" / "00-before.db")
        saved_runs = [row for row in saved.process_runs(run_id=saved_ids["run_id"])
                      if row["process_type"] == "team_hierarchy" and not row.get("superseded_by_id")]
        saved_outputs = saved.process_outputs(saved_runs[0]["id"]) or {}
        saved_checkpoint = saved_outputs.get("_lm_retry") or {}
        saved_request = saved_checkpoint.get("request")
        saved_frozen = saved.process_input_snapshot(saved_runs[0]["id"])
    if not isinstance(frozen, dict) or not isinstance(saved_frozen, dict):
        raise AssertionError("current and saved hierarchy inputs must be frozen")
    if _json_bytes(frozen) != _json_bytes(saved_frozen):
        raise AssertionError("second resume would alter the frozen hierarchy input")
    if not isinstance(saved_request, dict) or _json_bytes(retry_request) != _json_bytes(saved_request):
        raise AssertionError("second resume would alter the saved retry request")
    artifact = root / "resume-S-1m-2"
    with G2G4SecondResumeScenario(
        root, prior, "g2g4-S-resume-2", None, root.parent, 1_000_000,
        hold_spec={"codex_each": True, "claude_each": True, "codex_match": [TH_MARKER]}, roster="g2g4",
        campaign_ledger_path=artifact / "campaign-spend.jsonl", campaign_cap=1_000_000,
    ) as sc:
        sc.findings["historical_campaign_audit"] = {
            "first_resume_total": prior, "cap": 1_000_000, "warn": 800_000,
        }
        _run_resume_s(sc, frozen, retry_request)


@pytest.mark.live
@pytest.mark.skipif(not (LIVE_ENABLED and RESUME_S_3),
                    reason="requires both live flags and LIVE_ACCEPTANCE_G2G4_RESUME_S=3")
def test_live_run_g2g4_resume_s_1m_3():
    """Continue only the paused 1M retry checkpoint after ordinals 18--20 with HEAD architect wording."""
    if not RESUME_ROOT_RAW:
        pytest.skip("needs LIVE_ACCEPTANCE_G2G4_RESUME_ROOT")
    if not Path(RESUME_ROOT_RAW).is_absolute():
        raise AssertionError("LIVE_ACCEPTANCE_G2G4_RESUME_ROOT must be absolute")
    assert RESUME_ROOT is not None
    root = RESUME_ROOT.resolve()
    expected_root = _expected_resume_root()
    if root != expected_root:
        raise AssertionError(f"resume is authorized only for {expected_root}")
    ids, retry_request = _resume_preflight(root, previous="resume-S-1m-2", artifact="resume-S-1m-3")
    second = root / "resume-S-1m-2"
    campaign = _ledger_totals_readonly(second / "campaign-spend.jsonl")
    prior = campaign["global"]["non_cached"]
    if prior != 620_517:
        raise AssertionError(f"unexpected second-resume campaign spend: {prior}")
    with oracle.Oracle(root / "app.db") as current:
        runs = [row for row in current.process_runs(run_id=ids["run_id"])
                if row["process_type"] == "team_hierarchy" and not row.get("superseded_by_id")]
        frozen = current.process_input_snapshot(runs[0]["id"])
    with oracle.Oracle(second / "snapshots" / "00-before.db") as saved:
        saved_ids = _latest_ids(second / "snapshots" / "00-before.db")
        saved_runs = [row for row in saved.process_runs(run_id=saved_ids["run_id"])
                      if row["process_type"] == "team_hierarchy" and not row.get("superseded_by_id")]
        saved_outputs = saved.process_outputs(saved_runs[0]["id"]) or {}
        saved_request = (saved_outputs.get("_lm_retry") or {}).get("request")
        saved_frozen = saved.process_input_snapshot(saved_runs[0]["id"])
    if not isinstance(frozen, dict) or not isinstance(saved_frozen, dict):
        raise AssertionError("current and saved hierarchy inputs must be frozen")
    if _json_bytes(frozen) != _json_bytes(saved_frozen):
        raise AssertionError("third resume would alter the frozen hierarchy input")
    if not isinstance(saved_request, dict) or _json_bytes(retry_request) != _json_bytes(saved_request):
        raise AssertionError("third resume would alter the saved retry request")
    # Validate historical ordinals before any app.db mutation.
    G2G4ThirdResumeScenario.__new__(G2G4ThirdResumeScenario)._historical_codex(root)
    old_sha, new_sha = _refresh_retry_system_message(root / "app.db", runs[0]["id"])
    with oracle.Oracle(root / "app.db") as refreshed_db:
        refreshed_outputs = refreshed_db.process_outputs(runs[0]["id"]) or {}
        refreshed_request = (refreshed_outputs.get("_lm_retry") or {}).get("request")
        refreshed_frozen = refreshed_db.process_input_snapshot(runs[0]["id"])
    if _json_bytes(refreshed_frozen) != _json_bytes(frozen):
        raise AssertionError("system message refresh altered the frozen hierarchy input")
    if not isinstance(refreshed_request, dict):
        raise AssertionError("refreshed retry request is missing")
    architect = "Act as the team architect."
    if refreshed_request["messages"][1] != retry_request["messages"][1]:
        raise AssertionError("system message refresh altered the user payload")
    old_text, new_text = retry_request["messages"][0]["content"], refreshed_request["messages"][0]["content"]
    if new_text[:new_text.index(architect)] != old_text[:old_text.index(architect)]:
        raise AssertionError("system message refresh altered the preamble before the architect section")
    unchanged = ("model", "response_format", "temperature", "max_tokens")
    if {key: refreshed_request.get(key) for key in unchanged} != {key: retry_request.get(key) for key in unchanged}:
        raise AssertionError("system message refresh altered non-message request keys")
    artifact = root / "resume-S-1m-3"
    with G2G4ThirdResumeScenario(
        root, prior, "g2g4-S-resume-3", None, root.parent, 1_000_000,
        hold_spec={"codex_each": True, "claude_each": True, "codex_match": [TH_MARKER]}, roster="g2g4",
        campaign_ledger_path=artifact / "campaign-spend.jsonl", campaign_cap=1_000_000,
    ) as sc:
        sc.findings["historical_campaign_audit"] = {
            "second_resume_total": prior, "cap": 1_000_000, "warn": 800_000,
        }
        sc.findings["system_message_refresh"] = {
            "old_request_sha256": old_sha, "new_request_sha256": new_sha,
            "scope": "messages[0] architect section replaced with HEAD wording; preamble and user payload unchanged",
        }
        _run_resume_s(sc, frozen, refreshed_request)


@pytest.mark.live
@pytest.mark.skipif(not LIVE_ENABLED, reason="requires HUDDLEROOM_LIVE_ACCEPTANCE=1 and HUDDLEROOM_LIVE_EVAL=1")
@pytest.mark.skipif(not SNAP, reason="needs LIVE_ACCEPTANCE_SNAPSHOT_S from Run S")
def test_live_run_g2g4_l():
    if not ROOT:
        pytest.skip("needs LIVE_ACCEPTANCE_EVIDENCE_ROOT")
    if os.environ.get("LIVE_ACCEPTANCE_RETRY_FROM"):
        pytest.fail("automatic retry is not implemented; preserve evidence and start no replacement run")
    root = Path(ROOT)
    spec = {"codex_each": True, "claude_each": True, "codex_match": [PLAN_MARKER]}
    raw = os.environ.get("LIVE_ACCEPTANCE_G2G4_L_CAMPAIGN_CAP")
    if raw is None:
        seed_campaign(root / "campaign-spend.jsonl", CAMPAIGN_CAP, CAMPAIGN_PRIOR)
        with G2G4Scenario("g2g4-L", Path(SNAP), root, CAP, hold_spec=spec, roster="g2g4") as sc:
            _run_l(sc)
        return
    try:
        cap = int(raw)
    except ValueError:
        pytest.fail("LIVE_ACCEPTANCE_G2G4_L_CAMPAIGN_CAP must be a positive integer")
    if cap <= 0:
        pytest.fail("LIVE_ACCEPTANCE_G2G4_L_CAMPAIGN_CAP must be a positive integer")
    # Historical campaign ledger keeps its original cap on line 1; the authorized cap goes to a fresh ledger.
    historical = root / "campaign-spend.jsonl"
    prior = _ledger_totals_readonly(historical)["global"]["non_cached"]
    campaign_path = root / f"campaign-spend-g2g4-L-{cap}.jsonl"
    seed_campaign(campaign_path, cap, {"prior_campaign_total": prior})
    with G2G4Scenario("g2g4-L", Path(SNAP), root, CAP, hold_spec=spec, roster="g2g4",
                      campaign_ledger_path=campaign_path, campaign_cap=cap) as sc:
        sc.continue_through_warning = True
        if not hasattr(sc, "warning_recorded"):
            sc.warning_recorded = set()
        sc.findings["historical_campaign_audit"] = {"path": str(historical), "prior_total": prior, "cap": cap}
        _run_l(sc)
