"""Snapshot restore and the shared scenario fixture for live acceptance runs.

restore_snapshot() turns a saved snapshot (see dry_run._snapshot_dir) into a fresh run dir.
LiveScenario wires one scenario end to end: restore or seed, spend ledger, CLI shims with holds armed
before the server starts, the server, evidence snapshots, timeline, teardown, verdict.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sqlite3
import subprocess
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from tests.live_acceptance import auth_dir_watch, oracle
from tests.live_acceptance.cli_shim import TOOLS, make_shim, resolve_real
from tests.live_acceptance.dry_run import Api, _auth_roots, _leaks, _scrub, _sha256, _snapshot_dir
from tests.live_acceptance.live_env import server_env_for_cli
from tests.live_acceptance.run_dir import RunDir, assert_isolated, build_run_dir, init_workspace_repo
from tests.live_acceptance.seed import seed_minimal_roster
from tests.live_acceptance.server_supervisor import Supervisor, free_port
from tests.live_acceptance.spend_ledger import Ledger


class RestoreError(RuntimeError):
    pass


_MANIFEST_KEYS = ("db_file_sha256", "workspace_tar_sha256", "db_canonical_sha256")
_TERMINAL = ("failed", "blocked", "completed", "cancelled")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root_re(roots: list[str]) -> re.Pattern:
    """Matches an old run root only as a whole path prefix: not followed by a name character,
    so a sibling such as <root>-b does not match but <root>/x, <root>" and <root> at the end do."""
    return re.compile("(?:" + "|".join(re.escape(r) for r in sorted(roots, key=len, reverse=True)) + r")(?![\w.-])")


def find_path_columns(db: Path, needles: list[str]) -> list[tuple[str, str, int]]:
    """(table, column, row count) for every column whose text holds any root as a path prefix."""
    rx = _root_re(needles)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.create_function("hasroot", 1, lambda v: int(v is not None and rx.search(v) is not None))
    try:
        out = []
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for t in tables:
            for col in [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]:
                n = conn.execute(f'SELECT count(*) FROM "{t}" WHERE hasroot(CAST("{col}" AS TEXT))').fetchone()[0]
                if n:
                    out.append((t, col, n))
        return out
    finally:
        conn.close()


def _latest_ids(db: Path) -> dict:
    """run/goal/project ids (dashed) of the newest run, or {}."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT r.id, r.goal_id, g.project_id FROM orchestration_runs r "
                           "LEFT JOIN orchestration_goals g ON g.id = r.goal_id ORDER BY r.rowid DESC LIMIT 1").fetchone()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    if row is None:
        return {}
    u = lambda h: str(uuid.UUID(h)) if h and len(str(h)) == 32 else h
    return {k: u(v) for k, v in zip(("run_id", "goal_id", "project_id"), row) if v}


def snapshot_state(db: Path) -> dict:
    """Cleanliness of a DB: no active run blockers, no _lm_retry checkpoint in process outputs, no active
    *_analyzer_error warning. A dirty snapshot would stall the restored run (product waits for a retry)."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        kinds = []
        for (raw,) in conn.execute("SELECT active_blockers FROM orchestration_runs "
                                   "WHERE status NOT IN (?,?,?,?)", _TERMINAL):
            for b in json.loads(raw or "[]"):
                kinds.append(str((b.get("kind") or b.get("type") or b) if isinstance(b, dict) else b)[:80])
        retry = conn.execute("SELECT count(*) FROM orchestration_process_runs "
                             "WHERE instr(CAST(outputs AS TEXT), '_lm_retry') > 0").fetchone()[0]
        analyzer = [r[0] for r in conn.execute("SELECT warning_type FROM orchestration_warnings "
                                               "WHERE active = 1 AND warning_type LIKE '%\\_analyzer\\_error' ESCAPE '\\'")]
    finally:
        conn.close()
    return {"clean": not (kinds or retry or analyzer), "blocker_kinds": kinds + analyzer, "lm_retry_process_runs": retry}


def _git_text_files(ws: Path) -> list[Path]:
    git = ws / ".git"
    return [p for p in [git / "config", *git.glob("worktrees/*/gitdir")] if p.is_file()]


def restore_snapshot(snapshot_dir: Path, evidence_root: Path, label: str, old_root: Path | None = None) -> RunDir:
    """Restore <snapshot_dir>/{app.db,workspace.tar.gz,manifest.json} into a fresh run dir.

    Checks: manifest has all keys; file sha256s match; the tar holds only workspace/ members; the
    restored DB hashes to the manifest canonical hash; every non-terminal run is paused with
    pause_requested=1 (else RestoreError).

    Path rewrite: the only absolute-path column the product writes is projects.workspace_path, so a
    prefix UPDATE on that one column runs on the restored file BEFORE any server starts. The REST route
    (PUT /projects/{id}) would need a running server, and a server booted on stale paths could act on
    the old workspace first. Matching is boundary-aware (see _root_re). Text files in workspace/.git
    (config, worktrees/*/gitdir) are rewritten the same way. Any other old-root string in the DB or
    those files fails the restore. The pre-start DB is kept as snapshots/00-restored.db.

    Caveat: the snapshot may hold stale pids/sessions from snapshot time, and product recovery can act
    on server start (the scenario timeline records this).
    """
    snap = Path(snapshot_dir).resolve()
    manifest = json.loads((snap / "manifest.json").read_text())
    missing = [k for k in _MANIFEST_KEYS if not manifest.get(k)]
    if missing:
        raise RestoreError(f"manifest missing keys: {missing}")
    for name, key in (("app.db", "db_file_sha256"), ("workspace.tar.gz", "workspace_tar_sha256")):
        if _sha256(snap / name) != manifest[key]:
            raise RestoreError(f"{name} does not match manifest {key}")
    with tarfile.open(snap / "workspace.tar.gz") as tf:
        bad = [m.name for m in tf.getmembers() if m.name != "workspace" and not m.name.startswith("workspace/")]
    if bad:
        raise RestoreError(f"tar members outside workspace/: {bad[:5]}")
    old = Path(old_root) if old_root else snap.parent.parent
    needles = sorted({str(old), str(old.resolve())}, key=len, reverse=True)
    rx = _root_re(needles)

    rd = build_run_dir(evidence_root, label)
    new_root = str(rd.root.resolve())
    sub = lambda text: rx.sub(lambda _m: new_root, text)
    if oracle.snapshot(snap / "app.db", rd.db) != manifest["db_canonical_sha256"]:  # sqlite backup API copy
        raise RestoreError("restored DB does not match manifest db_canonical_sha256")
    with tarfile.open(snap / "workspace.tar.gz") as tf:
        tf.extractall(rd.root, filter="data")
    if not (rd.workspace / ".git").exists():
        raise RestoreError("restored workspace has no .git")

    conn = sqlite3.connect(rd.db)
    try:
        try:
            bad_runs = conn.execute("SELECT id, status, pause_requested FROM orchestration_runs "
                                    "WHERE status NOT IN (?,?,?,?) AND NOT (status = 'paused' AND pause_requested = 1)",
                                    _TERMINAL).fetchall()
        except sqlite3.OperationalError as e:
            raise RestoreError(f"cannot read orchestration_runs: {e}") from None
        if bad_runs:
            raise RestoreError(f"snapshot run is not paused with pause_requested=1: {bad_runs}")
        conn.create_function("rebase", 1, sub)
        conn.create_function("hasroot", 1, lambda v: int(v is not None and rx.search(v) is not None))
        n = conn.execute("UPDATE projects SET workspace_path = rebase(workspace_path) "
                         "WHERE hasroot(workspace_path)").rowcount
        conn.commit()
        if n < 1:
            raise RestoreError("no projects.workspace_path pointed at the old run root")
        paths = [r[0] for r in conn.execute("SELECT workspace_path FROM projects WHERE workspace_path IS NOT NULL")]
    finally:
        conn.close()
    outside = [p for p in paths if p != new_root and not p.startswith(new_root + "/")]
    if outside:
        raise RestoreError(f"project workspace_path not under the new run root: {outside}")
    for f in _git_text_files(rd.workspace):
        text = f.read_text(errors="replace")
        if rx.search(text):
            f.write_text(sub(text))
    left = find_path_columns(rd.db, needles)
    left += [("file", str(f.relative_to(rd.root)), 1) for f in _git_text_files(rd.workspace)
             if rx.search(f.read_text(errors="replace"))]
    if left:
        raise RestoreError(f"old run root still present after rewrite: {left}")
    git_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")} | {
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    st = subprocess.run(["git", "-C", str(rd.workspace), "status", "--porcelain"], capture_output=True, text=True,
                        env=git_env)
    if st.returncode != 0:
        raise RestoreError(f"git status failed in restored workspace: {st.stderr.strip()[:300]}")
    assert_isolated(rd, rd.db, rd.workspace)
    oracle.snapshot(rd.db, rd.snapshots / "00-restored.db")
    (rd.root / "restore.json").write_text(json.dumps(
        {"snapshot": str(snap), "old_root": str(old), "new_root": new_root, "rewrote": "projects.workspace_path",
         "projects_rewritten": n, "restored_at": _now()}, indent=2))
    return rd


class LiveScenario:
    """Context manager for one live scenario.

    hold_spec: {"codex": [ordinals...], "claude": [...]} for hold_nth, plus optional
    {"codex_match": [regex...], "claude_match": [...]}. Armed before the server starts.
    real_paths: {"codex": Path, "claude": Path} to override CLI resolution (tests pass fakes).
    artifact_root redirects human-readable run evidence; it defaults to the run directory.
    """

    def __init__(self, name: str, snapshot_dir: Path | None, evidence_root: Path, budget: int,
                 hold_spec: dict | None = None, real_paths: dict | None = None, port: int = 0,
                 output_ceiling: int = 3000, campaign_ledger_path: Path | None = None,
                 campaign_cap: int = 800_000, roster: str = "discovery",
                 artifact_root: Path | None = None):
        self.name, self.snapshot_dir, self.evidence_root = name, snapshot_dir, Path(evidence_root)
        self.budget, self.hold_spec, self.real_paths = budget, hold_spec or {}, real_paths or {}
        self.port, self.output_ceiling = port, output_ceiling
        self.campaign_ledger_path = Path(campaign_ledger_path or self.evidence_root / "campaign-spend.jsonl")
        self.campaign_cap = campaign_cap
        self.roster = roster
        self._artifact_root = Path(artifact_root) if artifact_root is not None else None
        self.errors: list[str] = []
        self.findings: dict = {"scenario": name, "budget": budget}
        self.verdict: dict | None = None
        self.ids: dict = {}
        self.snapshots: dict[str, str] = {}
        self.sup = self.ledger = self.oracle = None
        self.shims: dict = {}
        self.run_dir: RunDir

    # -- exposed helpers
    @property
    def base_url(self) -> str:
        return self.sup.base_url

    def mark(self, event: str) -> None:
        with open(self.artifact_root / "timeline.md", "a") as f:
            f.write(f"- {_now()} {_scrub(event)}\n")

    def oracle_dump(self, label: str, **ids) -> Path:
        known = {k: v for k, v in {**self.ids, **ids}.items() if k in ("run_id", "goal_id", "project_id", "task_id")}
        return self.oracle.dump(label, self.run_dir.snapshots, **known)

    def take_snapshot(self, label: str, require_clean: bool = True) -> dict:
        """Snapshot DB + workspace + manifest (with a clean flag) under snapshots/<label>/.
        Call while paused and BEFORE teardown; a dirty state raises unless require_clean=False."""
        state = snapshot_state(self.run_dir.db)
        if require_clean and not state["clean"]:
            raise RuntimeError(f"snapshot {label} is dirty: {state}")
        manifest = _snapshot_dir(self.run_dir, label, state)
        self.snapshots[label] = str(self.run_dir.snapshots / label)
        cfg = json.loads(self.run_dir.config.read_text()) if self.run_dir.config.exists() else {}
        self.run_dir.config.write_text(json.dumps({**cfg, "snapshots": self.snapshots}, indent=2))
        self.mark(f"snapshot {label} clean={state['clean']}")
        return manifest

    def _reap(self) -> int:
        kw = {}
        if "shim_log_paths" in inspect.signature(Ledger.reap).parameters:
            kw["shim_log_paths"] = [self.run_dir.shim / "shim.log"]
        return self.ledger.reap(**kw)

    def set_verdict(self, passed: bool, **notes) -> None:
        self.verdict = {"passed": bool(passed), **notes}

    def _set_artifact_root(self) -> None:
        self.artifact_root = (self._artifact_root or self.run_dir.root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    # -- lifecycle
    def __enter__(self) -> "LiveScenario":
        try:
            self._start()
        except BaseException:
            self._teardown(failed=True)
            raise
        return self

    def _start(self) -> None:
        if self.snapshot_dir is not None:
            self.run_dir = restore_snapshot(self.snapshot_dir, self.evidence_root, self.name)
            self._set_artifact_root()
            self.ids = _latest_ids(self.run_dir.db)
        else:
            self.run_dir = rd = build_run_dir(self.evidence_root, self.name)
            self._set_artifact_root()
            rd.workspace.rmdir()  # build_run_dir pre-creates it; init_workspace_repo wants to mkdir it
            init_workspace_repo(rd.workspace, rd)
        rd = self.run_dir
        self.mark(f"scenario {self.name} start; restored={self.snapshot_dir is not None}")
        if self.snapshot_dir is not None:
            self.mark("restored snapshot has stale pids/sessions from snapshot time; product recovery may act "
                      "when the server starts (pre-start DB: snapshots/00-restored.db)")
        self.auth_before = auth_dir_watch.listing(_auth_roots())
        auth_dir_watch.write_listing(self.artifact_root / "auth-before.json", self.auth_before)
        self.ledger = Ledger(rd.spend, self.budget, {self.name: self.budget})
        self._reap()
        if "campaign_ledger" not in inspect.signature(make_shim).parameters:
            raise RuntimeError("make_shim has no campaign_ledger support; the campaign spend cap is mandatory")
        extra = {"campaign_ledger": {"path": str(self.campaign_ledger_path.absolute()), "cap": self.campaign_cap,
                                     "scenario": self.name}}
        self.findings["campaign_ledger"] = {"active": True, "path": str(self.campaign_ledger_path),
                                            "cap": self.campaign_cap, "scenario": self.name}
        for tool in TOOLS:
            real = self.real_paths.get(tool) or resolve_real(tool, rd.shim)
            h = self.shims[tool] = make_shim(rd.shim, tool, Path(real), ledger_path=rd.spend, scenario=self.name,
                                             global_cap=self.budget, scenario_cap=self.budget,
                                             output_ceiling=self.output_ceiling,
                                             hold_each=bool(self.hold_spec.get(f"{tool}_each")), **extra)
            for n in self.hold_spec.get(tool, ()):
                h.hold_nth(n)
            for rx in self.hold_spec.get(f"{tool}_match", ()):
                h.hold_matching(rx)
        self.sup = Supervisor(rd, self.port or free_port(), env_overrides=server_env_for_cli(), shim_dir=rd.shim,
                              extra_pid_dirs=sorted({h.shim_dir / "pids" for h in self.shims.values()}))
        self.sup.start()
        self.mark(f"server up at {self.sup.base_url}")
        if self.snapshot_dir is None:
            self.ids = seed_minimal_roster(
                self.sup.base_url, rd, roster=self.roster,
                blank_definition_key=None if self.roster == "g2g4" else "dev1",
            )
        self.oracle = oracle.Oracle(rd.db)
        oracle.snapshot(rd.db, rd.snapshots / "00-before.db")
        self.mark("snapshot 00-before")

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.errors.append(f"body: {exc_type.__name__}: {exc}")
        self._teardown(failed=exc is not None)
        if self.errors and exc is None:
            raise RuntimeError("scenario teardown errors: " + "; ".join(self.errors))

    def _teardown(self, failed: bool = False) -> None:
        rd = getattr(self, "run_dir", None)
        if rd is None:
            return

        def attempt(label, fn):
            try:
                fn()
            except BaseException as e:  # noqa: BLE001 - each cleanup step is independent
                self.errors.append(f"{label}: {type(e).__name__}: {e}")

        if failed and self.sup is not None and self.sup._proc is not None:
            def pause():  # best effort: stop further spend while the server is still up
                ids = {**_latest_ids(rd.db), **{k: v for k, v in self.ids.items() if k in ("project_id", "goal_id")}}
                if ids.get("goal_id") and ids.get("project_id"):
                    self.findings["pause_error_path"] = Api(self.sup.base_url, ids["project_id"], ids["goal_id"]).pause()
            attempt("pause on error path", pause)
        # order matters: the server group dies FIRST so a held call dies with it and the product records no
        # analyzer-error blocker; then deny, then sweep live shims, and only then drop the gates
        if self.sup is not None:
            attempt("server kill", self.sup.kill)
            self.mark("teardown: server killed")
        for tool, h in self.shims.items():
            attempt(f"{tool} deny", h.deny)
        self.mark("teardown: deny")
        for tool, h in self.shims.items():
            attempt(f"{tool} kill_live", h.kill_live)
        for tool, h in self.shims.items():
            attempt(f"{tool} release_all", h.release_all)
        self.mark("teardown: gates released")

        def no_leftovers():
            left = {t: h.live_pids() for t, h in self.shims.items() if h.live_pids()}
            if left:
                raise RuntimeError(f"CLI processes alive after teardown: {left}")
        attempt("leftover check", no_leftovers)
        def hints():
            self.findings["timeout_hints"] = sum(
                1 for log in rd.root.glob("server-*.log") for l in log.read_text(errors="replace").splitlines()
                if re.search(r"timed out|timeout", l, re.I))
        attempt("timeout hints", hints)
        if self.oracle is not None:
            def end_dump():
                ids = {**_latest_ids(rd.db), **{k: v for k, v in self.ids.items() if k in ("project_id", "goal_id")}}
                self.oracle.dump("end", rd.snapshots, **ids)
            attempt("end oracle dump", end_dump)
            attempt("oracle close", self.oracle.close)
        if self.sup is not None:  # server is dead, so the DB is quiescent
            attempt("snapshot 90-after", lambda: oracle.snapshot(rd.db, rd.snapshots / "90-after.db"))
            self.mark("snapshot 90-after")
        if self.ledger is not None:
            def reap():
                self.findings["ledger_reaped"] = self._reap()
                self.findings["tokens"] = self.ledger.totals()
            attempt("ledger reap", reap)

        def auth_after():
            after = auth_dir_watch.listing(_auth_roots())
            auth_dir_watch.write_listing(self.artifact_root / "auth-after.json", after)
            cls = auth_dir_watch.classify(auth_dir_watch.diff(self.auth_before, after))
            self.findings["auth_diff"] = cls
            if cls["config_changed"]:
                raise RuntimeError(f"CLI config changed: {cls['config_changed']}")
        if hasattr(self, "auth_before"):
            attempt("auth after", auth_after)
        attempt("write verdict", self._write_verdict)

    def _write_verdict(self) -> None:
        rd = self.run_dir
        verdict = self.verdict or {"passed": False, "note": "no verdict set"}
        if self.errors:
            verdict["passed"] = False
        texts = {"findings.json": json.dumps({**self.findings, "errors": self.errors}, indent=2, default=str),
                 "verdict.json": json.dumps({**verdict, "errors": self.errors}, indent=2, default=str),
                 "timeline.md": (self.artifact_root / "timeline.md").read_text()
                 if (self.artifact_root / "timeline.md").exists() else ""}
        leaks = {n: _leaks(t) for n, t in texts.items() if _leaks(t)}
        if leaks:
            self.errors.append(f"secret-looking content scrubbed: {leaks}")
            verdict["passed"] = False
            texts["verdict.json"] = json.dumps({**verdict, "errors": self.errors}, indent=2, default=str)
            texts["findings.json"] = json.dumps({**self.findings, "errors": self.errors}, indent=2, default=str)
        for n, t in texts.items():
            (self.artifact_root / n).write_text(_scrub(t))
