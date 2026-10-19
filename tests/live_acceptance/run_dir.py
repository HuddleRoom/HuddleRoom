"""Run-directory builder and isolation guard for live acceptance runs.

Every live run writes into its own directory under an evidence root. The guard
refuses any database or workspace that could touch a developer's real data.
The supervisor calls assert_isolated_url() (for the DB URL) and assert_isolated()
(for the resolved paths) before every server start.
"""

from __future__ import annotations

import fnmatch
import os
import pwd
import re
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.engine import make_url

EVIDENCE_ROOT_ENV = "LIVE_ACCEPTANCE_EVIDENCE_ROOT"
_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_REPO_ROOT = Path(__file__).resolve().parents[2]  # HuddleRoom/ (contains pyproject.toml)
assert (_REPO_ROOT / "pyproject.toml").exists(), f"live_acceptance layout changed: no pyproject.toml in {_REPO_ROOT}"


class IsolationError(RuntimeError):
    """Raised when a live run would use a database or workspace outside its run dir."""


@dataclass(frozen=True)
class RunDir:
    root: Path
    db: Path
    workspace: Path
    snapshots: Path
    shim: Path
    proxy_log: Path
    spend: Path
    config: Path

    def server_log(self, n: int) -> Path:
        return self.root / f"server-{n}.log"


def real_home() -> Path:
    """Home directory from the passwd database; ignores $HOME, which tests override."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def _protected_dir() -> Path:
    return real_home() / ".huddleroom"


def _check_root_location(root: Path) -> None:
    if root.is_relative_to(_REPO_ROOT):
        raise IsolationError(f"run root {root} is inside the HuddleRoom repo {_REPO_ROOT}")
    if root.is_relative_to(_protected_dir()):
        raise IsolationError(f"run root {root} is inside {_protected_dir()}")


def _resolve_evidence_root(evidence_root: Path | None) -> Path:
    if evidence_root is None:
        env_value = os.environ.get(EVIDENCE_ROOT_ENV)
        if not env_value:
            raise ValueError(f"no evidence root: pass evidence_root or set {EVIDENCE_ROOT_ENV}")
        evidence_root = Path(env_value)
    path = Path(evidence_root).expanduser()
    if not path.is_absolute():
        raise ValueError(f"evidence root must be absolute: {path}")
    return path.resolve()


def build_run_dir(evidence_root: Path | None, label: str) -> RunDir:
    if not _LABEL_RE.fullmatch(label) or ".." in label:
        raise ValueError(f"invalid run label: {label!r}")
    root = _resolve_evidence_root(evidence_root)
    _check_root_location(root)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = root / f"run-{stamp}-{label}-{secrets.token_hex(3)}"
    _check_root_location(run_root)
    run_root.mkdir(parents=True, exist_ok=False)

    workspace = run_root / "workspace"
    snapshots = run_root / "snapshots"
    shim = run_root / "shim"
    for directory in (workspace, snapshots, shim):
        directory.mkdir()

    return RunDir(
        root=run_root,
        db=run_root / "app.db",
        workspace=workspace,
        snapshots=snapshots,
        shim=shim,
        proxy_log=run_root / "proxy.jsonl",
        spend=run_root / "spend.jsonl",
        config=run_root / "config.json",
    )


DENY_PATTERNS_ENV = "LIVE_ACCEPTANCE_DENY_PATH_PATTERNS"


def _deny_patterns() -> list[str]:
    """Comma-separated, case-insensitive path substrings that must never be used."""
    return [p.strip().lower() for p in os.environ.get(DENY_PATTERNS_ENV, "").split(",") if p.strip()]


def _check_path(raw: Path, root: Path, what: str) -> Path:
    if not raw.is_absolute():
        raise IsolationError(f"{what} path must be absolute: {raw}")
    if raw.is_symlink():
        raise IsolationError(f"{what} path is a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_relative_to(root):
        raise IsolationError(f"{what} {resolved} is outside run dir {root}")
    if resolved.is_relative_to(_protected_dir()):
        raise IsolationError(f"{what} {resolved} is inside {_protected_dir()}")
    for pattern in _deny_patterns():
        if pattern in str(resolved).lower():
            raise IsolationError(f"{what} path {resolved} contains denied pattern {pattern!r}")
    return resolved


def assert_isolated(run_dir: RunDir, db_path: Path, workspace: Path) -> None:
    if run_dir.root.is_symlink():
        raise IsolationError(f"run root is a symlink: {run_dir.root}")
    root = run_dir.root.resolve()
    _check_root_location(root)

    db = _check_path(Path(db_path), root, "database")
    if fnmatch.fnmatchcase(db.name.lower(), "rally*.db*"):
        raise IsolationError(f"database name {db.name} is not allowed")
    _check_path(Path(workspace), root, "workspace")


def assert_isolated_url(run_dir: RunDir, url: str) -> None:
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite":
        raise IsolationError(f"live runs only use sqlite, got {parsed.get_backend_name()!r}")
    database = parsed.database
    if not database or database == ":memory:":
        raise IsolationError("sqlite URL must name a file database")
    uri = parsed.query.get("uri")
    if uri is not None and str(uri).lower() not in ("", "0", "false", "no", "off"):
        raise IsolationError("sqlite URI mode (uri=true) is not allowed")
    assert_isolated(run_dir, Path(database), run_dir.workspace)


def _git(repo: Path, *args: str) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env |= {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=env, capture_output=True)


def init_workspace_repo(path: Path, run_dir: RunDir | None = None) -> None:
    path = Path(path)
    if run_dir is not None:
        if path.is_symlink() or not path.absolute().resolve().is_relative_to(run_dir.root.resolve()):
            raise IsolationError(f"workspace {path} is not inside run dir {run_dir.root}")
    path.mkdir(parents=True, exist_ok=False)
    (path / "README.md").write_text("# live acceptance workspace\n", encoding="utf-8")
    _git(path, "-c", "init.defaultBranch=main", "init", "-q")
    _git(path, "add", "README.md")
    _git(
        path,
        "-c", "user.name=live-acceptance",
        "-c", "user.email=live-acceptance@example.invalid",
        "-c", "commit.gpgsign=false",
        "commit", "-q", "-m", "init",
    )
