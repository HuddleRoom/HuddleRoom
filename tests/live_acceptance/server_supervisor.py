"""Start, kill and restart a HuddleRoom server confined to one run directory."""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from huddleroom.services.secret_redaction import redact_secrets
from tests.live_acceptance import run_dir as _run_dir
from tests.live_acceptance.run_dir import _REPO_ROOT, RunDir, assert_isolated, assert_isolated_url

_URL_CREDS = re.compile(r"([a-z][a-z0-9+.-]*://)[^/@\s]+@", re.I)
# CLI auth entries shared with the real login by design; add more only explicitly.
_SHARED_HOME_ENTRIES = (".claude", ".claude.json", ".codex")
_ALEMBIC_CODE = (
    "import sys; from alembic import command; from alembic.config import Config; "
    "c = Config(sys.argv[1]); c.set_main_option('script_location', sys.argv[2]); command.upgrade(c, 'head')"
)
_PASS_ENV = ("PATH", "USER", "LANG", "LC_ALL", "TMPDIR", "SHELL", "TERM", "SSL_CERT_FILE")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _redact(text: str) -> str:
    return _URL_CREDS.sub(r"\1[REDACTED]@", redact_secrets(text))


class Supervisor:
    """Runs one server per start() inside a per-run HOME (run_dir.root/home).

    Only the CLI auth entries in _SHARED_HOME_ENTRIES are symlinked from the real
    home (shared with the real login by design); everything else, including
    ~/.huddleroom, is invisible to the server.
    """

    def __init__(self, run_dir: RunDir, port: int, env_overrides: dict | None = None, shim_dir: Path | None = None,
                 extra_pid_dirs: list[Path] | None = None):
        # extra_pid_dirs: shim pid-file dirs (file name = pid) for CLI children that escape the server group
        self.extra_pid_dirs = [Path(d) for d in (extra_pid_dirs or [])]
        root = run_dir.root.resolve()
        for d in self.extra_pid_dirs:
            if not d.resolve().is_relative_to(root):
                raise ValueError(f"extra pid dir {d} is outside run dir {root}")
        self.run_dir, self.port = run_dir, port
        self.env_overrides = dict(env_overrides or {})
        self.shim_dir = shim_dir
        self.incarnation = 0
        self._migrated = False
        self._proc: subprocess.Popen | None = None
        self._pgid: int | None = None
        self.last_pgid: int | None = None
        self._log_file = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def log_path(self) -> Path:
        return self.run_dir.server_log(self.incarnation)

    def _prepare_home(self) -> tuple[Path, dict[str, str]]:
        home = self.run_dir.root / "home"
        home.mkdir(exist_ok=True)
        real = _run_dir.real_home()
        links = []
        for name in _SHARED_HOME_ENTRIES:
            src, dst = real / name, home / name
            if src.exists() and not dst.is_symlink() and not dst.exists():
                dst.symlink_to(src)
            if dst.is_symlink():
                links.append(f"{dst} -> {src}")
        extra = {}
        if (home / ".codex").is_symlink():
            extra["CODEX_HOME"] = str(home / ".codex")
        if (home / ".claude").is_symlink():
            extra["CLAUDE_CONFIG_DIR"] = str(home / ".claude")
        self.run_dir.config.write_text(json.dumps({"home": str(home), "shared_home_links": links}, indent=2))
        return home, extra

    def _env(self, db_url: str) -> dict[str, str]:
        env = {k: os.environ[k] for k in _PASS_ENV if k in os.environ}
        home, extra = self._prepare_home()
        env.update(extra)
        env.update(
            HOME=str(home),
            HUDDLEROOM_DATABASE_URL=db_url,
            HUDDLEROOM_WORKSPACE_DIR=str(self.run_dir.workspace),
            PYTHONPATH=str(_REPO_ROOT),
        )
        if self.shim_dir is not None:
            env["PATH"] = f"{self.shim_dir}{os.pathsep}{env.get('PATH', '')}"
        env.update(self.env_overrides)
        return env

    def start(self) -> None:
        if self._proc is not None:
            self.kill()  # full group cleanup; never poll() a possibly-zombie leader
            if self._leader_running():
                raise RuntimeError("server already running")
        db_url = f"sqlite+aiosqlite:///{self.run_dir.db}"
        assert_isolated_url(self.run_dir, db_url)
        assert_isolated(self.run_dir, self.run_dir.db, self.run_dir.workspace)
        env = self._env(db_url)
        if not self._migrated:
            try:
                subprocess.run(
                    [sys.executable, "-c", _ALEMBIC_CODE, str(_REPO_ROOT / "alembic.ini"), str(_REPO_ROOT / "alembic")],
                    cwd=self.run_dir.root, env=env, check=True, capture_output=True, text=True,
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"alembic upgrade failed:\n{_redact(error.stderr or '')[-3000:]}") from None
            self._migrated = True
        self.incarnation += 1
        self._close_log()
        self._log_file = open(self.log_path, "wb")
        # cwd=run root so a developer .env in the repo is never read
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "huddleroom.cli", "serve", "--host", "127.0.0.1", "--port", str(self.port)],
                cwd=self.run_dir.root, env=env, stdin=subprocess.DEVNULL, stdout=self._log_file, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self._close_log()
            raise
        self._pgid = self.last_pgid = self._proc.pid
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if not self._leader_running():
                    raise RuntimeError(f"server exited before healthy "
                                       f"(incarnation {self.incarnation}):\n{self.redacted_tail()}")
                try:
                    if httpx.get(f"{self.base_url}/health", timeout=0.5).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            raise RuntimeError(f"server not healthy in 30s (incarnation {self.incarnation}):\n{self.redacted_tail()}")
        except BaseException:
            self.kill(signal.SIGKILL)
            raise

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    # os.waitid(WNOWAIT) does not exist on macOS, so group state comes from `ps`.
    # The leader is never reaped (poll/wait) until the group is empty, so a zombie
    # leader keeps pinning the pgid and the id cannot be recycled under us.
    def _ps(self) -> list[tuple[int, int, str]]:
        try:
            res = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True)
        except OSError as error:
            raise RuntimeError(f"cannot run ps: {error}") from None
        out = res.stdout
        if res.returncode != 0 or not out.strip():
            raise RuntimeError(f"ps failed (rc={res.returncode}): {res.stderr.strip()}")
        rows = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                rows.append((int(parts[0]), int(parts[1]), parts[2]))
        return rows

    def _leader_running(self) -> bool:
        pid = self._pgid
        if pid is None:
            return False
        return any(p == pid and g == pid and not st.startswith("Z") for p, g, st in self._ps())

    def _members(self) -> list[int]:
        if self._pgid is None:
            return []
        return [p for p, g, _ in self._ps() if g == self._pgid and p != self._pgid]

    def _killpg(self, sig: int) -> None:
        if self._pgid is None:
            return
        try:
            os.killpg(self._pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _port_open(self) -> bool:
        with socket.socket() as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def kill(self, sig: int = signal.SIGTERM) -> None:
        if self._proc is not None:
            self._killpg(sig)
            deadline = time.monotonic() + (10 if sig == signal.SIGTERM else 5)
            while self._leader_running() and time.monotonic() < deadline:
                time.sleep(0.05)
            # leader is dead (zombie, unreaped) or out of time: SIGKILL everything left
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                self._killpg(signal.SIGKILL)
                if not self._members():
                    break
                time.sleep(0.05)
            self._proc.wait(timeout=5)  # reap the leader only now
            self._pgid = None  # reaped: the id may be recycled, never signal it again
            deadline = time.monotonic() + 5
            while self._port_open() and time.monotonic() < deadline:
                time.sleep(0.1)
        self._close_log()
        self._proc = None
        self._sweep_pid_dirs()

    def _sweep_pid_dirs(self) -> None:
        cmds = {}
        markers = {str(self.run_dir.shim)}
        for d in self.extra_pid_dirs:
            if d.parent != self.run_dir.root:
                markers.add(str(d.parent))
        for d in self.extra_pid_dirs:
            for f in d.glob("*") if d.is_dir() else ():
                if f.name.isdigit():
                    cmds[int(f.name)] = None
        if not cmds:
            return
        rows = {p: st for p, _, st in self._ps()}
        targets = []
        for pid in cmds:
            if pid not in rows or rows[pid].startswith("Z"):
                continue
            out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
            if any(m in out for m in markers):  # run's own shim path; guards against recycled pids
                targets.append(pid)
        for sig, wait in ((signal.SIGTERM, 3), (signal.SIGKILL, 3)):
            for pid in targets:
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                rows = {p: st for p, _, st in self._ps()}
                targets = [p for p in targets if p in rows and not rows[p].startswith("Z")]
                if not targets:
                    return
                time.sleep(0.1)

    stop = kill

    def restart(self, sig: int = signal.SIGTERM) -> None:
        self.kill(sig)
        self.start()

    def redacted_tail(self, n: int = 50) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()[-n:]
        except FileNotFoundError:
            return ""
        return _redact("\n".join(lines))

    def __enter__(self) -> "Supervisor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.kill()
