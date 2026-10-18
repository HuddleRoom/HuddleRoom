import signal
import sqlite3

import pytest

from tests.live_acceptance.run_dir import IsolationError, build_run_dir
from tests.live_acceptance.server_supervisor import Supervisor, free_port

pytestmark = pytest.mark.live


@pytest.fixture
def run_dir(tmp_path):
    return build_run_dir(tmp_path / "evidence", "supervisor")


@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM])
def test_kill_and_restart(run_dir, sig):
    import httpx

    with Supervisor(run_dir, free_port()) as sup:
        assert httpx.get(f"{sup.base_url}/health").status_code == 200
        sup.kill(sig)
        with pytest.raises(httpx.HTTPError):
            httpx.get(f"{sup.base_url}/health", timeout=1)
        sup.start()
        assert httpx.get(f"{sup.base_url}/health").status_code == 200
        assert sup.incarnation == 2
    assert run_dir.server_log(1).exists() and run_dir.server_log(2).exists()
    with sqlite3.connect(run_dir.db) as conn:
        assert conn.execute("select count(*) from alembic_version").fetchone()[0] == 1
    sup.kill()  # idempotent


def test_guard_refuses_symlinked_db(run_dir, tmp_path):
    real = tmp_path / "elsewhere.db"
    real.touch()
    run_dir.db.symlink_to(real)
    sup = Supervisor(run_dir, free_port())
    with pytest.raises(IsolationError):
        sup.start()
    assert sup._proc is None


def test_poisoned_real_home_is_not_used(run_dir, tmp_path, monkeypatch):
    import httpx
    from tests.live_acceptance import run_dir as rd

    fake = tmp_path / "fakehome"
    (fake / ".huddleroom").mkdir(parents=True)
    bad_db, bad_log = tmp_path / "poison.db", tmp_path / "poison.log"
    (fake / ".huddleroom" / "config.toml").write_text(
        f'database_url = "sqlite+aiosqlite:///{bad_db}"\nlog_file = "{bad_log}"\nopenai_api_key = "sk-bogus"\n'
    )
    (fake / ".codex").mkdir()
    monkeypatch.setattr(rd, "real_home", lambda: fake)
    with Supervisor(run_dir, free_port()) as sup:
        assert httpx.get(f"{sup.base_url}/health").status_code == 200
        home = run_dir.root / "home"
        assert not (home / ".huddleroom").exists()
        assert (home / ".codex").is_symlink()
    assert not bad_db.exists() and not bad_log.exists()
    with sqlite3.connect(run_dir.db) as conn:
        assert conn.execute("select count(*) from alembic_version").fetchone()[0] == 1
    assert ".codex" in run_dir.config.read_text()


def test_non_sqlite_url_refused_before_launch(run_dir, monkeypatch):
    from tests.live_acceptance import server_supervisor as ss

    monkeypatch.setattr(ss.Supervisor, "_env", lambda self, url: pytest.fail("launched"))
    sup = Supervisor(run_dir, free_port())
    monkeypatch.setattr(type(run_dir), "db", run_dir.root / "x.db", raising=False)
    object.__setattr__(run_dir, "db", run_dir.root.parent / "outside.db")
    with pytest.raises(IsolationError):
        sup.start()


def test_unhealthy_server_raises_and_leaves_no_process(run_dir):
    import os

    import socket

    with socket.socket() as busy:  # port already taken -> server exits before healthy
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        sup = Supervisor(run_dir, busy.getsockname()[1])
        with pytest.raises(RuntimeError):
            sup.start()
    assert sup.last_pgid > 0
    assert sup._proc is None
    pgid = sup.last_pgid
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


def test_alembic_failure_raises_redacted(run_dir):
    sup = Supervisor(run_dir, free_port(), env_overrides={"HUDDLEROOM_DATABASE_URL": "sqlite+aiosqlite:////nonexistent-dir/x/a.db"})
    with pytest.raises(RuntimeError, match="alembic upgrade failed"):
        sup.start()
    assert sup._proc is None


def test_sigterm_leaves_group_gone(run_dir):
    import os

    with Supervisor(run_dir, free_port()) as sup:
        pgid = sup._pgid
        sup.kill(signal.SIGTERM)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


def _fake_leader(run_dir, port, crash):
    """A supervisor whose 'server' is a script forking a TERM-trapping child."""
    import subprocess
    import sys

    sup = Supervisor(run_dir, port)
    script = "sh -c 'trap \"\" TERM; echo $$ > child.pid; sleep 60' & " + ("exit 0" if crash else "trap '' TERM; wait")
    sup._proc = subprocess.Popen(["sh", "-c", script], cwd=run_dir.root, start_new_session=True)
    sup._pgid = sup._proc.pid
    import time

    for _ in range(100):
        if (run_dir.root / "child.pid").exists() and (run_dir.root / "child.pid").read_text().strip():
            break
        time.sleep(0.05)
    return sup, int((run_dir.root / "child.pid").read_text())


@pytest.mark.parametrize("crash", [False, True])
def test_kill_takes_down_term_trapping_child(run_dir, crash):
    import os
    import time

    sup, child = _fake_leader(run_dir, free_port(), crash)
    if crash:
        time.sleep(0.5)  # leader exited on its own, child survives
    os.kill(child, 0)
    sup.kill(signal.SIGTERM)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
    assert sup._members() == []


def test_start_cleans_old_group_after_crash(run_dir):
    import os
    import time

    import httpx

    sup, child = _fake_leader(run_dir, free_port(), crash=True)
    time.sleep(0.5)
    os.kill(child, 0)
    try:
        sup.start()
        with pytest.raises(ProcessLookupError):
            os.kill(child, 0)
        assert httpx.get(f"{sup.base_url}/health").status_code == 200
    finally:
        sup.kill()


def test_server_stdin_is_devnull(run_dir, monkeypatch):
    import subprocess

    from tests.live_acceptance import server_supervisor as ss

    seen = {}
    real = subprocess.Popen

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(ss.subprocess, "Popen", spy)
    with Supervisor(run_dir, free_port()):
        pass
    assert seen["stdin"] == subprocess.DEVNULL


def test_kill_sweeps_shim_pid_dirs(run_dir):
    import subprocess
    import sys

    pid_dir = run_dir.shim / "pids"
    pid_dir.mkdir()
    shim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(run_dir.shim / "cli_shim_runner")],
                            start_new_session=True)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "/elsewhere/cli_shim_runner"],
                             start_new_session=True)
    try:
        (pid_dir / str(shim.pid)).touch()
        (pid_dir / str(other.pid)).touch()
        Supervisor(run_dir, free_port(), extra_pid_dirs=[pid_dir]).kill()
        assert shim.wait(timeout=5) == -signal.SIGTERM
        assert other.poll() is None  # same runner name, different path: left alone
    finally:
        other.kill()
        other.wait()
        shim.kill()


def test_pid_dir_outside_run_root_rejected(run_dir, tmp_path):
    with pytest.raises(ValueError):
        Supervisor(run_dir, free_port(), extra_pid_dirs=[tmp_path])
