import json
import math
import os
import pty
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.live_acceptance.cli_shim import GATE_TIMEOUT_EXIT, make_shim, resolve_real
from tests.live_acceptance.spend_ledger import Ledger

BUDGET_EXIT = 75

REAL_SCRIPT = """#!/bin/sh
echo "argc=$#"
for a in "$@"; do echo "arg=[$a]"; done
echo "stdin=[$(cat)]"
exit 3
"""

# Fake model CLI: behaviour is driven by argv flags so one script serves codex and claude.
FAKE_CLI = """#!{py}
import os
import sys
import time
argv = sys.argv[1:]

def opt(name):
    i = argv.index(name) if name in argv else -1
    return argv[i + 1] if 0 <= i and i + 1 < len(argv) else None

if "--version" in argv:
    sys.stdout.write("fake 0.0\\n")
    sys.exit(0)
if argv[:2] == ["login", "status"]:
    sys.stderr.write("Logged in using ChatGPT\\n")
    sys.exit(0)
if opt("--pidfile"):
    open(opt("--pidfile"), "w").write(str(os.getpid()))
data = sys.stdin.buffer.read()
if opt("--sleep"):
    time.sleep(float(opt("--sleep")))
if opt("--flood"):
    for _ in range(int(opt("--flood"))):
        sys.stdout.buffer.write(b"x" * 65536)
        sys.stdout.buffer.flush()
if opt("--stdin-out"):
    open(opt("--stdin-out"), "wb").write(data)
if opt("--mark"):
    open(opt("--mark"), "w").write("ran")
if opt("--emit-file"):
    sys.stdout.buffer.write(open(opt("--emit-file"), "rb").read())
if opt("--usage"):
    sys.stdout.write(opt("--usage") + "\\n")
if opt("--last"):
    open(opt("--last"), "w").write("LAST MESSAGE\\n")
if opt("--stderr"):
    sys.stderr.write(opt("--stderr"))
sys.stdout.flush()
sys.exit(int(opt("--exit") or 0))
"""

CODEX_TURN = json.dumps({"type": "turn.completed",
                         "usage": {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 7}})
CLAUDE_RESULT = json.dumps({"type": "result",
                            "usage": {"input_tokens": 50, "output_tokens": 5, "cache_read_input_tokens": 10}})


def _write_exec(d: Path, name: str, body: str) -> Path:
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text(body)
    p.chmod(0o755)
    return p


def _write_real(d: Path, name: str) -> Path:
    return _write_exec(d, name, REAL_SCRIPT)


def _write_fake(d: Path, name: str) -> Path:
    return _write_exec(d, name, FAKE_CLI.format(py=sys.executable))


def _write_fake_env_shebang(d: Path, name: str) -> Path:
    body = FAKE_CLI.format(py=sys.executable).split("\n", 1)[1]
    return _write_exec(d, name, "#!/usr/bin/env python3\n" + body)


def _spawn(path, args, data=b"", **kw):
    p = subprocess.Popen([str(path), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, **kw)
    p.stdin.write(data)
    p.stdin.close()
    return p


def _finish(p, timeout=10):
    """Drain stdout/stderr and wait; stdin was already closed by _spawn."""
    out = p.stdout.read()
    err = p.stderr.read()
    p.wait(timeout)
    return out, err


def _reap(*procs):
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()


@pytest.fixture
def real_tool(tmp_path):
    return _write_real(tmp_path / "real", "claude")


@pytest.fixture
def real_codex(tmp_path):
    return _write_real(tmp_path / "real", "codex")


@pytest.fixture
def fake_claude(tmp_path):
    return _write_fake(tmp_path / "fake", "claude")


@pytest.fixture
def fake_codex(tmp_path):
    return _write_fake(tmp_path / "fake", "codex")


@pytest.fixture
def shim_dir(tmp_path):
    d = tmp_path / "shim"
    d.mkdir()
    return d


@pytest.fixture
def shim(shim_dir, real_tool):
    h = make_shim(shim_dir, "claude", real_tool)
    yield h
    h.close()


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "spend.jsonl"


# --- basic pass-through and gating (sh fake, no metering) -------------------------------

@pytest.mark.live
def test_args_stdin_and_exit_code_pass_through(shim_dir, real_tool):
    h = make_shim(shim_dir, "claude", real_tool)
    r = subprocess.run(
        [str(h.path), "one", "two words", "$HOME", "x\\y"],
        input=b"hello\n",
        capture_output=True,
        timeout=30,
    )
    out = r.stdout.decode()
    assert r.returncode == 3
    assert "argc=4" in out
    assert "arg=[two words]" in out
    assert "arg=[$HOME]" in out
    assert "arg=[x\\y]" in out
    assert "stdin=[hello]" in out


@pytest.mark.live
def test_gate_blocks_until_release(shim):
    shim.hold()
    p = _spawn(shim.path, ["go"])
    try:
        assert shim.wait_started(10)
        time.sleep(0.6)
        assert p.poll() is None, "shim must block while gate is held"
        shim.release()
        out, _ = _finish(p)
    finally:
        _reap(p)
    assert p.returncode == 3
    assert b"arg=[go]" in out


def test_hold_each_releases_only_the_selected_call(shim_dir, real_tool):
    """A later call stays held when the first private ordinal gate is released."""
    shim = make_shim(shim_dir, "claude", real_tool, hold_each=True)
    first = _spawn(shim.path, ["first"])
    second = None
    try:
        assert shim.wait_invocations(1, timeout=10)
        second = _spawn(shim.path, ["second"])
        assert shim.wait_invocations(2, timeout=10)
        shim.release_invocation(1)
        _finish(first)
        time.sleep(0.3)
        assert second.poll() is None
        shim.release_invocation(2)
        _finish(second)
    finally:
        _reap(first, *([second] if second is not None else []))


def test_model_call_stdin_and_stdout_copied_to_calls_dir(shim_dir, fake_codex, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex)
    stdin = b"prompt \xff\n" * 3
    out = b"reply \x00\xfe\n"
    emit = tmp_path / "emit.bin"
    emit.write_bytes(out)
    r = subprocess.run([str(h.path), "exec", "--emit-file", str(emit)], input=stdin,
                       capture_output=True, timeout=10)
    assert r.returncode == 0
    assert r.stdout == out
    assert (shim_dir / "calls" / "codex.1.stdin").read_bytes() == stdin
    assert (shim_dir / "calls" / "codex.1.stdout").read_bytes() == out


@pytest.mark.live
def test_log_has_start_and_exec(shim):
    subprocess.run([str(shim.path), "a", "b c"], input=b"", capture_output=True, timeout=30)
    invs = shim.invocations()
    assert len(invs) == 1
    assert invs[0].args == "a b c"
    assert invs[0].ordinal == 1
    assert invs[0].exec_at is not None
    assert invs[0].started_at <= invs[0].exec_at
    assert not invs[0].timed_out and not invs[0].refused
    log = (shim.shim_dir / "shim.log").read_text()
    assert "start claude " in log and "exec claude " in log


@pytest.mark.live
def test_malformed_and_partial_log_lines_are_skipped(shim):
    with open(shim.shim_dir / "shim.log", "a") as f:
        f.write("garbage\n")
        f.write("start claude notanumber 1 1 x\n")
        f.write("start claude 100 4242 1 ok\n")
        f.write("start claude 10")  # half-written, no newline
    invs = shim.invocations()
    assert [(i.pid, i.args, i.ordinal) for i in invs] == [(4242, "ok", 1)]


@pytest.mark.live
def test_resolve_real_skips_shim_dir(shim_dir, real_tool, monkeypatch):
    make_shim(shim_dir, "claude", real_tool)
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{real_tool.parent}{os.pathsep}{os.environ['PATH']}")
    assert resolve_real("claude", shim_dir) == real_tool.absolute()


@pytest.mark.live
def test_resolve_real_skips_symlinked_shim_dir(tmp_path, shim_dir, real_tool, monkeypatch):
    make_shim(shim_dir, "claude", real_tool)
    link = tmp_path / "shim_link"
    link.symlink_to(shim_dir, target_is_directory=True)
    monkeypatch.setenv("PATH", f"{link}{os.pathsep}{real_tool.parent}{os.pathsep}{os.environ['PATH']}")
    assert resolve_real("claude", shim_dir) == real_tool.absolute()


@pytest.mark.live
def test_resolve_real_raises_when_result_is_inside_shim_dir(tmp_path, shim_dir, real_tool, monkeypatch):
    make_shim(shim_dir, "claude", real_tool)
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "claude").symlink_to(shim_dir / "claude")
    monkeypatch.setenv("PATH", f"{bad_dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(RuntimeError):
        resolve_real("claude", shim_dir)


@pytest.mark.live
def test_path_prefixed_which_resolves_shim(shim, monkeypatch):
    monkeypatch.setenv("PATH", f"{shim.shim_dir}{os.pathsep}{os.environ['PATH']}")
    assert shutil.which("claude") == str(shim.path)


@pytest.mark.live
def test_hold_codex_does_not_block_claude(shim_dir, real_tool, real_codex):
    claude = make_shim(shim_dir, "claude", real_tool)
    make_shim(shim_dir, "codex", real_codex)
    try:
        claude.hold("codex")
        r = subprocess.run([str(claude.path), "x"], input=b"", capture_output=True, timeout=5)
        assert r.returncode == 3
    finally:
        claude.close()


@pytest.mark.live
def test_hold_all_blocks_both_tools(shim_dir, real_tool, real_codex):
    claude = make_shim(shim_dir, "claude", real_tool)
    codex = make_shim(shim_dir, "codex", real_codex)
    procs = []
    try:
        claude.hold("all")
        procs = [_spawn(h.path, []) for h in (claude, codex)]
        assert claude.wait_started(10) and codex.wait_started(10)
        time.sleep(0.6)
        assert all(p.poll() is None for p in procs)
        claude.release_all()
        for p in procs:
            _finish(p)
            assert p.returncode == 3
    finally:
        claude.close()
        _reap(*procs)


@pytest.mark.live
def test_sigterm_to_process_group_exits_promptly(shim):
    shim.hold()
    p = _spawn(shim.path, [], start_new_session=True)
    try:
        assert shim.wait_started(10)
        pgid = p.pid
        os.killpg(pgid, signal.SIGTERM)
        t0 = time.monotonic()
        p.wait(timeout=3)
        assert time.monotonic() - t0 <= 3
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("lingering processes in shim process group")
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()


@pytest.mark.live
def test_gate_timeout_exits_97_without_exec(shim_dir, real_tool):
    h = make_shim(shim_dir, "claude", real_tool, gate_timeout=1)
    try:
        h.hold()
        r = subprocess.run([str(h.path), "x"], input=b"", capture_output=True, timeout=10)
        assert r.returncode == GATE_TIMEOUT_EXIT
        assert b"arg=[x]" not in r.stdout
        invs = h.invocations()
        assert len(invs) == 1
        assert invs[0].timed_out
        assert invs[0].exec_at is None
    finally:
        h.close()


@pytest.mark.live
def test_release_all_unblocks(shim):
    shim.hold()
    p = _spawn(shim.path, [])
    try:
        assert shim.wait_started(10)
        time.sleep(0.4)
        assert p.poll() is None
        shim.release_all()
        _finish(p)
        assert p.returncode == 3
    finally:
        _reap(p)


# --- fake model CLI: ordinal/content gates, stdout/stdin fidelity, metering ---------------

@pytest.mark.live
def test_ordinal_gate_holds_only_second_call(shim_dir, fake_codex):
    h = make_shim(shim_dir, "codex", fake_codex)
    p = None
    try:
        assert subprocess.run([str(h.path), "exec"], input=b"", capture_output=True, timeout=10).returncode == 0
        token = h.hold_nth(2)
        p = _spawn(h.path, ["exec"])
        assert h.wait_invocations(2, 10)
        time.sleep(0.6)
        assert p.poll() is None
        invs = h.invocations()
        assert invs[0].gates == [] and invs[1].gates == [token]
        h.release(token)
        _finish(p)
        assert p.returncode == 0
    finally:
        _reap(*([p] if p else []))
        h.close()


@pytest.mark.live
def test_content_gate_on_schema_file_marker(shim_dir, fake_codex, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex)
    schema = tmp_path / "schema.json"
    schema.write_text('{"marker": "MARK_SCHEMA_7"}')
    other = tmp_path / "other.json"
    other.write_text("{}")
    h.hold_matching("MARK_SCHEMA_7")
    p = None
    try:
        assert subprocess.run([str(h.path), "exec", "--output-schema", str(other)],
                              input=b"", capture_output=True, timeout=10).returncode == 0
        p = _spawn(h.path, ["exec", "--output-schema", str(schema)])
        assert h.wait_invocations(2, 10)
        time.sleep(0.6)
        assert p.poll() is None
        assert h.invocations()[1].gates
        h.release_all()
        _finish(p)
        assert p.returncode == 0
    finally:
        _reap(*([p] if p else []))
        h.close()


@pytest.mark.live
def test_content_gate_on_stdin_marker(shim_dir, fake_codex):
    h = make_shim(shim_dir, "codex", fake_codex)
    h.hold_matching("MARK_STDIN_9")
    p = None
    try:
        assert subprocess.run([str(h.path), "exec"], input=b"plain prompt",
                              capture_output=True, timeout=10).returncode == 0
        p = _spawn(h.path, ["exec"], data=b"head\nMARK_STDIN_9\n")
        assert h.wait_invocations(2, 10)
        time.sleep(0.6)
        assert p.poll() is None
        h.release_all()
        _finish(p)
        assert p.returncode == 0
    finally:
        _reap(*([p] if p else []))
        h.close()


@pytest.mark.live
def test_stdout_and_stdin_are_byte_identical(shim_dir, fake_codex, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex)
    payload = bytes(range(256)) * 4000 + "héllo ✓ 世界\n".encode() * 1000 + b"\xff\xfe\x00tail"
    emit = tmp_path / "emit.bin"
    emit.write_bytes(payload)
    stdin_out = tmp_path / "stdin.bin"
    stdin = bytes(range(256)) * 100 + b"\xff\xfe\n"
    r = subprocess.run([str(h.path), "exec", "--emit-file", str(emit), "--stdin-out", str(stdin_out)],
                       input=stdin, capture_output=True, timeout=60)
    assert r.returncode == 0
    assert r.stdout == payload
    assert stdin_out.read_bytes() == stdin


@pytest.mark.live
def test_stderr_and_output_last_message_file_untouched(shim_dir, fake_codex, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex)
    last = tmp_path / "last.txt"
    r = subprocess.run([str(h.path), "exec", "--last", str(last), "--stderr", "E-ERR"],
                       input=b"", capture_output=True, timeout=10)
    assert r.returncode == 0
    assert r.stderr == b"E-ERR"
    assert last.read_text() == "LAST MESSAGE\n"


@pytest.mark.live
def test_usage_settled_exactly(shim_dir, fake_codex, fake_claude, ledger_path):
    codex = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1", global_cap=10**6)
    claude = make_shim(shim_dir, "claude", fake_claude, ledger_path=ledger_path, scenario="s1", global_cap=10**6)
    assert subprocess.run([str(codex.path), "exec", "--usage", CODEX_TURN],
                          input=b"p", capture_output=True, timeout=10).returncode == 0
    assert subprocess.run([str(claude.path), "--print", "--usage", CLAUDE_RESULT],
                          input=b"p", capture_output=True, timeout=10).returncode == 0
    totals = Ledger(ledger_path, 10**6).totals()
    # codex: 100 - 40 + 7 = 67 (cached 40); claude: 50 - 10 + 5 = 45 (cached 10)
    assert totals["s1"] == {"non_cached": 112, "cached": 50}


@pytest.mark.live
def test_missing_usage_settles_estimate(shim_dir, fake_codex, ledger_path):
    h = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1",
                  global_cap=10**6, output_ceiling=100)
    r = subprocess.run([str(h.path), "exec"], input=b"x" * 300, capture_output=True, timeout=10)
    assert r.returncode == 0
    expected = math.ceil((300 + len("exec")) / 3) + 100
    assert Ledger(ledger_path, 10**6).totals()["s1"] == {"non_cached": expected, "cached": 0}


@pytest.mark.live
def test_cap_refusal_exits_75_and_never_runs_cli(shim_dir, fake_codex, ledger_path, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1",
                  global_cap=50, output_ceiling=100)
    mark = tmp_path / "ran.mark"
    r = subprocess.run([str(h.path), "exec", "--mark", str(mark)], input=b"p",
                       capture_output=True, timeout=10)
    assert r.returncode == BUDGET_EXIT
    assert b"budget refused:" in r.stderr
    assert not mark.exists(), "real CLI must not run when refused"
    assert h.invocations()[0].refused
    assert Ledger(ledger_path, 50).totals()["global"]["non_cached"] == 0


@pytest.mark.live
def test_login_status_and_version_pass_through_uncounted(shim_dir, fake_codex, ledger_path):
    h = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1",
                  global_cap=10**6)
    r = subprocess.run([str(h.path), "login", "status"], input=b"", capture_output=True, timeout=10)
    assert r.returncode == 0
    assert r.stderr == b"Logged in using ChatGPT\n"
    assert r.stdout == b""
    v = subprocess.run([str(h.path), "--version"], input=b"", capture_output=True, timeout=10)
    assert v.stdout == b"fake 0.0\n"
    assert h.invocations() == []
    assert not ledger_path.exists() or '"reserve"' not in ledger_path.read_text()


@pytest.mark.live
def test_two_tools_have_separate_gates_and_counters(shim_dir, fake_claude, fake_codex):
    claude = make_shim(shim_dir, "claude", fake_claude)
    codex = make_shim(shim_dir, "codex", fake_codex)
    p = None
    try:
        claude.hold_nth(1)
        p = _spawn(claude.path, ["--print"])
        assert claude.wait_started(10)
        r = subprocess.run([str(codex.path), "exec"], input=b"", capture_output=True, timeout=10)
        assert r.returncode == 0
        assert codex.invocations()[0].ordinal == 1
        assert p.poll() is None
        claude.release_all()
        _finish(p)
        assert p.returncode == 0
        assert claude.invocations()[0].ordinal == 1
    finally:
        _reap(*([p] if p else []))
        claude.close()
        codex.close()


def _wait_file(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not (path.exists() and path.read_text().strip()):
        assert time.monotonic() < deadline, f"{path} never written"
        time.sleep(0.05)


def _assert_gone(pid: int, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail(f"process {pid} still alive")


@pytest.mark.live
def test_closed_stdin_does_not_block(shim_dir, fake_codex):
    h = make_shim(shim_dir, "codex", fake_codex, stdin_timeout=30)
    p = subprocess.Popen([str(h.path), "exec"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, preexec_fn=lambda: os.close(0))
    out, _ = p.communicate(input=b"", timeout=10)
    assert p.returncode == 0
    assert h.invocations()[0].ordinal == 1


@pytest.mark.live
def test_tty_stdin_does_not_block(shim_dir, fake_codex):
    h = make_shim(shim_dir, "codex", fake_codex, stdin_timeout=30)
    master, slave = pty.openpty()
    try:
        p = subprocess.Popen([str(h.path), "exec"], stdin=slave, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        os.close(slave)
        slave = None
        out, _ = p.communicate(timeout=10)
        assert p.returncode == 0
        assert h.invocations()[0].ordinal == 1
    finally:
        if slave is not None:
            os.close(slave)
        os.close(master)


@pytest.mark.live
def test_open_stdin_pipe_times_out_and_still_forwards(shim_dir, fake_codex, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex, stdin_timeout=1.0)
    stdin_out = tmp_path / "stdin.bin"
    p = subprocess.Popen([str(h.path), "exec", "--stdin-out", str(stdin_out)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        p.stdin.write(b"partial")
        p.stdin.flush()
        assert h.wait_invocations(1, 10)
        deadline = time.monotonic() + 10
        while h.invocations()[0].exec_at is None:
            assert time.monotonic() < deadline, "call never proceeded after stdin timeout"
            time.sleep(0.05)
        assert "stdin_timeout codex" in (h.shim_dir / "shim.log").read_text()
        p.stdin.write(b"rest")
        p.stdin.close()
        _finish(p)
        assert p.returncode == 0
        assert stdin_out.read_bytes() == b"partialrest"
    finally:
        _reap(p)


@pytest.mark.live
def test_sigterm_to_runner_kills_child(shim_dir, fake_codex, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex)
    pidfile = tmp_path / "child.pid"
    p = _spawn(h.path, ["exec", "--pidfile", str(pidfile), "--sleep", "30"])
    try:
        _wait_file(pidfile)
        child = int(pidfile.read_text())
        os.kill(p.pid, signal.SIGTERM)
        p.wait(timeout=10)
        assert p.returncode == 128 + signal.SIGTERM
        _assert_gone(child)
    finally:
        _reap(p)


@pytest.mark.live
def test_stdout_reader_closes_early_terminates_child(shim_dir, fake_codex, ledger_path, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1",
                  global_cap=10**6, output_ceiling=100)
    pidfile = tmp_path / "child.pid"
    data = b"x" * 30
    p = _spawn(h.path, ["exec", "--pidfile", str(pidfile), "--flood", "100000"], data=data)
    try:
        _wait_file(pidfile)
        child = int(pidfile.read_text())
        assert p.stdout.read(1024)
        p.stdout.close()
        p.wait(timeout=20)
        err = p.stderr.read()
        assert p.returncode == 141
        assert b"Traceback" not in err
        _assert_gone(child)
        expected = math.ceil((len(data) + len("exec") + len("--pidfile") + len(str(pidfile))
                              + len("--flood") + len("100000")) / 3) + 100
        assert Ledger(ledger_path, 10**6).totals()["s1"] == {"non_cached": expected, "cached": 0}
    finally:
        _reap(p)


def _pid_files(shim_dir: Path) -> list:
    return sorted((shim_dir / "pids").iterdir())


@pytest.mark.live
def test_deny_exits_held_shim_76_without_running_cli(shim_dir, fake_codex, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex)
    mark = tmp_path / "ran.mark"
    h.hold()
    p = _spawn(h.path, ["exec", "--mark", str(mark)])
    try:
        assert h.wait_started(10)
        time.sleep(0.4)
        assert p.poll() is None
        t0 = time.monotonic()
        h.deny()
        p.wait(timeout=2)
        assert time.monotonic() - t0 <= 1.5
        assert p.returncode == 76
        assert not mark.exists(), "CLI must not run when denied"
        assert h.live_pids() == []
        assert _pid_files(shim_dir) == []
    finally:
        _reap(p)
        h.close()
        h.undeny()


@pytest.mark.live
def test_deny_before_start_exits_76_without_reserve_or_exec(shim_dir, fake_codex, ledger_path, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1", global_cap=10**6)
    mark = tmp_path / "ran.mark"
    h.deny()
    try:
        r = subprocess.run([str(h.path), "exec", "--mark", str(mark)], input=b"p",
                           capture_output=True, timeout=10)
        assert r.returncode == 76
        assert not mark.exists()
        assert not ledger_path.exists()
        assert _pid_files(shim_dir) == []
    finally:
        h.undeny()


@pytest.mark.live
def test_kill_live_stops_shim_outside_group_and_its_child(shim_dir, fake_codex, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex)
    pidfile = tmp_path / "child.pid"
    p = _spawn(h.path, ["exec", "--pidfile", str(pidfile), "--sleep", "30"], start_new_session=True)
    try:
        _wait_file(pidfile)
        child = int(pidfile.read_text())
        assert h.live_pids() == [p.pid]
        h.kill_live()
        p.wait(timeout=10)
        assert p.returncode == 128 + signal.SIGTERM
        _assert_gone(child)
        assert h.live_pids() == []
        assert _pid_files(shim_dir) == []
    finally:
        _reap(p)


@pytest.mark.live
def test_kill_live_sigkill_also_kills_recorded_child(shim_dir, fake_codex, tmp_path):
    h = make_shim(shim_dir, "codex", fake_codex)
    pidfile = tmp_path / "child.pid"
    p = _spawn(h.path, ["exec", "--pidfile", str(pidfile), "--sleep", "30"], start_new_session=True)
    try:
        _wait_file(pidfile)
        child = int(pidfile.read_text())
        h.kill_live(sig=signal.SIGKILL)
        p.wait(timeout=10)
        assert p.returncode == -signal.SIGKILL
        _assert_gone(child)
        assert _pid_files(shim_dir) == []
    finally:
        _reap(p)


@pytest.mark.live
def test_stale_pid_file_for_unrelated_process_is_not_signalled(shim_dir, fake_codex):
    h = make_shim(shim_dir, "codex", fake_codex)
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        (shim_dir / "pids" / str(sleeper.pid)).write_text(
            json.dumps({"tool": "codex", "child": None, "real": str(fake_codex)}))
        assert h.live_pids() == []
        assert not (shim_dir / "pids" / str(sleeper.pid)).exists()
        h.kill_live()
        time.sleep(0.2)
        assert sleeper.poll() is None, "unrelated process must not be signalled"
    finally:
        sleeper.kill()
        sleeper.wait()


@pytest.mark.live
def test_kill_live_sigkill_identifies_env_shebang_cli_by_ppid(shim_dir, tmp_path):
    real = _write_fake_env_shebang(tmp_path / "real_env", "codex")
    h = make_shim(shim_dir, "codex", real)
    pidfile = tmp_path / "child.pid"
    p = _spawn(h.path, ["exec", "--pidfile", str(pidfile), "--sleep", "30"], start_new_session=True)
    try:
        _wait_file(pidfile)
        child = int(pidfile.read_text())
        h.kill_live(sig=signal.SIGKILL)
        p.wait(timeout=10)
        assert p.returncode == -signal.SIGKILL
        _assert_gone(child)
        assert _pid_files(shim_dir) == []
    finally:
        _reap(p)


from tests.live_acceptance.cli_shim import seed_campaign  # noqa: E402
from tests.live_acceptance.spend_ledger import Ledger as _Ledger  # noqa: E402


def _camp(tmp_path, cap, scenario="campaign"):
    return {"path": tmp_path / "campaign.jsonl", "cap": cap, "scenario": scenario}


@pytest.mark.live
def test_campaign_cap_refuses_even_when_scenario_has_room(shim_dir, fake_codex, ledger_path, tmp_path):
    camp = _camp(tmp_path, 50)
    h = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1",
                  global_cap=10**6, output_ceiling=100, campaign_ledger=camp)
    mark = tmp_path / "ran.mark"
    r = subprocess.run([str(h.path), "exec", "--mark", str(mark)], input=b"p",
                       capture_output=True, timeout=10)
    assert r.returncode == BUDGET_EXIT
    assert b"budget refused: campaign" in r.stderr
    assert not mark.exists()
    assert _Ledger(ledger_path, 10**6).totals()["global"]["non_cached"] == 0
    assert _Ledger(camp["path"], 50).totals()["global"]["non_cached"] == 0


@pytest.mark.live
def test_campaign_and_scenario_settle_consistently(shim_dir, fake_codex, fake_claude, ledger_path, tmp_path):
    camp = _camp(tmp_path, 10**6)
    codex = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1",
                      global_cap=10**6, campaign_ledger=camp)
    claude = make_shim(shim_dir, "claude", fake_claude, ledger_path=ledger_path, scenario="s1",
                       global_cap=10**6, campaign_ledger=camp)
    assert subprocess.run([str(codex.path), "exec", "--usage", CODEX_TURN],
                          input=b"p", capture_output=True, timeout=10).returncode == 0
    assert subprocess.run([str(claude.path), "--print", "--usage", CLAUDE_RESULT],
                          input=b"p", capture_output=True, timeout=10).returncode == 0
    s = _Ledger(ledger_path, 10**6).totals()["s1"]
    c = _Ledger(camp["path"], 10**6).totals()["campaign"]
    assert s == {"non_cached": 112, "cached": 50}
    assert c == s


@pytest.mark.live
def test_scenario_refusal_releases_campaign_reservation(shim_dir, fake_codex, ledger_path, tmp_path):
    camp = _camp(tmp_path, 10**6)
    h = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1",
                  global_cap=50, output_ceiling=100, campaign_ledger=camp)
    mark = tmp_path / "ran.mark"
    r = subprocess.run([str(h.path), "exec", "--mark", str(mark)], input=b"p",
                       capture_output=True, timeout=10)
    assert r.returncode == BUDGET_EXIT
    assert b"budget refused: scenario" in r.stderr
    assert not mark.exists()
    assert _Ledger(camp["path"], 10**6).totals()["global"]["non_cached"] == 0
    rows = camp["path"].read_text()
    assert '"release"' in rows and '"reserve"' in rows


@pytest.mark.live
def test_seed_campaign_is_idempotent_and_counts_prior_spend(tmp_path):
    path = tmp_path / "campaign.jsonl"
    led = seed_campaign(path, 1000, {"dry-run": 300, "baseline": 100})
    assert led.totals()["global"]["non_cached"] == 400
    again = seed_campaign(path, 1000, {"dry-run": 300, "baseline": 100})
    assert again.totals()["global"]["non_cached"] == 400
    with pytest.raises(ValueError):
        seed_campaign(path, 999, {})


@pytest.mark.live
def test_held_then_killed_call_charges_nothing(shim_dir, fake_codex, ledger_path, tmp_path):
    camp = _camp(tmp_path, 10**6)
    h = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1",
                  global_cap=10**6, output_ceiling=100, campaign_ledger=camp)
    h.hold()
    p = _spawn(h.path, ["exec"])
    try:
        assert h.wait_started(10)
        p.kill()
        p.wait(timeout=10)
        assert _Ledger(ledger_path, 10**6).reap(stale_after_s=0, shim_log_paths=[h._log]) == 0
        assert _Ledger(ledger_path, 10**6).totals()["global"]["non_cached"] == 0
        assert _Ledger(camp["path"], 10**6).totals()["global"]["non_cached"] == 0
    finally:
        _reap(p)
        h.close()


@pytest.mark.live
def test_exec_then_killed_settles_estimate_after_reap(shim_dir, fake_codex, ledger_path, tmp_path):
    camp = _camp(tmp_path, 10**6)
    h = make_shim(shim_dir, "codex", fake_codex, ledger_path=ledger_path, scenario="s1",
                  global_cap=10**6, output_ceiling=100, campaign_ledger=camp)
    pidfile = tmp_path / "child.pid"
    args = ["exec", "--pidfile", str(pidfile), "--sleep", "30"]
    p = _spawn(h.path, args)
    child = None
    try:
        _wait_file(pidfile)
        child = int(pidfile.read_text())
        assert any("exec_start" in line for line in h._log.read_text().splitlines())
        os.kill(p.pid, signal.SIGKILL)
        p.wait(timeout=10)
        estimate = math.ceil(sum(len(a.encode()) for a in args) / 3) + 100
        assert _Ledger(ledger_path, 10**6).reap(stale_after_s=0, shim_log_paths=[h._log]) == 1
        assert _Ledger(camp["path"], 10**6).reap(stale_after_s=0, shim_log_paths=[h._log]) == 1
        assert _Ledger(ledger_path, 10**6).totals()["s1"] == {"non_cached": estimate, "cached": 0}
        assert _Ledger(camp["path"], 10**6).totals()["campaign"] == {"non_cached": estimate, "cached": 0}
    finally:
        _reap(p)
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.live
def test_reap_without_exec_start_settles_zero(tmp_path):
    led_path = tmp_path / "ledger.jsonl"
    led = _Ledger(led_path, 10**6)
    never = led.reserve("s1", 500)  # reserved, killed before exec_start
    log = tmp_path / "shim.log"
    log.write_text("start codex 1 1 1 exec\n")
    assert led.reap(stale_after_s=0, shim_log_paths=[log]) == 1
    assert led.totals()["s1"] == {"non_cached": 0, "cached": 0}
    rows = led_path.read_text()
    assert "never_executed" in rows and never in rows


@pytest.mark.live
def test_reap_with_exec_start_settles_estimate(tmp_path):
    led_path = tmp_path / "ledger.jsonl"
    led = _Ledger(led_path, 10**6)
    rid = led.reserve("s1", 500)
    log = tmp_path / "shim.log"
    log.write_text(f"exec_start codex 1 1 1 {rid}\n")
    assert led.reap(stale_after_s=0, shim_log_paths=[log]) == 1
    assert led.totals()["s1"] == {"non_cached": 500, "cached": 0}
