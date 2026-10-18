"""Runtime behind the wrappers written by cli_shim.make_shim.

Run as: python cli_shim_runner.py <config.json> <tool args...>

Passthrough (exec'd unchanged, not counted, gated, or metered): `login status`,
--version, --help. The process is replaced, so stdin/stdout/stderr are untouched.

Model calls:
- take the next ordinal under a lock;
- read stdin (empty if closed or a tty; at most stdin_timeout seconds before the
  bytes read so far are used and the rest is forwarded to the child as it arrives);
- evaluate gates once, before blocking: holds must be armed before the call starts;
- reserve budget in the spend ledger (if configured);
- run the real CLI with stdin forwarded byte-identical and stdout teed unchanged.
  stderr and --output-last-message files are left to the real CLI.
- settle usage from the terminal usage record, or the estimate if absent.

Signals SIGTERM/SIGINT/SIGHUP are forwarded to the child, which is then reaped.
A closed stdout pipe terminates the child and exits 141.

The ledger is never reaped here: other calls may be live. The harness calls
Ledger.reap() at run start and scenario end.
"""

import fcntl
import json
import math
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from spend_ledger import BudgetRefused, Ledger

GATE_TIMEOUT_EXIT = 97
BUDGET_EXIT = 75
DENY_EXIT = 76
DENY_FILE = "DENY"
BROKEN_PIPE_EXIT = 141
MATCH_BYTES = 64 * 1024
CHUNK = 64 * 1024
USAGE_TYPE = {"codex": "turn.completed", "claude": "result"}
PASSTHROUGH_FLAGS = ("--version", "--help")
FORWARD_WAIT = 2.0
STOP_SIGNALS = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
TERM_WAIT = 5.0

CHILD = {"pid": None, "pidfile": None}


def _exit(code):
    if CHILD["pidfile"] is not None:
        try:
            os.unlink(CHILD["pidfile"])
        except FileNotFoundError:
            pass
    os._exit(code)


def _register(cfg, child=None):
    path = Path(cfg["pids"]) / str(os.getpid())
    CHILD["pidfile"] = str(path)
    _atomic_json(path, {"tool": cfg["tool"], "child": child, "real": cfg["real"],
                        "rids": CHILD.get("rids", [])})


def _atomic_json(path, obj):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _denied(cfg):
    return (Path(cfg["gates"]) / DENY_FILE).exists()


def _deny_exit(cfg, ordinal):
    _log(cfg, f"denied {cfg['tool']} {int(time.time())} {os.getpid()} {ordinal}")
    sys.stderr.write("denied by DENY gate\n")
    sys.stderr.flush()
    _exit(DENY_EXIT)


def _log(cfg, line):
    fd = os.open(cfg["log"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, (line.replace("\n", " ") + "\n").encode())
    finally:
        os.close(fd)


def _ordinal(cfg):
    fd = os.open(cfg["counter"], os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # released on close
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 32).decode().strip()
        n = int(raw or 0) + 1
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, str(n).encode())
        return n
    finally:
        os.close(fd)


def _matched_gates(cfg, ordinal, text):
    gates = Path(cfg["gates"])
    tool = cfg["tool"]
    names = [n for n in (tool, "all", f"{tool}.nth.{ordinal}") if (gates / n).exists()]
    for f in sorted(gates.glob(f"{tool}.match.*")):
        try:
            pattern = f.read_text()
        except FileNotFoundError:
            continue
        if re.search(pattern, text):
            names.append(f.name)
    return names


def _block(cfg, ordinal, names, deadline):
    gates = Path(cfg["gates"])
    for name in names:
        _log(cfg, f"gate {cfg['tool']} {int(time.time())} {os.getpid()} {ordinal} {name}")
    while any((gates / n).exists() for n in names):
        if time.time() >= deadline:
            _log(cfg, f"gate_timeout {cfg['tool']} {int(time.time())} {os.getpid()}")
            _exit(GATE_TIMEOUT_EXIT)
        if _denied(cfg):
            _deny_exit(cfg, ordinal)
        time.sleep(0.2)


def _read_stdin(cfg, ordinal):
    """Return (prefix, rest_fd). rest_fd is set only if stdin timed out and is still open."""
    try:
        fd = sys.stdin.fileno()
        if os.isatty(fd):
            return b"", None
    except (AttributeError, ValueError, OSError):
        return b"", None
    buf = bytearray()
    deadline = time.monotonic() + cfg["stdin_timeout"]
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            _log(cfg, f"stdin_timeout {cfg['tool']} {int(time.time())} {os.getpid()} {ordinal}")
            return bytes(buf), fd
        ready, _, _ = select.select([fd], [], [], left)
        if not ready:
            continue
        chunk = os.read(fd, CHUNK)
        if not chunk:
            return bytes(buf), None
        buf += chunk


def _feed(proc, prefix, rest_fd):
    try:
        proc.stdin.write(prefix)
        proc.stdin.flush()
        if rest_fd is not None:
            while True:
                chunk = os.read(rest_fd, CHUNK)
                if not chunk:
                    break
                proc.stdin.write(chunk)
                proc.stdin.flush()
    except (OSError, ValueError):
        pass  # child exited or was stopped
    finally:
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass


def _usage(tool, stdout):
    want = USAGE_TYPE[tool]
    found = None
    for line in stdout.decode("utf-8", "replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict) or rec.get("type") != want or not isinstance(rec.get("usage"), dict):
            continue
        u = rec["usage"]
        i, o = u.get("input_tokens"), u.get("output_tokens")
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (i, o)):
            continue
        c = u.get("cached_input_tokens", u.get("cache_read_input_tokens", 0))
        if not (isinstance(c, int) and not isinstance(c, bool) and c >= 0):
            c = 0
        found = (i, o, c)
    return found


def _stop_child(pid, sig, wait_s):
    """Send sig, wait up to wait_s for exit (reaping it), then SIGKILL."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return
        if done:
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _on_signal(signum, _frame):
    if CHILD["pid"] is not None:
        _stop_child(CHILD["pid"], signum, FORWARD_WAIT)
    _exit(128 + signum)


def _passthrough(cfg, args):
    real = cfg["real"]
    _log(cfg, f"passthrough {cfg['tool']} {int(time.time())} {os.getpid()}")
    os.execv(real, [real, *args])


def _refuse(cfg, ordinal, which, err):
    sys.stderr.write(f"budget refused: {which} ({err})\n")
    sys.stderr.flush()
    _log(cfg, f"refused {cfg['tool']} {int(time.time())} {os.getpid()} {ordinal} {which} {err}")
    _exit(BUDGET_EXIT)


def _settle(cfg, targets, estimate, tool, ordinal, captured):
    """targets: list of (ledger, scenario, rid). Records the same usage on every ledger."""
    usage = _usage(tool, b"".join(captured))
    label = f"{tool} {ordinal}"
    if usage is not None:
        i, o, c = usage
        source = f"usage {i} {o} {c}"
        for led, scenario, rid in targets:
            led.record(scenario, label, i, o, cached_tokens=c, rid=rid)
    else:
        source = f"estimate {estimate}"
        for led, scenario, rid in targets:
            led.record(scenario, label, estimate, 0, rid=rid)
    _log(cfg, f"settled {tool} {int(time.time())} {os.getpid()} {ordinal} {source}")


def _model_call(cfg, args):
    tool = cfg["tool"]
    if _denied(cfg):
        _deny_exit(cfg, 0)
    _register(cfg)
    ordinal = _ordinal(cfg)
    prefix, rest_fd = _read_stdin(cfg, ordinal)

    schema = ""
    if "--output-schema" in args:
        i = args.index("--output-schema")
        if i + 1 < len(args):
            try:
                schema = Path(args[i + 1]).read_text(errors="replace")
            except OSError:
                schema = ""
    text = " ".join(args) + "\n" + schema + "\n" + prefix[:MATCH_BYTES].decode("utf-8", "replace")

    _log(cfg, f"start {tool} {int(time.time())} {os.getpid()} {ordinal} {' '.join(args)}")
    names = _matched_gates(cfg, ordinal, text)
    _block(cfg, ordinal, names, time.time() + cfg["gate_timeout"])
    if _denied(cfg):
        _deny_exit(cfg, ordinal)

    ledger_cfg = cfg.get("ledger")
    campaign_cfg = cfg.get("campaign")
    targets = []  # (ledger, scenario, rid) for every reservation actually held
    if ledger_cfg or campaign_cfg:
        prompt_bytes = len(prefix) + sum(len(a.encode()) for a in args)
        estimate = math.ceil(prompt_bytes / 3) + cfg["output_ceiling"]
    if campaign_cfg:
        cled = Ledger(campaign_cfg["path"], campaign_cfg["cap"])
        try:
            crid = cled.reserve(campaign_cfg["scenario"], estimate)
        except BudgetRefused as e:
            _refuse(cfg, ordinal, "campaign", e)
        targets.append((cled, campaign_cfg["scenario"], crid))
    if ledger_cfg:
        led = Ledger(ledger_cfg["path"], ledger_cfg["global_cap"], ledger_cfg["scenario_caps"])
        try:
            rid = led.reserve(ledger_cfg["scenario"], estimate)
        except BudgetRefused as e:
            for other, _s, r in targets:
                other.release(r)
            _refuse(cfg, ordinal, "scenario", e)
        targets.append((led, ledger_cfg["scenario"], rid))

    CHILD["rids"] = [r for _l, _s, r in targets]
    _register(cfg)

    # Mask stop signals across spawn and registration so a signal is delivered only after
    # CHILD["pid"] is known; the child itself gets the original mask back via preexec_fn.
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
    try:
        if _denied(cfg):
            for other, _s, r in targets:
                other.release(r)
            _deny_exit(cfg, ordinal)
        # exec_start names the reservations; reap() settles a hold with no exec_start at 0.
        rid_field = ",".join(CHILD["rids"]) or "-"
        _log(cfg, f"exec_start {tool} {int(time.time())} {os.getpid()} {ordinal} {rid_field}")
        _log(cfg, f"exec {tool} {int(time.time())} {os.getpid()}")
        # preexec_fn runs in the forked child before exec, so it must stay trivial. The stdin
        # forwarding thread is started only after Popen returns, never before fork.
        proc = subprocess.Popen([cfg["real"], *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask))
        CHILD["pid"] = proc.pid
        _register(cfg, child=proc.pid)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    writer = threading.Thread(target=_feed, args=(proc, prefix, rest_fd), daemon=True)
    writer.start()

    captured = []
    broken = False
    while True:
        chunk = proc.stdout.read1(CHUNK)
        if not chunk:
            break
        try:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        except BrokenPipeError:
            broken = True
            break
        captured.append(chunk)

    if broken:
        _stop_child(proc.pid, signal.SIGTERM, TERM_WAIT)
        if targets:
            _settle(cfg, targets, estimate, tool, ordinal, captured)
        _log(cfg, f"broken_pipe {tool} {int(time.time())} {os.getpid()} {ordinal}")
        _exit(BROKEN_PIPE_EXIT)

    proc.stdout.close()
    writer.join()
    rc = proc.wait()
    if targets:
        _settle(cfg, targets, estimate, tool, ordinal, captured)
    return rc


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    args = sys.argv[2:]
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    if (args[:2] == ["login", "status"]) or any(a in PASSTHROUGH_FLAGS for a in args):
        _passthrough(cfg, args)
    rc = _model_call(cfg, args)
    _exit(rc if rc >= 0 else 128 - rc)


if __name__ == "__main__":
    main()
