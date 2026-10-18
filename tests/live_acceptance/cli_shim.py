"""CLI shim for live acceptance runs.

make_shim() writes a /bin/sh wrapper named <tool> into shim_dir. The wrapper
execs the venv python on cli_shim_runner.py with a per-tool JSON config, all
paths absolute, so it works under the adapter's env whitelist.

Behaviour (see cli_shim_runner.py):
- `login status`, --version, --help: exec'd straight through, not counted/gated/metered.
- Model calls: counted per tool (ordinal), stdin read and forwarded byte-identical,
  gated, budget reserved in spend_ledger.Ledger (if configured), real CLI run with
  stdout teed unchanged; stderr and --output-last-message files untouched.
- stdin: empty if closed or a tty; otherwise read for at most stdin_timeout seconds
  (default 30). Past that, the bytes read so far are used for gating and budget, and the
  rest of stdin is still forwarded to the child as it arrives (nothing is dropped).
- Signals SIGTERM/SIGINT/SIGHUP to the shim are forwarded to the real CLI, which is
  reaped; the shim then exits 128+signal. Killing the shim alone does not orphan the CLI.
- Closed stdout (reader went away): the real CLI is terminated, usage settled, exit 141.
- Budget refusal: stderr "budget refused: <reason>", exit 75, real CLI not run.
- Gate timeout (gate_timeout, default 600 s): exit 97, real CLI not run.

Holds (files under shim_dir/gates). Holds must be armed BEFORE the call starts: gates
are evaluated once, when the call is counted, and a hold armed later does not block
that call.
- hold(name) / release(name): name is a tool name or "all"; blocks every call.
- hold_nth(n): blocks only the nth counted call of this tool.
- hold_matching(regex): blocks calls whose args, --output-schema file contents, or
  first 64 KiB of stdin match regex. Returns a token for release().
- release_all() clears all holds; close() is release_all().
- deny() / undeny(): global DENY gate for every tool. A held call exits 76 within
  about 0.2 s; a new call exits 76 before counting, reserving budget, or exec'ing.
- live_pids() lists running shim runners in this shim_dir (pid files under pids/,
  stale files removed). kill_live(sig, grace=3.0) signals each runner (the runner
  forwards to its CLI); SIGKILL also kills the recorded CLI pid. Survivors after
  grace are SIGKILLed. Works for shims outside the caller's process group.
- output_ceiling (make_shim, default 8192 tokens) is the per-call output reserve.
  Use 3000 for low-effort runs.
- The shim never calls Ledger.reap(): other calls may be live. The harness (fixture or
  orchestrator) calls reap() at run start and at scenario end.

Timing caveat: the product caps each orchestration CLI call at 120 s. A held
orchestration call must be released within that window or the product times it
out; tests holding such calls must release in time or expect a timeout.

Log format (shim_dir/shim.log), one line per event, tool name on every line:
    start <tool> <epoch> <pid> <ordinal> <args>
    gate <tool> <epoch> <pid> <ordinal> <gate name>
    exec <tool> <epoch> <pid>
    settled <tool> <epoch> <pid> <ordinal> <usage|estimate ...>
    refused <tool> <epoch> <pid> <ordinal> <reason>
    gate_timeout <tool> <epoch> <pid>
    passthrough <tool> <epoch> <pid>
    stdin_timeout <tool> <epoch> <pid> <ordinal>
    broken_pipe <tool> <epoch> <pid> <ordinal>

Clock resolution is 1 s (`time.time()` truncated); use a finer clock only if
sub-second timing matters.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .spend_ledger import Ledger

TOOLS = ("claude", "codex")
GATE_TIMEOUT_EXIT = 97
BUDGET_EXIT = 75
DEFAULT_GATE_TIMEOUT = 600
DEFAULT_OUTPUT_CEILING = 8192
RUNNER = Path(__file__).resolve().with_name("cli_shim_runner.py")


@dataclass
class Invocation:
    pid: int
    started_at: int
    ordinal: int
    args: str
    exec_at: int | None = None
    timed_out: bool = False
    refused: bool = False
    gates: list[str] = field(default_factory=list)


@dataclass
class ShimHandle:
    shim_dir: Path
    tool: str
    path: Path

    @property
    def _gates(self) -> Path:
        return self.shim_dir / "gates"

    @property
    def _log(self) -> Path:
        return self.shim_dir / "shim.log"

    def __enter__(self) -> "ShimHandle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def hold(self, name: str = "all") -> None:
        """Block calls for `name` (a tool name or "all") until released."""
        (self._gates / name).touch()

    def hold_nth(self, n: int) -> str:
        """Block only the nth counted call of this tool. Returns the release token."""
        if n < 1:
            raise ValueError("ordinal must be >= 1")
        token = f"{self.tool}.nth.{n}"
        (self._gates / token).touch()
        return token

    def hold_matching(self, regex: str) -> str:
        """Block calls whose args, --output-schema file, or first 64 KiB of stdin match regex.

        Returns the release token.
        """
        re.compile(regex)
        token = f"{self.tool}.match.{uuid.uuid4().hex}"
        (self._gates / token).write_text(regex)
        return token

    def release(self, name: str = "all") -> None:
        (self._gates / name).unlink(missing_ok=True)

    def release_all(self) -> None:
        for gate in self._gates.iterdir():
            gate.unlink(missing_ok=True)

    def close(self) -> None:
        self.release_all()

    def invocations(self) -> list[Invocation]:
        """Counted invocations of this handle's tool, parsed from the shared log.

        Malformed or half-written lines are skipped.
        """
        if not self._log.exists():
            return []
        text = self._log.read_text(errors="replace")
        lines = text.split("\n")
        if not text.endswith("\n"):
            lines = lines[:-1]  # last line still being written
        by_pid: dict[int, Invocation] = {}
        result: list[Invocation] = []
        for line in lines:
            parts = line.split(" ", 5)
            if len(parts) < 4 or parts[1] != self.tool:
                continue
            event = parts[0]
            try:
                pid = int(parts[3])
                if event == "start" and len(parts) >= 5:
                    inv = Invocation(
                        pid=pid,
                        started_at=int(parts[2]),
                        ordinal=int(parts[4]),
                        args=parts[5] if len(parts) == 6 else "",
                    )
                    by_pid[pid] = inv
                    result.append(inv)
                    continue
                inv = by_pid.get(pid)
                if inv is None:
                    continue
                if event == "exec":
                    inv.exec_at = int(parts[2])
                elif event == "gate_timeout":
                    inv.timed_out = True
                elif event == "refused":
                    inv.refused = True
                elif event == "gate" and len(parts) == 6:
                    inv.gates.append(parts[5])
            except ValueError:
                continue
        return result

    def deny(self) -> None:
        """Global DENY gate: every running and future call of any tool exits 76."""
        (self._gates / "DENY").touch()

    def undeny(self) -> None:
        (self._gates / "DENY").unlink(missing_ok=True)

    def live_pids(self) -> list[int]:
        return sorted(self._runners())

    def kill_live(self, sig: int = signal.SIGTERM, grace: float = 3.0) -> None:
        """Signal every verified live runner (and its CLI), escalating to SIGKILL after grace.

        Only processes whose ps command line still matches this shim's config (runner) or
        the real binary (CLI) are signalled; stale pid files are removed.
        """
        runners = self._runners()
        for pid, (child, _cfg, _real) in runners.items():
            if sig == signal.SIGKILL and child:
                _send(child, signal.SIGKILL)
            _send(pid, sig)
        pids_dir = self.shim_dir / "pids"
        end = time.monotonic() + grace
        while time.monotonic() < end and any(_owned(pid, cfg) for pid, (_c, cfg, _r) in runners.items()):
            time.sleep(0.05)
        for pid, (child, cfg, real) in runners.items():
            if _owned(pid, cfg):
                # Kill the CLI while the runner (its parent) is still alive, then the runner.
                if child and _child_owned(child, pid, real):
                    _send(child, signal.SIGKILL)
                _send(pid, signal.SIGKILL)
            (pids_dir / str(pid)).unlink(missing_ok=True)

    def _runners(self) -> dict[int, tuple[int | None, str, str]]:
        """pid -> (verified child pid or None, config path, real path) for live runners."""
        out: dict[int, tuple[int | None, str, str]] = {}
        for f in (self.shim_dir / "pids").glob("*"):
            try:
                pid = int(f.name)
                info = json.loads(f.read_text())
            except (ValueError, OSError, AttributeError):
                f.unlink(missing_ok=True)
                continue
            tool = info.get("tool") if isinstance(info, dict) else None
            if tool not in TOOLS:
                f.unlink(missing_ok=True)
                continue
            cfg = str(self.shim_dir / f"{tool}.json")
            if not _owned(pid, cfg):
                f.unlink(missing_ok=True)
                continue
            child = info.get("child")
            real = info.get("real")
            if not (isinstance(child, int) and isinstance(real, str) and _child_owned(child, pid, real)):
                child = None
            out[pid] = (child, cfg, real if isinstance(real, str) else "")
        return out

    def wait_invocations(self, n: int, timeout: float = 10.0) -> bool:
        """Wait until at least n counted invocations have started."""
        deadline = time.monotonic() + timeout
        while True:
            if len(self.invocations()) >= n:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def wait_started(self, timeout: float = 10.0) -> bool:
        return self.wait_invocations(1, timeout)


def make_shim(
    shim_dir: Path,
    tool: str,
    real_path: Path,
    *,
    gate_timeout: int = DEFAULT_GATE_TIMEOUT,
    ledger_path: Path | None = None,
    scenario: str = "default",
    stdin_timeout: float = 30.0,
    global_cap: int | None = None,
    scenario_cap: int | None = None,
    output_ceiling: int = DEFAULT_OUTPUT_CEILING,
    campaign_ledger: dict | None = None,
) -> ShimHandle:
    """campaign_ledger: optional {"path": Path, "cap": int, "scenario": str}.

    Every call reserves on the campaign ledger as well as the scenario ledger, and is
    refused if either would exceed its cap. Use seed_campaign() to create the campaign ledger.
    """
    if tool not in TOOLS:
        raise ValueError(f"unsupported tool: {tool!r}")
    if gate_timeout <= 0:
        raise ValueError("gate_timeout must be positive")
    if stdin_timeout <= 0:
        raise ValueError("stdin_timeout must be positive")
    if ledger_path is not None and global_cap is None:
        raise ValueError("global_cap is required when ledger_path is set")
    shim_dir = shim_dir.absolute()
    (shim_dir / "gates").mkdir(parents=True, exist_ok=True)
    (shim_dir / "pids").mkdir(parents=True, exist_ok=True)
    (shim_dir / "counters").mkdir(parents=True, exist_ok=True)

    ledger = None
    if ledger_path is not None:
        ledger = {
            "path": str(Path(ledger_path).absolute()),
            "global_cap": int(global_cap),
            "scenario_caps": {scenario: int(scenario_cap)} if scenario_cap is not None else {},
            "scenario": scenario,
        }
    campaign = None
    if campaign_ledger is not None:
        if ledger_path is None:
            raise ValueError("campaign_ledger requires ledger_path (scenario metering)")
        campaign = {
            "path": str(Path(campaign_ledger["path"]).absolute()),
            "cap": int(campaign_ledger["cap"]),
            "scenario": str(campaign_ledger["scenario"]),
        }
    config = {
        "campaign": campaign,
        "tool": tool,
        "real": str(real_path.absolute()),
        "log": str(shim_dir / "shim.log"),
        "gates": str(shim_dir / "gates"),
        "counter": str(shim_dir / "counters" / tool),
        "pids": str(shim_dir / "pids"),
        "gate_timeout": int(gate_timeout),
        "output_ceiling": int(output_ceiling),
        "stdin_timeout": float(stdin_timeout),
        "ledger": ledger,
    }
    config_path = shim_dir / f"{tool}.json"
    _atomic_write(config_path, json.dumps(config))

    path = shim_dir / tool
    q = shlex.quote
    script = f"#!/bin/sh\nexec {q(sys.executable)} {q(str(RUNNER))} {q(str(config_path))} \"$@\"\n"
    _atomic_write(path, script, mode=0o755)
    return ShimHandle(shim_dir=shim_dir, tool=tool, path=path)


def _ps(pid: int) -> tuple[int, str, str] | None:
    """(ppid, stat, command) for pid, or None if no such process."""
    r = subprocess.run(["/bin/ps", "-ww", "-o", "ppid=,stat=,command=", "-p", str(pid)],
                       capture_output=True, text=True)
    parts = r.stdout.strip().split(None, 2)
    if r.returncode != 0 or len(parts) < 2:
        return None
    try:
        ppid = int(parts[0])
    except ValueError:
        return None
    return ppid, parts[1], (parts[2] if len(parts) > 2 else "")


def _owned(pid: int, needle: str) -> bool:
    """True if pid is running (not a zombie) and its command line contains needle."""
    info = _ps(pid)
    return info is not None and not info[1].startswith("Z") and needle in info[2]


def _child_owned(child: int, runner: int, real: str) -> bool:
    """True if child is a live CLI of runner.

    Owned when its parent is the runner (survives shebang scripts whose ps command line
    lacks the real path), or when its command line names the real binary (path,
    resolved target, or basename).
    """
    info = _ps(child)
    if info is None or info[1].startswith("Z"):
        return False
    ppid, _stat, cmd = info
    if ppid == runner:
        return True
    needles = {real, os.path.realpath(real), os.path.basename(real)}
    return any(n and n in cmd for n in needles)


def _send(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def seed_campaign(path: Path, cap: int, prior_spend: dict[str, int],
                  scenario: str = "campaign"):
    """Open the campaign ledger, creating it with prior spend rows if it does not exist.

    prior_spend maps label -> non-cached tokens already spent. Idempotent: an existing
    non-empty ledger is opened unchanged (its caps must match cap). Creation is not
    atomic across concurrent seeders; seed once before starting parallel runs.
    """
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return Ledger(path, cap)
    led = Ledger(path, cap)
    for label, tokens in prior_spend.items():
        led.record(scenario, label, int(tokens), 0)
    return led


def _atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.chmod(mode)
    os.replace(tmp, path)


def resolve_real(tool: str, shim_dir: Path | None = None) -> Path:
    """Find the real binary on PATH, skipping shim_dir (symlinks included)."""
    skip = Path(shim_dir).resolve() if shim_dir is not None else None
    dirs = [
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and (skip is None or Path(p).resolve() != skip)
    ]
    found = shutil.which(tool, path=os.pathsep.join(dirs))
    if found is None:
        raise FileNotFoundError(f"{tool} not found on PATH (excluding {shim_dir})")
    if skip is not None and Path(found).resolve().is_relative_to(skip):
        raise RuntimeError(f"resolved {tool} to the shim itself: {found}")
    return Path(found).absolute()
