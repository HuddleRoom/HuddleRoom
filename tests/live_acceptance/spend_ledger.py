"""Multi-process-safe token spend ledger (jsonl, flock + fsync).

Line 1 persists the caps; reopening with different caps raises. reserve()
holds an estimate (with ts + pid) until record()/release()/reap() settles it,
so concurrent callers see each other's in-flight spend.
Non-cached tokens = input - cached + output.

Overshoot bound: a hold is sized by an estimate, but the real usage is only
known afterwards. Worst-case total overshoot of a cap is therefore
concurrency x (actual - estimate); callers must leave cap headroom for it.

A trailing torn line (crash mid-append) is skipped and truncated on the next
write; a torn line in the middle of the file raises.
"""
import fcntl
import json
import os
import time
import uuid
from collections import defaultdict


class BudgetRefused(Exception):
    pass


def _nonneg_int(v, name):
    v = int(v or 0)
    if v < 0:
        raise ValueError(f"{name} must be >= 0")
    return v


class Ledger:
    def __init__(self, path, global_cap, scenario_caps=None):
        self.path = str(path)
        want = {"k": "caps", "global": global_cap, "scenarios": dict(scenario_caps or {})}
        with self._locked() as f:
            rows = self._rows(f)
            if not rows:
                self._write(f, want)
            elif rows[0] != want:
                raise ValueError(f"ledger caps mismatch: {rows[0]} != {want}")
        self.caps = want

    # -- io
    def _locked(self):
        outer = self

        class Ctx:
            def __enter__(s):
                s.f = open(outer.path, "a+")
                fcntl.flock(s.f, fcntl.LOCK_EX)
                return s.f

            def __exit__(s, *a):
                s.f.close()  # releases the lock

        return Ctx()

    @staticmethod
    def _write(f, row):
        f.seek(0)
        data = f.read()
        if data and not data.endswith("\n"):
            tail = data[data.rfind("\n") + 1:]
            try:
                json.loads(tail)
                f.write("\n")  # complete row that only lacks its newline: keep it
            except ValueError:  # torn tail: drop it before appending
                os.ftruncate(f.fileno(), len(data[:data.rfind("\n") + 1].encode()))
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())

    def _rows(self, f=None):
        if f is None:
            with self._locked() as g:
                return self._rows(g)
        f.seek(0)
        lines = f.read().split("\n")
        last = max((i for i, x in enumerate(lines) if x.strip()), default=-1)
        rows = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                if i == last:
                    break
                raise
        return rows

    @staticmethod
    def _sum(rows):
        used = defaultdict(lambda: [0, 0])  # scenario -> [non_cached, cached]
        pending = {}
        for r in rows[1:]:
            if r["k"] == "reserve":
                pending[r["rid"]] = r
            elif r["k"] == "release":
                pending.pop(r["rid"], None)
            elif r["k"] == "usage":
                pending.pop(r.get("rid"), None)
                used[r["scenario"]][0] += r["input_tokens"] - r["cached_tokens"] + r["output_tokens"]
                used[r["scenario"]][1] += r["cached_tokens"]
        return used, pending

    # -- api
    def reserve(self, scenario, estimate):
        if estimate < 0:
            raise ValueError("negative estimate")
        with self._locked() as f:
            rows = self._rows(f)
            used, pending = self._sum(rows)
            s_tot = used[scenario][0] + sum(p["estimate"] for p in pending.values() if p["scenario"] == scenario)
            g_tot = sum(v[0] for v in used.values()) + sum(p["estimate"] for p in pending.values())
            cap = self.caps["scenarios"].get(scenario)
            if cap is not None and s_tot + estimate > cap:
                raise BudgetRefused(f"scenario {scenario} cap {cap}")
            if g_tot + estimate > self.caps["global"]:
                raise BudgetRefused(f"global cap {self.caps['global']}")
            rid = uuid.uuid4().hex
            self._write(f, {"k": "reserve", "rid": rid, "scenario": scenario, "estimate": estimate,
                            "ts": time.time(), "pid": os.getpid()})
            return rid

    def release(self, rid):
        """Only for a clean failure before anything was sent upstream."""
        with self._locked() as f:
            self._write(f, {"k": "release", "rid": rid})

    def record(self, scenario, label, input_tokens, output_tokens, cached_tokens=0, rid=None):
        i = _nonneg_int(input_tokens, "input_tokens")
        o = _nonneg_int(output_tokens, "output_tokens")
        c = min(_nonneg_int(cached_tokens, "cached_tokens"), i)
        with self._locked() as f:
            self._write(f, {"k": "usage", "rid": rid, "scenario": scenario, "label": label,
                            "input_tokens": i, "output_tokens": o, "cached_tokens": c})

    def reap(self, stale_after_s=3600, shim_log_paths=None):
        """Settle holds whose pid is dead or older than the TTL. Call at harness run start and end.

        With shim_log_paths, a hold whose rid has no "exec_start" line in those logs was never
        sent to the real CLI: it settles at 0 (label "never_executed"). Otherwise it settles at
        its estimate (label "reaped"). Without shim_log_paths every stale hold settles at its
        estimate. Returns the number of holds settled.
        """
        n = 0
        executed = _exec_rids(shim_log_paths) if shim_log_paths is not None else None
        with self._locked() as f:
            _, pending = self._sum(self._rows(f))
            for rid, p in pending.items():
                if time.time() - p["ts"] > stale_after_s or not _alive(p["pid"]):
                    if executed is not None and rid not in executed:
                        tokens, label = 0, "never_executed"
                    else:
                        tokens, label = p["estimate"], "reaped"
                    self._write(f, {"k": "usage", "rid": rid, "scenario": p["scenario"], "label": label,
                                    "input_tokens": tokens, "output_tokens": 0, "cached_tokens": 0})
                    n += 1
        return n

    def totals(self):
        used, _ = self._sum(self._rows())
        out = {s: {"non_cached": v[0], "cached": v[1]} for s, v in used.items()}
        out["global"] = {"non_cached": sum(v[0] for v in used.values()),
                         "cached": sum(v[1] for v in used.values())}
        return out

    def warn(self, fraction=0.8):
        t = self.totals()
        if t["global"]["non_cached"] >= fraction * self.caps["global"]:
            return True
        return any(t.get(s, {}).get("non_cached", 0) >= fraction * c
                   for s, c in self.caps["scenarios"].items())


def _exec_rids(paths):
    """rids named on shim 'exec_start' lines (comma-joined in the last field)."""
    rids = set()
    for path in paths:
        try:
            with open(path, errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) == 6 and parts[0] == "exec_start":
                rids.update(r for r in parts[5].split(",") if r and r != "-")
    return rids


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
