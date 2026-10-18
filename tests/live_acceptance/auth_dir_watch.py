"""Metadata-only watch over CLI auth directories.

A live run symlinks the real CLI auth entries (.codex, .claude, .claude.json)
into a per-run HOME so the CLIs can authenticate. This module lists those
entries by metadata only (size and mtime_ns; file contents are never opened)
and classifies what changed between two listings.

Caller contract: `listing` takes REAL paths (for example from
run_dir.real_home()), never the per-run HOME symlinks. A symlinked root raises
ValueError. Symlinks found inside a root are recorded with their readlink
target and never followed.

Buckets produced by `classify`. Config/state/auth matching only applies at
top-level depth (the root itself or one level below it); deeper files fall
through to other_changed.
- config_changed: config/settings files (config.toml, settings*.json).
  Flagged: an unexpected change fails the run.
- state_changed: top-level .claude.json. Allowed and recorded, never config.
- auth_refreshed: auth files (auth*). A token refresh is expected; it is
  recorded separately and is not a failure.
- other_changed: everything else, such as session, history, log and cache
  files. Allowed.

High-volume directories (PRUNE_DIRS, at any depth) are not listed per file.
They collapse to one entry "<key>/" {count, max_mtime_ns, total_size}.
"""

from __future__ import annotations

import errno
import fnmatch
import json
import os
from pathlib import Path

CONFIG_PATTERNS: tuple[str, ...] = ("config.toml", "settings*.json")
STATE_PATTERNS: tuple[str, ...] = (".claude.json",)
AUTH_PATTERNS: tuple[str, ...] = ("auth*",)
PRUNE_DIRS: tuple[str, ...] = (
    "sessions",
    "projects",
    "todos",
    "shell-snapshots",
    "cache",
    "logs",
    "history",
)


def _entry_stat(entry: os.DirEntry) -> os.stat_result:
    # Single seam for stat so tests can simulate a file vanishing mid-walk.
    return entry.stat(follow_symlinks=False)


def _error_name(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno or 0, type(exc).__name__)


def _summarize(root: Path) -> dict[str, int]:
    """Count files under a pruned directory without listing them one by one."""
    summary = {"count": 0, "max_mtime_ns": 0, "total_size": 0}
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                            continue
                        st = _entry_stat(entry)
                    except OSError:
                        continue
                    summary["count"] += 1
                    summary["total_size"] += st.st_size
                    summary["max_mtime_ns"] = max(summary["max_mtime_ns"], st.st_mtime_ns)
        except OSError:
            continue
    return summary


def _walk_root(root: Path, out: dict[str, dict], prune_dirs: tuple[str, ...]) -> None:
    stack: list[tuple[Path, str]] = [(root, root.name)]
    while stack:
        current, prefix = stack.pop()
        try:
            it = os.scandir(current)
        except OSError as exc:
            out[prefix] = {"error": _error_name(exc)}
            continue
        with it:
            for entry in it:
                key = f"{prefix}/{entry.name}"
                try:
                    if entry.is_symlink():
                        st = _entry_stat(entry)
                        out[key] = {
                            "size": st.st_size,
                            "mtime_ns": st.st_mtime_ns,
                            "symlink": os.readlink(entry.path),
                        }
                    elif entry.is_dir(follow_symlinks=False):
                        if entry.name in prune_dirs:
                            out[key + "/"] = _summarize(Path(entry.path))
                        else:
                            stack.append((Path(entry.path), key))
                    else:
                        st = _entry_stat(entry)
                        out[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
                except OSError as exc:
                    out[key] = {"error": _error_name(exc)}


def listing(
    paths: list[Path], prune_dirs: tuple[str, ...] = PRUNE_DIRS
) -> dict[str, dict]:
    """Metadata map for every existing root. Raises ValueError on symlinked roots.

    Keys are "<root name>[/<relpath>]". Directories are not entries; files and
    symlinks inside them are. Vanished or unreadable entries become
    {"error": "<errno name>"}.
    """
    roots = [Path(p) for p in paths]
    for p in roots:
        if os.path.islink(p):
            raise ValueError(f"listing root must be a real path, not a symlink: {p}")
    out: dict[str, dict] = {}
    for p in roots:
        if not os.path.lexists(p):
            continue
        try:
            if os.path.isdir(p):
                _walk_root(p, out, prune_dirs)
            else:
                st = os.lstat(p)
                out[p.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        except OSError as exc:
            out[p.name] = {"error": _error_name(exc)}
    return out


def diff(before: dict, after: dict) -> dict[str, list[str]]:
    added = sorted(k for k in after if k not in before)
    removed = sorted(k for k in before if k not in after)
    changed = sorted(k for k in before if k in after and before[k] != after[k])
    return {"added": added, "removed": removed, "changed": changed}


def _matches(key: str, patterns: tuple[str, ...]) -> bool:
    name = key.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def classify(
    d: dict[str, list[str]],
    config_patterns: tuple[str, ...] = CONFIG_PATTERNS,
    state_patterns: tuple[str, ...] = STATE_PATTERNS,
    auth_patterns: tuple[str, ...] = AUTH_PATTERNS,
) -> dict[str, list[str]]:
    """Split a diff into config_changed, state_changed, auth_refreshed, other_changed.

    Config, state and auth matches count only at top-level depth (key has at
    most one "/"). Deeper matches are other_changed.
    """
    result: dict[str, list[str]] = {
        "config_changed": [],
        "state_changed": [],
        "auth_refreshed": [],
        "other_changed": [],
    }
    for kind in ("added", "removed", "changed"):
        for key in d.get(kind, []):
            top_level = key.count("/") <= 1
            if top_level and _matches(key, state_patterns):
                result["state_changed"].append(key)
            elif top_level and _matches(key, config_patterns):
                result["config_changed"].append(key)
            elif top_level and _matches(key, auth_patterns):
                result["auth_refreshed"].append(key)
            else:
                result["other_changed"].append(key)
    for values in result.values():
        values.sort()
    return result


def write_listing(path: Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
