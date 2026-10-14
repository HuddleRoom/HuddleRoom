"""Input availability and content identity for artifacts, computed from current content."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from huddleroom.services.artifact_service import _hash_file
from huddleroom.services.orchestration_plan_ingestion import ingested_plan_hash

PRESENT = "present"
CONFIRMED_MISSING = "confirmed_missing"
UNINSPECTABLE = "uninspectable"

UNAVAILABLE = "artifact_identity_unavailable"
UNPINNED = "artifact_identity_unpinned"

# Directory hashing caps: above either, the directory is uninspectable.
MAX_DIR_FILES = 2000
MAX_DIR_BYTES = 64 * 1024 * 1024
GIT_TIMEOUT_SECONDS = 10

_PINNED_URL = re.compile(r"/(?:commit|blob)/([0-9a-fA-F]{40})(?:/|$)")
_SHA = re.compile(r"[0-9a-fA-F]{40}")


@dataclass(frozen=True)
class InputAvailability:
    state: str
    identity: str | None = None
    reason: str | None = None


def _missing(reason: str | None = None) -> InputAvailability:
    return InputAvailability(CONFIRMED_MISSING, None, reason)


def _unavailable(reason: str = UNAVAILABLE) -> InputAvailability:
    return InputAvailability(UNINSPECTABLE, None, reason)


def _within(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _hash_dir(root: Path, directory: Path) -> str:
    pairs: list[tuple[str, str]] = []
    total = 0

    def _raise(error: OSError) -> None:
        raise error

    for base, dirs, files in os.walk(directory, onerror=_raise):  # followlinks=False
        for name in dirs:
            full = Path(base) / name
            if full.is_symlink() and not _within(root, full.resolve()):
                raise PermissionError("path escapes workspace")
        for name in files:
            full = Path(base) / name
            real = full.resolve()
            if not _within(root, real):
                raise PermissionError("path escapes workspace")
            try:
                if not real.is_file():
                    raise PermissionError("not a regular file")
                total += real.stat().st_size
            except FileNotFoundError as exc:  # vanished mid-walk: not a confirmed absence
                raise RuntimeError("file vanished during hashing") from exc
            if len(pairs) + 1 > MAX_DIR_FILES or total > MAX_DIR_BYTES:
                raise OverflowError("directory over cap")
            try:
                file_hash = _hash_file(str(real))
            except FileNotFoundError as exc:
                raise RuntimeError("file vanished during hashing") from exc
            pairs.append((full.relative_to(directory).as_posix(), file_hash))
    digest = hashlib.sha256()
    for rel, file_hash in sorted(pairs):
        digest.update(f"{rel}\0{file_hash}\n".encode())
    return digest.hexdigest()


def _classify_path(path: str, workspace_root: str) -> InputAvailability:
    try:
        root = Path(workspace_root).resolve()
        if not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
            return _unavailable()
        candidate = Path(path)
        target = (candidate if candidate.is_absolute() else root / candidate).resolve()
        if not _within(root, target):
            return _unavailable()
        try:
            target.stat()
        except (FileNotFoundError, NotADirectoryError):
            return _missing()
        if target.is_dir():
            return InputAvailability(PRESENT, _hash_dir(root, target))
        if not target.is_file():
            return _unavailable()  # FIFO, device, ...
        return InputAvailability(PRESENT, _hash_file(str(target)))
    except FileNotFoundError:
        return _missing()  # top-level target vanished while hashing
    except (OSError, OverflowError, RuntimeError, ValueError):
        return _unavailable()


def _local_commit(metadata: dict, workspace_root: str) -> str | None:
    repo, commit = metadata.get("repo_path"), metadata.get("commit")
    if not isinstance(repo, str) or not isinstance(commit, str) or not _SHA.fullmatch(commit):
        return None
    try:
        root = Path(workspace_root).resolve()
        repo_path = (root / repo).resolve()
        if not _within(root, repo_path):
            return None
        result = subprocess.run(
            ["git", "-C", str(repo_path), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            env={k: v for k, v in os.environ.items() if k not in ("GIT_DIR", "GIT_WORK_TREE")},
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return commit.lower() if result.returncode == 0 else None


def _classify_url(url: str, metadata: dict, workspace_root: str) -> InputAvailability:
    match = _PINNED_URL.search(urlparse(url).path)
    if match:
        return InputAvailability(PRESENT, f"commit:{match.group(1).lower()}")
    commit = _local_commit(metadata, workspace_root)
    if commit:
        return InputAvailability(PRESENT, f"commit:{commit}")
    return _unavailable(UNPINNED)


def classify_input(artifact, workspace_root: str) -> InputAvailability:
    """Classify one artifact input; blocking IO, call via classify_input_async from async code."""
    if artifact.status == "deleted":
        return _missing()
    metadata = artifact.metadata_ or {}
    if artifact.path:
        return _classify_path(artifact.path, workspace_root)
    if artifact.url:
        return _classify_url(artifact.url, metadata, workspace_root)
    if artifact.artifact_type == "plan":
        if not metadata.get("plan_items"):
            return _missing()
        # Recomputed from stored content; the cached output_hash is never trusted.
        return InputAvailability(PRESENT, ingested_plan_hash(metadata))
    return _unavailable()


async def classify_input_async(artifact, workspace_root: str) -> InputAvailability:
    return await asyncio.to_thread(classify_input, artifact, workspace_root)


# --- binding a review/verification to the identities it saw ---------------------------------------------------

CHANGED = "changed"
MAX_REFERENCE_CHARS = 300


def pinned_reference(artifact, found: InputAvailability) -> str | None:
    """The immutable reference a reviewer and an approval consumer both receive; None when nothing is pinned."""
    if found.state != PRESENT or not found.identity:
        return None
    if artifact.path:
        reference = f"{artifact.path} (sha256 {found.identity})"
    elif artifact.url and _PINNED_URL.search(urlparse(artifact.url).path):
        reference = artifact.url
    elif artifact.url:
        reference = f"{(artifact.metadata_ or {}).get('repo_path')} at commit {found.identity.removeprefix('commit:')}"
    else:
        reference = f"stored {artifact.artifact_type} content (sha256 {found.identity})"
    return reference if len(reference) <= MAX_REFERENCE_CHARS else f"{reference[:MAX_REFERENCE_CHARS]}…"


async def _workspace_root(db, project_id) -> str | None:
    from huddleroom.models.project import Project

    project = await db.get(Project, project_id)
    return project.workspace_path if project is not None else None


async def identity_record(artifact, workspace_root: str | None) -> dict:
    # an unset workspace must not fall back to the process cwd
    found = await classify_input_async(artifact, workspace_root) if workspace_root else _unavailable()
    return {
        "artifact_id": str(artifact.id), "state": found.state, "identity": found.identity, "reason": found.reason,
        "reference": pinned_reference(artifact, found),
    }


async def bind_artifacts(db, project_id, artifacts) -> list[dict]:
    """Identity records for the artifacts a review/verification is dispatched with."""
    root = await _workspace_root(db, project_id)
    return [await identity_record(artifact, root) for artifact in artifacts]


async def recheck_binding(db, project_id, recorded) -> tuple[str, dict] | None:
    """Recompute every bound identity from current content. None when all still hold; else (state, record) for the
    first failure, by priority: CONFIRMED_MISSING (producer revision), UNINSPECTABLE (fail closed), CHANGED
    (approval no longer applies; review again)."""
    from huddleroom.models.artifact import Artifact

    root = await _workspace_root(db, project_id)
    failures: dict[str, dict] = {}
    for item in recorded or []:
        if not isinstance(item, dict):
            continue
        try:
            artifact = await db.get(Artifact, uuid.UUID(str(item.get("artifact_id"))))
        except ValueError:
            artifact = None
        if artifact is None or artifact.project_id != project_id:
            current = {"artifact_id": item.get("artifact_id"), "state": CONFIRMED_MISSING, "identity": None,
                       "reason": None, "reference": None}
        else:
            current = await identity_record(artifact, root)
        if current["state"] != PRESENT:
            failures.setdefault(current["state"], current)
        elif (item.get("state"), item.get("identity")) != (PRESENT, current["identity"]):
            failures.setdefault(CHANGED, current)
    for state in (CONFIRMED_MISSING, UNINSPECTABLE, CHANGED):
        if state in failures:
            return state, failures[state]
    return None
