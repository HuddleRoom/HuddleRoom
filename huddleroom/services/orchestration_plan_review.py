"""Readable, non-authoritative projections of canonical plan Artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any

from huddleroom.models.project import Project
from huddleroom.models.agent import Agent
from huddleroom.schemas.orchestration import PlanReviewIdentity, PlanReviewRecord
from huddleroom.services.orchestration_plan_ingestion import canonical_plan_content, ingested_plan_hash


RENDERER_VERSION = 1
EDIT_NOTICE = (
    "This file can be edited by you or agents. Approval applies to the version shown here. "
    "Edits require a revision before changed work can run."
)


class PlanReviewService:
    """Creates and checks an editable view without making it an Artifact input."""

    async def ensure(self, db, goal, run, artifact) -> PlanReviewRecord:
        state = run.plan_state if isinstance(run.plan_state, dict) else {}
        reviews = state.get("plan_reviews") if isinstance(state.get("plan_reviews"), dict) else {}
        stored = reviews.get(str(artifact.id))
        if isinstance(stored, dict):
            record = PlanReviewRecord.model_validate(stored)
            # An earlier observed mismatch is historical: matching bytes later cannot revive it.
            if record.state == "stale":
                return record
        else:
            record = None

        try:
            names = record.owner_names if record is not None else await self._owner_names(db, artifact)
            identity, markdown = self._identity(
                goal, run, artifact, review_cycle=(record.identity.review_cycle if record else 1), owner_names=names,
            )
        except ValueError:
            if record is not None:
                return self._stale(run, record, "canonical plan provenance changed")
            raise
        if record is not None and record.identity != identity:
            return self._stale(run, record, "canonical plan content changed")
        project = await db.get(Project, goal.project_id)
        export_path, reason = self._export(project.workspace_path if project else None, identity, markdown)
        current = PlanReviewRecord(
            identity=identity,
            state="ready" if reason is None else "stale",
            export_path=export_path,
            markdown=markdown,
            owner_names=names,
            reason=reason,
        )
        self._store(run, current)
        return current

    async def check(self, db, goal, run, artifact, *, expected: PlanReviewIdentity | None = None,
                    boundary: str) -> PlanReviewRecord:
        """Fail closed on an editable-copy mismatch; callers own commit/publication ordering."""
        record = await self.ensure(db, goal, run, artifact)
        if expected is not None and expected != record.identity:
            return self._stale(run, record, f"{boundary}: review candidate changed")
        if record.state == "stale":
            return record
        project = await db.get(Project, goal.project_id)
        reason = self._inspect(project.workspace_path if project else None, record)
        return self._stale(run, record, reason) if reason else record

    def _identity(self, goal, run, artifact, *, review_cycle: int, owner_names: dict[str, str] | None = None) -> tuple[PlanReviewIdentity, str]:
        metadata = artifact.metadata_ if isinstance(artifact.metadata_, dict) else {}
        canonical_hash = ingested_plan_hash(metadata)
        if artifact.content_hash != canonical_hash or metadata.get("output_hash") != canonical_hash:
            raise ValueError("canonical plan metadata is invalid")
        planning_task_id = metadata.get("planning_task_id") or artifact.linked_task_id
        planning_version = metadata.get("planning_version")
        if planning_task_id is None or not isinstance(planning_version, int) or planning_version < 1:
            raise ValueError("canonical plan provenance is incomplete")
        if artifact.linked_task_id is not None and str(planning_task_id) != str(artifact.linked_task_id):
            raise ValueError("canonical plan task provenance is invalid")
        if metadata.get("plan_author_agent_id") != str(artifact.created_by_agent):
            raise ValueError("canonical plan author provenance is invalid")
        provenance = {
            "artifact_id": str(artifact.id), "planning_task_id": str(planning_task_id),
            "planning_version": planning_version, "winning_session_id": metadata.get("winning_session_id"),
            "plan_author_agent_id": metadata.get("plan_author_agent_id"),
            "canonical_content_hash": canonical_hash,
        }
        recorded = (run.plan_state or {}).get("plan_provenance")
        if isinstance(recorded, dict) and recorded != provenance:
            raise ValueError("canonical plan provenance does not match ingestion")
        provisional = PlanReviewIdentity(
            project_id=goal.project_id, goal_id=goal.id, run_id=run.id, artifact_id=artifact.id,
            planning_task_id=planning_task_id, planning_version=planning_version,
            winning_session_id=metadata.get("winning_session_id"),
            provenance_hash=_sha(json.dumps(provenance, sort_keys=True, separators=(",", ":"))),
            canonical_content_hash=canonical_hash, renderer_version=RENDERER_VERSION,
            rendered_content_hash="0" * 64, review_cycle=review_cycle,
        )
        markdown = render_markdown(provisional, metadata, owner_names or {})
        return provisional.model_copy(update={"rendered_content_hash": _sha(markdown)}), markdown

    def _export(self, workspace: str | None, identity: PlanReviewIdentity, markdown: str) -> tuple[str | None, str | None]:
        if not workspace:
            return None, "project workspace is unavailable"
        try:
            root = Path(workspace).resolve(strict=True)
            if not root.is_dir():
                return None, "project workspace is unavailable"
            relative = Path("plans") / f"goal-{identity.goal_id}" / f"run-{identity.run_id}" / (
                f"plan-v{identity.planning_version}-review-{identity.review_cycle}.md"
            )
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            parent_fd = os.open(root, directory_flags)
            try:
                for part in relative.parent.parts:
                    try:
                        child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
                    except FileNotFoundError:
                        os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                        child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
                    except OSError:
                        return None, "plan export path is unsafe"
                    os.close(parent_fd)
                    parent_fd = child_fd
                target_name = relative.name
                payload = markdown.encode("utf-8")
                try:
                    existing_fd = os.open(target_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
                except FileNotFoundError:
                    existing_fd = None
                if existing_fd is not None:
                    try:
                        if not stat.S_ISREG(os.fstat(existing_fd).st_mode):
                            return None, "plan export is uninspectable"
                        existing = os.read(existing_fd, len(payload) + 1)
                    finally:
                        os.close(existing_fd)
                    return (relative.as_posix(), None) if existing == payload else (relative.as_posix(), "plan export diverged")
                temporary_name = f".plan-review-{os.urandom(16).hex()}"
                temporary_fd = os.open(temporary_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent_fd)
                try:
                    _write_all(temporary_fd, payload)
                    os.fsync(temporary_fd)
                finally:
                    os.close(temporary_fd)
                try:
                    os.link(temporary_name, target_name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
                except FileExistsError:
                    existing_fd = os.open(target_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
                    try:
                        if not stat.S_ISREG(os.fstat(existing_fd).st_mode):
                            return None, "plan export is uninspectable"
                        existing = os.read(existing_fd, len(payload) + 1)
                    finally:
                        os.close(existing_fd)
                    return (relative.as_posix(), None) if existing == payload else (relative.as_posix(), "plan export diverged")
                finally:
                    os.unlink(temporary_name, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
            return relative.as_posix(), None
        except (OSError, ValueError):
            return None, "plan export is uninspectable"

    def _inspect(self, workspace: str | None, record: PlanReviewRecord) -> str | None:
        if not workspace or not record.export_path:
            return record.reason or "plan export is unavailable"
        try:
            root = Path(workspace).resolve(strict=True)
            candidate = root / record.export_path
            if candidate.is_symlink():
                return "plan export is uninspectable"
            target = candidate.resolve(strict=True)
            if root not in target.parents or not target.is_file():
                return "plan export is uninspectable"
            return None if _sha(target.read_text(encoding="utf-8")) == record.identity.rendered_content_hash else "plan export diverged"
        except (OSError, UnicodeError, ValueError):
            return "plan export is uninspectable"

    def _stale(self, run, record: PlanReviewRecord, reason: str) -> PlanReviewRecord:
        stale = record.model_copy(update={"state": "stale", "reason": reason})
        self._store(run, stale)
        return stale

    @staticmethod
    def _store(run, record: PlanReviewRecord) -> None:
        state = dict(run.plan_state or {})
        reviews = dict(state.get("plan_reviews") or {})
        reviews[str(record.identity.artifact_id)] = record.model_dump(mode="json")
        run.plan_state = {**state, "plan_reviews": reviews}

    @staticmethod
    async def _owner_names(db, artifact) -> dict[str, str]:
        names: dict[str, str] = {}
        for item in (artifact.metadata_ or {}).get("plan_items") or []:
            agent_id = item.get("agent_id") if isinstance(item, dict) else None
            if not agent_id or str(agent_id) in names:
                continue
            try:
                agent = await db.get(Agent, uuid.UUID(str(agent_id)))
            except (TypeError, ValueError):
                agent = None
            if agent is not None and isinstance(agent.name, str):
                names[str(agent_id)] = agent.name
        return names


def render_markdown(identity: PlanReviewIdentity, metadata: dict[str, Any], owner_names: dict[str, str]) -> str:
    """Stable full projection; only stored envelope fields are rendered."""
    lines = [
        "# Plan candidate", "", EDIT_NOTICE, "",
        f"- Artifact: `{identity.artifact_id}`", f"- Planning task: `{identity.planning_task_id}`",
        f"- Planning version: {identity.planning_version}", f"- Canonical content SHA-256: `{identity.canonical_content_hash}`", "",
        "## Summary", "", _code_block(metadata.get("summary") or "", language="text"), "", "## Plan items", "",
    ]
    for index, item in enumerate(metadata.get("plan_items") or [], start=1):
        lines.extend([f"### Item {index}", ""])
        lines.append(f"- Owner: {_inline(_owner(item, owner_names))}")
        for label, key in (("Title", "title"), ("Owner type", "owner_type"),
                           ("Work function", "work_function"), ("Scope", "scope"), ("Deliverable", "deliverable"),
                           ("Dependencies", "depends_on"), ("Success evidence", "success_evidence"),
                           ("Required evidence", "required_evidence"), ("Forbidden work", "forbidden_work"),
                           ("Human prerequisite", "human_prerequisite")):
            lines.append(f"- {label}: {_inline(item.get(key))}")
        lines.append("")
    lines.extend(["## Canonical JSON appendix", "", _code_block(canonical_plan_content(metadata))])
    return "\n".join(lines).rstrip() + "\n"


def _inline(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _owner(item: dict[str, Any], owner_names: dict[str, str]) -> str:
    agent_id = item.get("agent_id")
    if agent_id:
        identifier = str(agent_id)
        return f"{owner_names[identifier]} ({identifier})" if identifier in owner_names else f"Unresolved agent ({identifier})"
    prerequisite = item.get("human_prerequisite")
    if isinstance(prerequisite, dict) and prerequisite.get("owner_user_id"):
        return f"Human prerequisite ({prerequisite['owner_user_id']})"
    return "Unassigned agent (blocked)"


def _code_block(value: Any, *, language: str = "json") -> str:
    content = value if isinstance(value, str) and language == "text" else json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)
    fence = "`" * (max((len(run) for run in re.findall(r"`+", content)), default=2) + 1)
    return f"{fence}{language}\n{content}\n{fence}"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_all(fd: int, payload: bytes) -> None:
    while payload:
        written = os.write(fd, payload)
        if written <= 0:
            raise OSError("could not write plan review")
        payload = payload[written:]
