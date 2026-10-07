from __future__ import annotations

import uuid
import os
import subprocess
from pathlib import Path
from sqlalchemy import delete, exists, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.models.session import Session
from huddleroom.models.event_log import EventLog
from huddleroom.models.meeting import (
    Meeting,
    MeetingActionItem,
    MeetingAgendaItem,
    MeetingDecision,
    MeetingEvent,
    MeetingParticipantSignal,
    MeetingRequest,
    MeetingTurn,
)
from huddleroom.models.graph import GraphRun, GraphRunStep, GraphRunTimeout
from huddleroom.models.channel import Channel
from huddleroom.models.message import Message
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationAgentSuggestion,
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.orchestration_process import (
    OrchestrationAgentReview,
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.models.memory_item import MemoryItem
from huddleroom.models.knowledge_item import KnowledgeItem
from huddleroom.models.artifact import Artifact, ArtifactWatcher
from huddleroom.models.hook import Hook
from huddleroom.models.optimization import CostMetric, Optimization, Pattern
from huddleroom.schemas.project import ProjectCreate, ProjectUpdate
from huddleroom.services.event_bus import emit_event


class ProjectService:
    @staticmethod
    def _git_output(path: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise HTTPException(status_code=409, detail="Roadmap staging boundary is not a Git worktree")
        return result.stdout.strip()

    @classmethod
    def _require_registered_worktree(cls, workspace: Path, candidate: Path) -> Path:
        """Accept only a distinct, existing worktree of the project's repository."""
        if candidate == workspace:
            raise HTTPException(status_code=409, detail="Roadmap worktree must differ from the project workspace")
        project_common = Path(cls._git_output(workspace, "rev-parse", "--git-common-dir"))
        candidate_common = Path(cls._git_output(candidate, "rev-parse", "--git-common-dir"))
        if not project_common.is_absolute():
            project_common = (workspace / project_common).resolve()
        if not candidate_common.is_absolute():
            candidate_common = (candidate / candidate_common).resolve()
        if project_common != candidate_common:
            raise HTTPException(status_code=409, detail="Roadmap worktree is not from the project repository")
        registered = {
            Path(line.removeprefix("worktree ")).resolve()
            for line in cls._git_output(workspace, "worktree", "list", "--porcelain").splitlines()
            if line.startswith("worktree ")
        }
        if candidate not in registered:
            raise HTTPException(status_code=409, detail="Roadmap worktree is not registered")
        return candidate

    async def lock_workspace_boundary(self, db: AsyncSession, project_id: uuid.UUID) -> None:
        """Serialize workspace changes with execution claims on every supported database."""
        result = await db.execute(
            update(Project).where(Project.id == project_id).values(id=Project.id)
        )
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Project not found")

    async def create(self, db: AsyncSession, data: ProjectCreate) -> Project:
        project = Project(
            name=data.name,
            description=data.description,
            workspace_path=data.workspace_path,
            config=data.config or {},
         )
        db.add(project)
        await db.flush()
        await emit_event(db, project.id, "project.created", {
            "project_id": str(project.id),
            "name": project.name,
        })
        return project

    async def get(self, db: AsyncSession, project_id: uuid.UUID) -> Project | None:
        result = await db.execute(select(Project).where(Project.id == project_id))
        return result.scalar_one_or_none()

    async def get_or_404(self, db: AsyncSession, project_id: uuid.UUID) -> Project:
        project = await self.get(db, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    async def require_runnable_project(self, db: AsyncSession, project_id: uuid.UUID) -> Path:
        """Return the verified workspace for an active project, or reject execution."""
        project = await db.get(Project, project_id, populate_existing=True)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if project.status != "active":
            raise HTTPException(status_code=409, detail={"code": "project_not_runnable", "reason": "project_inactive"})
        if not project.workspace_path:
            raise HTTPException(status_code=409, detail={"code": "project_not_runnable", "reason": "workspace_unset"})

        workspace = Path(project.workspace_path)
        if not workspace.is_absolute():
            raise HTTPException(status_code=409, detail={"code": "project_not_runnable", "reason": "workspace_invalid"})
        try:
            resolved_workspace = workspace.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail={"code": "project_not_runnable", "reason": "workspace_unavailable"}) from exc
        if not resolved_workspace.is_dir() or str(resolved_workspace) != project.workspace_path:
            raise HTTPException(status_code=409, detail={"code": "project_not_runnable", "reason": "workspace_invalid"})
        if not os.access(resolved_workspace, os.R_OK | os.W_OK | os.X_OK):
            raise HTTPException(status_code=409, detail={"code": "project_not_runnable", "reason": "workspace_unavailable"})
        return resolved_workspace

    async def require_roadmap_workspace(self, db: AsyncSession, project_id: uuid.UUID, boundary: dict) -> Path:
        """Resolve an already-existing reversible workspace; this never provisions or switches one."""
        workspace = await self.require_runnable_project(db, project_id)
        if not isinstance(boundary, dict) or not boundary.get("reversible"):
            raise HTTPException(status_code=409, detail="Roadmap staging boundary is not reversible")
        kind, identifier = boundary.get("type"), boundary.get("identifier")
        if not isinstance(identifier, str) or not identifier:
            raise HTTPException(status_code=409, detail="Roadmap staging boundary is unresolved")
        project = await self.get_or_404(db, project_id)
        config = project.config or {}
        if kind == "git_worktree":
            configured = (config.get("roadmap_worktrees") or {}).get(identifier)
            candidate = Path(configured or identifier)
            if not candidate.is_absolute():
                raise HTTPException(status_code=409, detail="Roadmap worktree is not configured")
            try:
                candidate = candidate.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise HTTPException(status_code=409, detail="Roadmap worktree is unavailable") from exc
            if not candidate.is_dir() or not os.access(candidate, os.R_OK | os.W_OK | os.X_OK):
                raise HTTPException(status_code=409, detail="Roadmap worktree is unavailable")
            return self._require_registered_worktree(workspace, candidate)
        if kind == "git_branch":
            current_branch = subprocess.run(
                ["git", "-C", str(workspace), "branch", "--show-current"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            if current_branch != identifier:
                raise HTTPException(status_code=409, detail="Roadmap branch is not the active workspace branch")
            return workspace
        raise HTTPException(status_code=409, detail="Roadmap staging boundary is unsupported by this adapter")

    async def require_frozen_roadmap_workspace(
        self, db: AsyncSession, project_id: uuid.UUID, boundary: dict, claimed_path: str,
    ) -> Path:
        """Revalidate the claim's exact path without consulting mutable workspace mappings."""
        await self.require_runnable_project(db, project_id)
        try:
            candidate = Path(claimed_path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail="Roadmap claimed worktree is unavailable") from exc
        if not candidate.is_dir() or not os.access(candidate, os.R_OK | os.W_OK | os.X_OK):
            raise HTTPException(status_code=409, detail="Roadmap claimed worktree is unavailable")
        project_workspace = await self.require_runnable_project(db, project_id)
        if (boundary or {}).get("type") == "git_worktree":
            return self._require_registered_worktree(project_workspace, candidate)
        if (boundary or {}).get("type") == "git_branch":
            branch = subprocess.run(["git", "-C", str(candidate), "branch", "--show-current"], capture_output=True, text=True, check=False).stdout.strip()
            if branch != boundary.get("identifier"):
                raise HTTPException(status_code=409, detail="Roadmap claimed branch drifted")
        return candidate

    async def list(
        self, db: AsyncSession, cursor: str | None = None, limit: int = 50
     ) -> tuple[list[Project], str | None]:
        from datetime import datetime
        from sqlalchemy import or_, and_
        query = select(Project).where(Project.status != "archived").order_by(Project.created_at.desc(), Project.id.desc()).limit(limit + 1)
        if cursor:
            try:
                cursor_dt_str, cursor_id_str = cursor.split("__", 1)
                cursor_dt = datetime.fromisoformat(cursor_dt_str)
                cursor_id = uuid.UUID(cursor_id_str)
            except (ValueError, AttributeError) as exc:
                raise HTTPException(status_code=400, detail="Invalid cursor") from exc
            query = query.where(
                or_(
                    Project.created_at < cursor_dt,
                    and_(Project.created_at == cursor_dt, Project.id < cursor_id),
                )
            )
        result = await db.execute(query)
        items = list(result.scalars().all())
        next_cursor = None
        if len(items) > limit:
            items = items[:limit]
            last = items[-1]
            next_cursor = f"{last.created_at.isoformat()}__{last.id}"
        return items, next_cursor

    async def update(self, db: AsyncSession, project_id: uuid.UUID, data: ProjectUpdate) -> Project:
        await self.lock_workspace_boundary(db, project_id)
        project = await self.get_or_404(db, project_id)
        if data.workspace_path is not None and data.workspace_path != project.workspace_path:
            if project.status != "active":
                raise HTTPException(status_code=409, detail="Cannot update workspace while project is not active")
            active_work = await db.execute(select(
                exists(select(Session.id).where(
                    Session.project_id == project.id, Session.status.in_(("pending", "running")),
                )),
                exists(select(Meeting.id).where(
                    Meeting.project_id == project.id, Meeting.status.in_(("preparing", "active", "concluding")),
                )),
                exists(select(GraphRun.id).where(
                    GraphRun.project_id == project.id, GraphRun.status.in_(("active", "paused")),
                )),
                exists(select(OrchestrationRun.id).join(OrchestrationGoal).where(
                    OrchestrationGoal.project_id == project.id,
                    OrchestrationRun.status.in_(("running", "blocked", "paused")),
                )),
            ))
            if any(active_work.one()):
                raise HTTPException(status_code=409, detail="Cannot update workspace while project work is active")
        if data.name is not None:
            project.name = data.name
        if data.description is not None:
            project.description = data.description
        if data.workspace_path is not None:
            project.workspace_path = data.workspace_path
        if data.config is not None:
            project.config = data.config
        await db.flush()
        return project

    async def archive(self, db: AsyncSession, project_id: uuid.UUID) -> Project:
        project = await self.get_or_404(db, project_id)
        project.status = "archived"
        await db.flush()
        return project

    async def reset(self, db: AsyncSession, project_id: uuid.UUID) -> dict[str, int]:
        """Delete project operational state without committing the caller's transaction."""
        counts: dict[str, int] = {}

        async def purge(name: str, model, condition) -> None:
            result = await db.execute(delete(model).where(condition))
            counts[name] = result.rowcount or 0

        meeting_ids = select(Meeting.id).where(Meeting.project_id == project_id)
        graph_run_ids = select(GraphRun.id).where(GraphRun.project_id == project_id)
        channel_ids = select(Channel.id).where(Channel.project_id == project_id)
        goal_ids = select(OrchestrationGoal.id).where(OrchestrationGoal.project_id == project_id)
        run_ids = select(OrchestrationRun.id).where(OrchestrationRun.goal_id.in_(goal_ids))
        artifact_ids = select(Artifact.id).where(Artifact.project_id == project_id)

        await purge("sessions", Session, Session.project_id == project_id)

        await purge("meeting_action_items", MeetingActionItem, MeetingActionItem.meeting_id.in_(meeting_ids))
        await purge("meeting_decisions", MeetingDecision, MeetingDecision.meeting_id.in_(meeting_ids))
        await purge("meeting_turns", MeetingTurn, MeetingTurn.meeting_id.in_(meeting_ids))
        await purge("meeting_agenda_items", MeetingAgendaItem, MeetingAgendaItem.meeting_id.in_(meeting_ids))
        await purge("meeting_events", MeetingEvent, MeetingEvent.meeting_id.in_(meeting_ids))
        await purge(
            "meeting_participant_signals",
            MeetingParticipantSignal,
            MeetingParticipantSignal.meeting_id.in_(meeting_ids),
        )
        await purge("meeting_requests", MeetingRequest, MeetingRequest.project_id == project_id)
        await purge("meetings", Meeting, Meeting.project_id == project_id)

        await purge(
            "graph_run_steps",
            GraphRunStep,
            GraphRunStep.graph_run_id.in_(graph_run_ids),
        )
        await purge(
            "graph_run_timeouts",
            GraphRunTimeout,
            GraphRunTimeout.graph_run_id.in_(graph_run_ids),
        )
        await purge("graph_runs", GraphRun, GraphRun.project_id == project_id)

        await purge("messages", Message, Message.channel_id.in_(channel_ids))
        await purge("channels", Channel, Channel.project_id == project_id)

        await purge("orchestration_evidence", OrchestrationEvidence, OrchestrationEvidence.run_id.in_(run_ids))
        await purge("orchestration_actions", OrchestrationAction, OrchestrationAction.run_id.in_(run_ids))
        await purge("orchestration_decisions", OrchestrationDecision, OrchestrationDecision.run_id.in_(run_ids))
        await purge("orchestration_gates", OrchestrationGate, OrchestrationGate.run_id.in_(run_ids))
        await purge(
            "orchestration_agent_suggestions",
            OrchestrationAgentSuggestion,
            OrchestrationAgentSuggestion.run_id.in_(run_ids),
        )
        await purge(
            "orchestration_memory_sections",
            OrchestrationMemorySection,
            OrchestrationMemorySection.project_id == project_id,
        )
        await purge("orchestration_warnings", OrchestrationWarning, OrchestrationWarning.goal_id.in_(goal_ids))
        await purge(
            "orchestration_authority_decisions",
            OrchestrationAuthorityDecision,
            OrchestrationAuthorityDecision.goal_id.in_(goal_ids),
        )
        await purge(
            "orchestration_agent_reviews",
            OrchestrationAgentReview,
            OrchestrationAgentReview.goal_id.in_(goal_ids),
        )
        await purge(
            "orchestration_process_runs",
            OrchestrationProcessRun,
            OrchestrationProcessRun.goal_id.in_(goal_ids),
        )
        await purge("orchestration_runs", OrchestrationRun, OrchestrationRun.goal_id.in_(goal_ids))
        await purge("orchestration_goals", OrchestrationGoal, OrchestrationGoal.project_id == project_id)

        await purge("memory_items", MemoryItem, MemoryItem.project_id == project_id)
        await purge("knowledge_items", KnowledgeItem, KnowledgeItem.project_id == project_id)
        await purge("artifact_watchers", ArtifactWatcher, ArtifactWatcher.artifact_id.in_(artifact_ids))
        await purge("artifacts", Artifact, Artifact.project_id == project_id)
        await purge("event_log", EventLog, EventLog.project_id == project_id)
        await purge("cost_metrics", CostMetric, CostMetric.project_id == project_id)
        await purge("optimizations", Optimization, Optimization.project_id == project_id)
        await purge("patterns", Pattern, Pattern.project_id == project_id)
        await purge("tasks", Task, Task.project_id == project_id)
        await db.execute(
            update(Hook)
            .where(Hook.project_id == project_id)
            .values(execution_count=0, error_count=0)
        )
        await db.flush()
        return counts
