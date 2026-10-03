from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.meeting import Meeting, MeetingDecision
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationEvidence, OrchestrationGate, OrchestrationRoadmapItem, OrchestrationRoadmapVersion, OrchestrationRun, OrchestrationWait
from huddleroom.models.orchestration_memory import OrchestrationMemorySection
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationProcessRun
from huddleroom.models.protocol import Protocol, ProtocolInstance
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.orchestration_steering import OrchestrationSteeringService


def _value(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value) if hasattr(value, "hex") else deepcopy(value)


def _row(row, *fields):
    return {field: _value(getattr(row, field)) for field in fields}


class OrchestrationSupervisionContextBuilder:
    """A deterministic, durable-only snapshot for orchestration decisions."""

    async def build(self, db: AsyncSession, goal, run: OrchestrationRun, *, body_limit: int | None = 1000) -> dict:
        if run.goal_id != goal.id:
            raise ValueError("orchestration run does not belong to goal")
        actions = list((await db.scalars(select(OrchestrationAction).where(OrchestrationAction.run_id == run.id).order_by(OrchestrationAction.created_at, OrchestrationAction.id))).all())
        action_task_ids = [row.target_id for row in actions if row.target_type == "task" and row.target_id]
        tasks = list((await db.scalars(select(Task).where(
            Task.id.in_(action_task_ids) if action_task_ids else False,
            Task.project_id == goal.project_id,
        ).order_by(Task.created_at, Task.id))).all())
        task_ids = [row.id for row in tasks]
        sessions = list((await db.scalars(select(Session).where(
            Session.task_id.in_(task_ids) if task_ids else False,
            Session.project_id == goal.project_id,
        ).order_by(Session.created_at, Session.id))).all())
        meetings = list((await db.scalars(select(Meeting).where(
            Meeting.source_task_id.in_(task_ids) if task_ids else False,
            Meeting.project_id == goal.project_id,
        ).order_by(Meeting.created_at, Meeting.id))).all())
        meeting_ids = [row.id for row in meetings]
        meeting_decisions = list((await db.scalars(select(MeetingDecision).where(MeetingDecision.meeting_id.in_(meeting_ids) if meeting_ids else False).order_by(MeetingDecision.created_at, MeetingDecision.id))).all())
        instances = list((await db.scalars(select(ProtocolInstance).where(
            ProtocolInstance.linked_task_id.in_(task_ids) if task_ids else False,
            ProtocolInstance.project_id == goal.project_id,
        ).order_by(ProtocolInstance.created_at, ProtocolInstance.id))).all())
        protocol_ids = [row.protocol_id for row in instances]
        protocols = list((await db.scalars(select(Protocol).where(
            Protocol.id.in_(protocol_ids) if protocol_ids else False,
            or_(Protocol.project_id == goal.project_id, Protocol.project_id.is_(None)),
        ).order_by(Protocol.created_at, Protocol.id))).all())
        gates = list((await db.scalars(select(OrchestrationGate).where(OrchestrationGate.run_id == run.id).order_by(OrchestrationGate.created_at, OrchestrationGate.id))).all())
        evidence = list((await db.scalars(select(OrchestrationEvidence).where(OrchestrationEvidence.run_id == run.id).order_by(OrchestrationEvidence.created_at, OrchestrationEvidence.id))).all())
        decisions = list((await db.scalars(select(OrchestrationAuthorityDecision).where(OrchestrationAuthorityDecision.goal_id == goal.id).order_by(OrchestrationAuthorityDecision.created_at, OrchestrationAuthorityDecision.id))).all())
        waits = list((await db.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run.id).order_by(OrchestrationWait.created_at, OrchestrationWait.id))).all())
        processes = list((await db.scalars(select(OrchestrationProcessRun).where(OrchestrationProcessRun.goal_id == goal.id).order_by(OrchestrationProcessRun.created_at, OrchestrationProcessRun.id))).all())
        versions = list((await db.scalars(select(OrchestrationRoadmapVersion).where(OrchestrationRoadmapVersion.goal_id == goal.id).order_by(OrchestrationRoadmapVersion.version, OrchestrationRoadmapVersion.id))).all())
        items = list((await db.scalars(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == goal.id).order_by(OrchestrationRoadmapItem.item_key, OrchestrationRoadmapItem.id))).all())
        memory_rows = list((await db.scalars(select(OrchestrationMemorySection).where(
            OrchestrationMemorySection.goal_id == goal.id,
            OrchestrationMemorySection.project_id == goal.project_id,
            OrchestrationMemorySection.fact_status != "superseded",
        ).order_by(OrchestrationMemorySection.always_load.desc(), OrchestrationMemorySection.toc_order, OrchestrationMemorySection.updated_at.desc(), OrchestrationMemorySection.id))).all())
        # Keep every unverified claim visible; only settled memory is bounded.
        memory = [row for row in memory_rows if row.fact_status == "unverified"]
        memory.extend([row for row in memory_rows if row.fact_status == "accepted"][:8])
        snapshot = {
            "goal": _row(goal, "id", "objective", "status", "goal_type", "success_criteria", "constraints"),
            "run": _row(run, "id", "goal_id", "status", "phase", "plan_state", "budget_state", "active_blockers"),
            "contracts": [{"action_id": str(row.id), "status": row.status, "contract": deepcopy(row.dispatch_contract)} for row in actions if row.dispatch_contract],
            "actions": [_row(row, "id", "action_type", "status", "request", "dispatch_contract", "budget_ledger", "error") for row in actions],
            "tasks": [_row(row, "id", "title", "status", "assigned_to", "depends_on", "metadata_") for row in tasks],
            "sessions": [_row(row, "id", "task_id", "agent_id", "status", "error", "origin") for row in sessions],
            "meetings": [_row(row, "id", "source_task_id", "title", "status", "meeting_type") for row in meetings],
            "meeting_decisions": [_row(row, "id", "meeting_id", "chosen_option", "is_vetoed", "is_partial") for row in meeting_decisions],
            "protocols": [_row(row, "id", "name", "version", "is_active", "definition") for row in protocols],
            "protocol_instances": [_row(row, "id", "protocol_id", "linked_task_id", "status", "current_state") for row in instances],
            "gates": [_row(row, "id", "gate_type", "success_criterion_key", "status", "required_evidence", "failure_reason") for row in gates],
            "evidence": [_row(row, "id", "gate_id", "source_type", "source_id", "producer_agent_id", "verdict", "evidence_metadata") for row in evidence],
            "authority_decisions": [_row(row, "id", "decision_key", "status", "authority", "question", "context", "options", "recommendation", "selected_option", "reason", "consequences", "runtime_identity", "contract_version", "continuation", "related_gate_id") for row in decisions],
            "waits": [_row(row, "id", "wait_key", "status", "owner", "awaited_event", "due_recheck_at", "fallback") for row in waits],
            "process_state": [_row(row, "id", "process_type", "status", "outputs", "superseded_by_id") for row in processes],
            "budget": deepcopy(run.budget_state),
            "memory": [{**_row(row, "id", "section_key", "title", "summary", "fact_status"), "provenance": deepcopy(row.provenance) if row.provenance else {"legacy": True, "section_id": str(row.id), "created_by": row.created_by}, "body": row.body if body_limit is None else row.body[:body_limit]} for row in memory],
            "roadmap": {
                "versions": [_row(row, "id", "run_id", "version", "snapshot", "approval_reference") for row in versions],
                "items": [_row(row, "id", "first_version_id", "item_key", "unit_type", "task_id", "child_goal_id", "gate_id", "completed_at") for row in items],
            },
        }
        steering = await OrchestrationSteeringService().context_snapshot(db, goal, run)
        if steering:
            snapshot["steering"] = steering
        return snapshot

    async def build_for_provider(self, db: AsyncSession, goal, run: OrchestrationRun, *, string_limit: int = 4000) -> dict:
        """Bound model input without mutating the complete durable snapshot."""
        return self.provider_snapshot(await self.build(db, goal, run, body_limit=None), string_limit=string_limit)

    async def fingerprint(self, db: AsyncSession, goal, run: OrchestrationRun) -> str:
        """Fingerprint the full state; truncation is only a provider transport concern."""
        return self.fingerprint_snapshot(await self.build(db, goal, run, body_limit=None))

    @staticmethod
    def fingerprint_snapshot(snapshot: dict) -> str:
        return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    @classmethod
    def provider_snapshot(cls, snapshot: dict, *, string_limit: int = 4000) -> dict:
        return cls._bound(deepcopy(snapshot), string_limit)

    @classmethod
    def _bound(cls, value, limit: int, key: str | None = None):
        if isinstance(value, str):
            return value[:limit]
        if isinstance(value, list):
            return [cls._bound(item, limit) for item in value]
        if isinstance(value, dict):
            return {name: cls._bound(item, limit, str(name)) for name, item in value.items()}
        return value
