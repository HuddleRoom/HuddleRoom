"""Read-only, bounded evidence for recovery analysis."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from copy import deepcopy
from datetime import date, datetime
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
    OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationProcessRun, OrchestrationWarning
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration_recovery import (
    EvidenceV1,
    RecoveryEvidenceQueryV1,
    RecoveryTriggerV1,
    ToolManifestV1,
)
from huddleroom.services.orchestration_agent_definition_review import accepted_work_functions
from huddleroom.services.orchestration_llm_decision_adapter import _redact_secrets
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService


RECOVERY_EVIDENCE_MAX_BYTES = 16_384
RECOVERY_CONTEXT_MAX_BYTES = 65_536
_SECRET_KEY = re.compile(r"(?:api[_-]?key|access[_-]?token|authorization|password|secret|token|bearer)", re.I)


def _missing() -> HTTPException:
    return HTTPException(status_code=404, detail="recovery evidence source not found")


def _json(value: Any) -> Any:
    """Convert rows to JSON while removing credentials before the size pass."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "hex"):
        return str(value)
    if isinstance(value, str):
        redacted = _redact_secrets(value)
        return "[redacted]" if redacted != value else redacted
    if isinstance(value, list):
        return [_json(item) for item in value]
    if isinstance(value, tuple):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items() if not _SECRET_KEY.search(str(key))}
    return deepcopy(value)


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, default=str).encode()


def _compact(value: Any, *, string_limit: int = 512, item_limit: int = 32) -> Any:
    if isinstance(value, str):
        return value[:string_limit]
    if isinstance(value, list):
        return [_compact(item, string_limit=string_limit, item_limit=item_limit) for item in value[:item_limit]]
    if isinstance(value, dict):
        return {
            key: _compact(value[key], string_limit=string_limit, item_limit=item_limit)
            for key in sorted(value)[:item_limit]
        }
    return value


def _bounded(value: dict[str, Any], limit: int, *, fixed_keys: tuple[str, ...] = (), normalize: bool = True) -> dict[str, Any]:
    if normalize:
        value = _json(value)
    raw = _encoded(value)
    if len(raw) <= limit:
        return value
    string_limit, item_limit = 512, 32
    while True:
        compact = (
            {key: _compact(value[key], string_limit=string_limit, item_limit=item_limit) for key in fixed_keys}
            if fixed_keys else _compact(value, string_limit=string_limit, item_limit=item_limit)
        )
        compact["truncated"] = True
        compact["omitted_bytes"] = max(1, len(raw) - len(_encoded(compact)))
        if len(_encoded(compact)) <= limit:
            return compact
        if string_limit <= 8 and item_limit <= 1:
            break
        string_limit = max(8, string_limit // 2)
        item_limit = max(1, item_limit // 2)
    if fixed_keys:
        compact = {key: compact.get(key, [] if key.endswith("s") else {}) for key in fixed_keys}
        compact.update(truncated=True, omitted_bytes=max(1, len(raw) - len(_encoded(compact))))
    return compact


def _row(row: Any, *fields: str) -> dict[str, Any]:
    return {field: _json(getattr(row, field)) for field in fields}


def _safe_process(row: OrchestrationProcessRun, *, json_safe: bool = True) -> dict[str, Any]:
    outputs = row.outputs if isinstance(row.outputs, dict) else {}
    safe_outputs = {
        key: outputs[key] for key in (
            "accepted", "fingerprint", "coverage_fingerprint", "gate_id", "gate_fingerprint",
            "team_structure_reviewed", "status", "gates",
        ) if key in outputs
    }
    value = {"id": row.id, "process_type": row.process_type, "status": row.status,
             "run_id": row.run_id, "outputs": safe_outputs}
    return _json(value) if json_safe else value


class _RecoveryScope:
    def __init__(self, goal: OrchestrationGoal, run: OrchestrationRun) -> None:
        self.goal = goal
        self.run = run


async def _scope(db: AsyncSession, trigger: RecoveryTriggerV1) -> _RecoveryScope:
    run = await db.get(OrchestrationRun, trigger.run_id)
    goal = await db.get(OrchestrationGoal, trigger.goal_id)
    if run is None or goal is None or run.id != trigger.run_id or run.goal_id != goal.id:
        raise _missing()
    return _RecoveryScope(goal, run)


class RecoveryEvidenceService:
    """Resolve only the finite, scoped recovery evidence surface."""

    async def query(self, db: AsyncSession, trigger: RecoveryTriggerV1, request: RecoveryEvidenceQueryV1) -> EvidenceV1:
        scope = await _scope(db, trigger)
        kind = request.kind
        if kind == "run_state":
            if request.run_id != scope.run.id:
                raise _missing()
            value = _bounded(_row(scope.run, "phase", "status", "active_blockers", "retry_state", "goal_contract_fingerprint"), RECOVERY_EVIDENCE_MAX_BYTES)
            return EvidenceV1(label="run state", value=value, source_kind="run", source_id=scope.run.id)
        if kind == "action":
            row = await db.get(OrchestrationAction, request.action_id)
            if row is None or row.run_id != scope.run.id:
                raise _missing()
            value = _bounded(_row(row, "action_type", "request", "status", "error", "target_type", "target_id", "decision_id"), RECOVERY_EVIDENCE_MAX_BYTES)
            return EvidenceV1(label="action", value=value, source_kind="action", source_id=row.id)
        if kind == "process":
            row = await OrchestrationProcessService().get_current(db, scope.goal.id, request.process_type)
            if row is None:
                raise _missing()
            value = _bounded(_safe_process(row), RECOVERY_EVIDENCE_MAX_BYTES)
            return EvidenceV1(label="process", value=value, source_kind="process", source_id=row.id)
        if kind == "team_configuration":
            value = await _team_configuration(db, scope)
            return EvidenceV1(label="team configuration", value=_bounded(value, RECOVERY_EVIDENCE_MAX_BYTES), source_kind="run", source_id=scope.run.id)
        if kind == "task":
            row = await db.get(Task, request.task_id)
            metadata = (row.metadata_ or {}).get("orchestration") if row is not None else None
            if row is None or row.project_id != scope.goal.project_id or not isinstance(metadata, dict) or metadata.get("run_id") != str(scope.run.id):
                raise _missing()
            value = _bounded(_row(row, "title", "description", "status", "assigned_to", "depends_on", "metadata_", "adapter_type_override"), RECOVERY_EVIDENCE_MAX_BYTES)
            return EvidenceV1(label="task", value=value, source_kind="task", source_id=row.id)
        if kind == "gate":
            row = await db.get(OrchestrationGate, request.gate_id)
            if row is None or row.run_id != scope.run.id:
                raise _missing()
            value = _bounded(_row(row, "success_criterion_key", "gate_type", "status", "required_evidence", "failure_reason"), RECOVERY_EVIDENCE_MAX_BYTES)
            return EvidenceV1(label="gate", value=value, source_kind="gate", source_id=row.id)
        if kind == "waits":
            rows = list((await db.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == scope.run.id, OrchestrationWait.status == "open").order_by(OrchestrationWait.created_at, OrchestrationWait.id))).all())
            value = _bounded({"waits": [_row(row, "id", "owner", "awaited_event", "due_recheck_at", "fallback") for row in rows]}, RECOVERY_EVIDENCE_MAX_BYTES)
            return EvidenceV1(label="open waits", value=value, source_kind="run", source_id=scope.run.id)
        if kind == "warning":
            row = await db.get(OrchestrationWarning, request.warning_id)
            if row is None or row.goal_id != scope.goal.id:
                raise _missing()
            value = _bounded(_row(row, "warning_type", "severity", "message", "active", "run_id", "related_gate_id", "related_action_id", "source_process_run_id"), RECOVERY_EVIDENCE_MAX_BYTES)
            return EvidenceV1(label="warning", value=value, source_kind="warning", source_id=row.id)
        if kind == "agent_execution":
            team = await _team_configuration(db, scope)
            if str(request.agent_id) not in set(team.get("agent_ids", [])) or await db.get(Agent, request.agent_id) is None:
                raise _missing()
            return EvidenceV1(label="agent execution", value={"status": "unproven", "agent_id": str(request.agent_id)}, source_kind="agent", source_id=request.agent_id)
        raise _missing()


async def _team_configuration(db: AsyncSession, scope: _RecoveryScope) -> dict[str, Any]:
    hierarchy = await OrchestrationProcessService().get_current(db, scope.goal.id, "team_hierarchy")
    team = (scope.goal.orchestrator_context or {}).get("team")
    accepted_hierarchy_id = None
    if not team and hierarchy is not None and hierarchy.status == "completed":
        team = hierarchy.outputs
        accepted_hierarchy_id = hierarchy.id
    team = team if isinstance(team, dict) else {}
    from huddleroom.services.orchestration_service import OrchestrationService
    agent_ids = sorted(OrchestrationService._roadmap_team_agent_ids(team) or set())
    roster = []
    for agent_id in agent_ids:
        try:
            agent = await db.get(Agent, UUID(agent_id))
        except (TypeError, ValueError):
            continue
        if agent is not None and agent.is_active:
            roster.append(_row(agent, "id", "name", "role", "provider", "model", "adapter_type", "cli_runtime"))
    return {
        "team": _json(team),
        "agent_ids": agent_ids,
        "roster": roster,
        "eligibility": _json(await accepted_work_functions(db, scope.goal.id)),
        "hierarchy_process_id": str(accepted_hierarchy_id) if accepted_hierarchy_id else None,
    }


def _attempt_summary(row: OrchestrationAction) -> dict[str, Any]:
    """Compact attempt identity: lineage/attempt and the sealed tools, never the full sealed request."""
    request, contract = row.request or {}, row.dispatch_contract or {}
    summary = {"lineage_id": contract.get("lineage_id"), "attempt": contract.get("attempt"), "kind": request.get("kind")}
    if isinstance(request.get("calls"), list):
        summary["calls"] = [{"call_id": call.get("call_id"), "tool": call.get("tool")}
                            for call in request["calls"] if isinstance(call, dict)]
    if row.action_type == "recover_blocker":
        summary["request"] = request
    return _json(summary)


class RecoveryContextBuilder:
    """Compose the fixed, durable recovery snapshot; no arbitrary model access."""

    async def build(self, db: AsyncSession, trigger: RecoveryTriggerV1, *, registry_manifest: Sequence[ToolManifestV1]) -> dict:
        scope = await _scope(db, trigger)
        source_error: dict[str, Any] = {"source_kind": trigger.source_kind, "source_id": str(trigger.source_id), "summary": trigger.summary}
        if trigger.source_kind == "failed_action":
            source = await db.get(OrchestrationAction, trigger.source_id)
            if source is None or source.run_id != scope.run.id:
                raise _missing()
            source_error["error"] = _json(source.error)
            source_error["status"] = source.status
        tasks = list((await db.scalars(select(Task).where(Task.project_id == scope.goal.project_id).order_by(Task.created_at, Task.id))).all())
        related_tasks = [row for row in tasks if isinstance((row.metadata_ or {}).get("orchestration"), dict) and (row.metadata_ or {})["orchestration"].get("run_id") == str(scope.run.id)]
        gates = list((await db.scalars(select(OrchestrationGate).where(OrchestrationGate.run_id == scope.run.id).order_by(OrchestrationGate.created_at, OrchestrationGate.id))).all())
        waits = list((await db.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == scope.run.id, OrchestrationWait.status == "open").order_by(OrchestrationWait.created_at, OrchestrationWait.id))).all())
        processes = list((await db.scalars(select(OrchestrationProcessRun).where(
            OrchestrationProcessRun.goal_id == scope.goal.id,
            OrchestrationProcessRun.superseded_by_id.is_(None),
        ).order_by(OrchestrationProcessRun.process_type, OrchestrationProcessRun.id))).all())
        warnings = await OrchestrationWarningService().list_warnings(db, scope.goal.id, active_only=True)
        lineage_run_ids = []
        lineage_run = scope.run
        while lineage_run.supersedes_run_id is not None:
            lineage_run = await db.get(OrchestrationRun, lineage_run.supersedes_run_id)
            if lineage_run is None:
                break
            lineage_run_ids.append(lineage_run.id)
        # The current run's attempts are the lineage too; a proposal belongs to this blocker only when
        # its sealed source matches the trigger's stable identity.
        source = {"kind": trigger.source_kind, "id": str(trigger.source_id), "blocker_code": trigger.blocker_code}
        actions = [row for row in (await db.scalars(select(OrchestrationAction).where(
            OrchestrationAction.run_id.in_([scope.run.id, *lineage_run_ids]),
            OrchestrationAction.action_type.in_(("recover_blocker", "recovery_proposal")),
        ).order_by(OrchestrationAction.created_at.desc(), OrchestrationAction.id))).all()
            if row.action_type == "recover_blocker" or row.run_id != scope.run.id
            or {key: str((row.request or {}).get("source", {}).get(key)) for key in source} == source]
        context = {
            "goal_contract": _row(scope.goal, "id", "objective", "success_criteria", "constraints", "budget", "status"),
            "active_run": _row(scope.run, "id", "phase", "status", "active_blockers", "retry_state", "goal_contract_fingerprint"),
            "source_error": source_error,
            "related_tasks": [_row(row, "id", "title", "status", "assigned_to", "metadata_") for row in related_tasks],
            "gates": [_row(row, "id", "success_criterion_key", "gate_type", "status", "failure_reason") for row in gates],
            "processes": [_safe_process(row) for row in processes],
            "waits": [_row(row, "id", "owner", "awaited_event", "due_recheck_at") for row in waits],
            "warnings": [_row(row, "id", "warning_type", "severity", "message", "run_id", "related_action_id") for row in warnings],
            "team_configuration": await _team_configuration(db, scope),
            "prior_lineage_attempts": [{**_row(row, "id", "run_id", "action_type", "status", "error"), **_attempt_summary(row)}
                                       for row in actions],
            "registry_manifest": [manifest.model_dump(mode="json") for manifest in sorted(registry_manifest, key=lambda item: (item.name, item.version))],
        }
        return _bounded(context, RECOVERY_CONTEXT_MAX_BYTES, fixed_keys=(
            "goal_contract", "active_run", "source_error", "related_tasks", "gates", "processes",
            "waits", "warnings", "team_configuration", "prior_lineage_attempts", "registry_manifest",
        ))
