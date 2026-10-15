"""Evidence-qualified, read-only current-operation projection."""
from __future__ import annotations

import uuid
import json
from datetime import datetime, timezone
from typing import Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import (
    OrchestrationAction, OrchestrationEvidence, OrchestrationGate, OrchestrationGoal,
    OrchestrationRun, OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.schemas.orchestration import (
    OrchestrationStatusActivity, OrchestrationStatusAttention, OrchestrationStatusCurrentState,
    OrchestrationStatusEvent, OrchestrationStatusLine, OrchestrationSupervisionCondition,
)
from huddleroom.services.orchestration_needs_you_presenter import (
    NeedsYouReferenceRequest, resolve_needs_you_references,
)


def _time(value: object) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _id(value: object) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


async def build_orchestration_status_line(
    db: AsyncSession, *, project_id: uuid.UUID, goal: OrchestrationGoal, run: OrchestrationRun,
    actions: Sequence[OrchestrationAction], gates: Sequence[OrchestrationGate],
    evidence: Sequence[OrchestrationEvidence], open_waits: Sequence[OrchestrationWait],
    worker_facts: Sequence[Mapping[str, object]],
) -> tuple[OrchestrationStatusLine, OrchestrationSupervisionCondition]:
    """Select current branches and the newest independently timed domain occurrence."""
    decisions = (await db.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
    ))).all()
    cleared = await db.scalar(select(OrchestrationWait).where(
        OrchestrationWait.run_id == run.id, OrchestrationWait.status == "cleared",
        OrchestrationWait.cleared_at.is_not(None),
    ).order_by(OrchestrationWait.cleared_at.desc(), OrchestrationWait.id.asc()).limit(1))
    requests: list[NeedsYouReferenceRequest] = []
    for fact in worker_facts:
        for kind, field in (("task", "task_id"), ("agent", "agent_id")):
            if (value := _id(fact.get(field))) is not None:
                requests.append(NeedsYouReferenceRequest(kind, value, goal.id))
    for gate in gates:
        requests.append(NeedsYouReferenceRequest("gate", gate.id, goal.id))
    for item in evidence:
        requests.append(NeedsYouReferenceRequest("gate", item.gate_id, goal.id))
    for action in actions:
        requests.append(NeedsYouReferenceRequest("action", action.id, goal.id))
    for decision in decisions:
        for kind, value in (("process", decision.source_process_run_id), ("gate", decision.related_gate_id),
                            ("action", decision.related_action_id)):
            if value is not None:
                requests.append(NeedsYouReferenceRequest(kind, value, goal.id))
    for blocker in run.active_blockers or []:
        if isinstance(blocker, dict):
            for kind, field in (("task", "task_id"), ("gate", "gate_id")):
                if (value := _id(blocker.get(field))) is not None:
                    requests.append(NeedsYouReferenceRequest(kind, value, goal.id))
    for wait in open_waits:
        owner = wait.owner if isinstance(wait.owner, dict) else {}
        if (value := _id(owner.get("task_id") or (owner.get("id") if owner.get("type") == "task" else None))):
            requests.append(NeedsYouReferenceRequest("task", value, goal.id))
    references = {(item.kind, item.id): item for item in await resolve_needs_you_references(
        db, project_id=project_id, requests=requests,
    )}

    current_gate_ids = {gate.id for gate in gates}

    def ref(kind: str, value: object):
        reference = references.get((kind, _id(value)))
        if reference is not None and kind == "gate" and reference.id not in current_gate_ids:
            return reference.model_copy(update={"goal_id": None})
        return reference

    def label(reference, fallback: str) -> str:
        return reference.label if reference is not None and reference.label else fallback

    activities = []
    for fact in worker_facts:
        if fact.get("session_status") not in {"pending", "running"}:
            continue
        observed = _time(fact.get("observed_at"))
        live = fact.get("observed_liveness") == "live" and observed is not None
        task = ref("task", fact.get("task_id"))
        agent = ref("agent", fact.get("agent_id"))
        subject_name = label(task, "a task")
        actor_name = label(agent, "A worker")
        activities.append(OrchestrationStatusActivity(
            key=f"session:{fact['session_id']}", state="reported_active" if live else "liveness_unknown",
            summary=(f"{actor_name} was reported working on {subject_name}." if live else
                     f"Liveness unknown for {actor_name} on {subject_name}."),
            subject=task, actor=agent, observed_at=observed if live else None,
        ))
    activities.sort(key=lambda row: row.key)
    stopped_at = _time((goal.continuous_state or {}).get("stopped_at")) if (
        goal.goal_type == "continuous" and isinstance(goal.continuous_state, dict)) else None
    control = ("completed" if goal.status == "completed" else "cancelled" if goal.status == "cancelled" else
               "stopped" if stopped_at else "paused" if goal.status == "paused" or run.status == "paused" else None)
    if control:
        activities = []
    reported = [activity for activity in activities if activity.state == "reported_active"]
    current_kind = control or ("working" if reported else "unknown")
    current_summary = {
        "completed": "Completed.", "cancelled": "Cancelled.", "stopped": "Stopped.", "paused": "Paused.",
        "unknown": "No active worker confirmed.",
    }.get(current_kind, " ".join(item.summary for item in reported))

    attention: list[tuple[int, datetime | None, OrchestrationStatusAttention]] = []
    for decision in decisions:
        if decision.status != "pending" or decision.authority != "human":
            continue
        subject = (ref("process", decision.source_process_run_id) or ref("gate", decision.related_gate_id)
                   or ref("action", decision.related_action_id))
        attention.append((0, _time(decision.asked_at or decision.created_at), OrchestrationStatusAttention(
            key=f"decision:{decision.id}:pending", kind="needs_you", summary="Owner decision needed.",
            subject=subject, unblock_actor="owner", unblock_actor_label="Owner",
            occurred_at=_time(decision.asked_at or decision.created_at),
        )))
    for blocker in run.active_blockers or []:
        if not isinstance(blocker, dict):
            continue
        subject = ref("task", blocker.get("task_id")) or ref("gate", blocker.get("gate_id"))
        reason = blocker.get("kind") if blocker.get("kind") in {"budget_measurement", "budget_integrity"} else None
        reason_summary = {
            "budget_measurement": "Budget measurement unavailable.",
            "budget_integrity": "Budget integrity needs review.",
        }.get(reason, "Reason unknown.")
        actor = "maintainer" if reason else "unknown"
        blocker_id = uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(blocker, sort_keys=True, default=str))
        attention.append((1, _time(blocker.get("occurred_at")), OrchestrationStatusAttention(
            key=f"blocker:{run.id}:{blocker_id}", kind="needs_attention",
            summary=f"{label(subject, 'Work')} needs attention. {reason_summary}",
            subject=subject, unblock_actor=actor, unblock_actor_label=actor.title() if reason else "Unknown",
            occurred_at=_time(blocker.get("occurred_at")),
        )))
    for wait in open_waits:
        owner = wait.owner if isinstance(wait.owner, dict) else {}
        subject = ref("task", owner.get("task_id") or (owner.get("id") if owner.get("type") == "task" else None))
        attention.append((2, _time(wait.created_at), OrchestrationStatusAttention(
            key=f"wait:{wait.id}:open", kind="waiting", summary=f"{label(subject, 'Work')} is waiting. Reason unknown.",
            subject=subject, unblock_actor="unknown", unblock_actor_label="Unknown",
            next_check_at=_time(wait.due_recheck_at), occurred_at=_time(wait.created_at),
        )))
    attention.sort(key=lambda row: (row[0], -(row[1].timestamp() if row[1] else float("-inf")), row[2].key))
    selected = attention[0][2].model_copy(update={"additional_count": len(attention) - 1}) if attention else None
    condition: OrchestrationSupervisionCondition = control or (selected.kind if selected else
        "working" if reported else "unknown")

    events: list[OrchestrationStatusEvent] = []

    def event(key: str, kind: str, summary: str, when: object, subject=None):
        if (occurred := _time(when)) is not None:
            events.append(OrchestrationStatusEvent(
                key=key, kind=kind, summary=summary, subject=subject, occurred_at=occurred,
            ))

    for decision in decisions:
        if decision.authority == "human":
            event(f"decision:{decision.id}:asked", "human_question", "Owner decision requested.",
                  decision.asked_at or decision.created_at)
            if decision.status == "answered":
                event(f"decision:{decision.id}:answered", "human_answer", "Owner decision answered.", decision.decided_at)
    for action in actions:
        event(f"action:{action.id}:reserved", "action_reserved", "Action reserved.", action.created_at,
              ref("action", action.id))
    for gate in gates:
        subject = ref("gate", gate.id)
        if gate.status in {"accepted", "failed"}:
            event(f"gate:{gate.id}:{gate.status}", f"{gate.status}_gate",
                  f"Verification {gate.status} for {label(subject, 'a criterion')}.",
                  gate.accepted_at if gate.status == "accepted" else gate.failed_at, subject)
    for item in evidence:
        if item.verdict == "accepted":
            subject = ref("gate", item.gate_id)
            event(f"evidence:{item.id}:accepted", "accepted_evidence",
                  f"Verification accepted for {label(subject, 'a criterion')}.", item.created_at, subject)
    for fact in worker_facts:
        for phase, field in (("started", "started_at"), ("ended", "ended_at")):
            event(f"session:{fact['session_id']}:{phase}", f"session_{phase}",
                  f"Worker session {phase} for {label(ref('task', fact.get('task_id')), 'a task')}.",
                  fact.get(field), ref("task", fact.get("task_id")))
    for wait in open_waits:
        event(f"wait:{wait.id}:open", "wait_opened", "Waiting for a recorded event.", wait.created_at)
    if cleared is not None:
        event(f"wait:{cleared.id}:cleared", "wait_cleared", "Recorded wait cleared.", cleared.cleared_at)
    if run.status in {"completed", "cancelled", "failed"}:
        event(f"run:{run.id}:{run.status}", f"run_{run.status}", f"Run {run.status}.", run.completed_at)
    if stopped_at:
        event(f"goal:{goal.id}:stopped", "stopped", "Continuous policy stopped.", stopped_at)
    events.sort(key=lambda row: (-row.occurred_at.timestamp(), row.kind, row.key))
    return OrchestrationStatusLine(
        current_state=OrchestrationStatusCurrentState(kind=current_kind, summary=current_summary, activities=activities),
        attention_summary=selected, last_material_event=events[0] if events else None,
    ), condition
