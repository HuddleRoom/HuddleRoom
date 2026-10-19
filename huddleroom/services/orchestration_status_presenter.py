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
from huddleroom.models.orchestration_process import (
    PROCESS_TYPE_VALUES, OrchestrationAuthorityDecision, OrchestrationProcessRun,
)
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import (
    OrchestrationStatusActivity, OrchestrationStatusAttention, OrchestrationStatusCurrentState,
    OrchestrationStatusEvent, OrchestrationStatusLine, OrchestrationSupervisionCondition,
    OrchestrationWorkflowApplication, OrchestrationWorkflowFacts, OrchestrationWorkflowGuidance,
    PlanReviewRecord,
)
from huddleroom.services.orchestration_needs_you_presenter import (
    _PROCESS_LABELS, NeedsYouReferenceRequest, resolve_needs_you_references,
)

_PHASE_LABELS = {
    "baseline": "Baseline setup", "ready": "Baseline complete", "waiting_activation": "Waiting for start",
    "authorized": "Executing", "completed": "Completed", "planning": "Planning",
}
_TERMINAL = frozenset({"completed", "cancelled", "stopped"})
_REASON_LIMIT = 160
_ADMISSION_REASON = "Waiting for the goal to be admitted."
_APPLICATION_FAILED = "Application failed."
_APPLICATION_REASONS = {
    "Orchestration admission paused": _ADMISSION_REASON,
    "Baseline operation discarded by reset": "The baseline operation was discarded by a reset.",
    "database is locked": "The database was busy; it will be retried.",
    "source_process_missing": "The step that asked is missing.",
    "source_process_superseded": "The step that asked was replaced.",
    "source_process_did_not_advance": "The step that asked did not advance.",
}


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


def _plan_review_current(decision: OrchestrationAuthorityDecision, run: OrchestrationRun) -> bool:
    """A G3 plan-approval question applies only to the persisted ready review it was asked about."""
    if not decision.decision_key.startswith("roadmap_plan:"):
        return True
    try:
        identity = json.loads(decision.context or "null")["plan_review_identity"]
        raw = ((run.plan_state or {}).get("plan_reviews") or {})[str(identity["artifact_id"])]
        record = PlanReviewRecord.model_validate(raw)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return record.state == "ready" and record.identity.model_dump(mode="json") == identity


async def pending_owner_decisions(
    db: AsyncSession, *, goal: OrchestrationGoal, run: OrchestrationRun,
    decisions: Sequence[OrchestrationAuthorityDecision],
) -> list[OrchestrationAuthorityDecision]:
    """Applicable pending human decisions, filtered before ordering (asked_at DESC, id ASC)."""
    pending = [d for d in decisions if d.status == "pending" and d.authority == "human" and d.run_id == run.id]
    source_ids = {d.source_process_run_id for d in pending if d.source_process_run_id is not None}
    live_sources = set(await db.scalars(select(OrchestrationProcessRun.id).where(
        OrchestrationProcessRun.id.in_(source_ids), OrchestrationProcessRun.goal_id == goal.id,
        OrchestrationProcessRun.run_id == run.id,
        OrchestrationProcessRun.superseded_by_id.is_(None),
    ))) if source_ids else set()
    applicable = [d for d in pending if (d.source_process_run_id is None or d.source_process_run_id in live_sources)
                  and _plan_review_current(d, run)]
    applicable.sort(key=lambda d: str(d.id))
    applicable.sort(key=lambda d: _time(d.asked_at or d.created_at) or datetime.min.replace(tzinfo=timezone.utc),
                    reverse=True)
    return applicable


def _unapplied_answers(actions: Sequence[OrchestrationAction]) -> list[OrchestrationWorkflowApplication]:
    """Saved (G1) answers whose application has not landed, in drain order."""
    rows = []
    for action in sorted(actions, key=lambda a: (_time(a.created_at) or datetime.min.replace(tzinfo=timezone.utc), str(a.id))):
        if action.action_type != "apply_authority_answer" or action.status not in {"reserved", "failed"}:
            continue
        contract = action.dispatch_contract if isinstance(action.dispatch_contract, dict) else {}
        try:
            ids = [uuid.UUID(str(i)) for i in action.request["submission"]["decision_ids"]]
        except (KeyError, TypeError, ValueError):
            continue
        state = contract.get("application_state")
        state = state if state in {"blocked", "failed"} else ("failed" if action.status == "failed" else "pending")
        reason = contract.get("reason")
        reason = _APPLICATION_REASONS.get(reason, _APPLICATION_FAILED) if reason else None
        rows.append(OrchestrationWorkflowApplication(
            action_id=action.id, decision_ids=ids, state=state, reason=reason,
        ))
    return rows


async def _operation(
    db: AsyncSession, *, goal: OrchestrationGoal, run: OrchestrationRun, actions: Sequence[OrchestrationAction],
) -> dict:
    """Qualified recorded operation (planner task or active process). Never claims live work."""
    tail = "Live activity unavailable."
    plan = run.plan_state if isinstance(run.plan_state, dict) else {}
    pending_replan = plan.get("pending_replan") if isinstance(plan.get("pending_replan"), dict) else {}
    if run.phase != "completed" and (pending_replan or plan.get("status") in {"requested", "revision_requested"}):
        task_id = _id(pending_replan.get("task_id") if pending_replan else plan.get("planning_task_id"))
        action_id = _id(pending_replan.get("action_id") if pending_replan else plan.get("request_action_id"))
        request = next((a for a in actions if a.id == action_id), None) if action_id else None
        task = await db.get(Task, task_id) if task_id else None
        orchestration = ((task.metadata_ or {}).get("orchestration") if task is not None
                         and isinstance(task.metadata_, dict) else None)
        orchestration = orchestration if isinstance(orchestration, dict) else {}
        bound = (task is not None and request is not None and task.project_id == goal.project_id
                 and orchestration.get("run_id") == str(run.id) and orchestration.get("work_function") == "planning"
                 and orchestration.get("action_id") == str(action_id))
        if not bound:
            return {"phase": "planning", "summary": f"Planning was requested; planner task unavailable. {tail}"}
        reference = (await resolve_needs_you_references(
            db, project_id=goal.project_id, requests=[NeedsYouReferenceRequest("task", task.id, goal.id)]))[0]
        return {"phase": "planning", "task": reference, "occurred_at": request.created_at,
                "summary": f"Last recorded: planning requested ({reference.label or 'planner task'}). "
                           f"Result not yet recorded; {tail[0].lower() + tail[1:]}"}
    rows = (await db.scalars(select(OrchestrationProcessRun).where(
        OrchestrationProcessRun.goal_id == goal.id, OrchestrationProcessRun.run_id == run.id,
        OrchestrationProcessRun.process_type.in_(PROCESS_TYPE_VALUES),
        OrchestrationProcessRun.superseded_by_id.is_(None),
        OrchestrationProcessRun.status.in_(("running", "waiting_decision")),
    ).order_by(OrchestrationProcessRun.started_at, OrchestrationProcessRun.id).limit(len(PROCESS_TYPE_VALUES)))).all()
    if not rows:
        return {"summary": f"No recorded operation. {tail}"}
    row = min(rows, key=lambda r: PROCESS_TYPE_VALUES.index(r.process_type))
    name = _PROCESS_LABELS.get(row.process_type, row.process_type.replace("_", " ")).lower()
    reference = (await resolve_needs_you_references(
        db, project_id=goal.project_id, requests=[NeedsYouReferenceRequest("process", row.id, goal.id)]))[0]
    state = ("waiting for a decision" if row.status == "waiting_decision"
             else "result not yet recorded")
    return {"process": reference, "row": row, "occurred_at": row.started_at,
            "summary": f"Last recorded: {name}. {state[0].upper() + state[1:]}; "
                       f"{tail[0].lower() + tail[1:]}"}


async def build_workflow_facts(
    db: AsyncSession, *, goal: OrchestrationGoal, run: OrchestrationRun, actions: Sequence[OrchestrationAction],
    condition: str = "unknown", has_next_check: bool = False,
    pending: Sequence[OrchestrationAuthorityDecision] | None = None,
) -> OrchestrationWorkflowFacts:
    """Compose phase, qualified operation, pending decision, saved-answer state and guidance (read-only)."""
    if pending is None:
        pending = await pending_owner_decisions(db, goal=goal, run=run, decisions=(await db.scalars(
            select(OrchestrationAuthorityDecision).where(OrchestrationAuthorityDecision.run_id == run.id))).all())
    terminal = condition in _TERMINAL
    operation = ({"summary": f"Goal {condition}."} if terminal
                 else await _operation(db, goal=goal, run=run, actions=actions))
    phase = operation.get("phase") or run.phase
    chosen = None if terminal else (pending[0] if pending else None)
    applications = _unapplied_answers(actions)
    app = applications[0] if applications else None
    unauthorized = (run.phase == "baseline" and not run.baseline_authorized and not run.pause_requested
                    and goal.status in {"active", "blocked"} and run.status in {"running", "blocked"})

    def guide(kind, label, decision_id=None, reason=None):
        return OrchestrationWorkflowGuidance(kind=kind, label=label, decision_id=decision_id, reason=reason)

    if terminal:
        guidance = guide("none", "No action")
    elif condition == "paused":
        reason = ("An owner decision is waiting." if chosen else "A saved answer is waiting to be applied."
                  if app else None)
        guidance = guide("resume", "Resume goal", chosen.id if chosen else None, reason)
    elif chosen:
        guidance = guide("review_decision", "Answer the owner decision", chosen.id)
    elif (app and unauthorized and condition != "needs_attention"
          and (app.state == "pending" or app.reason == _ADMISSION_REASON)):
        guidance = guide("authorize_baseline", "Authorize remaining baseline", reason="Baseline is not authorized.")
    elif condition == "needs_attention":
        guidance = guide("review_blocker", "Resolve current blocker")
    elif app and app.state == "pending":
        guidance = guide("wait_application", "Saved — waiting to be applied", reason=app.reason)
    elif app:
        guidance = guide("review_blocker", "Resolve saved answer blocker", reason=app.reason or "Application failed.")
    elif unauthorized and condition in {"unknown", "waiting"}:
        guidance = guide("authorize_baseline", "Authorize baseline", reason="Baseline is not authorized.")
    elif condition == "needs_attention":
        guidance = guide("review_blocker", "Resolve current blocker")
    elif condition == "waiting":
        guidance = guide("none", "Wait for scheduled check" if has_next_check else "Await recorded event")
    elif condition == "working":
        guidance = guide("none", "Await worker result")
    else:
        guidance = guide("none", "No action")
    return OrchestrationWorkflowFacts(
        run_id=run.id, control_version=run.control_version, phase=phase,
        phase_label=_PHASE_LABELS.get(phase, phase.replace("_", " ").capitalize()),
        process=operation.get("process"), process_type=operation["row"].process_type if "row" in operation else None,
        process_status=operation["row"].status if "row" in operation else None, task=operation.get("task"),
        operation_summary=operation["summary"], occurred_at=_time(operation.get("occurred_at")),
        pending_decision_id=chosen.id if chosen else None,
        additional_pending_count=max(len(pending) - 1, 0) if chosen else 0,
        application=app, additional_application_count=max(len(applications) - 1, 0), guidance=guidance,
    )


async def build_orchestration_status_line(
    db: AsyncSession, *, project_id: uuid.UUID, goal: OrchestrationGoal, run: OrchestrationRun,
    actions: Sequence[OrchestrationAction], gates: Sequence[OrchestrationGate],
    evidence: Sequence[OrchestrationEvidence], open_waits: Sequence[OrchestrationWait],
    worker_facts: Sequence[Mapping[str, object]], workflow: OrchestrationWorkflowFacts | None = None,
) -> tuple[OrchestrationStatusLine, OrchestrationSupervisionCondition]:
    """Select current branches and the newest independently timed domain occurrence."""
    decisions = (await db.scalars(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
    ))).all()
    pending = await pending_owner_decisions(db, goal=goal, run=run, decisions=decisions)
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
    if control in _TERMINAL:
        pending = []
    reported = [activity for activity in activities if activity.state == "reported_active"]
    current_kind = control or ("working" if reported else "unknown")
    current_summary = {
        "completed": "Completed.", "cancelled": "Cancelled.", "stopped": "Stopped.", "paused": "Paused.",
        "unknown": "No active worker confirmed.",
    }.get(current_kind, " ".join(item.summary for item in reported))

    attention: list[tuple[int, datetime | None, OrchestrationStatusAttention]] = []
    for decision in pending:
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
    if workflow is None:
        workflow = await build_workflow_facts(
            db, goal=goal, run=run, actions=actions, condition=condition, pending=pending,
            has_next_check=bool(selected and selected.next_check_at),
        )
    if current_kind == "unknown" and (workflow.process or workflow.task or workflow.phase == "planning"):
        current_summary = workflow.operation_summary
    return OrchestrationStatusLine(
        current_state=OrchestrationStatusCurrentState(kind=current_kind, summary=current_summary, activities=activities),
        attention_summary=selected, last_material_event=events[0] if events else None, workflow=workflow,
    ), condition
