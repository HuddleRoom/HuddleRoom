import asyncio
from datetime import timedelta

from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.services.orchestration_service import OrchestrationService


async def _run(db, project):
    goal = OrchestrationGoal(project_id=project.id, objective="Keep moving", status="active", goal_type="outcome")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="authorized")
    db.add(run)
    await db.flush()
    return goal, run


def _diff_paths(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in a or key not in b:
                out.append(path)
            else:
                out.extend(_diff_paths(a[key], b[key], path))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for index, (left, right) in enumerate(zip(a, b)):
            out.extend(_diff_paths(left, right, f"{prefix}[{index}]"))
        return out
    if a != b:
        return [prefix or "<root>"]
    return []


async def test_decision_context_is_identical_across_two_idle_local_ticks(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()
    await service.tick(db_session, run.id, local_only=True)
    await db_session.refresh(run)
    ctx1 = await service._decision_context(db_session, goal, run)
    await asyncio.sleep(0.01)
    await service.tick(db_session, run.id, local_only=True)
    await db_session.refresh(run)
    ctx2 = await service._decision_context(db_session, goal, run)
    assert ctx1 == ctx2, _diff_paths(ctx1, ctx2)


async def test_decision_context_is_identical_across_two_reconcile_local_calls_with_different_now(db_session, test_project):
    goal, run = await _run(db_session, test_project)
    service = OrchestrationService()
    t0 = _utcnow()
    await service.supervision.reconcile_local(db_session, goal, run, now=t0)
    ctx1 = await service._decision_context(db_session, goal, run)
    await service.supervision.reconcile_local(db_session, goal, run, now=t0 + timedelta(seconds=60))
    ctx2 = await service._decision_context(db_session, goal, run)
    assert ctx1 == ctx2, _diff_paths(ctx1, ctx2)


def test_stable_supervision_state_strips_volatile_keys_and_keeps_semantic_keys():
    state = {
        "evaluated_at": "2026-10-08T10:00:00Z",
        "judgment_due_at": "2026-10-08T10:05:00Z",
        "judgment_in_flight": True,
        "judgment_dirty": True,
        "judgment_failures": 2,
        "last_event_id": "evt-1",
        "context_fingerprint": "abc",
        "last_pass": {"at": "2026-10-08T10:00:00Z", "outcome": "ok"},
        "needs_judgment": True,
        "verified_progress": ["step-1"],
        "verification_candidates": [{"id": "c1", "assessed_at": "2026-10-08T10:00:00Z", "score": 3}],
        "memory_upgrade_reconciled": False,
        "last_assessment": {"at": "2026-10-08T10:00:00Z", "outcome": "wait", "wait_id": "w1", "count": 2},
        "recovery": {
            "sessions": {
                "s1": {"classification": "stalled", "disposition": "retry", "scheduler_ready_at": "2026-10-08T10:00:00Z"},
            },
        },
    }
    stable = OrchestrationService._stable_supervision_state(state)
    assert stable == {
        "needs_judgment": True,
        "verified_progress": ["step-1"],
        "verification_candidates": [{"id": "c1", "score": 3}],
        "memory_upgrade_reconciled": False,
        "last_assessment": {"outcome": "wait", "wait_id": "w1", "count": 2},
        "recovery": {"sessions": {"s1": {"classification": "stalled", "disposition": "retry"}}},
    }
    # deep copy: mutating the result must not touch the source
    stable["verified_progress"].append("step-2")
    assert state["verified_progress"] == ["step-1"]


def test_stable_supervision_state_handles_none_and_empty():
    assert OrchestrationService._stable_supervision_state(None) is None
    assert OrchestrationService._stable_supervision_state({}) == {}
    assert OrchestrationService._stable_supervision_state([]) == []
