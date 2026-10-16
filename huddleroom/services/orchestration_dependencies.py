"""Task dependency helpers for orchestration: a dependency is met only when its gate is accepted."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.orchestration import OrchestrationGate, OrchestrationGoal
from huddleroom.models.task import Task


ADHOC_GATE_TYPE = "adhoc_work_accepted"
RETAINED_GATE_TYPE = "retained_evidence"
NON_BLOCKING_GATE_TYPES = frozenset({ADHOC_GATE_TYPE, RETAINED_GATE_TYPE})
ADHOC_GATED_WORK_FUNCTIONS = {"investigation": "discovery", "review": "readiness"}
PLAN_IN_FLIGHT_STATUSES = frozenset({"requested", "revision_requested", "revision_required"})
EVIDENCE_ROLES = frozenset({"discovery", "readiness", "delivery", "verification", "integration"})
REVIEW_STAGES = frozenset({"preliminary", "final"})
# "plan_accepted" is orchestration_service.PLAN_GATE_TYPE (not imported: that module imports this one).
READINESS_GATE_TYPES = frozenset({"plan_accepted", "roadmap_plan_approval"})


def evidence_semantics(
    gate_type: str, required_evidence: Any, source_type: str | None, verdict: str, run_terminal_reason: str | None,
) -> dict:
    """Derived evidence metadata; declared gate values win, legacy gates get type-based defaults."""
    declared = _obj(required_evidence)
    role = declared.get("evidence_role")
    if role not in EVIDENCE_ROLES:
        role = (
            "integration" if gate_type == "roadmap_integration"
            else "readiness" if gate_type in READINESS_GATE_TYPES
            else "verification" if source_type == "verification"
            else "delivery"
        )
    stage = declared.get("review_stage")
    if stage not in REVIEW_STAGES:
        stage = "final"
    if verdict == "candidate":
        verdict = "superseded" if run_terminal_reason == "reclassified" else "pending"
    return {"evidence_role": role, "review_stage": stage, "acceptance_verdict": verdict}


def non_final_gate(gate: OrchestrationGate) -> bool:
    """Discovery/readiness or preliminary-stage gate: it satisfies its own item but never covers a criterion."""
    sem = evidence_semantics(gate.gate_type, gate.required_evidence, None, "accepted", None)
    return sem["evidence_role"] in {"discovery", "readiness"} or sem["review_stage"] == "preliminary"


def accepted_plan_items(plan_state: Any) -> list[dict]:
    """Items of the accepted plan snapshot; empty when no plan is accepted."""
    state = _obj(plan_state)
    items = _obj(state.get("accepted_plan_snapshot")).get("items") if state.get("status") == "accepted" else None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def criterion_state(key: str, plan_items: list[dict], gates: list[OrchestrationGate]) -> str:
    """`accepted` or `pending`; the one acceptance predicate shared by the progress view and the final manifest.

    Accepted needs every plan item mapped to the criterion satisfied by its own accepted gate (an unreleased
    item has no gate), every gate linked to the criterion accepted, and at least one accepted final-stage
    gate among them: readiness/discovery evidence alone never makes a criterion accepted.
    """
    by_key = {g.success_criterion_key: g for g in gates}
    for item in plan_items:
        if key in (item.get("success_criterion_keys") or []):
            gate = by_key.get(f"plan_item:{item.get('id')}")
            if gate is None or gate.status != "accepted":
                return "pending"
    linked = [
        g for g in gates
        if g.success_criterion_key == key or key in (_obj(g.required_evidence).get("success_criterion_keys") or [])
    ]
    if linked and all(g.status == "accepted" for g in linked) and any(not non_final_gate(g) for g in linked):
        return "accepted"
    return "pending"


def plan_in_flight(plan_state: Any) -> bool:
    return isinstance(plan_state, dict) and plan_state.get("status") in PLAN_IN_FLIGHT_STATUSES


def _uuid(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value).strip()) if value is not None else None
    except ValueError:
        return None


def _obj(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def dependency_ids(task: Task) -> list[uuid.UUID]:
    raw = task.depends_on if isinstance(task.depends_on, list) else []
    return [d for d in (_uuid(v) for v in raw) if d is not None]


def task_gate_id(task: Task, run_id: uuid.UUID) -> uuid.UUID | None:
    metadata = _obj(task.metadata_)
    orchestration = _obj(metadata.get("orchestration"))
    if orchestration.get("run_id") != str(run_id):
        return None
    return (
        _uuid(orchestration.get("gate_id"))
        or _uuid(orchestration.get("plan_item_gate_id"))
        or _uuid(_obj(metadata.get("orchestration_plan_item")).get("plan_item_gate_id"))
    )


def gate_satisfies(gate: OrchestrationGate | None, run_id: uuid.UUID) -> bool:
    return gate is not None and gate.run_id == run_id and gate.status == "accepted"


async def _gate_for_dependency(db: AsyncSession, run, task_id: uuid.UUID) -> OrchestrationGate | None:
    dep = await db.get(Task, task_id)
    gate_id = task_gate_id(dep, run.id) if dep is not None else None
    return await db.get(OrchestrationGate, gate_id) if gate_id is not None else None


async def dependency_satisfied(db: AsyncSession, run, task_id: uuid.UUID) -> bool:
    return gate_satisfies(await _gate_for_dependency(db, run, task_id), run.id)


async def first_unmet_dependency(db: AsyncSession, run, task: Task) -> tuple[uuid.UUID, str] | None:
    for dep_id in dependency_ids(task):
        gate = await _gate_for_dependency(db, run, dep_id)
        if not gate_satisfies(gate, run.id):
            return dep_id, gate.status if gate is not None else "none"
    return None


def _invalid(detail: str) -> HTTPException:
    return HTTPException(status_code=409, detail=f"dependency_invalid: {detail}")


async def validate_dependencies(
    db: AsyncSession,
    run,
    dep_ids: list[uuid.UUID],
    *,
    self_id: uuid.UUID | None = None,
) -> None:
    if not dep_ids:
        return
    if self_id is not None and self_id in dep_ids:
        raise _invalid("task cannot depend on itself")
    goal = await db.get(OrchestrationGoal, run.goal_id)
    tasks = {t.id: t for t in (await db.execute(select(Task).where(Task.id.in_(dep_ids)))).scalars()}
    for dep_id in dep_ids:
        dep = tasks.get(dep_id)
        if dep is None:
            raise _invalid(f"task {dep_id} not found")
        if dep.project_id != goal.project_id:
            raise _invalid(f"task {dep_id} belongs to another project")
        if _obj(_obj(dep.metadata_).get("orchestration")).get("run_id") != str(run.id):
            raise _invalid(f"task {dep_id} belongs to another run")
        if task_gate_id(dep, run.id) is None:
            raise _invalid(f"task {dep_id} has no orchestration gate; only gated work can be a dependency")
    # Cycle check: walk predecessors; stored data should be acyclic, so a revisit on the path is corruption.
    async def walk(node_id: uuid.UUID, path: set[uuid.UUID], done: set[uuid.UUID]) -> None:
        if node_id == self_id or node_id in path:
            raise _invalid(f"dependency cycle through task {node_id}")
        if node_id in done:
            return
        node = tasks.get(node_id) or await db.get(Task, node_id)
        for nxt in dependency_ids(node) if node is not None else []:
            await walk(nxt, path | {node_id}, done)
        done.add(node_id)

    done: set[uuid.UUID] = set()
    for dep_id in dep_ids:
        await walk(dep_id, set(), done)
