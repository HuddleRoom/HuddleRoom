import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationEvidence, OrchestrationGate, OrchestrationGoal
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_service import OrchestrationService


def _agent(name_prefix: str, role: str, capabilities: list[str]) -> Agent:
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config={},
        is_active=True,
    )


async def _make_run(db_session, project_id):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Ship gate validation",
            success_criteria=[{"key": "validated", "description": "Gates validate evidence."}],
            constraints={},
            budget={},
        ),
        created_by_user_id=None,
    )
    return service, goal, run


async def _make_gate(
    db_session,
    run_id,
    *,
    gate_type: str = "work_completed",
    required_evidence: dict | None = None,
) -> OrchestrationGate:
    gate = OrchestrationGate(
        run_id=run_id,
        success_criterion_key="plan_item:implement-api",
        gate_type=gate_type,
        required_evidence=required_evidence
        or {
            "required_source_types": ["task"],
            "min_count": 1,
            "plan_item_id": "implement-api",
        },
    )
    db_session.add(gate)
    await db_session.flush()
    return gate


async def _make_orchestrated_task(
    db_session,
    project_id,
    run_id,
    gate_id,
    agent_id,
) -> Task:
    task = Task(
        project_id=project_id,
        title="Implementation work",
        description="Work created from an accepted orchestration plan item.",
        status="in_progress",
        assigned_to=agent_id,
        metadata_={
            "orchestration": {
                "goal_id": "unused-by-gate-validation",
                "run_id": str(run_id),
                "work_function": "implementation",
                "plan_item_id": "implement-api",
                "plan_item_gate_id": str(gate_id),
                "expand_action_id": str(uuid.uuid4()),
            },
            "orchestration_plan_item": {
                "id": "implement-api",
                "work_function": "implementation",
                "accepted_plan_artifact_id": str(uuid.uuid4()),
            },
        },
    )
    db_session.add(task)
    await db_session.flush()
    return task


async def _add_evidence(
    db_session,
    run_id,
    gate_id,
    *,
    source_type: str,
    producer_agent_id,
    verdict: str = "candidate",
    event_seq: int,
    source_id=None,
    evidence_metadata: dict | None = None,
    created_at: datetime | None = None,
) -> OrchestrationEvidence:
    evidence = OrchestrationEvidence(
        run_id=run_id,
        gate_id=gate_id,
        source_type=source_type,
        source_id=source_id or uuid.uuid4(),
        observed_event_id=None,
        producer_agent_id=producer_agent_id,
        verdict=verdict,
        evidence_metadata=evidence_metadata or {"event_seq": event_seq, "source_type": source_type},
        created_at=created_at,
    )
    db_session.add(evidence)
    await db_session.flush()
    return evidence


async def _evidence_rows(db_session, gate_id):
    result = await db_session.execute(
        select(OrchestrationEvidence)
        .where(OrchestrationEvidence.gate_id == gate_id)
        .order_by(OrchestrationEvidence.created_at.asc(), OrchestrationEvidence.id.asc())
    )
    return list(result.scalars().all())


async def _override_events(db_session, project_id):
    result = await db_session.execute(
        select(EventLog)
        .where(
            EventLog.project_id == project_id,
            EventLog.event_type == "orchestration.gate_overridden",
        )
        .order_by(EventLog.seq.asc())
    )
    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_tick_keeps_gate_open_when_required_review_evidence_is_missing(db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(
        db_session,
        run.id,
        required_evidence={
            "required_source_types": ["task", "review"],
            "min_count": 2,
            "plan_item_id": "implement-api",
        },
    )
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    task.status = "done"
    task.completed_at = datetime.now(timezone.utc)
    await emit_event_once(
        db_session,
        test_project.id,
        "task.status_changed",
        {
            "task_id": str(task.id),
            "status": "done",
            "previous_status": "in_progress",
            "project_id": str(test_project.id),
        },
        dedup_key=f"phase14-task-done:{task.id}",
    )

    result = await service.tick(db_session, run.id)

    assert result["evidence_created"] == 1
    assert result["gates_validated"] == 0
    assert gate.status == "open"
    assert gate.failure_reason == "Missing evidence: review"
    assert run.status == "running"
    assert goal.status == "active"


@pytest.mark.asyncio
async def test_validate_open_gates_accepts_fresh_independent_evidence(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(
        db_session,
        run.id,
        required_evidence={
            "required_source_types": ["task", "review"],
            "min_count": 2,
            "requires_independent_agent": True,
        },
    )
    developer = _agent("developer", "developer", ["implementation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([developer, reviewer])
    await db_session.flush()
    task_evidence = await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        producer_agent_id=developer.id,
        event_seq=10,
    )
    review_evidence = await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        producer_agent_id=reviewer.id,
        event_seq=11,
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "accepted"
    assert gate.failure_reason is None
    assert gate.accepted_at is not None
    assert gate.failed_at is None
    assert task_evidence.verdict == "accepted"
    assert review_evidence.verdict == "accepted"


@pytest.mark.asyncio
async def test_validate_open_gates_uses_latest_evidence_for_required_source_type(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(
        db_session,
        run.id,
        required_evidence={
            "required_source_types": ["task", "review"],
            "min_count": 2,
            "requires_independent_agent": True,
        },
    )
    developer = _agent("developer", "developer", ["implementation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([developer, reviewer])
    await db_session.flush()
    task_evidence = await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        producer_agent_id=developer.id,
        event_seq=20,
    )
    stale_review = await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        producer_agent_id=reviewer.id,
        event_seq=19,
    )
    fresh_review = await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        producer_agent_id=reviewer.id,
        event_seq=21,
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "accepted"
    assert gate.failure_reason is None
    assert task_evidence.verdict == "accepted"
    assert fresh_review.verdict == "accepted"
    assert stale_review.verdict == "candidate"


@pytest.mark.asyncio
async def test_validate_open_gates_fails_same_agent_independent_verification(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(
        db_session,
        run.id,
        required_evidence={
            "required_source_types": ["task", "review"],
            "min_count": 2,
            "requires_independent_agent": True,
        },
    )
    developer = _agent("developer", "developer", ["implementation", "review"])
    db_session.add(developer)
    await db_session.flush()
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        producer_agent_id=developer.id,
        event_seq=10,
    )
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        producer_agent_id=developer.id,
        event_seq=11,
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "failed"
    assert gate.failed_at is not None
    assert gate.failure_reason == "Independent verification must come from a different agent"


@pytest.mark.asyncio
async def test_validate_open_gates_fails_stale_verification_evidence(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(
        db_session,
        run.id,
        required_evidence={
            "required_source_types": ["task", "review"],
            "min_count": 2,
        },
    )
    developer = _agent("developer", "developer", ["implementation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([developer, reviewer])
    await db_session.flush()
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        producer_agent_id=developer.id,
        event_seq=20,
    )
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        producer_agent_id=reviewer.id,
        event_seq=19,
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "failed"
    assert gate.failure_reason == "Evidence is stale"


@pytest.mark.asyncio
async def test_validate_open_gates_fails_equal_seq_verification_evidence(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(
        db_session,
        run.id,
        required_evidence={
            "required_source_types": ["task", "review"],
            "min_count": 2,
        },
    )
    developer = _agent("developer", "developer", ["implementation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([developer, reviewer])
    await db_session.flush()
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        producer_agent_id=developer.id,
        event_seq=20,
    )
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        producer_agent_id=reviewer.id,
        event_seq=20,
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "failed"
    assert gate.failure_reason == "Evidence is stale"


@pytest.mark.asyncio
async def test_validate_open_gates_fails_stale_verification_by_created_at_when_event_seq_missing(
    db_session,
    test_project,
):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(
        db_session,
        run.id,
        required_evidence={
            "required_source_types": ["task", "review"],
            "min_count": 2,
        },
    )
    developer = _agent("developer", "developer", ["implementation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([developer, reviewer])
    await db_session.flush()
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        producer_agent_id=developer.id,
        event_seq=20,
        evidence_metadata={"source_type": "task"},
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        producer_agent_id=reviewer.id,
        event_seq=19,
        evidence_metadata={"source_type": "review"},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "failed"
    assert gate.failure_reason == "Evidence is stale"


@pytest.mark.asyncio
async def test_validate_open_gates_fails_without_independent_verifier_evidence(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(
        db_session,
        run.id,
        required_evidence={
            "required_source_types": ["task"],
            "min_count": 1,
            "requires_independent_agent": True,
        },
    )
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        producer_agent_id=developer.id,
        event_seq=10,
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "failed"
    assert gate.failure_reason == "Independent verification evidence is missing"


@pytest.mark.asyncio
async def test_validate_open_gates_accepts_independent_verifier_from_configured_work_producer(
    db_session,
    test_project,
):
    service, _goal, run = await _make_run(db_session, test_project.id)
    developer = _agent("developer", "developer", ["implementation"])
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add_all([developer, reviewer])
    await db_session.flush()
    gate = await _make_gate(
        db_session,
        run.id,
        required_evidence={
            "required_source_types": ["review"],
            "min_count": 1,
            "requires_independent_agent": True,
            "work_producer_agent_id": str(developer.id),
        },
    )
    review_evidence = await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        producer_agent_id=reviewer.id,
        event_seq=11,
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "accepted"
    assert gate.failure_reason is None
    assert review_evidence.verdict == "accepted"


@pytest.mark.asyncio
async def test_validate_open_gates_fails_when_independent_verifier_producer_missing(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(
        db_session,
        run.id,
        required_evidence={
            "required_source_types": ["task", "review"],
            "min_count": 2,
            "requires_independent_agent": True,
        },
    )
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        producer_agent_id=developer.id,
        event_seq=10,
    )
    await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="review",
        producer_agent_id=None,
        event_seq=11,
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "failed"
    assert gate.failure_reason == "Independent verification producer is missing"


@pytest.mark.asyncio
async def test_validate_open_gates_fails_rejected_evidence(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    rejected = await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        producer_agent_id=developer.id,
        verdict="rejected",
        event_seq=10,
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "failed"
    assert gate.failed_at is not None
    assert gate.failure_reason == "Rejected evidence from task"
    assert rejected.verdict == "rejected"


@pytest.mark.asyncio
async def test_validate_open_gates_ignores_older_rejected_evidence_for_same_source(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    source_id = uuid.uuid4()
    rejected = await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        source_id=source_id,
        producer_agent_id=developer.id,
        verdict="rejected",
        event_seq=10,
    )
    latest = await _add_evidence(
        db_session,
        run.id,
        gate.id,
        source_type="task",
        source_id=source_id,
        producer_agent_id=developer.id,
        verdict="candidate",
        event_seq=11,
    )

    changed = await service.validate_open_gates(db_session, run.id)

    assert changed == 1
    assert gate.status == "accepted"
    assert gate.failure_reason is None
    assert rejected.verdict == "rejected"
    assert latest.verdict == "accepted"


@pytest.mark.asyncio
async def test_tick_ingests_task_evidence_and_accepts_single_source_gate(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    developer = _agent("developer", "developer", ["implementation"])
    db_session.add(developer)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, developer.id)
    task.status = "done"
    task.completed_at = datetime.now(timezone.utc)
    await emit_event_once(
        db_session,
        test_project.id,
        "task.status_changed",
        {
            "task_id": str(task.id),
            "status": "done",
            "previous_status": "in_progress",
            "project_id": str(test_project.id),
        },
        dedup_key=f"phase14-single-source-task:{task.id}",
    )

    result = await service.tick(db_session, run.id)
    rows = await _evidence_rows(db_session, gate.id)

    assert result["evidence_created"] == 1
    assert result["gates_validated"] == 1
    assert gate.status == "accepted"
    assert gate.accepted_at is not None
    assert rows[0].verdict == "accepted"


@pytest.mark.asyncio
async def test_human_override_accepts_failed_gate_with_reason(client, db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    gate.status = "failed"
    gate.failure_reason = "Independent verification must come from a different agent"
    gate.failed_at = datetime.now(timezone.utc)
    await db_session.flush()

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/override",
        json={
            "gate_id": str(gate.id),
            "decision": "accept",
            "reason": "Human reviewed the artifact and accepts the same-agent verification for this run.",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["gates"][0]["id"] == str(gate.id)
    assert body["gates"][0]["status"] == "accepted"
    assert body["gates"][0]["failure_reason"] is None
    rows = await _evidence_rows(db_session, gate.id)
    assert len(rows) == 1
    assert rows[0].source_type == "human_override"
    assert rows[0].verdict == "accepted"
    assert rows[0].evidence_metadata["reason"].startswith("Human reviewed")


@pytest.mark.asyncio
async def test_human_override_exact_replay_preserves_evidence_and_terminal_timestamp(
    client,
    db_session,
    test_project,
):
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    payload = {
        "gate_id": str(gate.id),
        "decision": "accept",
        "reason": "Human reviewed the failed gate and accepts it.",
    }

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/override",
        json=payload,
    )
    assert response.status_code == 200, response.text
    await db_session.refresh(gate)
    accepted_at = gate.accepted_at

    replay = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/override",
        json=payload,
    )

    assert replay.status_code == 200, replay.text
    await db_session.refresh(gate)
    rows = await _evidence_rows(db_session, gate.id)
    events = await _override_events(db_session, test_project.id)
    assert len(rows) == 1
    assert len(events) == 1
    assert rows[0].source_type == "human_override"
    assert gate.accepted_at == accepted_at


@pytest.mark.asyncio
async def test_human_override_repeated_transition_emits_event(client, db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    url = f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/override"

    accept = {
        "gate_id": str(gate.id),
        "decision": "accept",
        "reason": "Human accepts the gate.",
    }
    reject = {
        "gate_id": str(gate.id),
        "decision": "reject",
        "reason": "Human rejects the gate.",
    }

    for payload in (accept, reject, accept):
        response = await client.post(url, json=payload)
        assert response.status_code == 200, response.text

    await db_session.refresh(gate)
    rows = await _evidence_rows(db_session, gate.id)
    events = await _override_events(db_session, test_project.id)
    assert len(rows) == 1
    assert len(events) == 3
    assert gate.status == "accepted"
    assert gate.failure_reason is None


@pytest.mark.asyncio
async def test_human_override_rejects_completed_run(client, db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)
    run.status = "completed"
    run.completed_at = datetime.now(timezone.utc)
    await db_session.flush()

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/override",
        json={
            "gate_id": str(gate.id),
            "decision": "accept",
            "reason": "Human accepts the gate.",
        },
    )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_human_override_rejects_unknown_fields(client, db_session, test_project):
    service, goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id)

    response = await client.post(
        f"/api/v1/projects/{test_project.id}/orchestration/goals/{goal.id}/override",
        json={
            "gate_id": str(gate.id),
            "decision": "accept",
            "reason": "Human accepts the gate.",
            "unexpected": True,
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_synthesis_join_releases_once_only_after_every_audit_gate_is_accepted(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal
    from tests.test_orchestration_plan_to_work_expansion import _adopted_audit_plan

    from huddleroom.models.artifact import Artifact

    db_session.add(Artifact(
        id=uuid.UUID("00000000-0000-0000-0000-0000000000a1"), project_id=test_project.id,
        name="spec", artifact_type="document",
    ))
    await db_session.flush()
    service, goal, run, audits = await _adopted_audit_plan(
        db_session, test_project.id, audits=2,
        synthesis={"priority": 80, "why_runnable": "Both audits are accepted.",
                   "artifact_ids": ["00000000-0000-0000-0000-0000000000a1"]},
    )
    goal = await db_session.get(OrchestrationGoal, run.goal_id)

    async def release():
        return await service._release_ready_work(db_session, goal, run)

    async def synthesis_tasks():
        tasks = (await db_session.execute(select(Task).where(Task.project_id == test_project.id))).scalars().all()
        return [t for t in tasks if (t.metadata_ or {}).get("orchestration", {}).get("plan_item_id") == "synthesis"]

    async def gate(item_id):
        return (await db_session.execute(select(OrchestrationGate).where(
            OrchestrationGate.run_id == run.id, OrchestrationGate.success_criterion_key == f"plan_item:{item_id}",
        ))).scalar_one()

    assert await release() == 2  # binds both audits; the join is not runnable yet
    for audit in audits:
        audit.status = "done"
    await db_session.flush()

    (await gate("audit-0")).status = "accepted"  # 1 of 2 accepted
    await db_session.flush()
    assert await release() == 0 and await synthesis_tasks() == []

    (await gate("audit-1")).status = "failed"  # a failed predecessor blocks the join
    await db_session.flush()
    for _ in range(2):
        assert await release() == 0 and await synthesis_tasks() == []

    (await gate("audit-1")).status = "accepted"
    await db_session.flush()
    assert await release() == 1
    for _ in range(3):
        assert await release() == 0

    (task,) = await synthesis_tasks()
    assert task.depends_on == [str(audit.id) for audit in audits]
    assert task.priority == 80
    contract = task.metadata_["orchestration_contract"]
    assert contract["priority"] == 80 and contract["why_runnable"] == "Both audits are accepted."
    assert contract["artifact_ids"] == ["00000000-0000-0000-0000-0000000000a1"]
    assert contract["depends_on"] == [str(audit.id) for audit in audits]
    releases = (await db_session.execute(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "release_item",
    ))).scalars().all()
    assert sorted(a.request["plan_item_id"] for a in releases) == ["audit-0", "audit-1", "synthesis"]


@pytest.mark.asyncio
async def test_terminal_adopted_audit_without_verification_keeps_synthesis_queued(db_session, test_project):
    """Tester gap: a finished adopted audit whose gate still needs verification must not release the join."""
    from huddleroom.models.session import Session
    from huddleroom.services.event_bus import emit_event_once
    from tests.test_orchestration_plan_to_work_expansion import _adopted_audit_plan

    service, goal, run, (audit,) = await _adopted_audit_plan(
        db_session, test_project.id, audits=1, synthesis={"priority": 80},
    )
    audit.status = "done"
    session = Session(task_id=audit.id, agent_id=audit.assigned_to, project_id=test_project.id,
                      adapter_type="api", status="completed", output="Audit finished.")
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(db_session, test_project.id, "task.status_changed",
                          {"task_id": str(audit.id), "status": "done", "previous_status": "in_progress"},
                          dedup_key=f"done:{audit.id}")
    await emit_event_once(db_session, test_project.id, "session.completed",
                          {"session_id": str(session.id), "task_id": str(audit.id)},
                          dedup_key=f"completed:{session.id}")

    for _ in range(3):
        await service._release_ready_work(db_session, goal, run)

    gate = (await db_session.execute(select(OrchestrationGate).where(
        OrchestrationGate.run_id == run.id, OrchestrationGate.success_criterion_key == "plan_item:audit-0",
    ))).scalar_one()
    assert gate.status not in ("accepted", "failed")  # open: terminal task + report, verification still missing
    tasks = (await db_session.execute(select(Task).where(Task.project_id == test_project.id))).scalars().all()
    assert not [t for t in tasks if (t.metadata_ or {}).get("orchestration", {}).get("plan_item_id") == "synthesis"]


# --- ad-hoc discovery/readiness gates (T10c) --------------------------------------------

import json  # noqa: E402

from huddleroom.services.orchestration_dependencies import ADHOC_GATE_TYPE  # noqa: E402

VALID_REPORT = {
    "status": "done", "changes": [], "evidence": ["Found 3 failing call sites"], "criterion_progress": {},
    "decisions": [], "risks": [], "open_questions": [], "next_step": "Decide", "collaboration_need": None,
}


async def _adhoc_run(db_session, project_id):
    service, goal, run = await _make_run(db_session, project_id)
    goal.status, run.status, run.phase = "active", "running", "authorized"
    agent = _agent("auditor", "developer", ["investigation"])
    db_session.add(agent)
    await db_session.flush()
    return service, run, agent


async def _delegate_adhoc(db_session, service, run, agent, key, *, work_function="investigation", ad_hoc=True, **extra):
    action = await service.execute_create_delegation_task_action(
        db_session, run_id=run.id, skip_baseline_gate=True, ad_hoc=ad_hoc,
        request={
            "action_type": "create_delegation_task", "agent_id": str(agent.id), "work_function": work_function,
            "scope": f"Audit {key}", "deliverable": "Findings", **extra,
        },
        idempotency_key=f"run:{run.id}:kind:create_delegation_task:{key}",
    )
    return await db_session.get(Task, action.target_id)


async def _finish(db_session, project_id, task, report=VALID_REPORT, status="done"):
    task.status = status
    task.completed_at = datetime.now(timezone.utc)
    task.metadata_ = {
        **task.metadata_,
        "orchestration": {**task.metadata_["orchestration"], "terminal_output": report if isinstance(report, str) else json.dumps(report)},
    }
    await emit_event_once(
        db_session, project_id, "task.status_changed",
        {"task_id": str(task.id), "status": status, "previous_status": "in_progress", "project_id": str(project_id)},
        dedup_key=f"t10c-finish:{task.id}:{status}",
    )


async def _adhoc_gate(db_session, task):
    return await db_session.get(OrchestrationGate, uuid.UUID(task.metadata_["orchestration"]["gate_id"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(("work_function", "role"), [("investigation", "discovery"), ("review", "readiness")])
async def test_adhoc_investigation_and_review_get_a_deterministic_gate(db_session, test_project, work_function, role):
    service, run, agent = await _adhoc_run(db_session, test_project.id)
    agent.capabilities = ["investigation", "review"]

    task = await _delegate_adhoc(db_session, service, run, agent, "g1", work_function=work_function)

    gate = await _adhoc_gate(db_session, task)
    assert gate.id == uuid.uuid5(
        uuid.NAMESPACE_URL, f"rally:orchestration:gate:{run.id}:{ADHOC_GATE_TYPE}:adhoc_task:{task.id}"
    )
    assert gate.gate_type == ADHOC_GATE_TYPE and gate.status == "open"
    assert gate.required_evidence == {
        "required_source_types": ["task"], "min_count": 1, "evidence_role": role,
        "review_stage": "preliminary", "task_id": str(task.id),
    }


@pytest.mark.asyncio
async def test_plan_item_and_non_ad_hoc_work_get_no_adhoc_gate(db_session, test_project):
    service, run, agent = await _adhoc_run(db_session, test_project.id)
    agent.capabilities = ["investigation", "implementation"]

    plan_item_style = await _delegate_adhoc(db_session, service, run, agent, "p1", ad_hoc=False)
    implementation = await _delegate_adhoc(db_session, service, run, agent, "p2", work_function="implementation")

    for task in (plan_item_style, implementation):
        assert "gate_id" not in task.metadata_["orchestration"]
    assert (await db_session.scalars(select(OrchestrationGate).where(OrchestrationGate.run_id == run.id))).all() == []


@pytest.mark.asyncio
async def test_adhoc_gate_accepts_only_completed_task_with_valid_evidence_report(db_session, test_project):
    service, run, agent = await _adhoc_run(db_session, test_project.id)
    task = await _delegate_adhoc(db_session, service, run, agent, "a1")
    gate = await _adhoc_gate(db_session, task)
    await _finish(db_session, test_project.id, task)

    result = await service.tick(db_session, run.id)

    assert result["gates_validated"] == 1 and gate.status == "accepted"
    assert [e.verdict for e in await _evidence_rows(db_session, gate.id)] == ["accepted"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("report", "reason"),
    [
        ("not a report", "Task report is invalid"),
        ({**VALID_REPORT, "evidence": []}, "Task report has no evidence"),
    ],
)
async def test_adhoc_gate_fails_for_final_invalid_or_evidence_free_report(db_session, test_project, report, reason):
    service, run, agent = await _adhoc_run(db_session, test_project.id)
    task = await _delegate_adhoc(db_session, service, run, agent, "a2")
    gate = await _adhoc_gate(db_session, task)
    await _finish(db_session, test_project.id, task, report)

    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)

    assert gate.status == "failed" and gate.failure_reason == reason


@pytest.mark.asyncio
async def test_adhoc_gate_fails_when_task_fails(db_session, test_project):
    service, run, agent = await _adhoc_run(db_session, test_project.id)
    task = await _delegate_adhoc(db_session, service, run, agent, "a3")
    gate = await _adhoc_gate(db_session, task)
    await _finish(db_session, test_project.id, task, status="failed")

    await service.tick(db_session, run.id)

    assert gate.status == "failed"


@pytest.mark.asyncio
async def test_adhoc_gates_block_neither_closeout_manifest_nor_unverified_gate_follow_ups(db_session, test_project):
    from huddleroom.services.orchestration_progress_view import OrchestrationProgressView

    service, run, agent = await _adhoc_run(db_session, test_project.id)
    goal = await db_session.get(OrchestrationGoal, run.goal_id)
    open_task = await _delegate_adhoc(db_session, service, run, agent, "c1")
    failed_task = await _delegate_adhoc(db_session, service, run, agent, "c2")
    open_gate, failed_gate = await _adhoc_gate(db_session, open_task), await _adhoc_gate(db_session, failed_task)
    failed_gate.status = "failed"
    await db_session.flush()

    assert await service._current_run_gates(db_session, run.id) == []
    situation = await OrchestrationProgressView().build(db_session, goal, run)
    flagged = {item["id"] for item in situation.untracked_follow_ups if item["kind"] == "unverified_gate"}
    assert not flagged & {str(open_gate.id), str(failed_gate.id)}

    open_gate.status = "accepted"
    await db_session.flush()
    assert [g.id for g in await service._current_run_gates(db_session, run.id)] == [open_gate.id]


@pytest.mark.asyncio
async def test_adhoc_dependent_starts_once_only_after_predecessor_gate_accepted(db_session, test_project):
    from huddleroom.models.session import Session

    service, run, agent = await _adhoc_run(db_session, test_project.id)
    first = await _delegate_adhoc(db_session, service, run, agent, "d1")
    first.status = "done"
    second = await _delegate_adhoc(db_session, service, run, agent, "d2", depends_on=[str(first.id)])
    gate = await _adhoc_gate(db_session, first)

    async def sessions():
        return list(await db_session.scalars(select(Session).where(Session.task_id == second.id)))

    for _ in range(2):  # done but gate open: dependent stays in backlog
        await service.tick(db_session, run.id)
        await db_session.refresh(second)
        assert second.status == "backlog" and await sessions() == []

    await _finish(db_session, test_project.id, first)
    await service.tick(db_session, run.id)
    await service.tick(db_session, run.id)

    assert gate.status == "accepted"
    await db_session.refresh(second)
    assert second.status == "in_progress" and len(await sessions()) == 1


@pytest.mark.asyncio
async def test_adhoc_gate_accepts_report_repaired_on_the_clarification_task_marker(db_session, test_project):
    service, run, agent = await _adhoc_run(db_session, test_project.id)
    task = await _delegate_adhoc(db_session, service, run, agent, "r1")
    gate = await _adhoc_gate(db_session, task)
    clarification_id = uuid.uuid4()
    repair = await service.reserve_action(
        db_session, run.id, f"run:{run.id}:kind:report_clarification:task:{task.id}", "create_delegation_task", {},
    )
    repair.target_id = clarification_id
    await _finish(db_session, test_project.id, task, "not a report")
    await service.tick(db_session, run.id)
    # The repair is still pending, so the invalid first report is not final.
    assert gate.status == "open" and gate.failure_reason == "Task report is invalid"

    marker = await service.reserve_action(
        db_session, run.id, f"run:{run.id}:kind:report_consumed:task:{clarification_id}", "report_consumed",
        {"task_id": str(clarification_id), "session_id": None, "report": VALID_REPORT, "report_validation_errors": []},
    )
    await db_session.flush()
    assert marker.request["report"] == VALID_REPORT

    await service.validate_open_gates(db_session, run.id)

    assert gate.status == "accepted"
