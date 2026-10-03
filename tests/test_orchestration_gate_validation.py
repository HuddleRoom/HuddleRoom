import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationEvidence, OrchestrationGate
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
