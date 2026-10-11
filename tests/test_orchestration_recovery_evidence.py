"""Read-only, scope-safe inputs for recovery analysis."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from fastapi import HTTPException

from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
    OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationProcessRun, OrchestrationWarning
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration_recovery import (
    RECOVERY_EVIDENCE_QUERY_ADAPTER,
    RecoveryTriggerV1,
    ToolManifestV1,
)
from huddleroom.services.orchestration_recovery_evidence import (
    RECOVERY_CONTEXT_MAX_BYTES,
    RECOVERY_EVIDENCE_MAX_BYTES,
    RecoveryContextBuilder,
    RecoveryEvidenceService,
    _bounded,
)


pytestmark = pytest.mark.asyncio


def _trigger(goal, run, source_id) -> RecoveryTriggerV1:
    return RecoveryTriggerV1.model_validate({
        "schema_version": 1,
        "goal_id": str(goal.id),
        "run_id": str(run.id),
        "source_kind": "failed_action",
        "source_id": str(source_id),
        "blocker_code": "provider_failed",
        "summary": "The source action failed.",
        "source_fingerprint": "a" * 64,
        "retry_target": {"kind": "decision_loop"},
    })


def _request(kind: str, **values):
    return RECOVERY_EVIDENCE_QUERY_ADAPTER.validate_python({"kind": kind, **values})


async def _seed(db, project, test_agent):
    goal = OrchestrationGoal(
        project_id=project.id,
        objective="Recover the blocked action",
        success_criteria=[{"key": "done", "description": "The action succeeds."}],
        constraints={"secret": "api_key=never-return-this"},
    )
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(
        goal_id=goal.id,
        phase="authorized",
        active_blockers=[{"kind": "provider_failed", "detail": "token=never-return-this"}],
        retry_state={"attempt": 1},
    )
    db.add(run)
    await db.flush()
    action = OrchestrationAction(
        run_id=run.id,
        idempotency_key="recovery-source",
        action_type="dispatch_task",
        request={"task": "compile", "api_key": "never-return-this"},
        target_type="task",
        status="failed",
        error="Authorization: Bearer never-return-this",
    )
    task = Task(
        project_id=project.id,
        title="Compile the service",
        status="blocked",
        assigned_to=test_agent.id,
        metadata_={"orchestration": {"goal_id": str(goal.id), "run_id": str(run.id), "action_id": str(action.id)}},
    )
    gate = OrchestrationGate(
        run_id=run.id,
        success_criterion_key="done",
        gate_type="work_completed",
        status="failed",
        failure_reason="password=never-return-this",
    )
    process = OrchestrationProcessRun(
        goal_id=goal.id,
        run_id=run.id,
        process_type="authority_interview",
        trigger_reason="Recovery requires an answer.",
        status="completed",
        outputs={"accepted": True, "api_key": "never-return-this"},
    )
    warning = OrchestrationWarning(
        goal_id=goal.id,
        run_id=run.id,
        warning_type="recovery_attention",
        severity="warning",
        message="token=never-return-this",
        related_action_id=action.id,
    )
    db.add_all((action, task, gate, process, warning))
    await db.flush()
    wait = OrchestrationWait(
        run_id=run.id,
        wait_key="recovery-source",
        owner={"type": "action", "id": str(action.id)},
        awaited_event={"event_type": "task.status_changed", "matcher": {"task_id": str(task.id)}},
        due_recheck_at=action.created_at + timedelta(seconds=30),
        fallback={"action_type": "attention"},
        status="open",
    )
    db.add(wait)
    await db.flush()
    goal.orchestrator_context = {"team": {"role_to_agent": {"planning": str(test_agent.id)}}}
    return goal, run, action, task, gate, process, wait, warning


async def test_recovery_evidence_resolves_exactly_the_nine_scoped_kinds(db_session, test_project, test_agent):
    goal, run, action, task, gate, process, _wait, warning = await _seed(db_session, test_project, test_agent)
    trigger = _trigger(goal, run, action.id)
    service = RecoveryEvidenceService()

    cases = (
        ("run_state", {"run_id": run.id}, "run", run.id),
        ("action", {"action_id": action.id}, "action", action.id),
        ("process", {"process_type": process.process_type}, "process", process.id),
        ("team_configuration", {}, "run", run.id),
        ("task", {"task_id": task.id}, "task", task.id),
        ("gate", {"gate_id": gate.id}, "gate", gate.id),
        ("waits", {}, "run", run.id),
        ("warning", {"warning_id": warning.id}, "warning", warning.id),
        ("agent_execution", {"agent_id": test_agent.id}, "agent", test_agent.id),
    )
    evidence = [await service.query(db_session, trigger, _request(kind, **values)) for kind, values, _, _ in cases]

    assert [(item.source_kind, item.source_id) for item in evidence] == [(kind, source_id) for _, _, kind, source_id in cases]
    assert evidence[-1].value["status"] == "unproven"
    assert all("never-return-this" not in json.dumps(item.value).lower() for item in evidence)


async def test_recovery_evidence_rejects_foreign_goal_project_and_run_references(db_session, test_project, test_agent, tmp_path):
    goal, run, action, _task, _gate, _process, _wait, _warning = await _seed(db_session, test_project, test_agent)
    trigger = _trigger(goal, run, action.id)
    foreign_goal = OrchestrationGoal(project_id=test_project.id, objective="Foreign goal")
    foreign_project = Project(name="Other project", description="Other", workspace_path=str(tmp_path / "other"), config={})
    db_session.add_all((foreign_goal, foreign_project))
    await db_session.flush()
    foreign_run = OrchestrationRun(goal_id=foreign_goal.id, phase="authorized")
    db_session.add(foreign_run)
    await db_session.flush()
    foreign_action = OrchestrationAction(run_id=foreign_run.id, idempotency_key="foreign", action_type="x", status="failed")
    foreign_task = Task(
        project_id=foreign_project.id,
        title="Foreign task",
        status="blocked",
        metadata_={"orchestration": {"goal_id": str(goal.id), "run_id": str(run.id)}},
    )
    db_session.add_all((foreign_action, foreign_task))
    await db_session.flush()

    service = RecoveryEvidenceService()
    for request in (
        _request("run_state", run_id=foreign_run.id),
        _request("action", action_id=foreign_action.id),
        _request("task", task_id=foreign_task.id),
    ):
        with pytest.raises(HTTPException) as exc:
            await service.query(db_session, trigger, request)
        assert exc.value.status_code == 404


async def test_recovery_context_rejects_a_foreign_failed_action_source_before_snapshot(db_session, test_project, test_agent):
    goal, run, action, _task, _gate, _process, _wait, _warning = await _seed(db_session, test_project, test_agent)
    foreign_goal = OrchestrationGoal(project_id=test_project.id, objective="Foreign goal")
    db_session.add(foreign_goal)
    await db_session.flush()
    foreign_run = OrchestrationRun(goal_id=foreign_goal.id, phase="authorized")
    db_session.add(foreign_run)
    await db_session.flush()
    foreign_action = OrchestrationAction(run_id=foreign_run.id, idempotency_key="foreign-source", action_type="x", status="failed")
    db_session.add(foreign_action)
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await RecoveryContextBuilder().build(db_session, _trigger(goal, run, foreign_action.id), registry_manifest=())

    assert exc.value.status_code == 404


async def test_recovery_context_is_bounded_redacted_and_deterministic(db_session, test_project, test_agent):
    goal, run, action, _task, _gate, _process, _wait, _warning = await _seed(db_session, test_project, test_agent)
    prior_run = OrchestrationRun(goal_id=goal.id, status="completed", phase="completed")
    db_session.add(prior_run)
    await db_session.flush()
    db_session.add(OrchestrationAction(
        run_id=prior_run.id,
        idempotency_key="prior-recovery-attempt",
        action_type="recover_blocker",
        request={"kind": "proposal"},
        status="completed",
    ))
    run.supersedes_run_id = prior_run.id
    goal.constraints = {"api_key": "never-return-this", "padding": "x" * (RECOVERY_CONTEXT_MAX_BYTES * 2)}
    manifest = (
        ToolManifestV1(name="zeta", version="1", description="Last by name", arguments_schema={"type": "object"}),
        ToolManifestV1(name="alpha", version="1", description="First by name", arguments_schema={"type": "object"}),
    )

    builder = RecoveryContextBuilder()
    first = await builder.build(db_session, _trigger(goal, run, action.id), registry_manifest=manifest)
    second = await builder.build(db_session, _trigger(goal, run, action.id), registry_manifest=manifest)
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":")).encode()

    assert first == second
    assert len(encoded) <= RECOVERY_CONTEXT_MAX_BYTES
    assert "never-return-this" not in encoded.decode().lower()
    assert {"goal_contract", "active_run", "source_error", "related_tasks", "gates", "processes", "waits", "warnings", "team_configuration", "prior_lineage_attempts", "registry_manifest"} <= first.keys()
    assert [item["name"] for item in first["registry_manifest"]] == ["alpha", "zeta"]
    assert str(test_agent.id) in json.dumps(first["team_configuration"])
    assert str(prior_run.id) in json.dumps(first["prior_lineage_attempts"])
    assert first.get("truncated") is True


async def test_recovery_evidence_truncates_before_the_strict_schema_boundary(db_session, test_project, test_agent):
    goal, run, action, _task, _gate, _process, _wait, _warning = await _seed(db_session, test_project, test_agent)
    run.active_blockers = [{"detail": "x" * (RECOVERY_EVIDENCE_MAX_BYTES * 2)}]

    evidence = await RecoveryEvidenceService().query(db_session, _trigger(goal, run, action.id), _request("run_state", run_id=run.id))
    encoded = json.dumps(evidence.value, sort_keys=True, separators=(",", ":")).encode()

    assert len(encoded) <= RECOVERY_EVIDENCE_MAX_BYTES
    assert evidence.value["truncated"] is True
    assert evidence.value["omitted_bytes"] > 0


async def test_recovery_process_evidence_and_context_exclude_raw_retry_prompts(db_session, test_project, test_agent):
    goal, run, action, _task, _gate, process, _wait, _warning = await _seed(db_session, test_project, test_agent)
    process.input_snapshot = {"prompt_marker": "INPUT_PROMPT_MARKER"}
    process.outputs = {
        "gate": "planning_ready",
        "gates": [{"id": "planning", "status": "open"}],
        "fingerprint": "b" * 64,
        "safe_status": "awaiting approval",
        "_lm_retry": {"request": "OUTPUT_PROMPT_MARKER token=never-return-this"},
    }
    trigger = _trigger(goal, run, action.id)

    evidence = await RecoveryEvidenceService().query(
        db_session, trigger, _request("process", process_type=process.process_type),
    )
    context = await RecoveryContextBuilder().build(db_session, trigger, registry_manifest=())
    visible = json.dumps({"evidence": evidence.value, "context": context})

    assert "INPUT_PROMPT_MARKER" not in visible
    assert "OUTPUT_PROMPT_MARKER" not in visible
    assert "input_snapshot" not in evidence.value
    assert "_lm_retry" not in json.dumps(evidence.value)
    assert evidence.value["status"] == "completed"
    assert evidence.value["outputs"]["gates"] == [{"id": "planning", "status": "open"}]
    assert str(process.id) in {item["id"] for item in context["processes"]}
    assert next(item for item in context["processes"] if item["id"] == str(process.id))["outputs"]["gates"] == [
        {"id": "planning", "status": "open"}
    ]


async def test_recovery_team_uses_completed_hierarchy_and_established_manager_verifier_ids(db_session, test_project, test_agent):
    goal, run, action, _task, _gate, _process, _wait, _warning = await _seed(db_session, test_project, test_agent)
    goal.orchestrator_context = {}
    hierarchy = OrchestrationProcessRun(
        goal_id=goal.id,
        run_id=run.id,
        process_type="team_hierarchy",
        trigger_reason="Team review is pending.",
        status="running",
        outputs={
            "manager": {"kind": "agent", "id": str(test_agent.id)},
            "verifier_id": str(test_agent.id),
            "candidate_agents": [str(test_agent.id)],
        },
    )
    db_session.add(hierarchy)
    await db_session.flush()
    trigger = _trigger(goal, run, action.id)
    service = RecoveryEvidenceService()

    provisional = await service.query(db_session, trigger, _request("team_configuration"))
    assert provisional.value["team"] == {}
    assert provisional.value["hierarchy_process_id"] is None
    with pytest.raises(HTTPException) as exc:
        await service.query(db_session, trigger, _request("agent_execution", agent_id=test_agent.id))
    assert exc.value.status_code == 404

    hierarchy.status = "completed"
    accepted = await service.query(db_session, trigger, _request("team_configuration"))
    execution = await service.query(db_session, trigger, _request("agent_execution", agent_id=test_agent.id))

    assert str(test_agent.id) in accepted.value["agent_ids"]
    assert accepted.value["hierarchy_process_id"] == str(hierarchy.id)
    assert execution.value["status"] == "unproven"

    hierarchy.status = "waiting_decision"
    goal.orchestrator_context = {"team": {"role_to_agent": {"planning": str(test_agent.id)}}}
    persisted = await service.query(db_session, trigger, _request("team_configuration"))
    assert persisted.value["hierarchy_process_id"] is None


async def test_recovery_context_contains_only_current_processes_in_stable_order(db_session, test_project, test_agent):
    goal, run, action, _task, _gate, historical, _wait, _warning = await _seed(db_session, test_project, test_agent)
    historical.superseded_by_id = historical.id
    current = OrchestrationProcessRun(
        goal_id=goal.id,
        run_id=run.id,
        process_type=historical.process_type,
        trigger_reason="Current authority interview.",
        status="completed",
    )
    definition = OrchestrationProcessRun(
        goal_id=goal.id,
        run_id=run.id,
        process_type="goal_definition",
        trigger_reason="Current goal definition.",
        status="completed",
    )
    db_session.add_all((current, definition))
    await db_session.flush()
    historical.superseded_by_id = current.id
    await db_session.flush()

    context = await RecoveryContextBuilder().build(db_session, _trigger(goal, run, action.id), registry_manifest=())
    rows = context["processes"]

    assert str(historical.id) not in {row["id"] for row in rows}
    assert {str(current.id), str(definition.id)} <= {row["id"] for row in rows}
    assert [(row["process_type"], row["id"]) for row in rows] == sorted(
        (row["process_type"], row["id"]) for row in rows
    )


async def test_recovery_bounds_preserve_fixed_envelope_and_analyzer_json_size_with_unicode():
    sections = (
        "goal_contract", "active_run", "source_error", "related_tasks", "gates", "processes", "waits", "warnings",
        "team_configuration", "prior_lineage_attempts", "registry_manifest",
    )
    row = {field: "x" * 512 for field in ("first", "second", "third", "fourth")}
    oversized = {section: {"rows": [row for _ in range(32)]} for section in sections}
    bounded = _bounded(oversized, RECOVERY_CONTEXT_MAX_BYTES, fixed_keys=sections)
    emoji_evidence = _bounded({"payload": "😀" * 4_000}, RECOVERY_EVIDENCE_MAX_BYTES)

    assert set(sections) <= bounded.keys()
    assert bounded["truncated"] is True
    assert bounded["omitted_bytes"] > 0
    assert bounded == _bounded(oversized, RECOVERY_CONTEXT_MAX_BYTES, fixed_keys=sections)
    assert len(json.dumps(bounded, sort_keys=True, default=str).encode()) <= RECOVERY_CONTEXT_MAX_BYTES
    assert len(json.dumps(emoji_evidence, sort_keys=True, default=str).encode()) <= RECOVERY_EVIDENCE_MAX_BYTES
