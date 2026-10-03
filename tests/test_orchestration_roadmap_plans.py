import json
import uuid
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from huddleroom.models.artifact import Artifact
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationBudgetReservation,
    OrchestrationDecision,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationRoadmapItem,
    OrchestrationRoadmapVersion,
    OrchestrationRun,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.task import Task
from huddleroom.models.session import Session
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_budget_service import (
    BudgetMeasurementError,
    OrchestrationBudgetService,
)
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_roadmap_service import (
    OrchestrationRoadmapService,
    parse_roadmap_items,
)
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.schemas.orchestration import OrchestrationDelegationContract

from tests.test_orchestration_runtime_e2e import _agent, _authorized_run, _complete_task_session, _run_actions


pytestmark = pytest.mark.asyncio


async def test_ordinary_request_serializers_omit_empty_orchestrator_context():
    service = OrchestrationService()
    agent_id = uuid.uuid4()
    delegation = service._canonical_delegation_task_request({
        "agent_id": str(agent_id), "work_function": "implementation", "scope": "Do it",
        "deliverable": "Done", "orchestrator_context": {},
    })
    plan = service._canonical_plan_request({
        "agent_id": str(agent_id), "work_function": "planning", "scope": "Plan it",
        "orchestrator_context": {},
    })
    assert "orchestrator_context" not in delegation
    assert "orchestrator_context" not in plan


async def test_delegation_contract_dump_omits_empty_context_but_preserves_roadmap_context():
    base = {
        "goal_id": uuid.uuid4(), "run_id": uuid.uuid4(), "action_id": uuid.uuid4(),
        "agent_id": uuid.uuid4(), "work_function": "implementation", "scope": "Do it",
        "deliverable": "Done",
    }
    assert "orchestrator_context" not in OrchestrationDelegationContract(**base).model_dump(mode="json")
    assert OrchestrationDelegationContract(
        **base, orchestrator_context={"roadmap": {"roadmap_item_key": "item-1"}},
    ).model_dump(mode="json")["orchestrator_context"]["roadmap"]["roadmap_item_key"] == "item-1"


async def test_ordinary_plan_request_ignores_empty_baseline_bookkeeping(db_session, test_project):
    agent = _agent("ordinary-planner", ["planning"])
    db_session.add(agent)
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    goal.orchestrator_context = {"assumptions": [], "merged_decision_keys": []}

    action = await service.execute_request_plan_action(
        db_session, run.id,
        {"action_type": "request_plan", "agent_id": str(agent.id), "work_function": "planning", "scope": "Plan it"},
        f"run:{run.id}:kind:ordinary-request",
    )

    assert "orchestrator_context" not in action.request


async def test_plan_request_serializer_retains_nonempty_roadmap_context():
    agent_id = uuid.uuid4()
    request = OrchestrationService()._canonical_plan_request({
        "agent_id": str(agent_id), "work_function": "planning", "scope": "Plan it",
        "orchestrator_context": {"roadmap": {"roadmap_item_key": "item-1"}},
    })
    assert request["orchestrator_context"]["roadmap"]["roadmap_item_key"] == "item-1"


@pytest.fixture(autouse=True)
def decision_sessions_use_test_database(monkeypatch, test_engine):
    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.AsyncSessionLocal",
        async_sessionmaker(test_engine, expire_on_commit=False),
    )


def _roadmap_task(key="task-1", depends_on=(), *, mutates_shared_state=False, staging_boundary=None):
    return {
        "item_key": key,
        "unit_type": "task",
        "title": key,
        "depends_on": list(depends_on),
        "mutates_shared_state": mutates_shared_state,
        "staging_boundary": staging_boundary,
        "work_function": "implementation",
        "scope": f"Implement {key}",
        "deliverable": f"Verified {key}",
    }


def _roadmap_goal(key, *, allocation, depends_on=()):
    return {
        "item_key": key,
        "unit_type": "goal",
        "title": key,
        "objective": f"Deliver {key}",
        "success_criteria": [{"key": f"{key}-done"}],
        "depends_on": list(depends_on),
        "allocation": allocation,
        "mutates_shared_state": False,
    }


async def test_roadmap_goal_allocation_requires_every_parent_cap():
    """Dropping a capped dimension must not silently turn its spend into zero."""
    with pytest.raises(HTTPException, match="must include every parent budget dimension"):
        parse_roadmap_items([_roadmap_goal("child", allocation={"max_turns": 1})], {"max_turns", "max_tokens"})


async def test_accepted_team_parser_excludes_non_agent_identifiers():
    planner, assigned, listed, verifier, human_manager, candidate = (uuid.uuid4() for _ in range(6))
    team = {
        "agent_ids": [str(planner)],
        "role_to_agent": {"implementer": str(assigned)},
        "assignments": [{"work_function": "validation", "agent_ref": str(verifier)}],
        "hierarchy": {"agents": [str(listed)]},
        "verifier_agent_id": str(verifier),
        "manager": {"kind": "human", "id": str(human_manager)},
        "candidate_agents": {"implementation": [str(candidate)]},
        "approval": {"id": str(uuid.uuid4())},
    }

    assert OrchestrationService._roadmap_team_agent_ids(team) == {
        str(planner), str(assigned), str(listed), str(verifier),
    }
    assert OrchestrationService._roadmap_team_agent_ids(None) is None
    assert OrchestrationService._roadmap_team_agent_ids({"agent_ids": []}) == set()


async def _roadmap_acceptance(
    db_session, test_project, test_user, *, authority_model="human_manager", seed_approval=True,
    items=None, budget=None, team_agent_ids=(), use_hierarchy=False,
):
    planner = _agent("roadmap-planner", ["planning", "validation", "implementation"])
    db_session.add(planner)
    await db_session.flush()
    service, goal, run = await _authorized_run(db_session, test_project)
    if not use_hierarchy:
        goal.orchestrator_context = {"team": {"agent_ids": [str(planner.id), *map(str, team_agent_ids)]}}
    if budget is not None:
        goal.budget = {"caps": budget}
    request = await service.execute_request_plan_action(
        db_session, run.id,
        {"action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning", "scope": "Plan Roadmap work."},
        f"run:{run.id}:kind:roadmap-request",
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="roadmap-plan",
        artifact_type="plan",
        status="draft",
        linked_task_id=request.target_id,
        created_by_agent=planner.id,
        metadata_={"plan_items": items or [_roadmap_task()]},
    )
    db_session.add(artifact)
    await db_session.flush()
    goal.goal_type = "roadmap"
    goal.authority_model = authority_model
    goal.manager_user_id = goal.manager_user_id if authority_model == "human_manager" else None
    goal.manager_agent_id = planner.id if authority_model == "agent_manager" else None
    # Authority changes invalidate the completed baseline's roster/hierarchy
    # fingerprints. Use the supported human skip path for these prerequisite
    # reviews; the Roadmap approval under test still runs in full.
    if authority_model != "human_manager":
        for process_type in ("agent_definition_review", "team_hierarchy"):
            await OrchestrationProcessService().skip_process(
                db_session, goal.id, process_type=process_type, skipped_by=f"human:{test_user.id}",
                reason="Exercise Roadmap plan authority independently of baseline review.", run_id=run.id,
            )
    normalized = [item.model_dump(mode="json") for item in parse_roadmap_items(
        artifact.metadata_["plan_items"], set((goal.budget or {}).get("caps", {})),
    )]
    fingerprint = service._accepted_plan_fingerprint(normalized)
    if not seed_approval or goal.authority_model == "no_manager":
        decision = None
    else:
        decision = await OrchestrationAuthorityDecisionService().create_pending(
            db_session, goal.id, decision_key=f"roadmap_plan:{fingerprint}", title="Approve Roadmap plan",
            question="Approve?", authority="human" if goal.authority_model == "human_manager" else "manager",
            authority_agent_id=None if goal.authority_model == "human_manager" else goal.manager_agent_id,
            options=[{"key": "approve"}, {"key": "reject"}], run_id=run.id,
        )
    return service, goal, run, artifact, decision, fingerprint


async def _approve(db_session, goal, decision):
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve",
        decided_by_user_id=goal.manager_user_id if goal.authority_model == "human_manager" else None,
        decided_by_agent_id=goal.manager_agent_id if goal.authority_model == "agent_manager" else None,
    )


async def _assert_rejected_atomic(db_session, run):
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 0
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem)) == 0
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation)) == 0
    await db_session.refresh(run)
    assert run.plan_state["status"] == "revision_required"
    assert await db_session.scalar(select(OrchestrationAction.id).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "accept_plan",
        OrchestrationAction.status == "failed",
    )) is not None


async def _request_replan(db_session, service, goal, run, artifact):
    version = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    return await service.execute_request_roadmap_replan_action(
        db_session, run.id,
        {"action_type": "request_roadmap_replan", "agent_id": str(artifact.created_by_agent),
         "scope": "Revise remaining work.", "reason": "New information."},
        f"run:{run.id}:kind:request_roadmap_replan:version:{version.version}",
    )


def _roadmap_fingerprint(service, goal, items):
    return service._accepted_plan_fingerprint(_normalized_roadmap_items(goal, items))


def _normalized_roadmap_items(goal, items):
    return [
        item.model_dump(mode="json")
        for item in parse_roadmap_items(items, set((goal.budget or {}).get("caps", {})))
    ]


async def _accept_version_one(
    db_session, test_project, test_user, *, items, budget=None, authority_model="human_manager", team_agent_ids=(),
):
    service, goal, run, artifact, decision, _ = await _roadmap_acceptance(
        db_session, test_project, test_user, items=items, budget=budget, authority_model=authority_model,
        team_agent_ids=team_agent_ids,
    )
    await _approve(db_session, goal, decision)
    await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"run:{run.id}:kind:version-one",
    )
    return service, goal, run, artifact


async def _replan_artifact(db_session, test_project, service, goal, run, artifact, items):
    replan = await _request_replan(db_session, service, goal, run, artifact)
    candidate = Artifact(
        project_id=test_project.id, name="roadmap-replan", artifact_type="plan", status="draft",
        linked_task_id=replan.target_id, created_by_agent=artifact.created_by_agent,
        metadata_={"plan_items": items},
    )
    db_session.add(candidate)
    await db_session.flush()
    return replan, candidate


async def _accept_replan_after_selected_authority(db_session, service, goal, run, candidate, key):
    fingerprint = _roadmap_fingerprint(service, goal, candidate.metadata_["plan_items"])
    with pytest.raises(HTTPException, match="waiting_plan_authority"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(candidate.id)}, key,
        )
    approval = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key == f"roadmap_plan:{fingerprint}",
    ))
    await _approve(db_session, goal, approval)
    return await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(candidate.id)}, key,
    )


async def _release_roadmap_item(db_session, service, goal, run, key, unit_type="task"):
    result = await service.tick(db_session, run.id)
    assert result["authorized_execution"] == {"step": "release_item", "item_key": key, "unit_type": unit_type}
    return await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id, OrchestrationRoadmapItem.item_key == key,
    ))


async def test_initial_roadmap_rejects_item_agent_outside_accepted_team(
    db_session, test_project, test_user,
):
    insider = _agent("accepted-team-member", ["implementation"])
    outsider = _agent("unaccepted-plan-agent", ["implementation"])
    db_session.add_all([insider, outsider])
    await db_session.flush()
    item = {**_roadmap_task("outside-team"), "agent_id": str(outsider.id)}
    service, goal, run, artifact, decision, _ = await _roadmap_acceptance(
        db_session, test_project, test_user, items=[item],
    )
    goal.orchestrator_context = {"team": {"agent_ids": [str(insider.id)]}}
    await _approve(db_session, goal, decision)

    with pytest.raises(HTTPException, match="outside the accepted team"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            f"run:{run.id}:kind:outside-team-initial",
        )
    await _assert_rejected_atomic(db_session, run)


async def test_replan_cannot_expand_the_accepted_team(
    db_session, test_project, test_user,
):
    insider = _agent("replan-team-member", ["implementation"])
    outsider = _agent("replan-outsider", ["implementation"])
    db_session.add_all([insider, outsider])
    await db_session.flush()
    v1_item = {**_roadmap_task("v1"), "agent_id": str(insider.id)}
    service, goal, run, artifact, decision, _ = await _roadmap_acceptance(
        db_session, test_project, test_user, items=[v1_item],
    )
    goal.orchestrator_context = {"team": {"agent_ids": [str(insider.id)]}}
    await _approve(db_session, goal, decision)
    await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"run:{run.id}:kind:accepted-team-v1",
    )
    v1 = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    goal.orchestrator_context = {"team": {"agent_ids": [str(outsider.id)]}}
    _, candidate = await _replan_artifact(
        db_session, test_project, service, goal, run, artifact,
        [{**_roadmap_task("v2"), "agent_id": str(outsider.id)}],
    )
    fingerprint = _roadmap_fingerprint(service, goal, candidate.metadata_["plan_items"])
    approval = await OrchestrationAuthorityDecisionService().create_pending(
        db_session, goal.id, decision_key=f"roadmap_plan:{fingerprint}", title="Approve replan",
        question="Approve?", authority="human", options=[{"key": "approve"}], run_id=run.id,
    )
    await _approve(db_session, goal, approval)

    with pytest.raises(HTTPException, match="outside the accepted team"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(candidate.id)},
            f"run:{run.id}:kind:outside-team-replan",
        )
    assert (await OrchestrationRoadmapService(service).current_version(db_session, goal.id)).id == v1.id


async def test_initial_team_contract_comes_from_the_accepted_hierarchy(
    db_session, test_project, test_user,
):
    service, goal, run, artifact, decision, _ = await _roadmap_acceptance(
        db_session, test_project, test_user,
        items=[_roadmap_task("hierarchy-team")], use_hierarchy=True,
    )
    hierarchy = await OrchestrationProcessService().get_current(db_session, goal.id, "team_hierarchy")
    expected_ids = sorted(service._roadmap_team_agent_ids(hierarchy.outputs))
    await _approve(db_session, goal, decision)
    await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"run:{run.id}:kind:hierarchy-team-contract",
    )
    version = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    assert version.snapshot["team"]["agent_ids"] == expected_ids


async def test_accept_roadmap_plan_creates_version_one_and_projection(db_session, test_project, test_user):
    service, goal, run, artifact, decision, fingerprint = await _roadmap_acceptance(db_session, test_project, test_user)
    await _approve(db_session, goal, decision)

    action = await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"run:{run.id}:kind:roadmap-accept",
    )

    version = await db_session.scalar(select(OrchestrationRoadmapVersion))
    await db_session.refresh(run)
    assert version.version == 1
    assert version.plan_artifact_id == artifact.id
    assert version.fingerprint == fingerprint
    assert run.plan_state["roadmap_version_id"] == str(version.id)
    assert run.plan_state["accept_action_id"] == str(action.id)


async def test_accept_roadmap_plan_replay_returns_same_version(db_session, test_project, test_user):
    service, goal, run, artifact, decision, fingerprint = await _roadmap_acceptance(db_session, test_project, test_user)
    await _approve(db_session, goal, decision)
    key = f"run:{run.id}:kind:roadmap-accept"
    first = await service.execute_accept_plan_action(db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)}, key)
    replay = await service.execute_accept_plan_action(db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)}, key)

    assert replay.id == first.id
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 1


async def test_invalid_roadmap_plan_persists_no_version_or_lineage(db_session, test_project, test_user):
    service, _, run, artifact, _, _ = await _roadmap_acceptance(db_session, test_project, test_user)
    artifact.metadata_ = {"plan_items": [_roadmap_task("a", ["b"]), _roadmap_task("b", ["a"])]}

    with pytest.raises(HTTPException, match="Roadmap dependency cycle detected"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            f"run:{run.id}:kind:roadmap-invalid",
        )

    await _assert_rejected_atomic(db_session, run)


@pytest.mark.parametrize("status", ["missing", "pending", "cancelled"])
async def test_roadmap_plan_requires_selected_authority_approval(db_session, test_project, test_user, status):
    service, _, run, artifact, decision, _ = await _roadmap_acceptance(
        db_session, test_project, test_user, seed_approval=status != "missing",
    )
    if status == "missing":
        assert await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.run_id == run.id,
            OrchestrationAuthorityDecision.decision_key.like("roadmap_plan:%"),
        )) == 0
    if status == "cancelled" and decision is not None:
        await OrchestrationAuthorityDecisionService().cancel_decision(db_session, decision, reason="No")

    with pytest.raises(HTTPException, match="Roadmap plan approval"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            f"run:{run.id}:kind:roadmap-no-approval:{status}",
        )

    await _assert_rejected_atomic(db_session, run)


@pytest.mark.parametrize("producer", ["planner", "missing"])
async def test_no_manager_approval_gate_is_deterministic_and_requires_an_independent_verifier(
    db_session, test_project, test_user, producer,
):
    service, goal, run, artifact, _, fingerprint = await _roadmap_acceptance(
        db_session, test_project, test_user, authority_model="no_manager",
    )
    roadmap = OrchestrationRoadmapService(service)

    first = await roadmap.ensure_plan_approval(db_session, goal, run, artifact)
    second = await roadmap.ensure_plan_approval(db_session, goal, run, artifact)

    assert first.id == second.id
    assert first.required_evidence["requires_independent_agent"] is True
    assert first.required_evidence["work_producer_agent_id"] == str(artifact.created_by_agent)
    evidence = OrchestrationEvidence(
        run_id=run.id, gate_id=first.id, source_type="verification", source_id=first.id,
        producer_agent_id=artifact.created_by_agent if producer == "planner" else None,
        verdict="candidate", evidence_metadata={},
    )
    db_session.add(evidence)
    await db_session.flush()
    assert await service._validate_gate(db_session, first) is True
    assert first.status == "failed"
    assert first.failure_reason == (
        "Independent verification must come from a different agent"
        if producer == "planner" else "Independent verification producer is missing"
    )

    # Even an externally accepted gate/evidence cannot bypass final admission.
    first.status = "accepted"
    evidence.verdict = "accepted"
    await db_session.flush()
    with pytest.raises(HTTPException, match="Roadmap plan approval"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            f"run:{run.id}:kind:invalid-verifier:{producer}",
        )
    await _assert_rejected_atomic(db_session, run)


async def _pending_approval(db_session, run):
    return await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_plan:%"),
    ))


def _accept_decision(stub_decision, artifact):
    stub_decision(lambda _: {
        "action_type": "accept_plan", "plan_artifact_id": str(artifact.id), "reason": "Plan is ready.",
    })


@pytest.mark.parametrize("authority_model", ["human_manager", "agent_manager"])
async def test_managed_advance_waits_then_accepts_and_replays_once(
    db_session, test_project, test_user, stub_decision, authority_model,
):
    service, goal, run, artifact, _, fingerprint = await _roadmap_acceptance(
        db_session, test_project, test_user, authority_model=authority_model, seed_approval=False,
    )
    _accept_decision(stub_decision, artifact)
    approval_id = None
    for _ in range(3):
        assert await service._advance_authorized_execution(db_session, goal, run) == {
            "step": "waiting_plan_authority",
        }
        approval = await _pending_approval(db_session, run)
        approval_id = approval_id or approval.id
        assert approval.id == approval_id
        assert approval.status == "pending"
        assert not [a for a in await _run_actions(db_session, run.id) if a.action_type == "accept_plan"]
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key == f"roadmap_plan:{fingerprint}",
    )) == 1

    await _approve(db_session, goal, approval)
    result = await service._advance_authorized_execution(db_session, goal, run)
    assert result["step"] == "plan_decision"
    action = await db_session.get(OrchestrationAction, uuid.UUID(result["action_id"]))
    assert action.action_type == "accept_plan" and action.status == "completed"
    projection = deepcopy(run.plan_state)
    # Replay the actual dispatched action after acceptance, without entering Task 3 release work.
    for _ in range(2):
        replay = await service.execute_accept_plan_action(
            db_session, run.id, action.request, action.idempotency_key, action.decision_id,
        )
        assert replay.id == action.id
    assert run.plan_state == projection
    versions = list((await db_session.scalars(select(OrchestrationRoadmapVersion))).all())
    assert len(versions) == 1 and versions[0].fingerprint == fingerprint
    assert versions[0].approval_reference == {"kind": "authority_decision", "id": str(approval_id)}
    assert len([a for a in await _run_actions(db_session, run.id) if a.action_type == "accept_plan"]) == 1


@pytest.mark.parametrize("authority_model", ["human_manager", "agent_manager"])
async def test_advance_rejects_approval_from_wrong_selected_authority_before_reservation(
    db_session, test_project, test_user, stub_decision, authority_model,
):
    service, goal, run, artifact, _, fingerprint = await _roadmap_acceptance(
        db_session, test_project, test_user, authority_model=authority_model, seed_approval=False,
    )
    _accept_decision(stub_decision, artifact)
    assert (await service._advance_authorized_execution(db_session, goal, run))["step"] == "waiting_plan_authority"
    approval = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key == f"roadmap_plan:{fingerprint}",
    ))
    if authority_model == "human_manager":
        assert test_user.id != goal.manager_user_id
        await OrchestrationAuthorityDecisionService().answer_decision(
            db_session, approval, selected_option="approve", decided_by_user_id=test_user.id,
        )
    else:
        await _approve(db_session, goal, approval)
        # A once-valid manager answer must not authorize the newly selected manager's plan.
        replacement = _agent("replacement-manager", ["planning"])
        db_session.add(replacement)
        await db_session.flush()
        goal.manager_agent_id = replacement.id

    assert await service._advance_authorized_execution(db_session, goal, run) == {"step": "plan_revision_required"}
    assert run.plan_state["status"] == "revision_required"
    assert "not from the selected authority" in run.plan_state["revision_reason"]
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 0
    assert not [a for a in await _run_actions(db_session, run.id) if a.action_type == "accept_plan"]


async def test_advance_rejected_approval_requests_revision_without_accept_action(
    db_session, test_project, test_user, stub_decision,
):
    service, goal, run, artifact, _, _ = await _roadmap_acceptance(
        db_session, test_project, test_user, seed_approval=False,
    )
    _accept_decision(stub_decision, artifact)
    await service._advance_authorized_execution(db_session, goal, run)
    approval = await _pending_approval(db_session, run)
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, approval, selected_option="reject", decided_by_user_id=goal.manager_user_id,
    )
    assert await service._advance_authorized_execution(db_session, goal, run) == {"step": "plan_revision_required"}
    assert run.plan_state["status"] == "revision_required"
    assert run.plan_state["revision_reason"] == "Roadmap plan approval was rejected"
    stub_decision(lambda _: {
        "action_type": "request_plan_revision", "plan_task_id": str(artifact.linked_task_id),
        "revision_request": "Address the authority rejection.", "reason": "Authority rejected this plan.",
    })
    result = await service._advance_authorized_execution(db_session, goal, run)
    revision = await db_session.get(OrchestrationAction, uuid.UUID(result["action_id"]))
    assert revision.action_type == "request_plan_revision" and revision.status == "completed"
    assert run.plan_state["status"] == "revision_requested"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 0
    assert not [a for a in await _run_actions(db_session, run.id) if a.action_type == "accept_plan"]


async def test_no_manager_advance_ingests_independent_verification_and_accepts_once(
    db_session, test_project, test_user, stub_decision,
):
    verifier = _agent("roadmap-verifier", ["validation"])
    db_session.add(verifier)
    await db_session.flush()
    service, goal, run, artifact, _, fingerprint = await _roadmap_acceptance(
        db_session, test_project, test_user, authority_model="no_manager", seed_approval=False,
        team_agent_ids=[verifier.id],
    )
    _accept_decision(stub_decision, artifact)
    verification_id = None
    for _ in range(3):
        result = await service._advance_authorized_execution(db_session, goal, run)
        assert result["step"] == "waiting_plan_authority"
        verification_id = verification_id or result["action_id"]
        assert result["action_id"] == verification_id
        assert not [a for a in await _run_actions(db_session, run.id) if a.action_type == "accept_plan"]
    verification = await db_session.get(OrchestrationAction, uuid.UUID(verification_id))
    assert verification.status == "completed" and verification.action_type == "request_verification"
    task = await db_session.get(Task, verification.target_id)
    assert task.assigned_to == verifier.id != artifact.created_by_agent
    assert verification.request["producer_agent_id"] == str(artifact.created_by_agent)
    assert verification.request["source_task_id"] == str(artifact.linked_task_id)
    gate = await db_session.get(OrchestrationGate, uuid.UUID(verification.request["gate_id"]))
    assert gate.success_criterion_key == f"roadmap_plan:{fingerprint}"
    session = await _complete_task_session(
        db_session, test_project.id, task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["Reviewed each Roadmap item."]}),
    )
    events = await service._new_events(db_session, test_project.id, run.event_cursor)
    assert await service._ingest_evidence_from_events(db_session, run, events) >= 1
    assert await service.validate_open_gates(db_session, run.id) >= 1
    assert gate.status == "accepted"
    evidence = list((await db_session.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == gate.id,
    ))).all())
    assert len(evidence) == 1
    assert evidence[0].source_type == "verification" and evidence[0].source_id == session.id
    assert evidence[0].producer_agent_id == verifier.id and evidence[0].verdict == "accepted"
    assert evidence[0].evidence_metadata["verification_action_id"] == verification_id
    assert await service._ingest_evidence_from_events(db_session, run, events) == 0

    result = await service._advance_authorized_execution(db_session, goal, run)
    assert result["step"] == "plan_decision"
    acceptance = await db_session.get(OrchestrationAction, uuid.UUID(result["action_id"]))
    assert acceptance.action_type == "accept_plan" and acceptance.status == "completed"
    replay = await service.execute_accept_plan_action(
        db_session, run.id, acceptance.request, acceptance.idempotency_key, acceptance.decision_id,
    )
    assert replay.id == acceptance.id
    version = await db_session.scalar(select(OrchestrationRoadmapVersion))
    assert version.fingerprint == fingerprint
    assert version.approval_reference == {"kind": "gate", "id": str(gate.id)}
    assert run.plan_state["roadmap_version_id"] == str(version.id)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGate).where(
        OrchestrationGate.run_id == run.id, OrchestrationGate.gate_type == "roadmap_plan_approval",
    )) == 1
    actions = await _run_actions(db_session, run.id)
    assert len([a for a in actions if a.action_type == "request_verification"]) == 1
    assert len([a for a in actions if a.action_type == "accept_plan"]) == 1


async def test_tick_reconciles_terminal_roadmap_task_by_dispatching_one_verifier(
    db_session, test_project, test_user, stub_decision,
):
    """A terminal item reaches independent verification only through tick."""
    verifier = _agent("terminal-roadmap-verifier", ["validation"])
    db_session.add(verifier)
    await db_session.flush()
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=[_roadmap_task("terminal")], team_agent_ids=[verifier.id],
    )
    row = await _release_roadmap_item(db_session, service, goal, run, "terminal")
    producer_task = await db_session.get(Task, row.task_id)
    await _complete_task_session(
        db_session, test_project.id, producer_task, producer_task.assigned_to,
        json.dumps({"status": "done", "changes": ["terminal work"]}),
    )
    stub_decision(lambda _: {
        "action_type": "request_verification", "gate_id": str(row.gate_id),
        "work_function": "validation", "reason": "Verify terminal item.",
    })

    result = await service.tick(db_session, run.id)
    assert result["authorized_execution"]["step"] == "terminal_item_decision"
    verification = await db_session.get(
        OrchestrationAction, uuid.UUID(result["authorized_execution"]["action_id"]),
    )
    assert verification.action_type == "request_verification" and verification.status == "completed"
    assert len([action for action in await _run_actions(db_session, run.id)
                if action.action_type == "request_verification"]) == 1
    counts = [
        await db_session.scalar(select(func.count()).select_from(model).where(model.run_id == run.id))
        for model in (OrchestrationAction, OrchestrationDecision, OrchestrationGate, OrchestrationEvidence)
    ]
    waiting = await service.tick(db_session, run.id)
    assert waiting["authorized_execution"] == {"step": "waiting", "reason": "waiting_active_verification"}
    assert [
        await db_session.scalar(select(func.count()).select_from(model).where(model.run_id == run.id))
        for model in (OrchestrationAction, OrchestrationDecision, OrchestrationGate, OrchestrationEvidence)
    ] == counts


async def test_tick_releases_only_the_first_ready_roadmap_item(db_session, test_project, test_user):
    service, goal, run, _ = await _accept_version_one(
        db_session, test_project, test_user, items=[_roadmap_task("first"), _roadmap_task("second")],
    )

    result = await service.tick(db_session, run.id)

    assert result["authorized_execution"] == {"step": "release_item", "item_key": "first", "unit_type": "task"}
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    )) == 1


async def test_no_manager_replan_uses_pending_task_gate_and_accepts_v2_through_ticks(
    db_session, test_project, test_user, stub_decision,
):
    async def cardinalities():
        return (
            await db_session.scalar(select(func.count()).select_from(Task).where(
                Task.metadata_["orchestration"]["run_id"].as_string() == str(run.id),
            )),
            *[await db_session.scalar(select(func.count()).select_from(model).where(model.run_id == run.id))
              for model in (
                  OrchestrationDecision, OrchestrationAction, OrchestrationGate,
                  OrchestrationEvidence, OrchestrationRoadmapVersion,
              )],
        )

    verifier = _agent("replan-tick-verifier", ["validation"])
    wrong_verifier = _agent("wrong-replan-tick-verifier", ["validation"])
    db_session.add_all([verifier, wrong_verifier])
    await db_session.flush()
    service, goal, run, artifact, _, _ = await _roadmap_acceptance(
        db_session, test_project, test_user, authority_model="no_manager", seed_approval=False,
        team_agent_ids=[verifier.id],
    )
    artifact.metadata_["plan_items"][0]["agent_id"] = str(artifact.created_by_agent)
    _accept_decision(stub_decision, artifact)
    first_wait = await service.tick(db_session, run.id)
    first_verification = await db_session.get(
        OrchestrationAction, uuid.UUID(first_wait["authorized_execution"]["action_id"]),
    )
    first_task = await db_session.get(Task, first_verification.target_id)
    await _complete_task_session(
        db_session, test_project.id, first_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["V1 reviewed"]}),
    )
    await service.tick(db_session, run.id)
    v1 = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    assert v1 is not None and v1.version == 1

    released = await service.tick(db_session, run.id)
    assert released["authorized_execution"]["step"] == "release_item"
    v1_row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id, OrchestrationRoadmapItem.item_key == "task-1",
    ))
    v1_task = await db_session.get(Task, v1_row.task_id)
    await _complete_task_session(
        db_session, test_project.id, v1_task, v1_task.assigned_to,
        json.dumps({"status": "done", "changes": ["V1 delivered"]}),
    )
    stub_decision(lambda _: {
        "action_type": "request_verification", "gate_id": str(v1_row.gate_id),
        "work_function": "validation", "reason": "Verify V1 item.",
    })
    terminal = await service.tick(db_session, run.id)
    v1_item_verification = await db_session.get(
        OrchestrationAction, uuid.UUID(terminal["authorized_execution"]["action_id"]),
    )
    v1_item_verifier_task = await db_session.get(Task, v1_item_verification.target_id)
    await _complete_task_session(
        db_session, test_project.id, v1_item_verifier_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["V1 item reviewed"]}),
    )
    stub_decision(lambda _: {
        "action_type": "request_roadmap_replan", "agent_id": str(artifact.created_by_agent),
        "scope": "Revise remaining work.", "reason": "New information.",
    })
    integration_tick = await service.tick(db_session, run.id)
    assert integration_tick["authorized_execution"]["step"] == "integration_decision"
    integration_verification = await db_session.get(
        OrchestrationAction, uuid.UUID(integration_tick["authorized_execution"]["action_id"]),
    )
    integration_task = await db_session.get(Task, integration_verification.target_id)
    await _complete_task_session(
        db_session, test_project.id, integration_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["V1 integration reviewed"]}),
    )
    replan_tick = await service.tick(db_session, run.id)
    assert replan_tick["authorized_execution"]["step"] == "replan_decision"
    replan = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "request_roadmap_replan",
    ))
    assert replan is not None
    v1_gate = await db_session.get(OrchestrationGate, v1_row.gate_id)
    assert v1_gate.status == "accepted"
    candidate = Artifact(
        project_id=test_project.id, name="tick-v2", artifact_type="plan", status="draft",
        linked_task_id=replan.target_id, created_by_agent=artifact.created_by_agent,
        metadata_={"plan_items": [
            artifact.metadata_["plan_items"][0],
            {**_roadmap_task("v2"), "agent_id": str(artifact.created_by_agent)},
        ]},
    )
    db_session.add(candidate)
    await db_session.flush()
    _accept_decision(stub_decision, candidate)
    waiting = await service.tick(db_session, run.id)
    assert waiting["authorized_execution"]["step"] == "waiting_plan_authority"
    second_verification = await db_session.get(
        OrchestrationAction, uuid.UUID(waiting["authorized_execution"]["action_id"]),
    )
    approval_gate = await db_session.get(OrchestrationGate, uuid.UUID(second_verification.request["gate_id"]))
    assert second_verification.request["source_task_id"] == str(replan.target_id)
    assert approval_gate.required_evidence["planning_task_id"] == str(replan.target_id)
    assert run.plan_state["pending_replan"] == {
        "task_id": str(replan.target_id), "action_id": str(replan.id), "version_id": str(v1.id),
        "planner_agent_id": str(artifact.created_by_agent), "artifact_id": str(candidate.id),
        "fingerprint": _roadmap_fingerprint(service, goal, candidate.metadata_["plan_items"]),
        "gate_id": str(approval_gate.id),
    }
    counts = await cardinalities()
    replay_wait = await service.tick(db_session, run.id)
    assert replay_wait["authorized_execution"] == {"step": "waiting", "reason": "waiting_active_verification"}
    assert await cardinalities() == counts
    await _complete_task_session(
        db_session, test_project.id, first_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["Stale V1 reviewed"]}),
    )
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "waiting_active_verification",
    }
    assert await cardinalities() == counts
    second_task = await db_session.get(Task, second_verification.target_id)
    second_task_status = second_task.status
    await _complete_task_session(
        db_session, test_project.id, second_task, wrong_verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["Wrong V2 reviewed"]}),
    )
    second_task.status = second_task_status
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "waiting_active_verification",
    }
    assert await cardinalities() == counts
    await _complete_task_session(
        db_session, test_project.id, second_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["V2 reviewed"]}),
    )

    accepted = await service.tick(db_session, run.id)
    assert accepted["authorized_execution"]["step"] == "replan_decision"
    current = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    assert current.version == 2 and run.plan_state["pending_replan"] is None
    accepted_counts = await cardinalities()
    accept_action = await db_session.get(
        OrchestrationAction, uuid.UUID(accepted["authorized_execution"]["action_id"]),
    )
    replay = await service.execute_accept_plan_action(
        db_session, run.id, accept_action.request, accept_action.idempotency_key, accept_action.decision_id,
    )
    assert replay.id == accept_action.id and await cardinalities() == accepted_counts


async def test_no_manager_gate_collision_refetches_the_deterministic_winner(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, artifact, _, fingerprint = await _roadmap_acceptance(
        db_session, test_project, test_user, authority_model="no_manager",
    )
    roadmap = OrchestrationRoadmapService(service)
    winner = await roadmap.ensure_plan_approval(db_session, goal, run, artifact)
    gate_id = winner.id
    required_evidence = deepcopy(winner.required_evidence)
    db_session.expunge(winner)
    original_scalar = db_session.scalar
    reads = 0

    async def stale_first_read(statement, *args, **kwargs):
        nonlocal reads
        reads += 1
        # Simulate a writer winning immediately after the initial lookup.
        # The real INSERT then hits the database PK and must roll back/refetch.
        if reads == 1:
            return None
        return await original_scalar(statement, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(db_session, "scalar", stale_first_read)
        recovered = await roadmap.ensure_plan_approval(db_session, goal, run, artifact)
    assert reads == 2
    assert recovered.id == gate_id == uuid.uuid5(
        uuid.NAMESPACE_URL, f"rally:orchestration:gate:{run.id}:roadmap_plan_approval:{fingerprint}",
    )
    assert recovered.required_evidence == required_evidence
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGate).where(
        OrchestrationGate.id == gate_id,
    )) == 1
    assert db_session.is_active


async def test_unapproved_replan_preserves_accepted_version_gate_and_projection(
    db_session, test_project, test_user, test_engine,
):
    service, goal, run, artifact, decision, fingerprint = await _roadmap_acceptance(
        db_session, test_project, test_user,
    )
    await _approve(db_session, goal, decision)
    await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"run:{run.id}:kind:version-one",
    )
    version = await db_session.scalar(select(OrchestrationRoadmapVersion))
    version_id, run_id = version.id, run.id
    snapshot, projection = deepcopy(version.snapshot), deepcopy(run.plan_state)
    successor = Artifact(
        project_id=test_project.id, name="successor", artifact_type="plan", status="draft",
        linked_task_id=artifact.linked_task_id, created_by_agent=artifact.created_by_agent,
        metadata_={"plan_items": [_roadmap_task("different-item")]},
    )
    db_session.add(successor)
    await db_session.flush()
    successor_id = successor.id
    roadmap = OrchestrationRoadmapService(service)
    with pytest.raises(HTTPException, match="Roadmap plan approval is required"):
        async with db_session.begin_nested():
            await roadmap.accept_version(db_session, goal, run, successor, None)
    assert (await roadmap.accept_version(db_session, goal, run, artifact, None)).id == version_id
    with pytest.raises(HTTPException, match="Plan can only be accepted"):
        await service.execute_accept_plan_action(
            db_session, run_id, {"action_type": "accept_plan", "plan_artifact_id": str(successor_id)},
            f"run:{run_id}:kind:successor",
        )
    await db_session.commit()
    async with async_sessionmaker(test_engine)() as reader:
        versions = list((await reader.scalars(select(OrchestrationRoadmapVersion))).all())
        assert len(versions) == 1 and versions[0].id == version_id
        assert versions[0].snapshot == snapshot and versions[0].fingerprint == fingerprint
        assert (await reader.get(OrchestrationRun, run_id)).plan_state == projection
        assert (await reader.get(OrchestrationGate, uuid.UUID(projection["plan_gate_id"]))).status == "accepted"
        assert len([a for a in await _run_actions(reader, run_id) if a.action_type == "accept_plan"]) == 1


async def test_replan_appends_version_two_without_updating_version_one(
    db_session, test_project, test_user,
):
    service, goal, run, artifact, decision, _ = await _roadmap_acceptance(
        db_session, test_project, test_user,
    )
    await _approve(db_session, goal, decision)
    await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"run:{run.id}:kind:version-one",
    )
    first_snapshot = deepcopy((await db_session.scalar(select(OrchestrationRoadmapVersion))).snapshot)
    replan = await _request_replan(db_session, service, goal, run, artifact)
    successor = Artifact(
        project_id=test_project.id, name="successor", artifact_type="plan", status="draft",
        linked_task_id=replan.target_id, created_by_agent=artifact.created_by_agent,
        metadata_={"plan_items": [_roadmap_task("task-2")]},
    )
    db_session.add(successor)
    await db_session.flush()
    fingerprint = service._accepted_plan_fingerprint([
        item.model_dump(mode="json")
        for item in parse_roadmap_items(successor.metadata_["plan_items"], set())
    ])
    with pytest.raises(HTTPException, match="waiting_plan_authority"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(successor.id)},
            f"run:{run.id}:kind:replan-accept-wait",
        )
    approval = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key == f"roadmap_plan:{fingerprint}",
    ))
    await _approve(db_session, goal, approval)
    acceptance = await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(successor.id)},
        f"run:{run.id}:kind:replan-accept",
    )
    second = await db_session.get(OrchestrationRoadmapVersion, uuid.UUID(run.plan_state["roadmap_version_id"]))

    versions = list((await db_session.scalars(select(OrchestrationRoadmapVersion).order_by(
        OrchestrationRoadmapVersion.version
    ))).all())
    assert [version.version for version in versions] == [1, 2]
    assert versions[0].snapshot == first_snapshot
    assert second.snapshot["items"][0]["item_key"] == "task-2"
    assert acceptance.status == "completed" and run.plan_state["pending_replan"] is None


async def test_replan_may_change_add_or_remove_only_unstarted_items(db_session, test_project, test_user):
    original = [_roadmap_task("a"), _roadmap_task("b"), _roadmap_task("remove")]
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=original,
    )
    released = await _release_roadmap_item(db_session, service, goal, run, "a")
    changed = _roadmap_task("b")
    changed["scope"] = "Implement revised b"
    _, candidate = await _replan_artifact(
        db_session, test_project, service, goal, run, artifact,
        [deepcopy(original[0]), changed, _roadmap_task("added")],
    )
    await _accept_replan_after_selected_authority(
        db_session, service, goal, run, candidate, f"run:{run.id}:kind:accept-unstarted",
    )

    versions = list((await db_session.scalars(select(OrchestrationRoadmapVersion).order_by(
        OrchestrationRoadmapVersion.version
    ))).all())
    assert [version.version for version in versions] == [1, 2]
    assert versions[0].snapshot["items"] == _normalized_roadmap_items(goal, original)
    assert {item["item_key"] for item in versions[1].snapshot["items"]} == {"a", "b", "added"}
    assert released.item_snapshot == _normalized_roadmap_items(goal, [original[0]])[0]
    assert released.first_version_id == versions[0].id


async def test_replan_retains_the_initial_workspace_policy_when_project_config_drifts(
    db_session, test_project, test_user,
):
    test_project.config = {"workspace_policy": {"mode": "isolated"}}
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=[_roadmap_task("initial")],
    )
    first = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    test_project.config = {"workspace_policy": {"mode": "shared"}}
    _, candidate = await _replan_artifact(
        db_session, test_project, service, goal, run, artifact, [_roadmap_task("replacement")],
    )
    await _accept_replan_after_selected_authority(
        db_session, service, goal, run, candidate, f"run:{run.id}:kind:frozen-workspace-policy",
    )

    second = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    assert first.snapshot["workspace_policy"] == second.snapshot["workspace_policy"] == {"mode": "isolated"}


async def test_replan_rejects_started_item_removal(db_session, test_project, test_user):
    original = [_roadmap_task("a"), _roadmap_task("b")]
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=original,
    )
    await _release_roadmap_item(db_session, service, goal, run, "a")
    _, candidate = await _replan_artifact(db_session, test_project, service, goal, run, artifact, [_roadmap_task("b")])

    with pytest.raises(HTTPException, match="Released roadmap item 'a' cannot be removed"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(candidate.id)},
            f"run:{run.id}:kind:remove-started",
        )

    current = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    assert current.version == 1 and run.plan_state["roadmap_version_id"] == str(current.id)


@pytest.mark.parametrize("change", ["unit_type", "dependencies", "allocation", "staging"])
async def test_replan_rejects_started_item_snapshot_change(db_session, test_project, test_user, change):
    budget = {"max_tokens": 1_000, "max_turns": 10, "max_hours": 10}
    if change == "allocation":
        original = _roadmap_goal("a", allocation={"max_tokens": 100, "max_turns": 1, "max_hours": 1})
        candidate_item = deepcopy(original)
        candidate_item["allocation"]["max_tokens"] = 101
        items, candidate_items, unit_type = [original], [candidate_item], "goal"
    elif change == "dependencies":
        original = _roadmap_task("a")
        candidate_item = deepcopy(original)
        candidate_item["depends_on"] = ["b"]
        items, candidate_items, unit_type = [original, _roadmap_task("b")], [candidate_item, _roadmap_task("b")], "task"
    elif change == "unit_type":
        original = _roadmap_task("a")
        candidate_item = _roadmap_goal("a", allocation={"max_tokens": 100, "max_turns": 1, "max_hours": 1})
        items, candidate_items, unit_type = [original], [candidate_item], "task"
    else:
        original = _roadmap_task(
            "a", staging_boundary={"type": "git_worktree", "identifier": "roadmap/a", "reversible": True},
        )
        candidate_item = deepcopy(original)
        candidate_item["staging_boundary"]["identifier"] = "roadmap/changed-a"
        items, candidate_items, unit_type = [original], [candidate_item], "task"

    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=items, budget=budget,
    )
    released = await _release_roadmap_item(db_session, service, goal, run, "a", unit_type)
    _, candidate = await _replan_artifact(db_session, test_project, service, goal, run, artifact, candidate_items)

    with pytest.raises(HTTPException, match="Released roadmap item 'a' cannot be changed"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(candidate.id)},
            f"run:{run.id}:kind:changed-started:{change}",
        )

    assert released.item_snapshot == _normalized_roadmap_items(goal, [original])[0]
    assert (await OrchestrationRoadmapService(service).current_version(db_session, goal.id)).version == 1


async def test_replan_rejects_increased_allocation_beyond_remaining(db_session, test_project, test_user):
    budget = {"max_tokens": 600, "max_turns": 10, "max_hours": 10}
    held = _roadmap_goal("held", allocation={"max_tokens": 400, "max_turns": 1, "max_hours": 1})
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=[held], budget=budget,
    )
    await _release_roadmap_item(db_session, service, goal, run, "held", "goal")
    _, fitting = await _replan_artifact(
        db_session, test_project, service, goal, run, artifact,
        [deepcopy(held), _roadmap_goal("fits", allocation={"max_tokens": 200, "max_turns": 1, "max_hours": 1})],
    )
    fingerprint = _roadmap_fingerprint(service, goal, fitting.metadata_["plan_items"])
    with pytest.raises(HTTPException, match="waiting_human_scope_budget_authority"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(fitting.id)},
            f"run:{run.id}:kind:fits-wait-human",
        )
    expansion = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.decision_key == f"roadmap_expansion:{fingerprint}",
    ))
    await _approve(db_session, goal, expansion)
    await _accept_replan_after_selected_authority(
        db_session, service, goal, run, fitting, f"run:{run.id}:kind:fits",
    )
    _, over_budget = await _replan_artifact(
        db_session, test_project, service, goal, run, artifact,
        [deepcopy(held), _roadmap_goal("fits", allocation={"max_tokens": 200, "max_turns": 1, "max_hours": 1}),
         _roadmap_goal("over", allocation={"max_tokens": 201, "max_turns": 1, "max_hours": 1})],
    )

    with pytest.raises(HTTPException, match="Roadmap child allocation exceeds remaining parent budget"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(over_budget.id)},
            f"run:{run.id}:kind:over-budget",
        )
    assert (await OrchestrationRoadmapService(service).current_version(db_session, goal.id)).version == 2


@pytest.mark.parametrize("branch", ["integration", "terminal_item"])
async def test_fresh_session_replan_persists_measurement_attention_without_accept_action(
    db_session, test_project, test_user, test_engine, monkeypatch, branch,
):
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=[_roadmap_task("base")],
        budget={"max_tokens": 100, "max_turns": 10, "max_hours": 10},
    )
    row = await _release_roadmap_item(db_session, service, goal, run, "base")
    gate = await db_session.get(OrchestrationGate, row.gate_id)
    if branch == "integration":
        gate.status = "accepted"
    else:
        (await db_session.get(Task, row.task_id)).status = "done"
    _, candidate = await _replan_artifact(
        db_session, test_project, service, goal, run, artifact, [_roadmap_task("base")],
    )
    task = Task(
        project_id=test_project.id, title="unmeasured", assigned_to=artifact.created_by_agent, status="done",
        metadata_={"orchestration": {"run_id": str(run.id), "work_function": "planning"}},
    )
    db_session.add(task)
    await db_session.flush()
    db_session.add(OrchestrationAction(
        run_id=run.id, idempotency_key=f"test:claim-lineage:{run.id}:{task.id}",
        action_type="create_delegation_task", request={"agent_id": str(artifact.created_by_agent)},
        target_type="task", target_id=task.id, status="completed",
    ))
    db_session.add(Session(
        project_id=test_project.id, task_id=task.id, agent_id=artifact.created_by_agent,
        adapter_type="test", status="completed", started_at=run.started_at,
        ended_at=run.started_at + timedelta(minutes=1), metadata_={},
    ))
    await db_session.commit()

    original_remaining = OrchestrationRoadmapService.remaining_or_block
    summary = {"caps": goal.budget["caps"], "direct_spend": {key: "0" for key in goal.budget["caps"]},
               "settled_child_spend": {key: "0" for key in goal.budget["caps"]},
               "active_reservations": {key: "0" for key in goal.budget["caps"]},
               "remaining": {key: str(value) for key, value in goal.budget["caps"].items()}}
    calls = 0

    async def staged_remaining(*args):
        nonlocal calls
        calls += 1
        return summary if calls == 1 else await original_remaining(*args)

    async def accept_replan(*_args):
        return SimpleNamespace(
            id=uuid.uuid4(), validator_status="accepted",
            parsed_decision={"action_type": "accept_plan", "plan_artifact_id": str(candidate.id)},
        )

    monkeypatch.setattr(OrchestrationRoadmapService, "remaining_or_block", staged_remaining)
    monkeypatch.setattr(service, "request_llm_decision", accept_replan)
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as reader:
        result = await service.tick(reader, run.id)
        assert result["authorized_execution"] == {"step": "waiting", "reason": "needs_attention"}

    async with async_sessionmaker(test_engine)() as reader:
        persisted = await reader.get(OrchestrationRun, run.id)
        assert any(item.get("scope") == f"parent:{goal.id}" for item in persisted.active_blockers)
        assert await reader.scalar(select(func.count()).select_from(OrchestrationAction).where(
            OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "accept_plan",
        )) == 1


@pytest.mark.parametrize("expansion", ["new_goal", "increased_allocation"])
async def test_replan_new_goal_or_increased_allocation_requires_human_approval(
    db_session, test_project, test_user, expansion,
):
    budget = {"max_tokens": 1_000, "max_turns": 10, "max_hours": 10}
    base = _roadmap_goal("a", allocation={"max_tokens": 100, "max_turns": 1, "max_hours": 1})
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=[base], budget=budget, authority_model="agent_manager",
    )
    candidate_items = [deepcopy(base)]
    if expansion == "new_goal":
        candidate_items.append(_roadmap_goal("new", allocation={"max_tokens": 100, "max_turns": 1, "max_hours": 1}))
    else:
        candidate_items[0]["allocation"]["max_tokens"] = 101
    _, candidate = await _replan_artifact(db_session, test_project, service, goal, run, artifact, candidate_items)
    fingerprint = _roadmap_fingerprint(service, goal, candidate_items)

    with pytest.raises(HTTPException, match="waiting_human_scope_budget_authority"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(candidate.id)},
            f"run:{run.id}:kind:expansion-human:{expansion}",
        )
    expansion_decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key == f"roadmap_expansion:{fingerprint}",
    ))
    assert expansion_decision.authority == "human" and expansion_decision.status == "pending"
    assert len([action for action in await _run_actions(db_session, run.id) if action.action_type == "accept_plan"]) == 1
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, expansion_decision, selected_option="approve", decided_by_user_id=goal.created_by_user_id,
    )
    acceptance = await _accept_replan_after_selected_authority(
        db_session, service, goal, run, candidate, f"run:{run.id}:kind:expansion-accept:{expansion}",
    )
    assert acceptance.status == "completed"
    assert (await OrchestrationRoadmapService(service).current_version(db_session, goal.id)).fingerprint == fingerprint


async def test_rejected_replan_leaves_current_version_projection_unchanged(db_session, test_project, test_user):
    original = [_roadmap_task("a"), _roadmap_task("b")]
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=original,
    )
    released = await _release_roadmap_item(db_session, service, goal, run, "a")
    version = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    projection, released_snapshot = deepcopy(run.plan_state), deepcopy(released.item_snapshot)
    replan, candidate = await _replan_artifact(
        db_session, test_project, service, goal, run, artifact, [_roadmap_task("b")],
    )

    with pytest.raises(HTTPException, match="Released roadmap item 'a' cannot be removed"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(candidate.id)},
            f"run:{run.id}:kind:reject-preserves-v1",
        )

    current = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    persisted = await db_session.get(OrchestrationRoadmapItem, released.id)
    assert (current.id, current.fingerprint) == (version.id, version.fingerprint)
    assert {key: value for key, value in run.plan_state.items() if key != "pending_replan"} == projection
    assert run.plan_state["pending_replan"] == {
        "task_id": str(replan.target_id), "action_id": str(replan.id), "version_id": str(version.id),
        "planner_agent_id": str(artifact.created_by_agent),
    }
    assert (persisted.first_version_id, persisted.item_snapshot) == (version.id, released_snapshot)


async def test_historical_v1_fingerprint_replay_preserves_v2_projection_and_lineage(
    db_session, test_project, test_user,
):
    async def state():
        await db_session.refresh(run)
        return {
            "projection": deepcopy(run.plan_state),
            "versions": [
                (version.id, version.version, version.fingerprint)
                for version in await db_session.scalars(select(OrchestrationRoadmapVersion).where(
                    OrchestrationRoadmapVersion.goal_id == goal.id,
                ).order_by(OrchestrationRoadmapVersion.version))
            ],
            "items": [
                (item.id, item.first_version_id, item.item_key, deepcopy(item.item_snapshot))
                for item in await db_session.scalars(select(OrchestrationRoadmapItem).where(
                    OrchestrationRoadmapItem.goal_id == goal.id,
                ).order_by(OrchestrationRoadmapItem.item_key))
            ],
            "actions": [
                (action.id, action.status, action.error, action.target_id)
                for action in await db_session.scalars(select(OrchestrationAction).where(
                    OrchestrationAction.run_id == run.id,
                ).order_by(OrchestrationAction.id))
            ],
        }

    v1_items = [_roadmap_task("keep")]
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=v1_items,
    )
    v1 = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    released = await _release_roadmap_item(db_session, service, goal, run, "keep")
    v2_items = [*v1_items, _roadmap_task("added")]
    _, v2_candidate = await _replan_artifact(
        db_session, test_project, service, goal, run, artifact, v2_items,
    )
    await _accept_replan_after_selected_authority(
        db_session, service, goal, run, v2_candidate, f"run:{run.id}:kind:accept-v2",
    )
    v2 = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    _, historical_candidate = await _replan_artifact(
        db_session, test_project, service, goal, run, artifact, v1_items,
    )
    key = f"run:{run.id}:kind:reject-historical-v1"

    with pytest.raises(HTTPException, match="Historical Roadmap fingerprint cannot replace the current version"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(historical_candidate.id)},
            key,
        )

    current = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    assert (v1.version, current.id, current.fingerprint) == (1, v2.id, v2.fingerprint)
    assert {key: run.plan_state[key] for key in (
        "roadmap_version_id", "roadmap_version", "accepted_plan_fingerprint",
    )} == {
        "roadmap_version_id": str(v2.id), "roadmap_version": 2,
        "accepted_plan_fingerprint": v2.fingerprint,
    }
    assert run.plan_state["pending_replan"] == {
        "task_id": str(historical_candidate.linked_task_id),
        "action_id": str((await db_session.scalar(select(OrchestrationAction).where(
            OrchestrationAction.target_id == historical_candidate.linked_task_id,
        ))).id),
        "version_id": str(v2.id),
        "planner_agent_id": str(artifact.created_by_agent),
        "artifact_id": str(historical_candidate.id),
        "fingerprint": _roadmap_fingerprint(service, goal, v1_items),
    }
    stable = await state()
    replay = await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(historical_candidate.id)}, key,
    )
    assert replay.status == "failed"
    assert await state() == stable
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion).where(
        OrchestrationRoadmapVersion.goal_id == goal.id,
    )) == 2
    assert (await db_session.get(OrchestrationRoadmapItem, released.id)).item_snapshot == released.item_snapshot


async def test_replan_request_replay_creates_one_planning_task(db_session, test_project, test_user):
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=[_roadmap_task("a")],
    )
    current = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    key = f"run:{run.id}:kind:request_roadmap_replan:version:{current.version}"
    first = await service.execute_request_roadmap_replan_action(
        db_session, run.id,
        {"action_type": "request_roadmap_replan", "agent_id": str(artifact.created_by_agent),
         "scope": "Revise remaining work.", "reason": "New information."}, key,
    )
    replay = await service.execute_request_roadmap_replan_action(
        db_session, run.id,
        {"action_type": "request_roadmap_replan", "agent_id": str(artifact.created_by_agent),
         "scope": "ignored on replay", "reason": "ignored on replay"}, key,
    )
    task = await db_session.get(Task, first.target_id)
    assert replay.id == first.id and replay.target_id == first.target_id == task.id
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.idempotency_key == key,
    )) == 1
    assert await db_session.scalar(select(func.count()).select_from(Task).where(Task.id == first.target_id)) == 1


async def test_replan_task_budget_lineage_ignores_task_metadata(db_session, test_project, test_user):
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=[_roadmap_task("a")],
        budget={"max_tokens": 100, "max_turns": 10, "max_hours": 10},
    )
    replan = await _request_replan(db_session, service, goal, run, artifact)
    task = await db_session.get(Task, replan.target_id)
    task.metadata_ = {}
    db_session.add(Session(
        project_id=test_project.id, task_id=task.id, agent_id=artifact.created_by_agent,
        adapter_type="test", status="completed", started_at=run.started_at,
        ended_at=run.started_at + timedelta(hours=1),
        metadata_={"token_count_in": 20, "token_count_out": 30},
    ))
    await db_session.flush()

    budget = OrchestrationBudgetService()
    assert await budget.owning_roadmap_run(db_session, task) == run
    assert await budget.measured_run_spend(db_session, run, {"max_tokens", "max_turns", "max_hours"}) == {
        "max_tokens": "50", "max_turns": "1", "max_hours": "1",
    }

    task.metadata_ = {"orchestration": {"run_id": str(uuid.uuid4())}}
    active = Session(
        project_id=test_project.id, task_id=task.id, agent_id=artifact.created_by_agent,
        adapter_type="test", status="running",
        metadata_={"_run_config": {"max_tokens": 25, "timeout": 3600}},
    )
    db_session.add(active)
    await db_session.flush()
    assert await budget.owning_roadmap_run(db_session, task) == run
    assert await budget.active_run_commitments(
        db_session, run, {"max_tokens", "max_turns", "max_hours"}, goal.budget["caps"],
    ) == {"max_tokens": "25", "max_turns": "1", "max_hours": "1"}


async def test_replan_task_missing_telemetry_blocks_measured_spend(db_session, test_project, test_user):
    service, goal, run, artifact = await _accept_version_one(
        db_session, test_project, test_user, items=[_roadmap_task("a")],
    )
    replan = await _request_replan(db_session, service, goal, run, artifact)
    task = await db_session.get(Task, replan.target_id)
    task.metadata_ = {}
    db_session.add(Session(
        project_id=test_project.id, task_id=task.id, agent_id=artifact.created_by_agent,
        adapter_type="test", status="completed", metadata_={},
    ))
    await db_session.flush()

    with pytest.raises(BudgetMeasurementError):
        await OrchestrationBudgetService().measured_run_spend(db_session, run, {"max_tokens"})


async def test_pending_replan_rejects_stale_v1_artifact_and_wrong_planner(
    db_session, test_project, test_user,
):
    service, goal, run, v1_artifact = await _accept_version_one(
        db_session, test_project, test_user, items=[_roadmap_task("v1")],
    )
    replan = await _request_replan(db_session, service, goal, run, v1_artifact)
    other = _agent("wrong-replan-planner", ["planning"])
    db_session.add(other)
    wrong_planner = Artifact(
        project_id=test_project.id, name="wrong-planner", artifact_type="plan", status="draft",
        linked_task_id=replan.target_id, created_by_agent=other.id,
        metadata_={"plan_items": [_roadmap_task("v1")]},
    )
    db_session.add(wrong_planner)
    await db_session.flush()
    with pytest.raises(HTTPException, match="assigned planning agent"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(wrong_planner.id)},
            f"run:{run.id}:kind:wrong-replan-planner",
        )
    with pytest.raises(HTTPException, match="current planning task"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(v1_artifact.id)},
            f"run:{run.id}:kind:stale-v1-replan",
        )


@pytest.mark.parametrize("boundary", ["version_insert", "plan_gate_acceptance", "projection_flush"])
async def test_admission_failure_rolls_back_version_gate_and_projection_but_commits_failure_audit(
    db_session, test_project, test_user, test_engine, monkeypatch, boundary,
):
    service, goal, run, artifact, decision, fingerprint = await _roadmap_acceptance(
        db_session, test_project, test_user,
    )
    await _approve(db_session, goal, decision)
    run_id = run.id
    projection_before = deepcopy(run.plan_state)
    gate_id = uuid.UUID(projection_before["plan_gate_id"])
    gate = await db_session.get(OrchestrationGate, gate_id)
    gate_status_before = gate.status
    reason = f"Injected failure after {boundary}"
    injected = False
    original_accept = OrchestrationRoadmapService.accept_version
    original_gate = service._accept_plan_gate
    original_flush = db_session.flush

    def fail():
        nonlocal injected
        assert db_session.in_nested_transaction()
        injected = True
        raise HTTPException(status_code=409, detail=reason)

    async def fail_after_version(*args, **kwargs):
        version = await original_accept(*args, **kwargs)
        assert await db_session.get(OrchestrationRoadmapVersion, version.id) is version
        fail()

    async def fail_after_gate(*args, **kwargs):
        accepted_gate = await original_gate(*args, **kwargs)
        assert accepted_gate.status == "accepted"
        fail()

    async def fail_after_projection(*args, **kwargs):
        await original_flush(*args, **kwargs)
        if not injected and run.plan_state.get("roadmap_version_id"):
            assert run.plan_state["status"] == "accepted"
            fail()

    with monkeypatch.context() as patch:
        if boundary == "version_insert":
            patch.setattr(OrchestrationRoadmapService, "accept_version", fail_after_version)
        elif boundary == "plan_gate_acceptance":
            patch.setattr(service, "_accept_plan_gate", fail_after_gate)
        else:
            patch.setattr(db_session, "flush", fail_after_projection)
        with pytest.raises(HTTPException, match=reason):
            await service.execute_accept_plan_action(
                db_session, run_id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
                f"run:{run_id}:kind:atomic-failure:{boundary}",
            )
    assert injected
    await db_session.commit()
    async with async_sessionmaker(test_engine)() as reader:
        persisted_run = await reader.get(OrchestrationRun, run_id)
        await _assert_rejected_atomic(reader, persisted_run)
        assert persisted_run.plan_state == {
            **projection_before, "status": "revision_required", "revision_reason": reason,
        }
        persisted_gate = await reader.get(OrchestrationGate, gate_id)
        assert persisted_gate.status == gate_status_before != "accepted"
        assert persisted_gate.accepted_at is None
        assert await reader.scalar(select(func.count()).select_from(OrchestrationEvidence).where(
            OrchestrationEvidence.gate_id == gate_id, OrchestrationEvidence.verdict == "accepted",
        )) == 0
        actions = [a for a in await _run_actions(reader, run_id) if a.action_type == "accept_plan"]
        assert len(actions) == 1 and actions[0].status == "failed"
        assert actions[0].error == reason
        assert (await reader.get(OrchestrationAuthorityDecision, decision.id)).status == "answered"
        assert await reader.scalar(select(func.count()).select_from(EventLog).where(
            EventLog.project_id == test_project.id, EventLog.event_type == "orchestration.plan_accepted",
        )) == 0
