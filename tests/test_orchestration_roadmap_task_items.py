import asyncio
import json
import subprocess
import sys
import uuid
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.adapters.cli_adapter import CliAdapter
from huddleroom.config import settings
from huddleroom.models.artifact import Artifact
from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRoadmapItem,
    OrchestrationRun,
)
from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService, parse_roadmap_items
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_budget_service import OrchestrationBudgetService
from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper
from huddleroom.services.project_service import ProjectService
from huddleroom.services.session_service import SessionService
from huddleroom.services.task_service import TaskService
from huddleroom.schemas.session import SessionCreate
from huddleroom.workers.session_tasks import _require_runnable_session

from tests.test_orchestration_runtime_e2e import _agent, _authorized_run, _complete_task_session
from huddleroom.services.orchestration_service import OrchestrationService


@pytest.fixture(autouse=True)
def _manual_task_start(monkeypatch):
    # These tests drive the release-then-manual-run flow; auto-start is covered in test_orchestration_task_autostart.py.
    async def _noop(self, db, goal, run):
        return 0
    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", _noop)


pytestmark = pytest.mark.asyncio


def roadmap_task(key, *, depends_on=(), mutates_shared_state=False):
    return {
        "item_key": key,
        "unit_type": "task",
        "title": key,
        "depends_on": list(depends_on),
        "mutates_shared_state": mutates_shared_state,
        "staging_boundary": {"type": "git_worktree", "identifier": f"roadmap/{key}", "reversible": True},
        "work_function": "implementation",
        "scope": f"Implement {key}",
        "deliverable": f"Verified {key}",
    }


@pytest.fixture
async def accepted_roadmap(db_session, test_project):
    planner = _agent("roadmap-planner", ["planning", "implementation"])
    verifier = _agent("roadmap-verifier", ["validation"])
    db_session.add_all([planner, verifier])
    await db_session.flush()

    async def create(items, setup=None, assign_agents=True):
        from huddleroom.services.orchestration_roadmap_service import parse_roadmap_items

        if assign_agents:
            items = [{**item, "agent_id": str(planner.id)} for item in items]
        service, goal, run = await _authorized_run(db_session, test_project)
        goal.goal_type = "roadmap"
        if setup is not None:
            await setup(planner, goal, run)
        goal.orchestrator_context = {"team": {"agent_ids": [str(agent_id) for agent_id in await db_session.scalars(
            select(Agent.id).where(Agent.is_active.is_(True))
        )]}}
        request = await service.execute_request_plan_action(
            db_session, run.id,
            {"action_type": "request_plan", "agent_id": str(planner.id), "work_function": "planning", "scope": "Plan Roadmap work."},
            f"run:{run.id}:kind:roadmap-request",
        )
        artifact = Artifact(
            project_id=test_project.id, name="roadmap-plan", artifact_type="plan", status="draft",
            linked_task_id=request.target_id, created_by_agent=planner.id, metadata_={"plan_items": items},
        )
        db_session.add(artifact)
        await db_session.flush()
        normalized = [item.model_dump(mode="json") for item in parse_roadmap_items(items, set())]
        fingerprint = service._accepted_plan_fingerprint(normalized)
        decision = await OrchestrationAuthorityDecisionService().create_pending(
            db_session, goal.id, decision_key=f"roadmap_plan:{fingerprint}", title="Approve Roadmap plan",
            question="Approve?", authority="human", options=[{"key": "approve"}], run_id=run.id,
        )
        await OrchestrationAuthorityDecisionService().answer_decision(
            db_session, decision, selected_option="approve", decided_by_user_id=goal.manager_user_id,
        )
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
            f"run:{run.id}:kind:roadmap-accept",
        )
        return service, goal, run, verifier

    return create


async def test_roadmap_releases_one_dependency_ready_task_and_records_lineage(
    db_session, accepted_roadmap,
):
    service, goal, run, _ = await accepted_roadmap([
        roadmap_task("build"),
        roadmap_task("verify", depends_on=["build"]),
    ])

    result = await service.tick(db_session, run.id)
    rows = list((await db_session.scalars(
        select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == goal.id)
    )).all())

    assert result["authorized_execution"] == {"step": "release_item", "item_key": "build", "unit_type": "task"}
    assert [(row.item_key, row.unit_type) for row in rows] == [("build", "task")]
    assert rows[0].task_id is not None and rows[0].child_goal_id is None


async def test_capped_staged_cli_task_waits_for_budget_authority_before_release(
    db_session, test_project, accepted_roadmap, monkeypatch, tmp_path,
):
    """Unsupported CLI token caps need human authority before task artifacts exist."""
    async def workspace(*_args, **_kwargs):
        return tmp_path

    monkeypatch.setattr(ProjectService, "require_roadmap_workspace", workspace)
    monkeypatch.setattr(ProjectService, "require_frozen_roadmap_workspace", workspace)

    async def setup(planner, goal, _run):
        planner.adapter_type = "cli"
        goal.budget = {"caps": {"max_tokens": 400, "max_turns": 2, "max_hours": 1}}

    service, goal, run, _ = await accepted_roadmap([roadmap_task("capped")], setup=setup)
    before = await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem))

    result = await service.tick(db_session, run.id)

    assert result["authorized_execution"] == {"step": "waiting", "reason": "budget_wait"}
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem)) == before
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_budget_adapter:%:capped:%:cli:max_tokens"),
    ))
    assert decision is not None and decision.status == "pending"


async def test_capped_cli_root_creator_authority_binds_release_and_claim(
    db_session, test_project, accepted_roadmap, monkeypatch, tmp_path,
):
    async def workspace(*_args, **_kwargs):
        return tmp_path

    monkeypatch.setattr(ProjectService, "require_roadmap_workspace", workspace)
    monkeypatch.setattr(ProjectService, "require_frozen_roadmap_workspace", workspace)

    async def setup(planner, goal, _run):
        planner.adapter_type = "cli"
        goal.budget = {"caps": {"max_tokens": 400, "max_turns": 2, "max_hours": 1}}

    service, goal, run, _ = await accepted_roadmap([roadmap_task("forged")], setup=setup)
    creator = User(email="roadmap-root-creator@example.test", hashed_password="x")
    db_session.add(creator)
    await db_session.flush()
    goal.manager_user_id = None
    goal.created_by_user_id = creator.id
    await service.tick(db_session, run.id)
    version = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like(f"roadmap_budget_adapter:{version.id}:forged:%:cli:max_tokens"),
        OrchestrationAuthorityDecision.status == "pending",
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", decided_by_user_id=creator.id,
    )
    await service.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    ))
    task = await db_session.get(Task, row.task_id)
    context = task.metadata_["orchestration_contract"]["orchestrator_context"]
    agent = await db_session.get(Agent, task.assigned_to)
    assert (await service.roadmap_pre_release_authority(db_session, run.id, agent, context))["status"] == "approved"

    other_agent = _agent("wrong-claimant", ["implementation"])
    db_session.add(other_agent)
    await db_session.flush()
    with pytest.raises(HTTPException, match="immutable assigned team agent"):
        await SessionService().create(db_session, SessionCreate(
            agent_id=other_agent.id, task_id=task.id, project_id=test_project.id,
        ))

    wrong_user = User(email="forged-budget-authority@example.test", hashed_password="x")
    db_session.add(wrong_user)
    await db_session.flush()
    decision.decided_by_user_id = wrong_user.id

    with pytest.raises(HTTPException, match="CLI budget authority"):
        await SessionService().create(db_session, SessionCreate(
            agent_id=task.assigned_to, task_id=task.id, project_id=test_project.id,
        ))
    decision.decided_by_user_id = creator.id
    decision.decided_by_agent_id = task.assigned_to
    with pytest.raises(HTTPException, match="CLI budget authority"):
        await SessionService().create(db_session, SessionCreate(
            agent_id=task.assigned_to, task_id=task.id, project_id=test_project.id,
        ))
    decision.decided_by_agent_id = None
    decision.status = "pending"
    with pytest.raises(HTTPException, match="CLI budget authority"):
        await SessionService().create(db_session, SessionCreate(
            agent_id=task.assigned_to, task_id=task.id, project_id=test_project.id,
        ))
    decision.status = "answered"
    session = await SessionService().create(db_session, SessionCreate(
        agent_id=task.assigned_to, task_id=task.id, project_id=test_project.id,
    ))
    assert session.metadata_["_roadmap_cli_budget_approval"]["decision_id"] == str(decision.id)


@pytest.mark.parametrize("answer", ["approve", "reject"])
async def test_derived_cli_task_authority_replays_and_binds_effective_override(
    db_session, test_project, accepted_roadmap, answer, monkeypatch,
):
    """The release-time authority must bind the roster-derived claimant and CLI path."""
    derived_agent_id = None

    async def setup(planner, goal, _run):
        nonlocal derived_agent_id
        derived_agent_id = planner.id
        planner.adapter_type = "cli"
        goal.budget = {"caps": {"max_tokens": 400, "max_turns": 2, "max_hours": 1}}

    service, goal, run, _ = await accepted_roadmap(
        [roadmap_task("derived-cli")], setup=setup, assign_agents=False,
    )

    async def rank_derived(*_args, **_kwargs):
        return [SimpleNamespace(agent_id=derived_agent_id, weak=False)]

    monkeypatch.setattr(OrchestrationRosterMapper, "rank_agents", rank_derived)
    first = await service.tick(db_session, run.id)
    assert first["authorized_execution"] == {"step": "waiting", "reason": "budget_wait"}
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_budget_adapter:%:derived-cli:%:cli:max_tokens"),
    ))
    assert decision is not None and decision.status == "pending"
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "budget_wait",
    }
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.decision_key == decision.decision_key,
    )) == 1

    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option=answer, decided_by_user_id=goal.manager_user_id,
    )
    resolved = await service.tick(db_session, run.id)
    if answer == "reject":
        assert resolved["authorized_execution"] == {"step": "waiting", "reason": "needs_attention"}
        blockers = [item for item in run.active_blockers if item.get("decision_key") == decision.decision_key]
        assert len(blockers) == 1 and blockers[0]["kind"] == "budget_integrity"
        assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem)) == 0
        return

    assert resolved["authorized_execution"] == {
        "step": "release_item", "item_key": "derived-cli", "unit_type": "task",
    }
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    ))
    task = await db_session.get(Task, row.task_id)
    agent = await db_session.get(Agent, task.assigned_to)
    agent.adapter_type = "api"
    task.adapter_type_override = "cli"
    session = await SessionService().create(db_session, SessionCreate(
        agent_id=agent.id, task_id=task.id, project_id=test_project.id,
    ))
    assert session.adapter_type == "cli"
    assert session.metadata_["_roadmap_cli_budget_approval"]["decision_id"] == str(decision.id)
    assert session.metadata_["token_usage_complete"] is False


async def test_direct_cli_terminal_spend_keeps_known_overage_and_incomplete_audit(
    db_session, test_project, accepted_roadmap,
):
    async def setup(planner, goal, _run):
        planner.adapter_type = "cli"
        goal.budget = {"caps": {"max_tokens": 400, "max_turns": 2, "max_hours": 1}}

    service, goal, run, _ = await accepted_roadmap([roadmap_task("direct-overage")], setup=setup)
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "budget_wait",
    }
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_budget_adapter:%:direct-overage:%:cli:max_tokens"),
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", decided_by_user_id=goal.manager_user_id,
    )
    await service.tick(db_session, run.id)
    item = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    ))
    task = await db_session.get(Task, item.task_id)
    session = await SessionService().create(db_session, SessionCreate(
        agent_id=task.assigned_to, task_id=task.id, project_id=test_project.id,
    ))
    session.status = "completed"
    session.started_at = run.started_at
    session.ended_at = run.started_at + timedelta(minutes=1)
    session.metadata_ = {**session.metadata_, "token_count_in": 450, "token_count_out": 0}

    known, complete = await OrchestrationBudgetService().known_run_spend(
        db_session, run, {"max_tokens"},
    )
    summary = await OrchestrationBudgetService().remaining(db_session, goal)
    assert (known, complete, summary["direct_spend"]["max_tokens"]) == ({"max_tokens": "450"}, False, "450")
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {
        "step": "waiting", "reason": "needs_attention",
    }
    assert any(item["kind"] == "budget_integrity" for item in run.active_blockers)


async def test_task_run_rejects_staged_adapter_override_before_mutating_task(
    db_session, test_project, accepted_roadmap, monkeypatch, tmp_path,
):
    async def workspace(*_args, **_kwargs):
        return tmp_path

    monkeypatch.setattr(ProjectService, "require_roadmap_workspace", workspace)
    monkeypatch.setattr(ProjectService, "require_frozen_roadmap_workspace", workspace)

    async def setup(planner, goal, _run):
        planner.adapter_type = "cli"
        goal.budget = {"caps": {"max_tokens": 400, "max_turns": 2, "max_hours": 1}}

    item = roadmap_task("override", mutates_shared_state=True)
    service, goal, run, _ = await accepted_roadmap([item], setup=setup)
    await service.tick(db_session, run.id)
    version = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    approval = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like(f"roadmap_budget_adapter:{version.id}:override:%:cli:max_tokens"),
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, approval, selected_option="approve", decided_by_user_id=goal.manager_user_id,
    )
    await service.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    ))
    task = await db_session.get(Task, row.task_id)
    before = task.status

    with pytest.raises(HTTPException, match="requires a CLI adapter"):
        await TaskService().run(
            db_session, test_project.id, task.id, adapter_type_override="api",
        )

    assert task.status == before


async def _complete_producer(db_session, test_project, service, run, task):
    await _complete_task_session(db_session, test_project.id, task, task.assigned_to, json.dumps({"status": "done"}))
    return await service.tick(db_session, run.id)


async def _request_and_complete_verification(db_session, test_project, service, run, row, agent_id):
    action = await service.execute_request_verification_action(
        db_session, run.id,
        {"action_type": "request_verification", "gate_id": str(row.gate_id), "work_function": "validation"},
        f"run:{run.id}:kind:request_verification:roadmap_item:{row.item_key}",
    )
    verifier_task = await db_session.get(Task, action.target_id)
    verifier_task.assigned_to = agent_id
    await _complete_task_session(
        db_session, test_project.id, verifier_task, agent_id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["verified"]}),
    )
    return await service.tick(db_session, run.id)


def _local_decisions(monkeypatch, service, decision_fn):
    """Keep the production dispatcher in the test transaction's SQLite database."""
    async def request(db, run_id, adapter=None):
        run = await db.get(OrchestrationRun, run_id)
        goal = await db.get(OrchestrationGoal, run.goal_id)
        context = await service._decision_context(db, goal, run)
        return await service.record_validated_decision(db, run_id, context, {}, decision_fn(context))

    monkeypatch.setattr(service, "request_llm_decision", request)


async def test_roadmap_task_successor_waits_for_accepted_predecessor_gate(
    db_session, test_project, accepted_roadmap, monkeypatch,
):
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", 1)
    service, goal, run, verifier = await accepted_roadmap([
        roadmap_task("build"), roadmap_task("verify", depends_on=["build"]),
    ])
    await service.tick(db_session, run.id)
    build = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id, OrchestrationRoadmapItem.item_key == "build",
    ))
    task = await db_session.get(Task, build.task_id)

    _local_decisions(monkeypatch, service, lambda _: {
        "action_type": "request_verification", "gate_id": str(build.gate_id),
        "work_function": "validation", "reason": "Verify build.",
    })
    dispatched = await _complete_producer(db_session, test_project, service, run, task)
    assert dispatched["authorized_execution"]["step"] == "terminal_item_decision"
    verification = await db_session.get(
        OrchestrationAction, uuid.UUID(dispatched["authorized_execution"]["action_id"]),
    )
    verifier_task = await db_session.get(Task, verification.target_id)
    producer_evidence = list((await db_session.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == build.gate_id,
    ))).all())
    assert ("task", task.id, task.assigned_to) in [
        (evidence.source_type, evidence.source_id, evidence.producer_agent_id) for evidence in producer_evidence
    ]
    assert await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id, OrchestrationRoadmapItem.item_key == "verify",
    )) is None

    await _complete_task_session(
        db_session, test_project.id, verifier_task, verifier.id,
        json.dumps({"status": "done", "verdict": "accepted", "evidence": ["verified"]}),
    )
    result = await service.tick(db_session, run.id)
    gate = await db_session.get(OrchestrationGate, build.gate_id)
    assert gate.status == "accepted"
    evidence = list((await db_session.scalars(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == build.gate_id,
    ).order_by(OrchestrationEvidence.created_at))).all())
    assert ("task", task.assigned_to) in [(item.source_type, item.producer_agent_id) for item in evidence]
    assert ("verification", verifier.id) in [(item.source_type, item.producer_agent_id) for item in evidence]
    assert result["authorized_execution"]["item_key"] == "verify"
    await service.tick(db_session, run.id)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "request_verification",
    )) == 1


async def test_terminal_roadmap_decision_commits_authority_wait(
    db_session, test_project, accepted_roadmap, monkeypatch,
):
    service, goal, run, _ = await accepted_roadmap([roadmap_task("build")])
    await service.tick(db_session, run.id)
    item = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    ))
    task = await db_session.get(Task, item.task_id)
    await _complete_task_session(
        db_session, test_project.id, task, task.assigned_to, json.dumps({"status": "done"}),
    )
    _local_decisions(monkeypatch, service, lambda _: {
        "action_type": "request_verification", "gate_id": str(item.gate_id),
        "work_function": "validation", "reason": "Verify build.",
    })

    async def wait_for_authority(*_args, **_kwargs):
        raise HTTPException(status_code=409, detail="waiting_plan_authority")

    monkeypatch.setattr(service, "_dispatch_execution_decision", wait_for_authority)
    result = await service.tick(db_session, run.id)
    assert result["authorized_execution"] == {
        "step": "waiting", "reason": "waiting_plan_authority",
    }


async def test_roadmap_task_release_replay_duplicates_nothing(db_session, accepted_roadmap):
    service, goal, run, _ = await accepted_roadmap([roadmap_task("build")])
    await service.tick(db_session, run.id)
    counts = [
        await db_session.scalar(select(func.count()).select_from(model))
        for model in (Task, OrchestrationAction, OrchestrationGate, OrchestrationRoadmapItem)
    ]
    await service.tick(db_session, run.id)
    assert [
        await db_session.scalar(select(func.count()).select_from(model))
        for model in (Task, OrchestrationAction, OrchestrationGate, OrchestrationRoadmapItem)
    ] == counts


async def test_concurrent_approved_roadmap_task_claim_creates_one_active_session(
    db_session, test_project, accepted_roadmap, concurrent_sessions, test_engine, monkeypatch,
):
    """Two committed claimers for one approved Roadmap task cannot both hold execution."""
    service, goal, run, _ = await accepted_roadmap([roadmap_task("one-claim")])
    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "one-claim"
    item = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    ))
    task = await db_session.get(Task, item.task_id)
    task_id, agent_id, project_id = task.id, task.assigned_to, test_project.id
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *_args: None)
    await db_session.commit()
    first, second = concurrent_sessions

    async def claim(db):
        try:
            session = await SessionService().create(db, SessionCreate(
                agent_id=agent_id, task_id=task_id, project_id=project_id,
            ))
            await db.commit()
            return session.id
        except HTTPException as exc:
            await db.rollback()
            return exc

    results = await asyncio.gather(claim(first), claim(second))
    created = [result for result in results if isinstance(result, uuid.UUID)]
    rejected = [result for result in results if isinstance(result, HTTPException)]
    assert len(created) == len(rejected) == 1
    assert rejected[0].status_code == 409 and rejected[0].detail == "Task already has an active session"
    async with async_sessionmaker(test_engine)() as reader:
        active = list(await reader.scalars(select(Session).where(
            Session.task_id == task_id, Session.status.in_(("pending", "running")),
        )))
    assert [session.id for session in active] == created


async def test_roadmap_task_gate_requires_independent_verification(
    db_session, test_project, accepted_roadmap, monkeypatch,
):
    service, goal, run, verifier = await accepted_roadmap([roadmap_task("build")])
    _local_decisions(monkeypatch, service, lambda _ctx: {
        "action_type": "noop", "reason": "Integration verification is pending.",
    })
    await service.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == goal.id))
    task = await db_session.get(Task, row.task_id)
    await _complete_producer(db_session, test_project, service, run, task)
    await _request_and_complete_verification(db_session, test_project, service, run, row, task.assigned_to)
    same_agent_gate = await db_session.get(OrchestrationGate, row.gate_id)
    assert same_agent_gate.status == "open"

    await _request_and_complete_verification(db_session, test_project, service, run, row, verifier.id)
    accepted_gate = await db_session.get(OrchestrationGate, row.gate_id)
    assert accepted_gate.status == "accepted"


async def test_roadmap_mutable_task_items_release_serially(db_session, accepted_roadmap):
    service, goal, run, _ = await accepted_roadmap([
        roadmap_task("build", mutates_shared_state=True), roadmap_task("verify", mutates_shared_state=True),
    ])
    await service.tick(db_session, run.id)
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_unstaged_mutation:%:build"),
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", decided_by_user_id=goal.manager_user_id,
    )
    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "build"
    result = await service.tick(db_session, run.id)
    assert result["authorized_execution"] == {"step": "waiting", "reason": "waiting_shared_workspace"}
    assert [(row.item_key, row.unit_type) for row in (await db_session.scalars(
        select(OrchestrationRoadmapItem).where(OrchestrationRoadmapItem.goal_id == goal.id)
    )).all()] == [("build", "task")]


async def test_roadmap_task_release_stops_at_two_active_tasks(db_session, accepted_roadmap):
    service, goal, run, _ = await accepted_roadmap([
        roadmap_task("one", mutates_shared_state=False),
        roadmap_task("two", mutates_shared_state=False),
        roadmap_task("three", mutates_shared_state=False),
    ])
    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "one"
    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "two"
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {"step": "waiting", "reason": "waiting_active_work"}
    first = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id, OrchestrationRoadmapItem.item_key == "one",
    ))
    (await db_session.get(Task, first.task_id)).status = "done"
    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "three"


async def test_roadmap_dependency_uses_predecessor_lineage_version(db_session, accepted_roadmap):
    service, goal, run, _ = await accepted_roadmap([roadmap_task("build")])
    await service.tick(db_session, run.id)
    lineage = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id, OrchestrationRoadmapItem.item_key == "build",
    ))
    gate = await db_session.get(OrchestrationGate, lineage.gate_id)
    gate.status = "accepted"
    successor_version = SimpleNamespace(id=uuid.uuid4())
    from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
    assert await OrchestrationRoadmapService(service)._dependencies_accepted(
        db_session, goal.id, run.id, successor_version, ["build"]
    )


async def test_roadmap_unstaged_mutation_waits_for_version_bound_human_authority(db_session, accepted_roadmap):
    item = roadmap_task("unstaged", mutates_shared_state=True)
    item["staging_boundary"] = None
    service, goal, run, _ = await accepted_roadmap([item])
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {"step": "waiting", "reason": "waiting_unstaged_approval"}
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_unstaged_mutation:%:unstaged"),
    ))
    assert str(decision.run_id) == str(run.id) and decision.status == "pending"
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", decided_by_user_id=goal.manager_user_id,
    )
    assert (await service.tick(db_session, run.id))["authorized_execution"]["item_key"] == "unstaged"


async def test_roadmap_unstaged_mutation_rejection_blocks_dispatch(db_session, accepted_roadmap):
    item = roadmap_task("unstaged", mutates_shared_state=True)
    item["staging_boundary"] = None
    service, goal, run, _ = await accepted_roadmap([item])
    await service.tick(db_session, run.id)
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_unstaged_mutation:%:unstaged"),
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="reject", decided_by_user_id=goal.manager_user_id,
    )
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {"step": "waiting", "reason": "needs_attention"}
    assert any(item["kind"] == "staging_boundary" for item in run.active_blockers)
    assert await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id
    )) is None


async def test_roadmap_task_contract_preserves_staging_and_workspace_policy(
    db_session, accepted_roadmap,
):
    service, goal, run, _ = await accepted_roadmap([roadmap_task("staged", mutates_shared_state=True)])
    await service.tick(db_session, run.id)
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_unstaged_mutation:%:staged"),
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", decided_by_user_id=goal.manager_user_id,
    )
    await service.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    ))
    task = await db_session.get(Task, row.task_id)
    context = task.metadata_["orchestration_contract"]["orchestrator_context"]

    assert context["roadmap"] == {
        "roadmap_version_id": str(row.first_version_id),
        "roadmap_item_key": "staged",
        "staging_boundary": None,
        "mutates_shared_state": True,
        "no_publish_before_integration": True,
        "unstaged_authority_decision_id": str(decision.id),
    }
    assert "workspace_policy" in context
    assert "Roadmap execution policy:" in task.description


async def test_roadmap_mutable_claim_uses_contract_workspace_and_rejects_override(
    db_session, test_project, accepted_roadmap,
):
    service, goal, run, _ = await accepted_roadmap([roadmap_task("claim", mutates_shared_state=True)])
    await service.tick(db_session, run.id)
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_unstaged_mutation:%:claim"),
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", decided_by_user_id=goal.manager_user_id,
    )
    await service.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    ))
    task = await db_session.get(Task, row.task_id)
    agent = await db_session.get(Agent, task.assigned_to)
    agent.adapter_type = "cli"

    session = await SessionService().create(db_session, SessionCreate(
        agent_id=agent.id, task_id=task.id, project_id=test_project.id,
    ))
    assert "_roadmap_workspace" not in session.metadata_
    for key, value in task.metadata_["orchestration_contract"]["orchestrator_context"]["roadmap"].items():
        assert session.input_context["orchestrator_context"]["roadmap"][key] == value
    with pytest.raises(HTTPException, match="cannot be overridden"):
        await SessionService().create(db_session, SessionCreate(
            agent_id=agent.id, task_id=task.id, project_id=test_project.id,
            context_override={"orchestrator_context": {"roadmap": {}}},
        ))


async def test_roadmap_claim_uses_version_bound_approval_after_task_metadata_is_stripped(
    db_session, test_project, accepted_roadmap,
):
    item = roadmap_task("immutable-approval", mutates_shared_state=True)
    item["staging_boundary"] = None
    service, goal, run, _ = await accepted_roadmap([item])
    await service.tick(db_session, run.id)
    decision = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.goal_id == goal.id,
        OrchestrationAuthorityDecision.decision_key.like("roadmap_unstaged_mutation:%:immutable-approval"),
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option="approve", decided_by_user_id=goal.manager_user_id,
    )
    await service.tick(db_session, run.id)
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    ))
    task = await db_session.get(Task, row.task_id)
    task.metadata_ = {}
    agent = await db_session.get(Agent, task.assigned_to)
    agent.adapter_type = "api"

    session = await SessionService().create(db_session, SessionCreate(
        agent_id=agent.id, task_id=task.id, project_id=test_project.id,
    ))

    assert session.input_context["orchestrator_context"]["roadmap"]["unstaged_authority_decision_id"] == str(decision.id)


async def test_registered_worktree_roadmap_release_and_claim_ignore_mapping_drift(
    db_session, test_project, accepted_roadmap, tmp_path, test_engine, monkeypatch,
):
    workspace = tmp_path / "repo"
    worktree = tmp_path / "registered-worktree"
    workspace.mkdir()
    for command in (("git", "init"), ("git", "config", "user.email", "test@example.com"),
                    ("git", "config", "user.name", "Test")):
        subprocess.run(command, cwd=workspace, check=True, capture_output=True)
    (workspace / "README.md").write_text("test\n")
    subprocess.run(("git", "add", "README.md"), cwd=workspace, check=True, capture_output=True)
    subprocess.run(("git", "commit", "-m", "initial"), cwd=workspace, check=True, capture_output=True)
    subprocess.run(("git", "worktree", "add", "--detach", str(worktree)), cwd=workspace, check=True, capture_output=True)
    script = tmp_path / "agent.py"
    script.write_text(f"#!{sys.executable}\nimport os\nprint(os.getcwd())\n")
    script.chmod(0o755)

    async def setup(planner, _goal, _run):
        planner.role = "developer"
        planner.adapter_type = "cli"
        planner.config = {"cli_runtime": "custom", "script_path": str(script)}
        test_project.workspace_path = str(workspace)
        test_project.config = {"roadmap_worktrees": {"roadmap/registered": str(worktree)}}

    service, goal, run, _ = await accepted_roadmap(
        [roadmap_task("registered", mutates_shared_state=True)], setup=setup, assign_agents=False,
    )
    first = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    first_artifact = await db_session.get(Artifact, first.plan_artifact_id)
    # Post-acceptance configuration cannot redirect the frozen registered path,
    # including through an otherwise unrelated replan.
    test_project.config = {"roadmap_worktrees": {"roadmap/registered": str(workspace)}}
    replan = await service.execute_request_roadmap_replan_action(
        db_session, run.id,
        {"action_type": "request_roadmap_replan", "agent_id": str(first_artifact.created_by_agent),
         "scope": "Add independent follow-up.", "reason": "New information."},
        f"run:{run.id}:kind:request_roadmap_replan:version:{first.version}",
    )
    candidate_items = [
        roadmap_task("registered", mutates_shared_state=True),
        roadmap_task("unrelated", mutates_shared_state=False),
    ]
    candidate = Artifact(
        project_id=test_project.id, name="roadmap-replan", artifact_type="plan", status="draft",
        linked_task_id=replan.target_id, created_by_agent=first_artifact.created_by_agent,
        metadata_={"plan_items": candidate_items},
    )
    db_session.add(candidate)
    await db_session.flush()
    fingerprint = service._accepted_plan_fingerprint([
        item.model_dump(mode="json") for item in parse_roadmap_items(candidate_items, set())
    ])
    with pytest.raises(HTTPException, match="waiting_plan_authority"):
        await service.execute_accept_plan_action(
            db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(candidate.id)},
            f"run:{run.id}:kind:registered-replan",
        )
    approval = await db_session.scalar(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key == f"roadmap_plan:{fingerprint}",
    ))
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, approval, selected_option="approve", decided_by_user_id=goal.manager_user_id,
    )
    await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(candidate.id)},
        f"run:{run.id}:kind:registered-replan",
    )
    second = await OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    assert second.snapshot["workspace_bindings"]["registered"] == first.snapshot["workspace_bindings"]["registered"]
    assert (await service.tick(db_session, run.id))["authorized_execution"] == {
        "step": "release_item", "item_key": "registered", "unit_type": "task",
    }
    row = await db_session.scalar(select(OrchestrationRoadmapItem).where(
        OrchestrationRoadmapItem.goal_id == goal.id,
    ))
    task = await db_session.get(Task, row.task_id)
    agent_id, task_id, project_id = task.assigned_to, task.id, test_project.id
    # Claim and launch use a separate committed session: CliAdapter itself
    # commits between its two frozen-workspace revalidations.
    monkeypatch.setattr(SessionService, "_schedule_dispatch_after_commit", lambda *_args: None)
    await db_session.commit()
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as fresh_db:
        session = await SessionService().create(fresh_db, SessionCreate(
            agent_id=agent_id, task_id=task_id, project_id=project_id,
        ))
        await fresh_db.commit()
        assert await _require_runnable_session(fresh_db, session.id, session.runner_task_id)
        await CliAdapter().run(session.id, fresh_db, session.runner_task_id)
        await fresh_db.refresh(session)

    assert session.metadata_["_roadmap_workspace"] == str(worktree.resolve())
    assert session.input_context["orchestrator_context"]["roadmap"]["workspace_path"] == str(worktree.resolve())
    assert session.sandbox_path.startswith(str(worktree.resolve()))
    assert session.output.strip() == str(worktree.resolve())
    assert session.status == "completed" and session.ended_at is not None

async def test_roadmap_rejects_delegation_to_agent_outside_inherited_team(
    db_session, accepted_roadmap,
):
    service, goal, run, _ = await accepted_roadmap([roadmap_task("team")])
    outsider = _agent("roadmap-outsider", ["implementation"])
    db_session.add(outsider)
    await db_session.flush()

    with pytest.raises(HTTPException, match="outside the inherited team"):
        await service._validate_delegation_targets(db_session, run.id, {
            "agent_id": str(outsider.id), "work_function": "implementation",
            "orchestrator_context": {"team": {"contributors": [str(uuid.uuid4())]}},
        })


async def test_roadmap_parent_contract_excludes_active_agents_outside_accepted_team(
    db_session, accepted_roadmap, monkeypatch,
):
    service, goal, _, _ = await accepted_roadmap([roadmap_task("team-contract")])
    insider = _agent("accepted-insider", ["implementation"])
    outsider = _agent("active-outsider", ["implementation"])
    db_session.add_all([insider, outsider])
    await db_session.flush()

    from huddleroom.services.orchestration_process_service import OrchestrationProcessService
    monkeypatch.setattr(
        OrchestrationProcessService,
        "get_current",
        lambda *_args: __import__("asyncio").sleep(0, result=SimpleNamespace(outputs={"manager_agent_id": str(insider.id)})),
    )
    roadmap = __import__("huddleroom.services.orchestration_roadmap_service", fromlist=["OrchestrationRoadmapService"])
    version = await roadmap.OrchestrationRoadmapService(service).current_version(db_session, goal.id)
    snapshot = await roadmap.OrchestrationRoadmapService(service)._parent_contract_snapshot(
        db_session, goal, version, roadmap.parse_roadmap_items(version.snapshot["items"], set())[0]
    )

    assert str(outsider.id) not in snapshot["team"]["agent_ids"]
