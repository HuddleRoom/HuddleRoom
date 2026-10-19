"""G3 Task 4 authority boundaries exercised through real service paths."""

import uuid
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models import User
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationBudgetReservation,
    OrchestrationGate,
    OrchestrationRoadmapItem,
    OrchestrationRoadmapVersion,
    OrchestrationRun,
)
from huddleroom.models.orchestration_process import OrchestrationProcessRun
from huddleroom.models.project import Project
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.orchestration_chat import ProjectChatTurn
from huddleroom.models.task import Task
from huddleroom.models.session import Session
from huddleroom.schemas.orchestration import OrchestrationDecisionAnswerRequest
from huddleroom.services import orchestration_decision_application as application
from huddleroom.services.orchestration_plan_review import PlanReviewService
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_roadmap_service import OrchestrationRoadmapService
from huddleroom.services.orchestration_service import OrchestrationService
from tests.conftest import adopt_session
from tests import test_orchestration_roadmap_plans as roadmap_plan_helpers
from tests.test_orchestration_roadmap_plans import _agent, _approve, _roadmap_acceptance, _roadmap_task


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _local_roadmap_helper_setup(db_session, test_engine, monkeypatch):
    """Scope only the imported roadmap helper's transaction fixtures to this file."""
    original_authorized = roadmap_plan_helpers._authorized_run
    marker = "orchestration_tick_owns_transaction"
    previous = db_session.info.get(marker)

    async def authorized(db, *args, **kwargs):
        root = db.sync_session.get_transaction()
        if db is db_session and not db.in_nested_transaction() and (root is None or root.origin.name == "AUTOBEGIN"):
            kwargs["owns_setup"] = True
            adopt_session(db)
            db.info[marker] = True
        return await original_authorized(db, *args, **kwargs)

    async def no_auto_start(_self, _db, _goal, _run):
        return 0

    monkeypatch.setattr(roadmap_plan_helpers, "_authorized_run", authorized)
    monkeypatch.setattr(
        "huddleroom.services.orchestration_service.AsyncSessionLocal",
        async_sessionmaker(test_engine, expire_on_commit=False),
    )
    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", no_auto_start)
    yield
    if previous is None:
        db_session.info.pop(marker, None)
    else:
        db_session.info[marker] = previous


@pytest.fixture
async def db_session(test_engine):
    async with AsyncSession(test_engine, expire_on_commit=False) as db:
        yield db
        await db.rollback()


@pytest.fixture(autouse=True)
def safe_execution_boundary(monkeypatch):
    """Record and neutralize every path that could launch external work."""
    calls = []

    def record_dispatch(session_id, _adapter_type, _project_id, task_id=None):
        calls.append(("session_dispatch", str(session_id), task_id))
        return task_id or "test-blocked-dispatch"

    async def blocked_provider(*_args, **_kwargs):
        calls.append(("provider",))
        raise AssertionError("test attempted a provider/model call")

    async def blocked_session_worker(*_args, **_kwargs):
        calls.append(("session_worker",))
        raise AssertionError("test attempted a session worker launch")

    monkeypatch.setattr("huddleroom.workers.task_runner.register_session", record_dispatch)
    monkeypatch.setattr("huddleroom.workers.task_runner.publish_session_dispatch", record_dispatch)
    monkeypatch.setattr("huddleroom.workers.session_tasks.execute_api_session", blocked_session_worker)
    monkeypatch.setattr("huddleroom.workers.session_tasks.execute_cli_session", blocked_session_worker)
    monkeypatch.setattr("huddleroom.services.orchestration_llm_decision_adapter.OrchestrationDecisionAdapter.decide", blocked_provider)
    return calls


async def _stale_export(db_session, goal, run, artifact):
    review = await PlanReviewService().ensure(db_session, goal, run, artifact)
    assert review.export_path is not None
    path = Path((await db_session.get(Project, goal.project_id)).workspace_path) / review.export_path
    path.write_text("edited after review", encoding="utf-8")


async def _committed_stale_roadmap_seed(test_engine, tmp_path):
    """Seed the candidate in one committed writer for real request-facade tests."""
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    workspace = tmp_path / f"g3-authority-{uuid.uuid4()}"
    workspace.mkdir()
    async with sessions() as setup:
        project = Project(name="G3 authority", description="", workspace_path=str(workspace), config={})
        user = User(email=f"g3-authority-{uuid.uuid4()}@test.invalid", hashed_password="x", display_name="G3")
        setup.add_all([project, user])
        await setup.flush()
        _, goal, run, artifact, decision, _ = await _roadmap_acceptance(setup, project, user)
        review = await PlanReviewService().ensure(setup, goal, run, artifact)
        assert review.export_path is not None
        export = workspace / review.export_path
        original = export.read_text(encoding="utf-8")
        ids = (project.id, user.id, goal.id, run.id, artifact.id, decision.id)
        action_count = await setup.scalar(select(func.count()).select_from(OrchestrationAction))
        await setup.commit()
    return sessions, ids, export, original, action_count


async def _assert_only_durable_stale_marker(sessions, ids, action_count, *, turn_id=None, frozen_context=None):
    project_id, _, goal_id, run_id, artifact_id, decision_id = ids
    async with sessions() as fresh:
        run = await fresh.get(OrchestrationRun, run_id)
        decision = await fresh.get(OrchestrationAuthorityDecision, decision_id)
        assert run.goal_id == goal_id and run.plan_state["plan_reviews"][str(artifact_id)]["state"] == "stale"
        assert (decision.status, decision.selected_option, decision.reason) == ("pending", None, None)
        assert await fresh.scalar(select(func.count()).select_from(OrchestrationAction)) == action_count
        assert await fresh.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 0
        if turn_id is not None:
            turn = await fresh.get(ProjectChatTurn, turn_id)
            assert turn.action_receipt is None
            assert turn.frozen_context == frozen_context
        assert await fresh.get(Project, project_id) is not None


async def test_stale_approval_answer_is_rejected_before_decision_write_or_roadmap_release(
    db_session, test_project, test_user,
):
    service, goal, run, artifact, decision, _ = await _roadmap_acceptance(
        db_session, test_project, test_user,
    )
    await _stale_export(db_session, goal, run, artifact)

    with pytest.raises(HTTPException, match="Roadmap plan") as error:
        await application.answer_decision(
            db_session, test_project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Approve the reviewed candidate."),
            test_user,
        )

    assert error.value.status_code == 409
    await db_session.refresh(decision)
    assert decision.status == "pending" and decision.selected_option is None
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 0


async def test_stale_candidate_cannot_admit_an_independent_roadmap_verifier(
    db_session, test_project, test_user,
):
    verifier = _agent("current-candidate-verifier", ["validation"])
    db_session.add(verifier)
    await db_session.flush()
    service, goal, run, artifact, _, _ = await _roadmap_acceptance(
        db_session, test_project, test_user, authority_model="no_manager", seed_approval=False,
        team_agent_ids=[verifier.id],
    )
    gate = await OrchestrationRoadmapService(service).ensure_plan_approval(db_session, goal, run, artifact)
    await _stale_export(db_session, goal, run, artifact)

    with pytest.raises(HTTPException, match="Roadmap plan") as error:
        await service.execute_request_verification_action(
            db_session, run.id,
            {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
            f"g3:stale-verifier:{uuid.uuid4()}",
        )

    assert error.value.status_code == 409
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "request_verification",
        OrchestrationAction.status == "completed",
    )) == 0


async def test_stale_export_still_allows_a_revision_request(
    db_session, test_project, test_user,
):
    service, goal, run, artifact, _, _ = await _roadmap_acceptance(
        db_session, test_project, test_user,
    )
    await _stale_export(db_session, goal, run, artifact)

    revision = await service.execute_request_plan_revision_action(
        db_session, run.id,
        {
            "action_type": "request_plan_revision",
            "plan_task_id": str(artifact.linked_task_id),
            "revision_request": "The reviewed export changed; provide a fresh candidate.",
            "reason": "Reviewed plan is stale.",
        },
        f"g3:stale-revision:{uuid.uuid4()}",
    )

    assert revision.status == "completed"
    assert revision.target_id is not None


async def test_role_map_change_blocks_frozen_owner_before_roadmap_release(
    db_session, test_project, test_user,
):
    owner = _agent("frozen-owner", ["implementation"])
    replacement = _agent("replacement-owner", ["implementation"])
    db_session.add_all([owner, replacement])
    await db_session.flush()
    item = {**_roadmap_task("frozen-owner-task"), "agent_id": str(owner.id)}
    service, goal, run, artifact, decision, _ = await _roadmap_acceptance(
        db_session, test_project, test_user, items=[item], team_agent_ids=[owner.id, replacement.id],
    )
    await _approve(db_session, goal, decision)
    await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"g3:frozen-owner:accept:{run.id}",
    )
    hierarchy = await db_session.scalar(select(OrchestrationProcessRun).where(
        OrchestrationProcessRun.goal_id == goal.id,
        OrchestrationProcessRun.process_type == "team_hierarchy",
        OrchestrationProcessRun.superseded_by_id.is_(None),
    ))
    hierarchy.outputs = {
        **hierarchy.outputs,
        "role_to_agent": {**hierarchy.outputs["role_to_agent"], "implementation": str(replacement.id)},
        "responsibility_coverage": [
            {**row, "agent_ref": str(replacement.id)}
            if row["work_function"] == "implementation" else row
            for row in hierarchy.outputs["responsibility_coverage"]
        ],
    }
    agent_review = await db_session.scalar(select(OrchestrationProcessRun).where(
        OrchestrationProcessRun.goal_id == goal.id,
        OrchestrationProcessRun.process_type == "agent_definition_review",
        OrchestrationProcessRun.superseded_by_id.is_(None),
    ))
    agent_review.outputs = {
        "eligibility": {
            **{agent_id: functions for agent_id, functions in agent_review.outputs["eligibility"].items()
               if agent_id != str(owner.id)},
            str(replacement.id): ["implementation"],
        }
    }
    await db_session.flush()
    action_count = await db_session.scalar(select(func.count()).select_from(OrchestrationAction))
    item_count = await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem))
    gate_count = await db_session.scalar(select(func.count()).select_from(OrchestrationGate))

    roadmap = OrchestrationRoadmapService(service)
    version = await roadmap.current_version(db_session, goal.id)
    plan_item = roadmap.parse_items(artifact, goal)[0]
    with pytest.raises(HTTPException, match="Roadmap plan ownership is blocked") as error:
        await roadmap.release_task_item(db_session, goal, run, version, plan_item)

    assert error.value.status_code == 409
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction)) == action_count
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem)) == item_count
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGate)) == gate_count


async def test_planning_only_verdict_accepts_candidate_but_releases_work_only_after_execution_upgrade(
    db_session, test_project, test_user, monkeypatch, safe_execution_boundary,
):
    from huddleroom.services.orchestration_authority_service import (
        compute_goal_contract_fingerprint,
        execution_grant,
        effective_authorization,
    )
    from tests.test_orchestration_runtime_e2e import _complete_task_session

    verifier = _agent("planning-only-verifier", ["validation"])
    db_session.add(verifier)
    await db_session.flush()
    service, goal, run, artifact, _, _ = await _roadmap_acceptance(
        db_session, test_project, test_user, authority_model="no_manager", seed_approval=False,
        team_agent_ids=[verifier.id],
    )

    async def unexpected_execution(*_args, **_kwargs):
        raise AssertionError("planning-only roadmap advance must not launch execution")

    # The all-test boundary above blocks provider/session dispatch before fixture
    # setup.  This stricter guard covers the implementation auto-start boundary.
    monkeypatch.setattr(OrchestrationService, "_start_released_tasks", unexpected_execution)
    grant = await db_session.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.action_type == "authorize_execution",
    ))
    fingerprint = compute_goal_contract_fingerprint(goal)
    await db_session.execute(update(OrchestrationRun).where(OrchestrationRun.id == run.id).values(
        goal_contract_fingerprint=fingerprint,
    ))
    await db_session.refresh(run)
    grant.request = {"scope": "planning_only", "contract_fingerprint": fingerprint}
    run.plan_state = {**run.plan_state, "scope_contract_version": 1}
    gate = await OrchestrationRoadmapService(service).ensure_plan_approval(db_session, goal, run, artifact)
    verification = await service.execute_request_verification_action(
        db_session, run.id,
        {"action_type": "request_verification", "gate_id": str(gate.id), "work_function": "validation"},
        f"g3:planning-only-verifier:{run.id}",
    )
    verifier_task = await db_session.get(Task, verification.target_id)
    assert (await effective_authorization(db_session, goal, run, verifier_task)).allowed
    session = await _complete_task_session(
        db_session, test_project.id, verifier_task, verifier.id,
        '{"status":"done","verdict":"accepted","evidence":["Reviewed current candidate."]}',
    )
    events = await service._new_events(db_session, test_project.id, run.event_cursor)
    await service._ingest_evidence_from_events(db_session, run, events)
    assert await service.validate_open_gates(db_session, run.id) >= 1
    assert gate.status == "accepted" and session.id is not None
    await service.execute_accept_plan_action(
        db_session, run.id, {"action_type": "accept_plan", "plan_artifact_id": str(artifact.id)},
        f"g3:planning-only-accept:{run.id}",
    )

    assert (await execution_grant(db_session, goal, run)).scope == "planning_only"
    tasks_before = await db_session.scalar(select(func.count()).select_from(Task))
    sessions_before = await db_session.scalar(select(func.count()).select_from(Session))
    reservations_before = await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation))
    actions_before = await db_session.scalar(select(func.count()).select_from(OrchestrationAction))
    gates_before = await db_session.scalar(select(func.count()).select_from(OrchestrationGate))
    result = await OrchestrationRoadmapService(service).advance(db_session, goal, run)
    released = list(await db_session.scalars(select(OrchestrationRoadmapItem)))
    assert len(released) == 0, {
        "advance": result,
        "items": [(row.item_key, row.status, row.task_id, row.gate_id, row.child_goal_id) for row in released],
        "new_tasks": (await db_session.scalar(select(func.count()).select_from(Task))) - tasks_before,
        "new_actions": (await db_session.scalar(select(func.count()).select_from(OrchestrationAction))) - actions_before,
        "new_gates": (await db_session.scalar(select(func.count()).select_from(OrchestrationGate))) - gates_before,
    }
    assert await db_session.scalar(select(func.count()).select_from(Task)) == tasks_before, result
    assert await db_session.scalar(select(func.count()).select_from(Session)) == sessions_before, result
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationBudgetReservation)) == reservations_before, result
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction)) == actions_before, result
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationGate)) == gates_before, result
    assert (await effective_authorization(db_session, goal, run)).reason_code == "execution_scope_required"
    assert safe_execution_boundary == []

    await service.start_run(
        db_session, test_project.id, goal.id, actor="human:g3", scope="execution",
        request_id=uuid.uuid4(), expected_run_id=run.id, expected_control_version=run.control_version,
    )
    assert (await effective_authorization(db_session, goal, run)).allowed
    await OrchestrationRoadmapService(service).advance(db_session, goal, run)
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapItem)) == 1


async def test_http_stale_answer_marker_survives_rollback_and_restored_export_stays_rejected(
    test_engine, tmp_path, safe_execution_boundary,
):
    """The HTTP owner must preserve stale observation after its request rollback."""
    from httpx import ASGITransport, AsyncClient
    from huddleroom.database import get_db
    from huddleroom.dependencies import get_current_user
    from huddleroom.main import create_app

    sessions, ids, export, original, action_count = await _committed_stale_roadmap_seed(test_engine, tmp_path)
    project_id, user_id, goal_id, _, _, decision_id = ids
    export.write_text("changed after reviewed export", encoding="utf-8")
    app = create_app()

    async def request_db():
        async with sessions() as db:
            yield db

    async def actor():
        async with sessions() as db:
            return await db.get(User, user_id)

    app.dependency_overrides[get_db] = request_db
    app.dependency_overrides[get_current_user] = actor
    url = f"/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/decisions/{decision_id}/answer"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = await client.post(url, json={"selected_option": "approve", "reason": "Approve reviewed candidate."})
            assert first.status_code == 409, first.text
            await _assert_only_durable_stale_marker(sessions, ids, action_count)
            export.write_text(original, encoding="utf-8")
            retry = await client.post(url, json={"selected_option": "approve", "reason": "Approve reviewed candidate."})
            assert retry.status_code == 409, retry.text
    finally:
        app.dependency_overrides.clear()
    await _assert_only_durable_stale_marker(sessions, ids, action_count)
    assert safe_execution_boundary == []


async def test_chat_stale_answer_marker_survives_rollback_without_receipt_or_confirmation_writes(
    test_engine, tmp_path, safe_execution_boundary,
):
    """Chat must roll back its outer receipt before persisting only the stale marker."""
    from huddleroom.schemas.orchestration_chat import ChatActionDraft, ChatInterpretation
    from huddleroom.services.orchestration_chat_actions import OrchestrationChatActionService

    sessions, ids, export, original, action_count = await _committed_stale_roadmap_seed(test_engine, tmp_path)
    project_id, user_id, goal_id, run_id, _, decision_id = ids
    async with sessions() as prepared:
        turn = ProjectChatTurn(
            project_id=project_id, actor_id=user_id, client_request_id=uuid.uuid4(),
            content=f"Answer {decision_id} approve", context_goal_id=goal_id, source_kind="action",
        )
        prepared.add(turn)
        await prepared.flush()
        draft = ChatActionDraft(
            action_id=uuid.uuid4(), operation="answer_authority", project_id=project_id,
            goal_id=goal_id, run_id=run_id, target_id=decision_id,
            payload={"selected_option": "approve"}, summary="Answer the reviewed Roadmap approval.",
        )
        turn.interpretation = ChatInterpretation(
            intent="instruction", operation="answer_authority", goal_id=goal_id, payload=draft.payload,
        ).model_dump(mode="json")
        action = await OrchestrationChatActionService(session_factory=sessions).prepare(prepared, turn, draft)
        turn_id = turn.id
        frozen_context = turn.frozen_context
        await prepared.commit()

    export.write_text("changed after reviewed export", encoding="utf-8")
    service = OrchestrationChatActionService(session_factory=sessions)
    async with sessions() as submit:
        with pytest.raises(HTTPException, match="Roadmap plan") as error:
            await service.execute(submit, await submit.get(ProjectChatTurn, turn_id), action)
        assert error.value.status_code == 409
    await _assert_only_durable_stale_marker(
        sessions, ids, action_count, turn_id=turn_id, frozen_context=frozen_context,
    )
    export.write_text(original, encoding="utf-8")
    async with sessions() as retry:
        with pytest.raises(HTTPException, match="Roadmap plan") as error:
            await service.execute(retry, await retry.get(ProjectChatTurn, turn_id), action)
        assert error.value.status_code == 409
    await _assert_only_durable_stale_marker(
        sessions, ids, action_count, turn_id=turn_id, frozen_context=frozen_context,
    )
    assert safe_execution_boundary == []


async def _manager_wait(db_session, test_project, test_user, option):
    service, goal, run, artifact, decision, fingerprint = await _roadmap_acceptance(
        db_session, test_project, test_user, authority_model="agent_manager",
    )
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, decision, selected_option=option, decided_by_agent_id=goal.manager_agent_id,
    )
    run.plan_state = {**run.plan_state, "roadmap_authority_wait": {
        "artifact_id": str(artifact.id),
        "fingerprint": await OrchestrationRoadmapService(service).plan_fingerprint(
            db_session, run, [i.model_dump(mode="json") for i in OrchestrationRoadmapService(service).parse_items(artifact, goal)],
        ),
        "approval_id": str(decision.id), "approval_kind": "authority_decision",
    }}
    await db_session.flush()
    return service, goal, run, artifact


def _no_llm(service, monkeypatch):
    calls = []

    async def fake(*_args, **_kwargs):
        calls.append(1)
        raise TimeoutError("llm must not be needed")

    monkeypatch.setattr(service, "request_llm_decision", fake)
    return calls


async def test_answered_roadmap_approval_is_applied_without_llm_and_idempotently(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, artifact = await _manager_wait(db_session, test_project, test_user, "approve")
    calls = _no_llm(service, monkeypatch)

    first = await service._advance_authorized_execution_once(db_session, goal, run)
    await db_session.refresh(run)
    assert first["step"] == "plan_decision" and calls == []
    assert run.plan_state["status"] == "accepted"
    assert run.plan_state["accepted_artifact_id"] == str(artifact.id)
    action = await db_session.get(OrchestrationAction, uuid.UUID(first["action_id"]))
    assert action.action_type == "accept_plan" and action.status == "completed"

    # Replay: a second advance is a no-LLM-needed accepted step and creates nothing new.
    again = await service._apply_answered_roadmap_approval(db_session, goal, run, run.plan_state)
    assert again is not None and again["action_id"] == first["action_id"]
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 1
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id, OrchestrationAction.action_type == "accept_plan",
    )) == 1


async def test_answered_roadmap_rejection_requires_revision_without_llm(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, _ = await _manager_wait(db_session, test_project, test_user, "reject")
    calls = _no_llm(service, monkeypatch)

    result = await service._advance_authorized_execution_once(db_session, goal, run)
    await db_session.refresh(run)
    assert result["step"] == "plan_revision_required" and calls == []
    assert run.plan_state["status"] == "revision_required"
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 0


async def test_stale_roadmap_wait_falls_through_to_llm(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, _ = await _manager_wait(db_session, test_project, test_user, "approve")
    wait = run.plan_state["roadmap_authority_wait"]
    run.plan_state = {**run.plan_state, "roadmap_authority_wait": {**wait, "fingerprint": "stale"}}
    await db_session.flush()
    calls = _no_llm(service, monkeypatch)

    with pytest.raises(TimeoutError):
        await service._advance_authorized_execution_once(db_session, goal, run)
    assert calls == [1]
    assert await db_session.scalar(select(func.count()).select_from(OrchestrationRoadmapVersion)) == 0


async def test_failed_accept_action_under_same_key_falls_through_to_llm(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, _ = await _manager_wait(db_session, test_project, test_user, "approve")
    calls = _no_llm(service, monkeypatch)

    async def failed(*_args, **_kwargs):
        return SimpleNamespace(id=uuid.uuid4(), status="failed")

    monkeypatch.setattr(service, "execute_accept_plan_action", failed)
    assert await service._apply_answered_roadmap_approval(db_session, goal, run, run.plan_state) is None
    with pytest.raises(TimeoutError):
        await service._advance_authorized_execution_once(db_session, goal, run)
    assert calls == [1]


async def test_accept_validation_error_requires_revision_under_goal_lock(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, _ = await _manager_wait(db_session, test_project, test_user, "approve")
    calls = _no_llm(service, monkeypatch)
    locked = []
    real_lock = service._lock_goal_for_baseline_transition

    def spy(db, goal_id, **kw):
        locked.append(goal_id)
        return real_lock(db, goal_id, **kw)

    async def boom(*_a, **_k):
        raise HTTPException(409, "owners invalid")

    monkeypatch.setattr(service, "_lock_goal_for_baseline_transition", spy)
    from huddleroom.services.orchestration_plan_ownership import PlanOwnershipService
    monkeypatch.setattr(PlanOwnershipService, "validate_plan_owners", boom)
    result = await service._advance_authorized_execution_once(db_session, goal, run)
    assert result["step"] == "plan_revision_required" and calls == []
    assert goal.id in locked
    assert run.plan_state["status"] == "revision_required"


async def test_revision_required_clears_roadmap_authority_wait(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, _ = await _manager_wait(db_session, test_project, test_user, "reject")
    _no_llm(service, monkeypatch)
    await service._advance_authorized_execution_once(db_session, goal, run)
    assert "roadmap_authority_wait" not in run.plan_state


async def _pending_manager_wait(db_session, test_project, test_user):
    service, goal, run, artifact, decision, _ = await _roadmap_acceptance(
        db_session, test_project, test_user, authority_model="agent_manager",
    )
    roadmap = OrchestrationRoadmapService(service)
    run.plan_state = {**run.plan_state, "roadmap_authority_wait": {
        "artifact_id": str(artifact.id),
        "fingerprint": await roadmap.plan_fingerprint(
            db_session, run, [i.model_dump(mode="json") for i in roadmap.parse_items(artifact, goal)],
        ),
        "approval_id": str(decision.id), "approval_kind": "authority_decision",
    }}
    await db_session.flush()
    return service, goal, run, decision


async def test_pending_plan_authority_freezes_orchestrator_without_llm(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, _ = await _pending_manager_wait(db_session, test_project, test_user)
    before = dict(run.plan_state)
    calls = _no_llm(service, monkeypatch)

    result = await service._advance_authorized_execution_once(db_session, goal, run)
    assert result == {"step": "waiting_plan_authority"} and calls == []
    assert run.plan_state == before


async def test_cancelled_plan_authority_unfreezes_llm_path(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, decision = await _pending_manager_wait(db_session, test_project, test_user)
    decision.status = "cancelled"
    await db_session.flush()
    calls = _no_llm(service, monkeypatch)

    with pytest.raises(TimeoutError):
        await service._advance_authorized_execution_once(db_session, goal, run)
    assert calls == [1]


async def test_stale_pending_plan_authority_wait_falls_through_to_llm(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, _ = await _pending_manager_wait(db_session, test_project, test_user)
    wait = run.plan_state["roadmap_authority_wait"]
    run.plan_state = {**run.plan_state, "roadmap_authority_wait": {**wait, "fingerprint": "stale"}}
    await db_session.flush()
    calls = _no_llm(service, monkeypatch)

    with pytest.raises(TimeoutError):
        await service._advance_authorized_execution_once(db_session, goal, run)
    assert calls == [1]


async def _human_decision_after_escalation(db_session, service, goal, run, decision):
    await service._escalate_authority_decision(db_session, goal, run, decision, reason="test escalation")
    await db_session.flush()
    return (await db_session.execute(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.run_id == run.id,
        OrchestrationAuthorityDecision.decision_key == decision.decision_key,
        OrchestrationAuthorityDecision.status == "pending",
    ))).scalar_one()


async def test_escalated_plan_authority_keeps_freeze_and_routes_human_answer_without_llm(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, decision = await _pending_manager_wait(db_session, test_project, test_user)
    human = await _human_decision_after_escalation(db_session, service, goal, run, decision)
    assert decision.status == "cancelled" and human.authority == "human"
    calls = _no_llm(service, monkeypatch)

    assert await service._advance_authorized_execution_once(db_session, goal, run) == {"step": "waiting_plan_authority"}
    assert calls == []
    await OrchestrationAuthorityDecisionService().answer_decision(
        db_session, human, selected_option="approve", decided_by_user_id=test_user.id,
    )
    result = await service._advance_authorized_execution_once(db_session, goal, run)
    await db_session.refresh(run)
    # Applied deterministically (no LLM). Under agent_manager a human approval is, by existing
    # _require_plan_approval design, not "from the selected authority", so it requires revision.
    assert result["step"] == "plan_revision_required" and calls == []
    assert "not from the selected authority" in run.plan_state["revision_reason"]


async def test_failed_delegation_task_without_report_escalates_and_freeze_follows(
    db_session, test_project, test_user, monkeypatch,
):
    service, goal, run, decision = await _pending_manager_wait(db_session, test_project, test_user)
    await service._sync_agent_authority_decisions(db_session, goal, run)
    await db_session.refresh(decision)
    assert decision.related_action_id is not None and decision.status == "pending"
    action = await db_session.get(OrchestrationAction, decision.related_action_id)
    (await db_session.get(Task, action.target_id)).status = "failed"
    await db_session.flush()

    summary = await service._sync_agent_authority_decisions(db_session, goal, run)
    await db_session.refresh(decision)
    assert summary["escalated"] == 1 and decision.status == "cancelled"
    assert (await db_session.execute(select(OrchestrationAuthorityDecision).where(
        OrchestrationAuthorityDecision.decision_key == decision.decision_key,
        OrchestrationAuthorityDecision.status == "pending", OrchestrationAuthorityDecision.authority == "human",
    ))).scalar_one()
    calls = _no_llm(service, monkeypatch)
    assert await service._advance_authorized_execution_once(db_session, goal, run) == {"step": "waiting_plan_authority"}
    assert calls == []
