"""Legacy sources remain durable and retain their original visibility in project chat."""
import os
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from huddleroom.models import Project, User
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_advisor import ProjectAdvisorTurn
from huddleroom.models.orchestration_conversation import (
    ConversationFeedback, ConversationInvestigation, ConversationInvestigationReservation,
    ConversationMessage, ConversationReservation, ConversationResponse,
)
from huddleroom.models.orchestration_steering import (
    OrchestrationSteeringProposal, OrchestrationSteeringRequest, OrchestrationSteeringResultLink,
    OrchestrationSteeringState, OrchestrationSteeringTransition, steering_state_id,
)


def _alembic(path, *args):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args], capture_output=True, text=True,
        env={**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{path}"}, check=False,
    )
    assert result.returncode == 0, result.stderr


def _snapshot(path):
    with sqlite3.connect(path) as db:
        names = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {name: db.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall()
                for name in names if name != "alembic_version" and not name.startswith("project_chat")}


async def _seed_sources(db, *, commit=True):
    now = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
    actors = [User(email=f"chat-history-{i}@test.dev", hashed_password="unused") for i in range(2)]
    projects = [Project(name=f"history-{i}") for i in range(2)]
    db.add_all(actors + projects)
    await db.flush()
    goals = [OrchestrationGoal(project_id=projects[i // 2].id, objective=f"Goal {i}",
                               created_by_user_id=actors[0].id) for i in range(3)]
    db.add_all(goals)
    await db.flush()
    runs = [OrchestrationRun(goal_id=goal.id) for goal in goals]
    db.add_all(runs)
    await db.flush()
    advisors = [ProjectAdvisorTurn(project_id=projects[project].id, actor_id=actors[actor].id,
                                   question=question, answer="legacy answer" if status == "completed" else None,
                                   status=status, tokens_used=tokens,
                                   citations=[{"type": "goal", "id": str(goals[0].id),
                                               "goal_id": str(goals[0].id), "label": "Goal 0"}], created_at=now)
                for actor, project, question, status, tokens in (
                    (0, 0, "private A", "completed", 11), (1, 0, "private B", "completed", 22),
                    (0, 0, "unknown usage", "failed", None), (0, 0, "pending advisor", "pending", None),
                    (0, 1, "foreign advisor", "completed", 33))]
    db.add_all(advisors)
    responses = []
    for sequence, (actor, status, reservation_status) in enumerate(
        ((0, "completed", "settled"), (1, "completed", "held_unknown"),
         (0, "pending", "reserved"), (0, "failed", "released")), start=1
    ):
        message = ConversationMessage(goal_id=goals[0].id, actor_id=actors[actor].id,
                                      client_request_id=uuid.uuid4(), sequence=sequence,
                                      content=f"shared {sequence}", created_at=now)
        db.add(message)
        await db.flush()
        response = ConversationResponse(message_id=message.id, run_id=runs[0].id, status=status,
                                        dossier={"original": True}, context_manifest={}, context_version="old-v1",
                                        provider_request_id=f"legacy-{sequence}",
                                        answer="shared answer" if status == "completed" else None,
                                        started_at=now if status == "completed" else None,
                                        deadline_at=now if status == "completed" else None,
                                        finished_at=now if status in {"completed", "failed"} else None,
                                        created_at=now)
        db.add(response)
        await db.flush()
        responses.append(response)
        db.add(ConversationReservation(response_id=response.id, goal_id=goals[0].id,
                                       actor_id=actors[actor].id, ceiling_snapshot=100, reserved_tokens=100,
                                       status=reservation_status,
                                       settled_tokens=10 if reservation_status == "settled" else 0,
                                       released_tokens=90 if reservation_status == "settled" else
                                       100 if reservation_status == "released" else 0,
                                       committed_at=now if reservation_status in {"settled", "held_unknown"} else None,
                                       settled_at=now if reservation_status == "settled" else None,
                                       released_at=now if reservation_status in {"settled", "released"} else None))
    feedback_ids = [uuid.uuid4() for actor in actors]
    for actor, feedback_id in zip(actors, feedback_ids):
        db.add(ConversationFeedback(id=feedback_id, response_id=responses[0].id,
                                   actor_id=actor.id, rating="helpful"))
    investigation = ConversationInvestigation(
        id=uuid.uuid4(), response_id=responses[0].id, goal_id=goals[0].id, actor_id=actors[0].id,
        context_version="old-v1", objective="legacy investigation", scope=[], input_manifest={},
        provider_identity="legacy-investigation", status="completed", attempt_count=1,
        accumulated_tokens=5, started_at=now, deadline_at=now, finished_at=now,
        provider_request_id="legacy-investigation:1", report={"summary": "legacy report"},
    )
    db.add(investigation)
    await db.flush()
    db.add(ConversationInvestigationReservation(
        id=uuid.uuid4(), investigation_id=investigation.id, goal_id=goals[0].id, actor_id=actors[0].id,
        ceiling_snapshot=20, reserved_tokens=20, settled_tokens=5, released_tokens=15,
        status="settled", committed_at=now, settled_at=now, released_at=now,
    ))
    proposals = [OrchestrationSteeringProposal(id=uuid.uuid4(), response_id=responses[i].id,
                                              goal_id=goals[0].id, actor_id=actors[0].id,
                                              draft={"directive": "legacy direction"}) for i in range(2)]
    db.add_all(proposals)
    await db.flush()
    request = OrchestrationSteeringRequest(
        id=uuid.uuid4(), goal_id=goals[0].id, actor_id=actors[0].id, client_request_id=uuid.uuid4(),
        sequence=1, submitted_run_id=runs[0].id, directive="legacy direction", target_type="goal",
        target_id=str(goals[0].id), scope="run", lifetime="remaining_current_run", impact_summary="legacy",
        source_proposal_id=proposals[1].id, status="pending", reason_code="submitted",
        contract_version="old-contract", plan_version="old-plan",
    )
    db.add(request)
    await db.flush()
    active_request = OrchestrationSteeringRequest(
        id=uuid.uuid4(), goal_id=goals[0].id, actor_id=actors[0].id, client_request_id=uuid.uuid4(),
        sequence=2, submitted_run_id=runs[0].id, directive="active legacy direction", target_type="goal",
        target_id=str(goals[0].id), scope="run", lifetime="remaining_current_run", impact_summary="legacy",
        status="applied", reason_code="applied", contract_version="old-contract", plan_version="old-plan",
        considered_at=now, finished_at=now,
    )
    db.add(active_request)
    foreign_message = ConversationMessage(goal_id=goals[2].id, actor_id=actors[0].id,
                                          client_request_id=uuid.uuid4(), sequence=1, content="foreign goal")
    db.add(foreign_message)
    await db.flush()
    db.add(ConversationResponse(message_id=foreign_message.id, run_id=runs[2].id, dossier={},
                                context_manifest={}, context_version="foreign-v1",
                                provider_request_id="foreign-goal-provider"))
    proposals[1].status = "promoted"
    proposals[1].promoted_request_id = request.id
    decision = OrchestrationDecision(run_id=runs[0].id, decision_type="steering", input_snapshot={"audit": True})
    db.add(decision)
    await db.flush()
    action = OrchestrationAction(run_id=runs[0].id, decision_id=decision.id,
                                 idempotency_key="legacy-result", action_type="steering")
    db.add(action)
    await db.flush()
    db.add_all([
        OrchestrationSteeringState(id=steering_state_id(goals[0].id), goal_id=goals[0].id,
                                  inbox_version=3, direction_version=2),
        OrchestrationSteeringTransition(id=uuid.uuid4(), request_id=request.id, sequence=1,
                                       to_status="pending", reason_code="submitted", actor="operator"),
        OrchestrationSteeringResultLink(id=uuid.uuid4(), request_id=request.id,
                                       decision_id=decision.id, action_id=action.id),
    ])
    await db.flush()
    if commit:
        await db.commit()
    return {"actors": [actor.id for actor in actors], "project": projects[0].id,
            "goal": goals[0].id, "advisors": [turn.id for turn in advisors],
            "responses": [response.id for response in responses],
            "proposals": [proposal.id for proposal in proposals], "request": request.id,
            "feedback": feedback_ids, "result_action": action.id, "active_request": active_request.id}


@pytest.mark.asyncio
async def test_upgrade_preserves_sources_and_visibility(tmp_path, monkeypatch):
    from huddleroom.services.orchestration_chat_history import history
    from huddleroom.config import settings

    monkeypatch.setattr(settings, "orchestration_advisor_allowance_tokens", 1000)
    monkeypatch.setattr(settings, "orchestration_conversation_allowance_tokens", 1000)

    path = tmp_path / "chat-upgrade.db"
    _alembic(path, "upgrade", "049")
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as db:
            seeded = await _seed_sources(db)
        before = _snapshot(path)
        _alembic(path, "upgrade", "050")
        from huddleroom.models.orchestration_chat import ProjectChatTurn
        with sqlite3.connect(path) as connection:
            assert {row[1] for row in connection.execute("PRAGMA table_info(project_chat_turns)")} == set(
                ProjectChatTurn.__table__.columns.keys())
            schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='project_chat_turns'"
            ).fetchone()[0]
            for constraint in ProjectChatTurn.__table__.constraints:
                if constraint.name:
                    assert constraint.name in schema
            indexes = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")}
            assert {index.name for index in ProjectChatTurn.__table__.indexes} <= indexes
        assert {table: rows for table, rows in _snapshot(path).items() if table in before} == before
        async with sessions() as db:
            first = await history(db, seeded["project"], seeded["actors"][0])
            second = await history(db, seeded["project"], seeded["actors"][0])
        assert first.model_dump() == second.model_dump()
        by_content = {item.content: item for item in first.items}
        assert set(by_content) == {"private A", "unknown usage", "pending advisor", "shared 1",
                                   "shared 2", "shared 3", "shared 4"}
        assert first.total == 7 and first.omitted == 0
        assert all(item.historical for item in first.items)
        assert by_content["private A"].source_references[0].id == seeded["advisors"][0]
        assert by_content["private A"].citations[0].goal_id == str(seeded["goal"])
        assert by_content["shared 2"].actor_id == seeded["actors"][1]
        assert [by_content[f"shared {i}"].status for i in range(1, 5)] == [
            "completed", "completed", "pending", "failed"]
        for i in range(1, 5):
            assert by_content[f"shared {i}"].source_references[0].id == seeded["responses"][i - 1]
            assert by_content[f"shared {i}"].source_goal_ids == [seeded["goal"]]
        assert by_content["shared 1"].goal_turn.feedback.rating == "helpful"
        assert by_content["shared 1"].goal_turn.feedback.feedback_id == seeded["feedback"][0]
        assert by_content["shared 1"].goal_turn.feedback_eligible is False
        assert by_content["shared 2"].goal_turn.feedback is None
        assert by_content["shared 2"].goal_turn.feedback_eligible is False
        assert by_content["shared 1"].goal_turn.investigation.status == "completed"
        assert by_content["shared 1"].goal_turn.proposed_steering.status == "proposed"
        assert by_content["shared 2"].goal_turn.proposed_steering.status == "promoted"
        ledger = first.steering[str(seeded["goal"])]
        assert (ledger.inbox_version, ledger.direction_version) == (3, 2)
        assert ledger.requests[0].request_id == seeded["request"]
        assert ledger.requests[0].status == "pending"
        assert len(ledger.requests[0].transitions) == 1
        assert ledger.requests[0].result_action_ids == [seeded["result_action"]]
        assert ledger.requests[1].request_id == seeded["active_request"]
        assert ledger.requests[1].status == "applied"
        assert first.goal_allowances[str(seeded["goal"])].used == 115
        assert first.allowance.remaining == 0  # Unknown historical advisor usage stays unknown/exhausted.
        async with sessions() as db:
            other = await history(db, seeded["project"], seeded["actors"][1])
        other_content = {item.content: item for item in other.items}
        assert set(other_content) == {"private B", "shared 1", "shared 2", "shared 3", "shared 4"}
        assert other_content["shared 1"].goal_turn.feedback.feedback_id == seeded["feedback"][1]
        assert other_content["shared 1"].goal_turn.feedback_eligible is False
        assert other_content["shared 2"].goal_turn.feedback_eligible is True
        assert {table: rows for table, rows in _snapshot(path).items() if table in before} == before
        _alembic(path, "downgrade", "049")
        assert _snapshot(path) == before
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_index_deduplicates_original_sources_and_filters_private_threads(db_session):
    from huddleroom.models.orchestration_chat import ProjectChatTurn
    from huddleroom.services.orchestration_chat_history import history

    seeded = await _seed_sources(db_session, commit=False)
    indexes = [ProjectChatTurn(project_id=seeded["project"], actor_id=seeded["actors"][actor],
                               client_request_id=uuid.uuid4(), content=content, source_kind=kind,
                               source_record_id=source_id, context_goal_id=goal_id,
                               answer="stale index answer", status="pending")
               for actor, content, kind, source_id, goal_id in (
                   (0, "indexed advisor", "project_advisor", seeded["advisors"][0], None),
                   (0, "indexed shared", "goal_conversation", seeded["responses"][0], seeded["goal"]),
                   (1, "other private index", "clarification", uuid.uuid4(), None))]
    db_session.add_all(indexes)
    await db_session.flush()
    projected = await history(db_session, seeded["project"], seeded["actors"][0])
    content = {item.content: item for item in projected.items}
    assert "private A" not in content and "shared 1" not in content
    assert "other private index" not in content
    assert len(projected.items) == 7
    assert content["indexed advisor"].answer == "legacy answer"
    assert content["indexed advisor"].status == "completed"
    assert content["indexed shared"].answer == "shared answer"
    assert content["indexed shared"].goal_turn.response_id == seeded["responses"][0]
    assert content["indexed shared"].historical is False
    assert content["indexed shared"].source_references[0].id == seeded["responses"][0]
    # Actor B still sees the shared source, not actor A's indexed private content.
    other = await history(db_session, seeded["project"], seeded["actors"][1])
    assert "shared 1" in {item.content for item in other.items}
    assert "indexed advisor" not in {item.content for item in other.items}
    assert "indexed shared" not in {item.content for item in other.items}


@pytest.mark.asyncio
async def test_history_rejects_missing_project_without_indexing(db_session, test_user):
    from huddleroom.services.orchestration_chat_history import history
    from huddleroom.services.orchestration_conversation_service import ConversationDomainError

    with pytest.raises(ConversationDomainError) as error:
        await history(db_session, uuid.uuid4(), test_user.id)
    assert error.value.code == "project_not_found"
