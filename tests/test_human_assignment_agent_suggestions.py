import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql.functions import count

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationAgentSuggestion
from huddleroom.models.task import Task
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.orchestration_service import OrchestrationService


async def _make_run(db_session, project_id):
    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=project_id,
        data=OrchestrationGoalCreate(
            objective="Handle weak roster fit",
            success_criteria=[{"key": "assigned", "description": "Weak roster fit is escalated safely."}],
        ),
        created_by_user_id=None,
    )
    return service, run


async def _event_rows(db_session, project_id, event_type):
    result = await db_session.execute(
        select(EventLog)
        .where(EventLog.project_id == project_id, EventLog.event_type == event_type)
        .order_by(EventLog.seq.asc())
    )
    return list(result.scalars().all())


def _agent(name_prefix: str, role: str, capabilities: list[str], *, is_active: bool = True) -> Agent:
    return Agent(
        name=f"{name_prefix}-{uuid.uuid4()}",
        role=role,
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=capabilities,
        config={},
        is_active=is_active,
    )


@pytest.mark.asyncio
async def test_execute_ask_human_action_reserves_completes_and_emits_once(db_session, test_project):
    service, run = await _make_run(db_session, test_project.id)
    candidate_id = uuid.uuid4()
    request = {
        "action_type": "ask_human",
        "question": "  No strong validation fit exists. Which existing agent should validate this gate?  ",
        "work_function": " validation ",
        "required_capabilities": ["validation", " ", "testing"],
        "candidate_agent_ids": [str(candidate_id), " "],
        "reason": "  Best available roster fit is weak.  ",
    }

    first = await service.execute_ask_human_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key="run:phase9:kind:ask_human:validation",
    )
    second = await service.execute_ask_human_action(
        db_session,
        run_id=run.id,
        request={**request, "question": "Replay should not replace original request."},
        idempotency_key="run:phase9:kind:ask_human:validation",
    )

    assert second.id == first.id
    assert first.status == "completed"
    assert first.action_type == "ask_human"
    assert first.target_type == "authority_decision"
    assert first.target_id is not None
    assert first.request == {
        "action_type": "ask_human",
        "question": "No strong validation fit exists. Which existing agent should validate this gate?",
        "work_function": "validation",
        "required_capabilities": ["validation", "testing"],
        "candidate_agent_ids": [str(candidate_id)],
        "gate_id": None,
        "reason": "Best available roster fit is weak.",
    }

    events = await _event_rows(db_session, test_project.id, "orchestration.human_input_required")
    assert len(events) == 1
    assert events[0].payload["run_id"] == str(run.id)
    assert events[0].payload["action_id"] == str(first.id)
    assert events[0].payload["question"] == first.request["question"]
    assert events[0].payload["work_function"] == "validation"
    assert events[0].payload["candidate_agent_ids"] == [str(candidate_id)]


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", ["ask_human", "suggest_agent"])
async def test_execute_action_malformed_completed_replay_returns_existing_action(db_session, test_project, action_type):
    service, run = await _make_run(db_session, test_project.id)

    if action_type == "ask_human":
        request = {
            "action_type": "ask_human",
            "question": "No strong validation fit exists. Which existing agent should validate this gate?",
            "work_function": "validation",
            "required_capabilities": ["validation", "testing"],
            "reason": "Best available roster fit is weak.",
        }
        first = await service.execute_ask_human_action(
            db_session,
            run_id=run.id,
            request=request,
            idempotency_key="run:phase9:kind:ask_human:malformed",
        )
        second = await service.execute_ask_human_action(
            db_session,
            run_id=run.id,
            request={"action_type": "ask_human"},
            idempotency_key="run:phase9:kind:ask_human:malformed",
        )
        event_type = "orchestration.human_input_required"
    else:  # suggest_agent
        request = {
            "action_type": "suggest_agent",
            "missing_work_function": "validation",
            "reason": "No active agent can provide independent validation.",
            "suggested_role": "validator",
            "suggested_capabilities": ["validation", "testing"],
        }
        first = await service.execute_suggest_agent_action(
            db_session,
            run_id=run.id,
            request=request,
            idempotency_key="run:phase9:kind:suggest_agent:malformed",
        )
        second = await service.execute_suggest_agent_action(
            db_session,
            run_id=run.id,
            request={"action_type": "suggest_agent"},
            idempotency_key="run:phase9:kind:suggest_agent:malformed",
        )
        event_type = "orchestration.agent_suggested"

    assert second.id == first.id
    assert second.status == "completed"

    events = await _event_rows(db_session, test_project.id, event_type)
    assert len(events) == 1

    if action_type == "suggest_agent":
        suggestion_count = await db_session.scalar(select(count(OrchestrationAgentSuggestion.id)))
        assert suggestion_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("action_type", ["ask_human", "suggest_agent"])
async def test_execute_action_rejects_malformed_request_before_reserving(db_session, test_project, action_type):
    service, run = await _make_run(db_session, test_project.id)

    with pytest.raises(HTTPException) as exc_info:
        if action_type == "ask_human":
            await service.execute_ask_human_action(
                db_session,
                run_id=run.id,
                request={"action_type": "ask_human"},
                idempotency_key="run:phase9:kind:ask_human:invalid-first",
            )
        else:  # suggest_agent
            await service.execute_suggest_agent_action(
                db_session,
                run_id=run.id,
                request={"action_type": "suggest_agent"},
                idempotency_key="run:phase9:kind:suggest_agent:invalid-first",
            )

    assert exc_info.value.status_code == 400
    action_count = await db_session.scalar(select(count(OrchestrationAction.id)))
    assert action_count == 0
    event_count = await db_session.scalar(
        select(count(EventLog.id)).where(EventLog.project_id == test_project.id)
    )
    assert event_count == 0

    if action_type == "suggest_agent":
        suggestion_count = await db_session.scalar(select(count(OrchestrationAgentSuggestion.id)))
        assert suggestion_count == 0


@pytest.mark.asyncio
async def test_execute_suggest_agent_action_persists_suggestion_and_emits_once(db_session, test_project):
    agent_count_before = await db_session.scalar(select(count(Agent.id)))
    service, run = await _make_run(db_session, test_project.id)
    request = {
        "action_type": "suggest_agent",
        "missing_work_function": " validation ",
        "reason": "  No active agent can provide independent validation. ",
        "suggested_role": " validator ",
        "suggested_capabilities": ["validation", " ", "testing"],
        "suggested_adapter_type": " api ",
        "suggested_model": " gpt-4o-mini ",
        "suggested_system_prompt_outline": (
            "  Validate completed work and report evidence without editing artifacts.  "
        ),
    }

    first = await service.execute_suggest_agent_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key="run:phase9:kind:suggest_agent:validation",
    )
    second = await service.execute_suggest_agent_action(
        db_session,
        run_id=run.id,
        request={**request, "suggested_role": "different"},
        idempotency_key="run:phase9:kind:suggest_agent:validation",
    )

    assert second.id == first.id
    assert first.status == "completed"
    assert first.action_type == "suggest_agent"
    assert first.target_type == "agent_suggestion"
    assert first.target_id is not None
    assert first.request == {
        "action_type": "suggest_agent",
        "missing_work_function": "validation",
        "reason": "No active agent can provide independent validation.",
        "suggested_role": "validator",
        "suggested_capabilities": ["validation", "testing"],
        "suggested_adapter_type": "api",
        "suggested_model": "gpt-4o-mini",
        "suggested_system_prompt_outline": "Validate completed work and report evidence without editing artifacts.",
    }

    suggestion = await db_session.get(OrchestrationAgentSuggestion, first.target_id)
    assert suggestion is not None
    assert suggestion.run_id == run.id
    assert suggestion.status == "open"
    assert suggestion.missing_work_function == "validation"
    assert suggestion.reason == "No active agent can provide independent validation."
    assert suggestion.suggested_role == "validator"
    assert suggestion.suggested_capabilities == ["validation", "testing"]
    assert suggestion.suggested_adapter_type == "api"
    assert suggestion.suggested_model == "gpt-4o-mini"
    assert suggestion.suggested_system_prompt_outline == (
        "Validate completed work and report evidence without editing artifacts."
    )

    # suggest_agent only records an OrchestrationAgentSuggestion; it must not
    # also create a real Agent row as a side effect.
    agent_count = await db_session.scalar(select(count(Agent.id)))
    assert agent_count == agent_count_before

    events = await _event_rows(db_session, test_project.id, "orchestration.agent_suggested")
    assert len(events) == 1
    assert events[0].payload["run_id"] == str(run.id)
    assert events[0].payload["action_id"] == str(first.id)
    assert events[0].payload["suggestion_id"] == str(suggestion.id)
    assert events[0].payload["missing_work_function"] == "validation"


@pytest.mark.asyncio
async def test_handle_weak_roster_fit_records_suggestion_for_existing_weak_roster(db_session, test_project):
    finance_agent = _agent("finance", "finance", ["invoicing"])
    db_session.add(finance_agent)
    await db_session.flush()
    service, run = await _make_run(db_session, test_project.id)

    action = await service.handle_weak_roster_fit(
        db_session,
        run_id=run.id,
        work_function="validation",
        required_capabilities=["validation"],
        reason="No strong validation fit exists.",
    )

    assert action.action_type == "suggest_agent"
    assert action.status == "completed"
    assert action.target_type == "agent_suggestion"

    suggestion = await db_session.get(OrchestrationAgentSuggestion, action.target_id)
    assert suggestion is not None
    assert suggestion.missing_work_function == "validation"
    assert suggestion.reason == "No strong validation fit exists."
    assert suggestion.suggested_role == "validator"
    assert suggestion.suggested_capabilities == ["validation"]

    human_events = await _event_rows(db_session, test_project.id, "orchestration.human_input_required")
    assert human_events == []
    events = await _event_rows(db_session, test_project.id, "orchestration.agent_suggested")
    assert len(events) == 1
    assert events[0].payload["suggestion_id"] == str(suggestion.id)


@pytest.mark.asyncio
async def test_handle_weak_roster_fit_replays_when_roster_load_changes(db_session, test_project):
    first_candidate = _agent("a-finance", "finance", ["invoicing"])
    second_candidate = _agent("b-finance", "finance", ["billing"])
    db_session.add_all([first_candidate, second_candidate])
    await db_session.flush()
    service, run = await _make_run(db_session, test_project.id)

    first = await service.handle_weak_roster_fit(
        db_session,
        run_id=run.id,
        work_function="validation",
        required_capabilities=["validation"],
        reason="No strong validation fit exists.",
    )
    db_session.add(
        Task(
            project_id=test_project.id,
            title="Load shift",
            status="in_progress",
            assigned_to=first_candidate.id,
        )
    )
    await db_session.flush()

    second = await service.handle_weak_roster_fit(
        db_session,
        run_id=run.id,
        work_function="validation",
        required_capabilities=["validation"],
        reason="No strong validation fit exists after load changed.",
    )

    assert second.id == first.id
    assert second.request["reason"] == "No strong validation fit exists."
    suggestion_count = await db_session.scalar(select(count(OrchestrationAgentSuggestion.id)))
    assert suggestion_count == 1
    events = await _event_rows(db_session, test_project.id, "orchestration.agent_suggested")
    assert len(events) == 1


@pytest.mark.asyncio
async def test_handle_weak_roster_fit_replays_existing_action_when_roster_becomes_strong(db_session, test_project):
    finance_agent = _agent("finance", "finance", ["invoicing"])
    db_session.add(finance_agent)
    await db_session.flush()
    service, run = await _make_run(db_session, test_project.id)

    first = await service.handle_weak_roster_fit(
        db_session,
        run_id=run.id,
        work_function="validation",
        required_capabilities=["validation"],
        reason="No strong validation fit exists.",
    )
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()

    second = await service.handle_weak_roster_fit(
        db_session,
        run_id=run.id,
        work_function="validation",
        required_capabilities=["validation"],
        reason="A strong validator now exists, but this need was already escalated.",
    )

    assert second.id == first.id
    assert second.request["reason"] == "No strong validation fit exists."
    suggestion_count = await db_session.scalar(select(count(OrchestrationAgentSuggestion.id)))
    assert suggestion_count == 1
    events = await _event_rows(db_session, test_project.id, "orchestration.agent_suggested")
    assert len(events) == 1


def test_stable_need_hash_normalizes_capability_tokens():
    service = OrchestrationService()

    first = service._stable_need_hash("validation", [" Validation ", "code-review", "testing"])
    second = service._stable_need_hash("validation", ["validation", "code_review", " testing "])

    assert first == second


@pytest.mark.asyncio
async def test_handle_weak_roster_fit_records_suggestion_when_no_active_agent(db_session, test_project):
    inactive_agent = _agent("inactive", "validator", ["validation"], is_active=False)
    db_session.add(inactive_agent)
    await db_session.flush()
    service, run = await _make_run(db_session, test_project.id)

    action = await service.handle_weak_roster_fit(
        db_session,
        run_id=run.id,
        work_function="validation",
        required_capabilities=["validation", "testing"],
        reason="No active validation agent exists.",
    )

    assert action.action_type == "suggest_agent"
    assert action.status == "completed"
    assert action.target_type == "agent_suggestion"

    suggestion = await db_session.get(OrchestrationAgentSuggestion, action.target_id)
    assert suggestion is not None
    assert suggestion.missing_work_function == "validation"
    assert suggestion.reason == "No active validation agent exists."
    assert suggestion.suggested_role == "validator"
    assert suggestion.suggested_capabilities == ["validation", "testing"]
    assert suggestion.status == "open"

    active_agent_count = await db_session.scalar(
        select(count(Agent.id)).where(Agent.is_active.is_(True))
    )
    assert active_agent_count == 0

    events = await _event_rows(db_session, test_project.id, "orchestration.agent_suggested")
    assert len(events) == 1
    assert events[0].payload["suggestion_id"] == str(suggestion.id)


@pytest.mark.asyncio
async def test_handle_weak_roster_fit_rejects_strong_fit(db_session, test_project):
    validator = _agent("validator", "validator", ["validation"])
    db_session.add(validator)
    await db_session.flush()
    service, run = await _make_run(db_session, test_project.id)

    with pytest.raises(Exception) as exc_info:
        await service.handle_weak_roster_fit(
            db_session,
            run_id=run.id,
            work_function="validation",
            required_capabilities=["validation"],
            reason="This should not escalate because the fit is strong.",
        )

    assert getattr(exc_info.value, "status_code", None) == 409
    assert getattr(exc_info.value, "detail", "") == "Roster fit is not weak"

    events = await _event_rows(db_session, test_project.id, "orchestration.human_input_required")
    assert events == []


@pytest.mark.asyncio
async def test_execute_suggest_agent_action_reuses_reserved_action_suggestion(db_session, test_project):
    service, run = await _make_run(db_session, test_project.id)
    request = {
        "action_type": "suggest_agent",
        "missing_work_function": "validation",
        "reason": "No active agent can provide independent validation.",
        "suggested_role": "validator",
        "suggested_capabilities": ["validation", "testing"],
        "suggested_adapter_type": "api",
        "suggested_model": "gpt-4o-mini",
    }
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase9:kind:suggest_agent:reserved-replay",
        action_type="suggest_agent",
        request=request,
    )
    db_session.add(
        OrchestrationAgentSuggestion(
            id=action.id,
            run_id=run.id,
            missing_work_function="validation",
            reason="No active agent can provide independent validation.",
            suggested_role="validator",
            suggested_capabilities=["validation", "testing"],
            suggested_adapter_type="api",
            suggested_model="gpt-4o-mini",
        )
    )
    await db_session.flush()

    replay = await service.execute_suggest_agent_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key="run:phase9:kind:suggest_agent:reserved-replay",
    )

    assert replay.id == action.id
    assert replay.status == "completed"
    assert replay.target_type == "agent_suggestion"
    assert replay.target_id == action.id

    suggestion_count = await db_session.scalar(select(count(OrchestrationAgentSuggestion.id)))
    assert suggestion_count == 1

    events = await _event_rows(db_session, test_project.id, "orchestration.agent_suggested")
    assert len(events) == 1


@pytest.mark.asyncio
async def test_execute_suggest_agent_action_reselects_after_raced_suggestion_insert(
    db_session, test_project, monkeypatch
):
    service, run = await _make_run(db_session, test_project.id)
    request = {
        "action_type": "suggest_agent",
        "missing_work_function": "validation",
        "reason": "No active agent can provide independent validation.",
        "suggested_role": "validator",
        "suggested_capabilities": ["validation", "testing"],
        "suggested_adapter_type": "api",
        "suggested_model": "gpt-4o-mini",
    }
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:phase9:kind:suggest_agent:raced-insert",
        action_type="suggest_agent",
        request=request,
    )
    existing_suggestion = OrchestrationAgentSuggestion(
        id=action.id,
        run_id=run.id,
        missing_work_function="validation",
        reason="No active agent can provide independent validation.",
        suggested_role="validator",
        suggested_capabilities=["validation", "testing"],
        suggested_adapter_type="api",
        suggested_model="gpt-4o-mini",
    )
    original_get = db_session.get
    get_attempts = 0
    flush_raised = False

    async def fake_get(model, ident, *args, **kwargs):
        nonlocal get_attempts
        if model is OrchestrationAgentSuggestion and ident == action.id:
            get_attempts += 1
            return None if get_attempts == 1 else existing_suggestion
        return await original_get(model, ident, *args, **kwargs)

    original_flush = db_session.flush

    async def fake_flush(*args, **kwargs):
        nonlocal flush_raised
        if not flush_raised and any(
            isinstance(obj, OrchestrationAgentSuggestion) and obj.id == action.id for obj in db_session.new
        ):
            flush_raised = True
            raise IntegrityError("insert", {}, Exception("duplicate"))
        return await original_flush(*args, **kwargs)

    monkeypatch.setattr(db_session, "get", fake_get)
    monkeypatch.setattr(db_session, "flush", fake_flush)

    completed = await service.execute_suggest_agent_action(
        db_session,
        run_id=run.id,
        request=request,
        idempotency_key="run:phase9:kind:suggest_agent:raced-insert",
    )

    assert completed.id == action.id
    assert completed.status == "completed"
    assert completed.target_type == "agent_suggestion"
    assert completed.target_id == action.id
    assert flush_raised is True

    events = await _event_rows(db_session, test_project.id, "orchestration.agent_suggested")
    assert len(events) == 1
