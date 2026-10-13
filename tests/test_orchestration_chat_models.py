import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from huddleroom.models.orchestration_chat import ProjectChatTurn
from huddleroom.schemas.orchestration_chat import ChatActionDraft, ChatInterpretation, ChatSubmitRequest


@pytest.mark.parametrize("content", ["", " \n\t", "x" * 12001])
def test_submit_rejects_empty_or_unbounded_input(content):
    with pytest.raises(ValidationError):
        ChatSubmitRequest(content=content, client_request_id=uuid.uuid4())


def test_submit_trims_content_and_rejects_extra_authority_fields():
    request_id = uuid.uuid4()
    assert ChatSubmitRequest(content="  pause Alpha  ", client_request_id=request_id).content == "pause Alpha"
    with pytest.raises(ValidationError):
        ChatSubmitRequest(content="pause", client_request_id=request_id, confirmed=True)


@pytest.mark.parametrize("operation", ["create_goal", "stop_continuous", "reset", "waive_gate", "shell"])
def test_interpretation_and_draft_reject_unsupported_operations(operation):
    with pytest.raises(ValidationError):
        ChatInterpretation(intent="instruction", operation=operation)
    with pytest.raises(ValidationError):
        ChatActionDraft(action_id=uuid.uuid4(), operation=operation, project_id=uuid.uuid4(),
                        goal_id=uuid.uuid4(), summary="unsupported")


def test_action_payload_rejects_arbitrary_execution_and_unbounded_batches():
    with pytest.raises(ValidationError):
        ChatInterpretation(intent="instruction", operation="resume", payload={"command": "rm -rf"})
    with pytest.raises(ValidationError):
        ChatInterpretation(intent="instruction", operation="answer_agent_definition_batch",
                           payload={"answers": [{}] * 101})
    with pytest.raises(ValidationError):
        ChatInterpretation(intent="instruction", operation="answer_agent_definition_batch", payload={
            "answers": [{"decision_id": str(uuid.uuid4()), "selected_option": "approve", "waive_gates": True}],
        })


def _agent_edit(**changes):
    return {"name": "Reviewer", "role": "reviewer", "provider": "mock", "model": "mock-model", **changes}


@pytest.mark.parametrize("edit", [
    _agent_edit(description="x" * 12001),
    _agent_edit(system_prompt="x" * 12001),
    _agent_edit(capabilities=["read"] * 101),
    _agent_edit(capabilities=["x" * 12001]),
    _agent_edit(config={"value": "x" * 12001}),
    _agent_edit(config={str(i): i for i in range(101)}),
    _agent_edit(config={"value": [0] * 101}),
    _agent_edit(config={"a": "x" * 12000, "b": "x" * 12000}),
    _agent_edit(config={"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": 1}}}}}}}}}}),
    _agent_edit(command="run arbitrary tool"),
])
def test_chat_agent_edit_rejects_unbounded_or_unknown_nested_values(edit):
    with pytest.raises(ValidationError):
        ChatInterpretation(intent="instruction", operation="answer_authority", payload={"edited_agent": edit})


@pytest.mark.parametrize("field", ["reason", "edited_description", "edited_persona", "selected_option"])
def test_chat_batch_answer_rejects_unbounded_nested_text(field):
    answer = {"decision_id": str(uuid.uuid4()), "selected_option": "approve", field: "x" * 12001}
    with pytest.raises(ValidationError):
        ChatInterpretation(intent="instruction", operation="answer_agent_definition_batch", payload={"answers": [answer]})


@pytest.mark.parametrize("versions", [{"x" * 129: "v1"}, {"contract": "x" * 12001},
                                       {str(i): "v1" for i in range(31)}])
def test_chat_frozen_versions_rejects_unbounded_keys_values_and_count(versions):
    with pytest.raises(ValidationError):
        ChatActionDraft(action_id=uuid.uuid4(), operation="resume", project_id=uuid.uuid4(),
                        goal_id=uuid.uuid4(), summary="Resume", frozen_versions=versions)


def test_chat_specialized_edits_preserve_existing_authority_fields():
    decision_id = uuid.uuid4()
    payload = {
        "selected_option": "approve_with_edits", "reason": "Clarify boundaries", "contract_version": "v3",
        "edited_description": "Review changes", "edited_persona": "Precise",
        "edited_agent": _agent_edit(description="Reviewer", system_prompt="Review carefully",
                                   capabilities=["read"], config={"limits": {"tokens": 100}}),
        "answers": [{"decision_id": str(decision_id), "selected_option": "approve_with_edits",
                     "reason": "Clarify boundaries", "edited_description": "Review changes", "edited_persona": "Precise"}],
    }
    parsed = ChatInterpretation(intent="instruction", operation="answer_authority", payload=payload)
    actual = parsed.payload.model_dump(mode="json", exclude_none=True)
    for field, value in payload.items():
        if field == "edited_agent":
            for nested, nested_value in value.items():
                assert actual[field][nested] == nested_value
        else:
            assert actual[field] == value


@pytest.mark.asyncio
async def test_chat_identity_and_preallocated_provider_authority_persist(
    db_session, test_project, test_user, conversation_goal_run
):
    goal, run = conversation_goal_run
    request_id, source_id = uuid.uuid4(), uuid.uuid4()
    deadline = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
    row = ProjectChatTurn(project_id=test_project.id, actor_id=test_user.id,
                          client_request_id=request_id, content="Question", context_goal_id=goal.id,
                          run_id=run.id, frozen_context={"contract_version": "frozen-v1"},
                          source_kind="project_advisor", source_record_id=source_id,
                          provider_request_id="chat-provider-test", provider_status="pending",
                          deadline_at=deadline, accounting_snapshot={"scope": "project", "reserved_tokens": 100})
    db_session.add(row)
    await db_session.flush()
    db_session.expire(row)
    await db_session.refresh(row)
    assert row.client_request_id == request_id
    assert row.source_record_id == source_id
    assert row.provider_request_id == "chat-provider-test"
    assert row.provider_status == "pending"
    assert row.context_goal_id == goal.id and row.run_id == run.id
    assert row.frozen_context == {"contract_version": "frozen-v1"}
    assert row.accounting_snapshot == {"scope": "project", "reserved_tokens": 100}
    assert row.deadline_at.replace(tzinfo=timezone.utc) == deadline
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(ProjectChatTurn(project_id=test_project.id, actor_id=test_user.id,
                                           client_request_id=request_id, content="conflicting request"))
            await db_session.flush()
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(ProjectChatTurn(project_id=test_project.id, actor_id=test_user.id,
                                           client_request_id=uuid.uuid4(), content="other",
                                           provider_request_id="chat-provider-test"))
            await db_session.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("status", "applied"), ("provider_status", "uncertain"),
                                        ("source_kind", "arbitrary_tool")])
async def test_chat_storage_rejects_invalid_lifecycle_literals(db_session, test_project, test_user, field, value):
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(ProjectChatTurn(project_id=test_project.id, actor_id=test_user.id,
                                           client_request_id=uuid.uuid4(), content="test", **{field: value}))
            await db_session.flush()
