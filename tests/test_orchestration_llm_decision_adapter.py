# pylint: disable=redefined-outer-name

import asyncio
import json
from types import SimpleNamespace
import uuid

from fastapi import HTTPException
import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.orchestration import OrchestrationDecision, OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.schemas.orchestration import OrchestrationGoalCreate
from huddleroom.services.event_bus import emit_event_once
from huddleroom.services.orchestration_llm_decision_adapter import (
    OrchestrationDecisionAdapter,
    OrchestrationDecisionAdapterResult,
    build_orchestration_decision_messages,
    parse_decision_content,
)
from huddleroom.services.orchestration_service import OrchestrationService


@pytest_asyncio.fixture(autouse=True)
async def use_test_database(test_engine, monkeypatch):
    import huddleroom.services.orchestration_service as orchestration_service_module

    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", _test_session_factory(test_engine))


def test_prompt_states_hard_boundary_and_allowed_schema():
    messages = build_orchestration_decision_messages(
        {
            "goal": {"objective": "Ship a feature"},
            "run": {"status": "running"},
            "recent_events": [],
        }
    )

    assert messages[0]["role"] == "system"
    system_text = messages[0]["content"]
    assert "must not write plans" in system_text
    assert "must not write code" in system_text
    assert "must not write tests" in system_text
    assert "must not write reviews" in system_text
    assert "must not write validation reports" in system_text
    assert "must not write meeting decisions" in system_text
    assert "must not write project artifacts" in system_text
    assert "must not write final summaries" in system_text
    assert "request_plan" in system_text
    assert "ask_human" in system_text
    assert "parent_task_id" in system_text
    assert "source_session_id" in system_text
    assert "Return exactly one JSON object" in system_text
    assert "Top-level reason is optional for every action." in system_text
    assert messages[1]["role"] == "user"
    assert '"objective": "Ship a feature"' in messages[1]["content"]


def test_parse_decision_content_accepts_wrapped_decision():
    parsed = parse_decision_content(
        json.dumps(
            {
                "decision": {
                    "action_type": "ask_human",
                    "question": "Which existing agent should validate this gate?",
                    "reason": "No clear validator exists.",
                }
            }
        )
    )

    assert parsed == {
        "action_type": "ask_human",
        "question": "Which existing agent should validate this gate?",
        "reason": "No clear validator exists.",
    }


@pytest.mark.parametrize(
    ("raw_content", "reason"),
    [
        ("not json", "LLM output was not valid JSON"),
        (json.dumps(["noop"]), "LLM output JSON root must be an object"),
        (json.dumps({"action_type": "noop"}), "LLM output must contain a decision object"),
        (json.dumps({"decision": ["noop"]}), "LLM output decision must be an object"),
    ],
)
def test_parse_decision_content_rejects_malformed_contract(raw_content, reason):
    parsed = parse_decision_content(raw_content)

    assert parsed == {"action_type": "invalid_llm_output", "reason": reason}


@pytest.mark.asyncio
async def test_adapter_decide_calls_litellm_boundary_with_json_response_format():
    calls = []

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "decision": {
                                    "action_type": "noop",
                                    "reason": "Waiting for new evidence.",
                                }
                            }
                        )
                    )
                )
            ]
        )

    adapter = OrchestrationDecisionAdapter(model="openai/test-model", completion_fn=fake_completion)
    result = await adapter.decide({"goal": {"objective": "Wait"}, "run": {"status": "running"}})

    assert result.input_snapshot["goal"]["objective"] == "Wait"
    assert result.llm_output["raw_content"] == json.dumps(
        {"decision": {"action_type": "noop", "reason": "Waiting for new evidence."}}
    )
    assert result.parsed_decision == {"action_type": "noop", "reason": "Waiting for new evidence."}
    assert calls[0]["model"] == "openai/test-model"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["temperature"] == 0
    assert calls[0]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_adapter_decide_accepts_fenced_json_but_retains_raw_diagnostics():
    raw_content = ' \n```json\n{"decision":{"action_type":"noop","reason":"Wait."}}\n```\n '

    async def fake_completion(**_kwargs):
        return {"choices": [{"message": {"content": raw_content}}]}

    result = await OrchestrationDecisionAdapter(completion_fn=fake_completion).decide({"run": {"status": "running"}})

    assert result.llm_output["raw_content"] == raw_content
    assert result.parsed_decision == {"action_type": "noop", "reason": "Wait."}


@pytest.mark.asyncio
async def test_adapter_decide_accepts_mapping_response_shape():
    async def fake_completion(**_kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "decision": {
                                    "action_type": "noop",
                                    "reason": "Mapping responses should work too.",
                                }
                            }
                        )
                    }
                }
            ]
        }

    adapter = OrchestrationDecisionAdapter(completion_fn=fake_completion)
    result = await adapter.decide({"goal": {"objective": "Wait"}, "run": {"status": "running"}})

    assert result.parsed_decision == {"action_type": "noop", "reason": "Mapping responses should work too."}


@pytest.mark.parametrize(
    "content_value",
    [None, ["not", "a", "string"]],
    ids=["none", "list"],
)
@pytest.mark.asyncio
async def test_adapter_decide_rejects_non_string_content(content_value):
    async def fake_completion(**_kwargs):
        return {"choices": [{"message": {"content": content_value}}]}

    adapter = OrchestrationDecisionAdapter(completion_fn=fake_completion)
    result = await adapter.decide({"goal": {"objective": "Wait"}, "run": {"status": "running"}})

    assert result.llm_output == {
        "raw_content": None,
        "response_error": "LLM response content must be a string",
    }
    assert result.parsed_decision == {
        "action_type": "invalid_llm_output",
        "reason": "LLM response content must be a string",
    }


@pytest.mark.asyncio
async def test_adapter_decide_maps_completion_failures_to_invalid_output():
    async def fake_completion(**_kwargs):
        raise RuntimeError("rate limited")

    adapter = OrchestrationDecisionAdapter(completion_fn=fake_completion)
    result = await adapter.decide({"goal": {"objective": "Wait"}, "run": {"status": "running"}})

    assert result.llm_output == {
        "raw_content": None,
        "completion_error": "RuntimeError: rate limited",
    }
    assert result.parsed_decision == {
        "action_type": "invalid_llm_output",
        "reason": "LLM completion failed: RuntimeError: rate limited",
    }


@pytest.mark.asyncio
async def test_adapter_decide_redacts_completion_failure_secrets():
    async def fake_completion(**_kwargs):
        raise RuntimeError("Authorization: Bearer super-secret-token api_key=abc123")

    adapter = OrchestrationDecisionAdapter(completion_fn=fake_completion)
    result = await adapter.decide({"goal": {"objective": "Wait"}, "run": {"status": "running"}})

    assert result.llm_output == {
        "raw_content": None,
        "completion_error": "RuntimeError: Authorization=[REDACTED] api_key=[REDACTED]",
    }
    assert result.parsed_decision == {
        "action_type": "invalid_llm_output",
        "reason": "LLM completion failed: RuntimeError: Authorization=[REDACTED] api_key=[REDACTED]",
    }


@pytest.mark.asyncio
async def test_adapter_decide_redacts_json_quoted_completion_failure_secrets():
    async def fake_completion(**_kwargs):
        raise RuntimeError(
            'request body {"api_key":"sk-secret","secret": "top-secret","token":"abc123"}'
        )

    adapter = OrchestrationDecisionAdapter(completion_fn=fake_completion)
    result = await adapter.decide({"goal": {"objective": "Wait"}, "run": {"status": "running"}})

    assert result.llm_output == {
        "raw_content": None,
        "completion_error": (
            'RuntimeError: request body {"api_key":"[REDACTED]","secret": '
            '"[REDACTED]","token":"[REDACTED]"}'
        ),
    }
    assert result.parsed_decision == {
        "action_type": "invalid_llm_output",
        "reason": (
            'LLM completion failed: RuntimeError: request body '
            '{"api_key":"[REDACTED]","secret": "[REDACTED]","token":"[REDACTED]"}'
        ),
    }


@pytest.mark.asyncio
async def test_adapter_decide_redacts_bare_provider_keys_in_completion_failures():
    async def fake_completion(**_kwargs):
        raise RuntimeError("Incorrect API key provided: sk-abcdefghijklmnopqrstuvwxyz123456")

    adapter = OrchestrationDecisionAdapter(completion_fn=fake_completion)
    result = await adapter.decide({"goal": {"objective": "Wait"}, "run": {"status": "running"}})

    assert result.llm_output == {
        "raw_content": None,
        "completion_error": "RuntimeError: Incorrect API key provided: [REDACTED]",
    }
    assert result.parsed_decision == {
        "action_type": "invalid_llm_output",
        "reason": "LLM completion failed: RuntimeError: Incorrect API key provided: [REDACTED]",
    }


@pytest.mark.asyncio
async def test_adapter_decide_truncates_long_completion_failures():
    async def fake_completion(**_kwargs):
        raise RuntimeError("x" * 200)

    adapter = OrchestrationDecisionAdapter(completion_fn=fake_completion)
    result = await adapter.decide({"goal": {"objective": "Wait"}, "run": {"status": "running"}})

    completion_error = result.llm_output["completion_error"]
    assert completion_error.startswith("RuntimeError: ")
    assert completion_error.endswith("...")
    assert len(completion_error) <= len("RuntimeError: ") + 120


@pytest.mark.asyncio
async def test_adapter_decide_maps_structural_response_failures_to_invalid_output():
    async def fake_completion(**_kwargs):
        return {"choices": []}

    adapter = OrchestrationDecisionAdapter(completion_fn=fake_completion)
    result = await adapter.decide({"goal": {"objective": "Wait"}, "run": {"status": "running"}})

    assert result.llm_output == {
        "raw_content": None,
        "response_error": "LLM response did not include choices[0]",
    }
    assert result.parsed_decision == {
        "action_type": "invalid_llm_output",
        "reason": "LLM response did not include choices[0]",
    }


async def _complete_goal_definition(db_session, goal, run):
    from tests.conftest import complete_baseline_processes

    run.baseline_authorized = True
    await complete_baseline_processes(db_session, goal, run)


async def _make_run(db_session, test_project):
    service = OrchestrationService()
    goal, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Choose the next coordination move",
            success_criteria=[{"key": "done", "description": "The goal is proven complete."}],
            constraints={"max_parallel_work": 1},
            budget={"llm_calls": 3},
        ),
        created_by_user_id=None,
    )
    await _complete_goal_definition(db_session, goal, run)
    return service, run


class RecordingAdapter:
    def __init__(self, parsed_decision):
        self.parsed_decision = parsed_decision
        self.contexts = []

    async def decide(self, context, project=None, goal=None):
        self.contexts.append(context)
        return OrchestrationDecisionAdapterResult(
            input_snapshot=context,
            llm_output={"raw_content": json.dumps({"decision": self.parsed_decision})},
            parsed_decision=self.parsed_decision,
        )


def _test_session_factory(test_engine):
    return async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def _make_committed_run(test_engine, workspace):
    session_factory = _test_session_factory(test_engine)
    async with session_factory() as session:
        project = Project(
            name="Committed Run Project",
            description="test",
            workspace_path=str(workspace),
            config={},
        )
        session.add(project)
        await session.flush()
        service, run = await _make_run(session, project)
        await session.commit()
        return service, session_factory, project.id, run.goal_id, run.id


@pytest_asyncio.fixture(name="committed_run")
async def committed_run_fixture(test_engine, tmp_path):
    service, session_factory, project_id, goal_id, run_id = await _make_committed_run(test_engine, tmp_path)
    try:
        yield service, session_factory, run_id
    finally:
        async with session_factory() as cleanup_db:
            await cleanup_db.execute(delete(OrchestrationDecision).where(OrchestrationDecision.run_id == run_id))
            await cleanup_db.execute(delete(OrchestrationRun).where(OrchestrationRun.id == run_id))
            await cleanup_db.execute(delete(OrchestrationGoal).where(OrchestrationGoal.id == goal_id))
            await cleanup_db.execute(delete(Project).where(Project.id == project_id))
            await cleanup_db.commit()


@pytest.mark.asyncio
async def test_request_llm_decision_records_accepted_adapter_decision(db_session, test_project):
    service, run = await _make_run(db_session, test_project)
    adapter = RecordingAdapter(
        {
            "action_type": "ask_human",
            "question": "Which existing agent should validate this gate?",
            "reason": "No safe automatic fit exists.",
        }
    )

    decision = await service.request_llm_decision(db_session, run.id, adapter=adapter)

    assert decision.run_id == run.id
    assert decision.decision_type == "ask_human"
    assert decision.validator_status == "accepted"
    assert decision.rejection_reason is None
    assert decision.reason == "No safe automatic fit exists."
    assert adapter.contexts[0]["goal"]["objective"] == "Choose the next coordination move"
    assert adapter.contexts[0]["run"]["id"] == str(run.id)


@pytest.mark.asyncio
async def test_request_llm_decision_records_malformed_adapter_output_as_rejected(db_session, test_project):
    service, run = await _make_run(db_session, test_project)
    adapter = RecordingAdapter(
        {
            "action_type": "invalid_llm_output",
            "reason": "LLM output was not valid JSON",
        }
    )

    decision = await service.request_llm_decision(db_session, run.id, adapter=adapter)

    assert decision.decision_type == "invalid_llm_output"
    assert decision.validator_status == "rejected"
    assert decision.rejection_reason == "Unknown orchestration action type 'invalid_llm_output'"
    assert decision.parsed_decision == {
        "action_type": "invalid_llm_output",
        "reason": "LLM output was not valid JSON",
    }


@pytest.mark.asyncio
async def test_decision_context_excludes_orchestration_events_to_keep_cache_stable(db_session, test_project):
    service, run = await _make_run(db_session, test_project)
    await emit_event_once(
        db_session,
        test_project.id,
        "orchestration.tick",
        {"run_id": str(run.id)},
        dedup_key=f"phase7-tick-{uuid.uuid4()}",
    )
    await emit_event_once(
        db_session,
        test_project.id,
        "task.created",
        {"title": "Existing work"},
        dedup_key=f"phase7-task-{uuid.uuid4()}",
    )
    adapter = RecordingAdapter({"action_type": "noop", "reason": "Only inspecting context."})

    await service.request_llm_decision(db_session, run.id, adapter=adapter)

    assert "recent_events" not in adapter.contexts[0]


@pytest.mark.asyncio
async def test_request_llm_decision_allows_blocked_run(db_session, test_project):
    service, run = await _make_run(db_session, test_project)
    run.status = "blocked"
    await db_session.flush()
    adapter = RecordingAdapter({"action_type": "noop", "reason": "Blocked runs can still coordinate recovery."})

    decision = await service.request_llm_decision(db_session, run.id, adapter=adapter)

    assert decision.validator_status == "accepted"
    assert adapter.contexts[0]["run"]["status"] == "blocked"


@pytest.mark.asyncio
async def test_request_llm_decision_rejects_inactive_run(db_session, test_project):
    service, run = await _make_run(db_session, test_project)
    run.status = "paused"
    await db_session.flush()

    with pytest.raises(HTTPException) as exc:
        await service.request_llm_decision(
            db_session,
            run.id,
            adapter=RecordingAdapter({"action_type": "noop", "reason": "Paused."}),
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == "Orchestration run is paused"


@pytest.mark.asyncio
async def test_request_llm_decision_fallback_404s_when_goal_missing(db_session, test_project, monkeypatch):
    service, run = await _make_run(db_session, test_project)
    adapter = RecordingAdapter({"action_type": "noop", "reason": "Should not reach adapter."})

    original_get = db_session.get

    async def fake_get(model, ident, *args, **kwargs):
        if model.__name__ == "OrchestrationGoal" and ident == run.goal_id:
            return None
        return await original_get(model, ident, *args, **kwargs)

    monkeypatch.setattr(db_session, "get", fake_get)

    with pytest.raises(HTTPException) as exc:
        await service.request_llm_decision(db_session, run.id, adapter=adapter)

    assert exc.value.status_code == 404
    assert exc.value.detail == "Orchestration goal not found"
    assert not adapter.contexts


@pytest.mark.asyncio
async def test_request_llm_decision_uses_committed_sessions_for_committed_run(committed_run, monkeypatch):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, session_factory, run_id = committed_run
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)

    async with session_factory() as caller_db:
        run = await caller_db.get(orchestration_service_module.OrchestrationRun, run_id)
        goal = await caller_db.get(orchestration_service_module.OrchestrationGoal, run.goal_id)
        goal.objective = "Caller-only objective"
        adapter = RecordingAdapter({"action_type": "noop", "reason": "Committed context should win."})

        decision = await service.request_llm_decision(caller_db, run_id, adapter=adapter)

    assert decision.run_id == run_id
    assert adapter.contexts[0]["goal"]["objective"] == "Choose the next coordination move"


@pytest.mark.asyncio
async def test_request_llm_decision_uses_caller_when_factory_has_another_engine(db_session, test_project, monkeypatch):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, run = await _make_run(db_session, test_project)

    class ForeignSession:
        def get_bind(self):
            return object()

        async def get(self, *_args, **_kwargs):
            raise AssertionError("foreign factory must not be queried")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", ForeignSession)
    decision = await service.request_llm_decision(
        db_session, run.id, adapter=RecordingAdapter({"action_type": "noop", "reason": "Caller owns run."}),
    )

    assert decision.run_id == run.id


@pytest.mark.asyncio
async def test_request_llm_decision_committed_path_returns_cached_accepted_decision_sequentially(
    committed_run,
    monkeypatch,
):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, session_factory, run_id = committed_run
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)

    class CountingAdapter:
        def __init__(self):
            self.calls = 0

        async def decide(self, context, project=None, goal=None):
            self.calls += 1
            return OrchestrationDecisionAdapterResult(
                input_snapshot=context,
                llm_output={"raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Cached"}})},
                parsed_decision={"action_type": "noop", "reason": "Cached"},
            )

    adapter = CountingAdapter()

    async with session_factory() as caller_db:
        first = await service.request_llm_decision(caller_db, run_id, adapter=adapter)
        second = await service.request_llm_decision(caller_db, run_id, adapter=adapter)

    assert adapter.calls == 1
    assert first.id == second.id
    assert first.decision_type == "noop"
    assert first.validator_status == "accepted"
    assert second.validator_status == "accepted"

    async with session_factory() as verify_db:
        result = await verify_db.execute(
            orchestration_service_module.select(orchestration_service_module.OrchestrationDecision).where(
                orchestration_service_module.OrchestrationDecision.run_id == run_id
            )
        )
        decisions = result.scalars().all()
        assert len(decisions) == 1


@pytest.mark.asyncio
async def test_request_llm_decision_committed_path_does_not_block_pause_midflight(committed_run, monkeypatch):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, session_factory, run_id = committed_run
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingAdapter:
        async def decide(self, context, project=None, goal=None):
            started.set()
            await release.wait()
            return OrchestrationDecisionAdapterResult(
                input_snapshot=context,
                llm_output={"raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Late"}})},
                parsed_decision={"action_type": "noop", "reason": "Late"},
            )

    async def pause_run():
        await started.wait()
        async with session_factory() as db:
            run = await db.get(orchestration_service_module.OrchestrationRun, run_id)
            run.status = "paused"
            await db.commit()

    async def request_once():
        async with session_factory() as caller_db:
            return await service.request_llm_decision(caller_db, run_id, adapter=BlockingAdapter())

    decision_task = asyncio.create_task(request_once())
    await started.wait()
    pause_task = asyncio.create_task(pause_run())
    await asyncio.sleep(0.05)
    assert pause_task.done()
    release.set()
    decision = await decision_task
    await pause_task

    assert decision.run_id == run_id
    assert decision.decision_type == "noop"

    async with session_factory() as verify_db:
        result = await verify_db.execute(
            orchestration_service_module.select(orchestration_service_module.OrchestrationDecision).where(
                orchestration_service_module.OrchestrationDecision.run_id == run_id
            )
        )
        decisions = result.scalars().all()
        assert len(decisions) == 1

        run = await verify_db.get(orchestration_service_module.OrchestrationRun, run_id)
        assert run.status == "paused"


@pytest.mark.asyncio
async def test_request_llm_decision_committed_path_rejects_terminal_run_before_write(committed_run, monkeypatch):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, session_factory, run_id = committed_run
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingAdapter:
        async def decide(self, context, project=None, goal=None):
            started.set()
            await release.wait()
            return OrchestrationDecisionAdapterResult(
                input_snapshot=context,
                llm_output={"raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Late"}})},
                parsed_decision={"action_type": "noop", "reason": "Late"},
            )

    async def complete_run():
        await started.wait()
        async with session_factory() as db:
            run = await db.get(orchestration_service_module.OrchestrationRun, run_id)
            run.status = "completed"
            await db.commit()

    async with session_factory() as caller_db:
        decision_task = asyncio.create_task(
            service.request_llm_decision(caller_db, run_id, adapter=BlockingAdapter())
        )
        await started.wait()
        await complete_run()
        release.set()

        with pytest.raises(HTTPException) as exc:
            await decision_task

    assert exc.value.status_code == 409
    assert exc.value.detail == "Orchestration run is completed"

    async with session_factory() as verify_db:
        result = await verify_db.execute(
            orchestration_service_module.select(orchestration_service_module.OrchestrationDecision).where(
                orchestration_service_module.OrchestrationDecision.run_id == run_id
            )
        )
        decisions = result.scalars().all()
        assert not decisions


@pytest.mark.asyncio
async def test_lock_run_for_committed_decision_requires_fresh_sqlite_session(committed_run, monkeypatch):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, session_factory, run_id = committed_run
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)

    async with session_factory() as db:
        await db.execute(orchestration_service_module.select(orchestration_service_module.OrchestrationRun.id))
        with pytest.raises(RuntimeError) as exc:
            await service._lock_run_for_committed_decision(db, run_id)

    assert str(exc.value) == (
        "_lock_run_for_committed_decision requires a fresh sqlite session; "
        "BEGIN IMMEDIATE must be the first statement"
    )


@pytest.mark.asyncio
async def test_request_llm_decision_committed_path_retries_after_rejected_duplicate_context(committed_run, monkeypatch):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, session_factory, run_id = committed_run
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)

    class FailThenSucceedAdapter:
        def __init__(self):
            self.calls = 0

        async def decide(self, context, project=None, goal=None):
            self.calls += 1
            if self.calls == 1:
                return OrchestrationDecisionAdapterResult(
                    input_snapshot=context,
                    llm_output={"raw_content": None, "completion_error": "RuntimeError: rate limited"},
                    parsed_decision={"action_type": "invalid_llm_output", "reason": "LLM completion failed"},
                )
            return OrchestrationDecisionAdapterResult(
                input_snapshot=context,
                llm_output={
                    "raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Recovered"}})
                },
                parsed_decision={"action_type": "noop", "reason": "Recovered"},
            )

    adapter = FailThenSucceedAdapter()

    async with session_factory() as caller_db:
        first = await service.request_llm_decision(caller_db, run_id, adapter=adapter)
        second = await service.request_llm_decision(caller_db, run_id, adapter=adapter)

    assert adapter.calls == 2
    assert first.id != second.id
    assert first.validator_status == "rejected"
    assert second.validator_status == "accepted"


@pytest.mark.asyncio
async def test_request_llm_decision_committed_path_deduplicates_concurrent_requests(committed_run, monkeypatch):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, session_factory, run_id = committed_run
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingAdapter:
        def __init__(self):
            self.calls = 0

        async def decide(self, context, project=None, goal=None):
            self.calls += 1
            if self.calls > 2:
                raise AssertionError("adapter called more than twice")
            started.set()
            await release.wait()
            return OrchestrationDecisionAdapterResult(
                input_snapshot=context,
                llm_output={"raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Once"}})},
                parsed_decision={"action_type": "noop", "reason": "Once"},
            )

    adapter = BlockingAdapter()

    async def request_once():
        async with session_factory() as db:
            return await service.request_llm_decision(db, run_id, adapter=adapter)

    first_task = asyncio.create_task(request_once())
    await started.wait()
    second_task = asyncio.create_task(request_once())
    await asyncio.sleep(0.05)
    release.set()

    first_decision, second_decision = await asyncio.gather(first_task, second_task)

    assert adapter.calls == 2
    assert first_decision.id == second_decision.id

    async with session_factory() as verify_db:
        result = await verify_db.execute(
            orchestration_service_module.select(orchestration_service_module.OrchestrationDecision).where(
                orchestration_service_module.OrchestrationDecision.run_id == run_id
            )
        )
        decisions = result.scalars().all()
        assert len(decisions) == 1
