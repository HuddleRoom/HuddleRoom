# pylint: disable=redefined-outer-name

import asyncio
import json
from types import SimpleNamespace
import uuid

from fastapi import HTTPException
import pytest
import pytest_asyncio
from sqlalchemy import delete, text
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
from huddleroom.services.orchestration_wake_when import clamp_recheck_seconds
from huddleroom.services.project_service import ProjectService
from tests.orchestration_wake_helpers import NOOP_WAKE_WHEN


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
    assert REASON_RULE in system_text
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
                                    "wake_when": {"recheck_after_seconds": 300, "expected_result": "New evidence."},
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
        {
            "decision": {
                "action_type": "noop",
                "reason": "Waiting for new evidence.",
                "wake_when": {"recheck_after_seconds": 300, "expected_result": "New evidence."},
            }
        }
    )
    assert result.parsed_decision == {
        "action_type": "noop",
        "reason": "Waiting for new evidence.",
        "wake_when": {"events": [], "recheck_after_seconds": clamp_recheck_seconds(300), "expected_result": "New evidence."},
    }
    assert calls[0]["model"] == "openai/test-model"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["temperature"] == 0
    assert calls[0]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_adapter_decide_accepts_fenced_json_but_retains_raw_diagnostics():
    raw_content = (
        ' \n```json\n{"decision":{"action_type":"noop","reason":"Wait.",'
        '"wake_when":{"recheck_after_seconds":300,"expected_result":"test wait"}}}\n```\n '
    )

    async def fake_completion(**_kwargs):
        return {"choices": [{"message": {"content": raw_content}}]}

    result = await OrchestrationDecisionAdapter(completion_fn=fake_completion).decide({"run": {"status": "running"}})

    assert result.llm_output["raw_content"] == raw_content
    assert result.parsed_decision == {
        "action_type": "noop",
        "reason": "Wait.",
        "wake_when": {"events": [], "recheck_after_seconds": clamp_recheck_seconds(300), "expected_result": "test wait"},
    }


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
                                    "wake_when": NOOP_WAKE_WHEN,
                                }
                            }
                        )
                    }
                }
            ]
        }

    adapter = OrchestrationDecisionAdapter(completion_fn=fake_completion)
    result = await adapter.decide({"goal": {"objective": "Wait"}, "run": {"status": "running"}})

    assert result.parsed_decision == {
        "action_type": "noop",
        "reason": "Mapping responses should work too.",
        "wake_when": {"events": [], "recheck_after_seconds": clamp_recheck_seconds(300), "expected_result": "test wait"},
    }


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


@pytest_asyncio.fixture(name="committed_run_details")
async def committed_run_details_fixture(test_engine, tmp_path):
    service, session_factory, project_id, goal_id, run_id = await _make_committed_run(test_engine, tmp_path)
    try:
        yield service, session_factory, project_id, goal_id, run_id
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
    adapter = RecordingAdapter({"action_type": "noop", "reason": "Only inspecting context.", "wake_when": NOOP_WAKE_WHEN})

    await service.request_llm_decision(db_session, run.id, adapter=adapter)

    assert "recent_events" not in adapter.contexts[0]


@pytest.mark.asyncio
async def test_request_llm_decision_allows_blocked_run(db_session, test_project):
    service, run = await _make_run(db_session, test_project)
    run.status = "blocked"
    await db_session.flush()
    adapter = RecordingAdapter({"action_type": "noop", "reason": "Blocked runs can still coordinate recovery.", "wake_when": NOOP_WAKE_WHEN})

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
            adapter=RecordingAdapter({"action_type": "noop", "reason": "Paused.", "wake_when": NOOP_WAKE_WHEN}),
        )

    assert exc.value.status_code == 409
    assert exc.value.detail == "Orchestration run is paused"


@pytest.mark.asyncio
async def test_request_llm_decision_fallback_404s_when_goal_missing(db_session, test_project, monkeypatch):
    service, run = await _make_run(db_session, test_project)
    adapter = RecordingAdapter({"action_type": "noop", "reason": "Should not reach adapter.", "wake_when": NOOP_WAKE_WHEN})

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
        adapter = RecordingAdapter({"action_type": "noop", "reason": "Committed context should win.", "wake_when": NOOP_WAKE_WHEN})

        decision = await service.request_llm_decision(caller_db, run_id, adapter=adapter)
        assert caller_db.is_modified(goal)

        async with session_factory() as verify_db:
            persisted_goal = await verify_db.get(orchestration_service_module.OrchestrationGoal, goal.id)
            assert persisted_goal.objective == "Choose the next coordination move"

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
        db_session, run.id, adapter=RecordingAdapter({"action_type": "noop", "reason": "Caller owns run.", "wake_when": NOOP_WAKE_WHEN}),
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
                llm_output={"raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Cached", "wake_when": NOOP_WAKE_WHEN}})},
                parsed_decision={"action_type": "noop", "reason": "Cached", "wake_when": NOOP_WAKE_WHEN},
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
                llm_output={"raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Late", "wake_when": NOOP_WAKE_WHEN}})},
                parsed_decision={"action_type": "noop", "reason": "Late", "wake_when": NOOP_WAKE_WHEN},
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
async def test_request_llm_decision_releases_sqlite_autobegin_workspace_lock_before_provider_call(
    committed_run_details,
    monkeypatch,
):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, session_factory, project_id, goal_id, run_id = committed_run_details
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)
    started = asyncio.Event()
    release = asyncio.Event()
    writer_error = None

    class BlockingAdapter:
        async def decide(self, context, project=None, goal=None):
            started.set()
            await release.wait()
            if writer_error is not None:
                raise writer_error
            return OrchestrationDecisionAdapterResult(
                input_snapshot=context,
                llm_output={"raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Done", "wake_when": NOOP_WAKE_WHEN}})},
                parsed_decision={"action_type": "noop", "reason": "Done", "wake_when": NOOP_WAKE_WHEN},
            )

    async def commit_other_writer():
        async with session_factory() as writer:
            await writer.execute(text("PRAGMA busy_timeout = 100"))
            project = await writer.get(Project, project_id)
            project.description = "writer committed while provider waited"
            await writer.commit()

    async def commit_while_provider_waits():
        nonlocal writer_error
        await started.wait()
        try:
            await commit_other_writer()
        except Exception as exc:  # Re-raise inside the provider so the pre-fix path exits cleanly.
            writer_error = exc
        finally:
            release.set()

    async with session_factory() as caller_db:
        async with service._lock_goal_for_baseline_transition(
            caller_db, goal_id, tick_owns_transaction=True
        ):
            await ProjectService().lock_workspace_boundary(caller_db, project_id)
            writer_task = asyncio.create_task(commit_while_provider_waits())
            try:
                decision = await service.request_llm_decision(caller_db, run_id, adapter=BlockingAdapter())
            finally:
                await writer_task

        assert "orchestration_tick_owns_transaction" not in caller_db.info

    assert decision.run_id == run_id
    assert decision.validator_status == "accepted"
    async with session_factory() as verify_db:
        decision_count = await verify_db.scalar(
            orchestration_service_module.select(orchestration_service_module.func.count())
            .select_from(orchestration_service_module.OrchestrationDecision)
            .where(orchestration_service_module.OrchestrationDecision.run_id == run_id)
        )
        project = await verify_db.get(Project, project_id)

    assert decision_count == 1
    assert project.description == "writer committed while provider waited"


@pytest.mark.asyncio
async def test_request_llm_decision_preserves_explicit_nested_caller_transaction(
    committed_run_details,
    monkeypatch,
):
    import huddleroom.services.orchestration_service as orchestration_service_module

    service, session_factory, _project_id, _goal_id, run_id = committed_run_details
    monkeypatch.setattr(orchestration_service_module, "AsyncSessionLocal", session_factory)
    adapter = RecordingAdapter({"action_type": "noop", "reason": "Caller owns this transaction.", "wake_when": NOOP_WAKE_WHEN})

    async with session_factory() as caller_db:
        async with caller_db.begin():
            async with caller_db.begin_nested():
                decision = await service.request_llm_decision(caller_db, run_id, adapter=adapter)
                assert caller_db.in_transaction()
                assert caller_db.in_nested_transaction()

    assert decision.run_id == run_id
    assert decision.validator_status == "accepted"


@pytest.mark.asyncio
async def test_goal_lock_restores_tick_transaction_scope_after_success_and_failure(db_session, test_project):
    service = OrchestrationService()
    key = "orchestration_tick_owns_transaction"
    db_session.info[key] = "outer"

    async with service._lock_goal_for_baseline_transition(
        db_session, uuid.uuid4(), tick_owns_transaction=True
    ):
        assert db_session.info[key] is True
    assert db_session.info[key] == "outer"

    with pytest.raises(RuntimeError, match="scope failure"):
        async with service._lock_goal_for_baseline_transition(
            db_session, uuid.uuid4(), tick_owns_transaction=True
        ):
            assert db_session.info[key] is True
            raise RuntimeError("scope failure")
    assert db_session.info[key] == "outer"
    db_session.info.pop(key)


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
                llm_output={"raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Late", "wake_when": NOOP_WAKE_WHEN}})},
                parsed_decision={"action_type": "noop", "reason": "Late", "wake_when": NOOP_WAKE_WHEN},
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
                    "raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Recovered", "wake_when": NOOP_WAKE_WHEN}})
                },
                parsed_decision={"action_type": "noop", "reason": "Recovered", "wake_when": NOOP_WAKE_WHEN},
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
                llm_output={"raw_content": json.dumps({"decision": {"action_type": "noop", "reason": "Once", "wake_when": NOOP_WAKE_WHEN}})},
                parsed_decision={"action_type": "noop", "reason": "Once", "wake_when": NOOP_WAKE_WHEN},
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


REASON_RULE = (
    "Top-level reason is required where the action schema lists it; otherwise it is optional but recommended. "
    "When present it states how the action advances or unblocks the goal."
)

# Action names that the dispatcher no longer executes (validator rejects them). None may appear in the prompt.
NON_DISPATCHABLE_ACTION_NAMES = (
    "request_human_decision",
    "request_manager_decision",
    "request_split",
    "complete_run",
    "request_final_summary",
    "expand_plan_item",
    "open_gate",
    "record_authority_decision",
    "cancel_pending_decision",
    "acknowledge_warning",
    "resolve_warning",
)


def _system_text(**kwargs) -> str:
    return build_orchestration_decision_messages({"run": {"phase": "authorized"}}, **kwargs)[0]["content"]


def test_prompt_includes_progress_contract_sections():
    from huddleroom.services.orchestration_llm_decision_adapter import PROGRESS_CONTRACT

    assert PROGRESS_CONTRACT in _system_text()
    for section in ("Progress duty", "Fresh run", "Run with history", "Authority", "Waiting", "Reason"):
        assert section in PROGRESS_CONTRACT
    assert "request_plan" in PROGRESS_CONTRACT
    assert "ask_human" in PROGRESS_CONTRACT
    assert "request_human_decision" not in PROGRESS_CONTRACT
    assert "progress_view" in PROGRESS_CONTRACT
    assert "untracked_follow_ups" in PROGRESS_CONTRACT


def test_progress_contract_routes_owner_decisions_to_ask_human():
    from huddleroom.services.orchestration_llm_decision_adapter import PROGRESS_CONTRACT

    assert "Use ask_human only when a missing owner decision truly blocks starting" in PROGRESS_CONTRACT
    assert "Route those actions to ask_human" in PROGRESS_CONTRACT


def test_progress_contract_follow_up_wording_is_ranked_and_non_repeating():
    from huddleroom.services.orchestration_llm_decision_adapter import PROGRESS_CONTRACT

    assert "Handle the highest-impact actionable follow-up; the list is already ranked." in PROGRESS_CONTRACT
    assert "A blocked item names its dependency" in PROGRESS_CONTRACT
    assert "defer it and continue independent work" in PROGRESS_CONTRACT
    assert "After an unsuccessful wake, your diagnosis or action must differ from last time." in PROGRESS_CONTRACT
    assert "Handle the first untracked follow-up" not in PROGRESS_CONTRACT
    assert "progress_view" in PROGRESS_CONTRACT
    assert "untracked_follow_ups" in PROGRESS_CONTRACT


def test_prompt_reason_rule_is_single_and_agrees_in_contract_and_shape():
    from huddleroom.services.orchestration_llm_decision_adapter import PROGRESS_CONTRACT

    system_text = _system_text()
    assert REASON_RULE in PROGRESS_CONTRACT
    assert REASON_RULE in system_text
    assert "Top-level reason is optional for every action" not in system_text
    assert "the reason states how the action advances the goal" not in system_text


def test_prompt_states_authority_rule_and_keeps_specific_restrictions():
    from huddleroom.services.orchestration_llm_decision_adapter import PROGRESS_CONTRACT

    system_text = _system_text()
    assert 'run.phase == "authorized"' in system_text
    assert "prepare only" in system_text
    for restriction in ("live sends", "spending", "purchases", "account changes", "publishing", "destructive"):
        assert restriction in PROGRESS_CONTRACT


def test_prompt_has_neutral_example_not_noop_only():
    system_text = _system_text()
    assert '{"decision":{"action_type":"noop","reason":"short coordination reason"}}' not in system_text
    assert '"action_type":"<allowed type>"' in system_text
    assert '"reason":"how this advances the goal"' in system_text


def test_prompt_documents_wake_when_and_event_matcher_table():
    system_text = _system_text()
    assert "wake_when" in system_text
    assert "recheck_after_seconds" in system_text
    assert "task.status_changed" in system_text
    assert "orchestration.steering_changed" not in system_text


def test_prompt_starts_with_preamble():
    from huddleroom.services.orchestration_llm_decision_adapter import orchestrator_preamble

    assert _system_text().startswith(orchestrator_preamble())


def test_progress_contract_is_importable_constant():
    from huddleroom.services.orchestration_llm_decision_adapter import PROGRESS_CONTRACT

    assert isinstance(PROGRESS_CONTRACT, str)
    assert PROGRESS_CONTRACT.strip()


def test_decision_prompt_instructs_applies_decision_id_for_follow_ups():
    from huddleroom.services.orchestration_llm_decision_adapter import (
        DECISION_FOLLOW_UP_CONTRACT,
        PROGRESS_CONTRACT,
    )

    system_text = _system_text()
    assert DECISION_FOLLOW_UP_CONTRACT in system_text
    assert "set top-level applies_decision_id" in system_text
    assert "applies_decision_id" not in PROGRESS_CONTRACT
    assert (
        system_text.index(PROGRESS_CONTRACT)
        < system_text.index(DECISION_FOLLOW_UP_CONTRACT)
        < system_text.index("Return exactly one JSON object")
    )


def test_prompt_tells_model_to_tag_inputs():
    from huddleroom.services.orchestration_llm_decision_adapter import PROGRESS_CONTRACT

    system_text = _system_text()
    assert PROGRESS_CONTRACT in system_text
    assert "criterion:<key>" in PROGRESS_CONTRACT
    assert "meeting_action_item:<id>" in PROGRESS_CONTRACT


def test_prompt_section_order_restriction_contract_shape_wake_schemas():
    from huddleroom.services.orchestration_llm_decision_adapter import PROGRESS_CONTRACT

    system_text = _system_text()
    first_line = PROGRESS_CONTRACT.splitlines()[0]
    assert (
        system_text.index("You must not write plans")
        < system_text.index(first_line)
        < system_text.index("Return exactly one JSON object")
        < system_text.index("A wait requires wake_when")
        < system_text.index("Allowed action schemas:")
    )


def test_prompt_mentions_no_non_dispatchable_action_names():
    from huddleroom.services.orchestration_decision_validator import ALLOWED_ACTION_SCHEMAS

    system_text = _system_text()
    for name in NON_DISPATCHABLE_ACTION_NAMES:
        assert name not in ALLOWED_ACTION_SCHEMAS
        assert name not in system_text, f"prompt advertises non-dispatchable action {name}"


def test_prompt_action_name_mentions_are_all_dispatchable():
    import re

    from huddleroom.services.orchestration_decision_validator import ALLOWED_ACTION_SCHEMAS

    system_text = _system_text()
    verb_prefixed = re.compile(
        r"\b(?:request|complete|open|expand|record|cancel|acknowledge|resolve|ask|start|pause|suggest|"
        r"create|retry|reassign|schedule|accept|noop)_[a-z_]+\b"
    )
    mentioned = set(verb_prefixed.findall(system_text))
    unknown = mentioned - set(ALLOWED_ACTION_SCHEMAS) - {"noop"}
    assert not unknown, f"prompt names actions outside ALLOWED_ACTION_SCHEMAS: {sorted(unknown)}"


def _completion_returning(*decisions):
    calls = []
    queue = list(decisions)

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        content = queue.pop(0) if len(queue) > 1 else queue[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return fake_completion, calls


@pytest.mark.asyncio
async def test_adapter_noop_without_wake_when_is_repaired():
    fake_completion, calls = _completion_returning(
        json.dumps({"decision": {"action_type": "noop", "reason": "Waiting."}}),
        json.dumps({"decision": {"action_type": "noop", "reason": "Waiting.", "wake_when": NOOP_WAKE_WHEN}}),
    )
    adapter = OrchestrationDecisionAdapter(model="openai/test-model", completion_fn=fake_completion)

    result = await adapter.decide({"goal": {"objective": "Wait"}})

    assert len(calls) == 2
    assert result.parsed_decision["action_type"] == "noop"
    assert result.parsed_decision["wake_when"]["expected_result"] == "test wait"


@pytest.mark.asyncio
async def test_adapter_noop_exhausts_repairs_to_invalid_llm_output():
    fake_completion, calls = _completion_returning(
        json.dumps({"decision": {"action_type": "noop", "reason": "Waiting."}}),
    )
    adapter = OrchestrationDecisionAdapter(model="openai/test-model", completion_fn=fake_completion)

    result = await adapter.decide({"goal": {"objective": "Wait"}})

    assert len(calls) == 3
    assert result.parsed_decision["action_type"] == "invalid_llm_output"
    assert "wake_when" in result.parsed_decision["reason"]


@pytest.mark.asyncio
async def test_adapter_repairs_unknown_top_level_key_with_validator_message():
    good = {"action_type": "ask_human", "question": "Choose scope.", "reason": "Need input."}
    fake_completion, calls = _completion_returning(
        json.dumps({"decision": {**good, "bogus_key": 1}}),
        json.dumps({"decision": good}),
    )
    adapter = OrchestrationDecisionAdapter(model="openai/test-model", completion_fn=fake_completion)

    result = await adapter.decide({"goal": {"objective": "Wait"}})

    assert len(calls) == 2
    assert "includes unknown top-level key: bogus_key" in json.dumps(calls[1]["messages"])
    assert result.parsed_decision == good


@pytest.mark.asyncio
async def test_adapter_repairs_wake_when_on_non_noop_with_validator_message():
    good = {"action_type": "ask_human", "question": "Choose scope.", "reason": "Need input."}
    fake_completion, calls = _completion_returning(
        json.dumps({"decision": {**good, "wake_when": NOOP_WAKE_WHEN}}),
        json.dumps({"decision": good}),
    )
    adapter = OrchestrationDecisionAdapter(model="openai/test-model", completion_fn=fake_completion)

    result = await adapter.decide({"goal": {"objective": "Wait"}})

    assert len(calls) == 2
    assert "includes unknown top-level key: wake_when" in json.dumps(calls[1]["messages"])
    assert result.parsed_decision == good


@pytest.mark.asyncio
async def test_adapter_clamps_wake_when_recheck():
    fake_completion, _calls = _completion_returning(
        json.dumps({"decision": {"action_type": "noop", "reason": "Waiting.", "wake_when": {
            "recheck_after_seconds": 10 ** 9, "expected_result": "Later."}}}),
    )
    adapter = OrchestrationDecisionAdapter(model="openai/test-model", completion_fn=fake_completion)

    result = await adapter.decide({"goal": {"objective": "Wait"}})

    assert result.parsed_decision["wake_when"]["recheck_after_seconds"] == clamp_recheck_seconds(10 ** 9)


@pytest.mark.asyncio
async def test_adapter_leaves_non_noop_decisions_untouched():
    decision = {"action_type": "ask_human", "question": "Choose scope.", "reason": "Need input."}
    fake_completion, _calls = _completion_returning(json.dumps({"decision": decision}))
    adapter = OrchestrationDecisionAdapter(model="openai/test-model", completion_fn=fake_completion)

    result = await adapter.decide({"goal": {"objective": "Wait"}})

    assert result.parsed_decision == decision


def test_prompt_schema_lists_wake_when_as_noop_required():
    system_text = _system_text()
    schemas = json.loads(system_text.split("Allowed action schemas: ", 1)[1])

    assert schemas["noop"]["required"] == ["wake_when"]
