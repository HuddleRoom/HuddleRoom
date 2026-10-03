"""Public conversation endpoint contract."""

# pylint: disable=redefined-outer-name

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

import huddleroom.routers.orchestration_goals as goals_router
from huddleroom.models.base import _utcnow
from huddleroom.models.orchestration_conversation import (
    ConversationInvestigation,
    ConversationInvestigationReservation,
    ConversationMessage,
    ConversationReservation,
    ConversationResponse,
    conversation_investigation_reservation_id,
    conversation_investigation_provider_request_id,
    conversation_reservation_id,
    conversation_response_id,
)
from huddleroom.models.project import Project
from huddleroom.services.orchestration_conversation_service import (
    OrchestrationConversationService,
)
from huddleroom.services.orchestration_conversation_dossier import DossierBuild
from huddleroom.services.orchestration_conversation_investigation import InvestigationInput
from huddleroom.services.orchestration_service import OrchestrationService
from huddleroom.database import get_db
from huddleroom.main import create_app


ANON_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


def _contains_forbidden(value, forbidden, values=()):
    if isinstance(value, dict):
        return any(key in forbidden or _contains_forbidden(item, forbidden, values) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden(item, forbidden, values) for item in value)
    return value in values


def _utc_timestamp(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.fixture(autouse=True)
def conversation_service(test_engine, monkeypatch):
    async def completion(**_kwargs):
        return {
            "choices": [{"message": {"content": "Safe answer"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }

    monkeypatch.setattr(
        goals_router,
        "conversation_service",
        OrchestrationConversationService(
            async_sessionmaker(test_engine, expire_on_commit=False),
            completion,
            orchestration_service=OrchestrationService(),
        ),
    )
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_allowance_tokens",
        1_000,
    )


@pytest.fixture
async def conversation_client(test_engine):
    app = create_app()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)

    async def override_get_db():
        async with factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_submit_trims_outer_whitespace_and_returns_only_safe_turn(
    conversation_client, conversation_goal_run, db_session, monkeypatch, test_engine, tmp_path
):
    goal, run = conversation_goal_run
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = await db_session.get(Project, goal.project_id)
    project.workspace_path = str(workspace)
    await db_session.commit()
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_allowance_tokens",
        50_000,
    )
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_investigation_enabled",
        True,
    )

    async def completion(**kwargs):
        if "tools" in kwargs:
            return {
                "choices": [{"message": {"content": None, "tool_calls": [{"function": {
                    "name": "request_investigation",
                    "arguments": json.dumps({
                        "objective": "Check release risk",
                        "requests": [{"operation": "read", "path": "risk.txt", "query": None}],
                    }),
                }}]}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            }
        return {
            "choices": [{"message": {"content": json.dumps({
                "findings": "Gate is pending.",
                "uncertainty": "No validator output.",
                "sources": ["risk.txt#L1-L1"],
            })}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }

    service = OrchestrationConversationService(
        async_sessionmaker(test_engine, expire_on_commit=False), completion,
        orchestration_service=OrchestrationService(),
    )

    def collect(_workspace, _request):
        identity = workspace.stat()
        return InvestigationInput(
            scope=({"path": "PRIVATE_SCOPE_SENTINEL"},),
            sources=({
                "reference": "risk.txt#L1-L1", "operation": "read", "status": "included",
                "freshness_at": "2026-09-14T08:00:00+00:00", "truncated": False,
                "excerpt": "PRIVATE_EXCERPT_SENTINEL", "workspace_root": "PRIVATE_ROOT_SENTINEL",
                "provider_identity": "PRIVATE_PROVIDER_SENTINEL", "reservation": "PRIVATE_RESERVATION_SENTINEL",
            },),
            omissions=({
                "reference": "[restricted source]", "operation": "read", "status": "restricted",
                "freshness_at": None, "truncated": False, "excerpt": "PRIVATE_OMISSION_SENTINEL",
            },),
            root_identity=(identity.st_dev, identity.st_ino),
        )

    monkeypatch.setattr(service._investigations._reader, "collect", collect)
    monkeypatch.setattr(goals_router, "conversation_service", service)

    async def build_with_private_values(_self, _goal, _run, _question, _prior_turns):
        return DossierBuild(
            dossier={"dossier": "POST_DOSSIER_SENTINEL", "raw_prompt": "POST_PROMPT_SENTINEL"},
            manifest={
                "run_id": str(run.id), "excluded_categories": ["provider"], "truncated": False,
                "sources": [{
                    "source": "goal", "status": "included", "freshness_at": "2026-09-14T00:00:00+00:00",
                    "available": 1, "included": 1, "omitted": 0, "truncated": False,
                    "references": [str(goal.id)], "provider_payload": "POST_PROVIDER_SENTINEL",
                    "raw_provider_error": "POST_ERROR_SENTINEL",
                    "reservation": "POST_RESERVATION_SENTINEL",
                    "excluded_source_data": "POST_EXCLUDED_SENTINEL",
                }],
                "raw_prompt": "POST_PROMPT_SENTINEL", "provider_payload": "POST_PROVIDER_SENTINEL",
                "raw_provider_error": "POST_ERROR_SENTINEL", "reservation": "POST_RESERVATION_SENTINEL",
                "excluded_source_data": "POST_EXCLUDED_SENTINEL",
            },
            context_version="v1", run_id=run.id,
            provider_messages=[{"role": "system", "content": "safe"}, {"role": "user", "content": "safe"}],
        )

    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.ConversationDossierBuilder.build",
        build_with_private_values,
    )

    response = await conversation_client.post(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation",
        json={"client_request_id": str(uuid.uuid4()), "content": "  hello\n  world  "},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "hello\n  world"
    assert body["answer"] == "Gate is pending."
    async with async_sessionmaker(test_engine, expire_on_commit=False)() as db:
        investigation = await db.scalar(select(ConversationInvestigation))
    assert body["investigation"] == {
        "investigation_id": str(investigation.id),
        "status": "completed",
        "objective": "Check release risk",
        "attempt_count": 1,
        "repair_count": 0,
        "retry_count": 0,
        "sources": [
            {
                "reference": "risk.txt#L1-L1", "operation": "read", "status": "included",
                "freshness_at": "2026-09-14T08:00:00+00:00", "truncated": False,
            },
            {
                "reference": "[restricted source]", "operation": "read", "status": "restricted",
                "freshness_at": None, "truncated": False,
            },
        ],
        "report": {
            "findings": "Gate is pending.", "uncertainty": "No validator output.",
            "sources": ["risk.txt#L1-L1"],
        },
        "error": None,
        "started_at": _utc_timestamp(investigation.started_at),
        "deadline_at": _utc_timestamp(investigation.deadline_at),
        "finished_at": _utc_timestamp(investigation.finished_at),
        "created_at": _utc_timestamp(investigation.created_at),
        "updated_at": _utc_timestamp(investigation.updated_at),
    }
    assert not _contains_forbidden(
        body,
        {
            "dossier", "raw_prompt", "provider_payload", "provider_request_id",
            "raw_provider_error", "reservation", "excluded_source_data", "input_manifest",
            "excerpt", "workspace_root", "root_identity", "provider_identity", "provider_request_id",
            "raw_provider_result", "raw_provider_error",
        },
        {
            "POST_DOSSIER_SENTINEL", "POST_PROMPT_SENTINEL", "POST_PROVIDER_SENTINEL",
            "POST_ERROR_SENTINEL", "POST_RESERVATION_SENTINEL", "POST_EXCLUDED_SENTINEL",
            "PRIVATE_SCOPE_SENTINEL", "PRIVATE_EXCERPT_SENTINEL", "PRIVATE_ROOT_SENTINEL",
            "PRIVATE_PROVIDER_SENTINEL", "PRIVATE_RESERVATION_SENTINEL", "PRIVATE_OMISSION_SENTINEL",
        },
    )
    assert body["context_manifest"]["sources"][0]["source"] == "goal"


@pytest.mark.parametrize("content", ["  \n\t ", "x" * 4_001])
async def test_submit_rejects_blank_and_overlong_content(
    conversation_client, conversation_goal_run, db_session, content
):
    goal, _ = conversation_goal_run
    await db_session.commit()

    response = await conversation_client.post(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation",
        json={"client_request_id": str(uuid.uuid4()), "content": content},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "conversation_invalid_content"


async def test_invalid_body_is_rejected_before_recovery(
    conversation_client, conversation_goal_run, db_session, monkeypatch
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    recover = AsyncMock()
    monkeypatch.setattr(goals_router.conversation_service, "recover_goal", recover)

    response = await conversation_client.post(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation",
        json={"client_request_id": str(uuid.uuid4()), "content": "  \n\t "},
    )

    assert response.status_code == 422
    recover.assert_not_awaited()


async def test_submit_rejects_an_investigation_request_field_before_recovery(
    conversation_client, conversation_goal_run, db_session, monkeypatch
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    recover = AsyncMock()
    monkeypatch.setattr(goals_router.conversation_service, "recover_goal", recover)

    response = await conversation_client.post(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation",
        json={
            "client_request_id": str(uuid.uuid4()), "content": "question",
            "investigation": {"objective": "must not widen the request"},
        },
    )

    assert response.status_code == 422
    recover.assert_not_awaited()


async def test_history_returns_latest_fifty_in_order_and_current_actor_allowance(
    conversation_client, conversation_goal_run, test_user, db_session
):
    goal, run = conversation_goal_run
    now = _utcnow()
    response_ids = []
    for sequence in range(1, 53):
        message = ConversationMessage(
            id=uuid.uuid4(),
            goal_id=goal.id,
            actor_id=test_user.id if sequence % 2 else ANON_USER_ID,
            client_request_id=uuid.uuid4(),
            sequence=sequence,
            content=f"question {sequence}",
        )
        response = ConversationResponse(
            id=conversation_response_id(message.id),
            message_id=message.id,
            run_id=run.id,
            status="completed",
            dossier={"raw_prompt": "never serialize", "provider_payload": {"secret": "x"}},
            context_manifest={
                "run_id": str(run.id),
                "excluded_categories": ["provider"],
                "sources": [{
                    "source": "goal", "status": "included", "freshness_at": now.isoformat(),
                    "available": 1, "included": 1, "omitted": 0, "truncated": False,
                    "references": [str(goal.id)], "raw_provider_error": "never serialize",
                }],
                "reservation": {"secret": "never serialize"},
            },
            context_version="v1",
            provider_request_id=f"private-{sequence}",
            answer=f"answer {sequence}",
            started_at=now - timedelta(seconds=2),
            deadline_at=now - timedelta(seconds=1),
            finished_at=now,
        )
        response_ids.append(response.id)
        db_session.add_all((message, response))
    await db_session.flush()
    for response_id, actor_id, used in (
        (response_ids[0], ANON_USER_ID, 100),
        (response_ids[1], test_user.id, 900),
    ):
        db_session.add(
            ConversationReservation(
                id=conversation_reservation_id(response_id),
                response_id=response_id,
                goal_id=goal.id,
                actor_id=actor_id,
                ceiling_snapshot=1_000,
                reserved_tokens=used,
                settled_tokens=used,
                status="settled",
                committed_at=now,
                settled_at=now,
                released_at=now,
            )
        )
    await db_session.commit()

    response = await conversation_client.get(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation"
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["total"], body["omitted"]) == (52, 2)
    assert [item["sequence"] for item in body["items"]] == list(range(3, 53))
    assert {item["actor_id"] for item in body["items"]} == {
        str(test_user.id), str(ANON_USER_ID)
    }
    assert all(item["feedback"] is None for item in body["items"])
    assert [item["feedback_eligible"] for item in body["items"]] == [
        sequence % 2 == 0 for sequence in range(3, 53)
    ]
    assert not _contains_forbidden(
        body,
        {"dossier", "raw_prompt", "provider_payload", "provider_request_id", "raw_provider_error", "reservation"},
    )
    assert body["items"][0]["context_manifest"]["sources"][0]["source"] == "goal"
    assert all(item["investigation"] is None for item in body["items"])
    assert body["allowance"] == {"enabled": True, "limit": 1_000, "used": 100, "remaining": 900}


async def test_history_projects_every_investigation_status_with_only_safe_manifest_metadata(
    conversation_client, conversation_goal_run, db_session, monkeypatch
):
    """Hostile schema-valid projection fixture only; lifecycle/accounting reachability is covered separately."""
    goal, run = conversation_goal_run
    now = _utcnow()
    source_statuses = {
        "pending": ("included", "risk.txt#L1-L1"),
        "running": ("restricted", "[restricted source]"),
        "completed": ("included", "risk.txt#L1-L1"),
        "limited": ("unsafe", "[unsafe path]"),
        "failed": ("binary", "binary.dat"),
        "cancelled": ("changed", "risk.txt"),
        "unavailable": ("too_large", "large.txt"),
        "interrupted_unknown": ("omitted_by_limit", "[additional sources]"),
    }
    errors = {
        "pending": None, "running": None, "completed": None,
        "limited": "conversation_allowance_exhausted", "failed": "invalid_investigation_report",
        "cancelled": "request_cancelled", "unavailable": "workspace_unavailable",
        "interrupted_unknown": "provider_outcome_unknown",
    }
    response_errors = {**errors, "cancelled": "investigation_cancelled"}
    investigations = []
    timestamps = {}
    for sequence, (status, (source_status, reference)) in enumerate(source_statuses.items(), start=1):
        occurred_at = now + timedelta(seconds=sequence)
        active = status in {"pending", "running"}
        started = occurred_at if status in {"running", "completed", "interrupted_unknown"} else None
        deadline = occurred_at + timedelta(seconds=30) if started else None
        finished = None if active else occurred_at
        timestamps[status] = {
            "started_at": started,
            "deadline_at": deadline,
            "finished_at": finished,
            "created_at": occurred_at,
            "updated_at": occurred_at,
        }
        message = ConversationMessage(
            id=uuid.uuid4(), goal_id=goal.id, actor_id=ANON_USER_ID,
            client_request_id=uuid.uuid4(), sequence=sequence, content=status,
        )
        response = ConversationResponse(
            id=conversation_response_id(message.id), message_id=message.id, run_id=run.id,
            status=("pending" if status == "pending" else "running" if status == "running" else "completed" if status == "completed" else
                    "interrupted_unknown" if status == "interrupted_unknown" else "failed"),
            dossier={}, context_manifest={}, context_version=f"status-{status}",
            provider_request_id=f"response-{status}", started_at=started, deadline_at=deadline,
            finished_at=finished, error=({"code": response_errors[status]} if response_errors[status] else None),
        )
        source = {
            "reference": reference, "operation": "read", "status": source_status,
            "freshness_at": "2026-09-14T08:00:00+00:00", "truncated": source_status == "included",
            "excerpt": "PRIVATE_EXCERPT_SENTINEL", "workspace_root": "PRIVATE_ROOT_SENTINEL",
        }
        investigation = ConversationInvestigation(
            id=uuid.uuid4(), response_id=response.id, goal_id=goal.id, actor_id=ANON_USER_ID,
            context_version=response.context_version, status=status, objective=f"Inspect {status}",
            scope=[{"path": "PRIVATE_SCOPE_SENTINEL"}],
            input_manifest={
                "sources": [source] if source_status == "included" else [],
                "omissions": [] if source_status == "included" else [source],
                "root_identity": [1, 2], "provider_identity": "PRIVATE_PROVIDER_SENTINEL",
                "raw_provider_result": "PRIVATE_REPORT_SENTINEL",
                "raw_provider_error": "PRIVATE_ERROR_SENTINEL",
            },
            provider_identity=f"provider-{status}",
            provider_request_id=f"request-{status}" if started else None,
            attempt_count=1 if started else 0,
            report=(
                {
                    "findings": "Gate is pending.", "uncertainty": "No validator output.",
                    "sources": [reference],
                }
                if status == "completed" else None
            ),
            error={"code": errors[status]} if errors[status] else None,
            started_at=started, deadline_at=deadline, finished_at=finished,
            cancelled_at=occurred_at if status == "cancelled" else None,
            created_at=occurred_at, updated_at=occurred_at,
        )
        db_session.add_all((message, response))
        investigations.append(investigation)
    await db_session.flush()
    db_session.add_all(investigations)
    await db_session.commit()
    monkeypatch.setattr(goals_router.conversation_service, "recover_goal", AsyncMock())

    response = await conversation_client.get(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation"
    )

    assert response.status_code == 200
    rows = {item["investigation"]["status"]: item for item in response.json()["items"]}
    assert set(rows) == set(source_statuses)
    for status, (source_status, reference) in source_statuses.items():
        turn = rows[status]
        investigation = turn["investigation"]
        assert investigation["sources"] == [{
            "reference": reference, "operation": "read", "status": source_status,
            "freshness_at": "2026-09-14T08:00:00+00:00", "truncated": source_status == "included",
        }]
        assert investigation["error"] == (
            {"code": errors[status]} if errors[status] else None
        )
        assert investigation["report"] == (
            {
                "findings": "Gate is pending.", "uncertainty": "No validator output.",
                "sources": [reference],
            }
            if status == "completed" else None
        )
        for field, value in timestamps[status].items():
            assert investigation[field] == _utc_timestamp(value)
        assert turn["error"] == (
            {"code": response_errors[status]} if response_errors[status] else None
        )
    assert not _contains_forbidden(
        response.json(),
        {
            "input_manifest", "excerpt", "workspace_root", "root_identity", "provider_identity",
            "provider_request_id", "raw_provider_result", "raw_provider_error", "reservation",
        },
        {
            "PRIVATE_EXCERPT_SENTINEL", "PRIVATE_ROOT_SENTINEL", "PRIVATE_SCOPE_SENTINEL",
            "PRIVATE_PROVIDER_SENTINEL", "PRIVATE_REPORT_SENTINEL", "PRIVATE_ERROR_SENTINEL",
        },
    )


async def test_history_allowance_combines_current_actor_reservations_from_both_tables(
    conversation_client, conversation_goal_run, test_user, db_session, monkeypatch
):
    """Breaks if either reservation table mischarges a current actor lifecycle state."""
    goal, run = conversation_goal_run
    now = _utcnow()
    chat_states = {
        "reserved": (11, 0, 0),
        "committed": (23, 0, 0),
        "held_unknown": (31, 0, 0),
        "settled": (59, 17, 42),
        "released": (43, 0, 43),
    }
    investigation_states = {
        "reserved": (7, 0, 0),
        "committed": (19, 0, 0),
        "held_unknown": (29, 0, 0),
        "settled": (47, 13, 34),
        "released": (37, 0, 37),
    }
    state_names = tuple(chat_states)

    def response_for(actor_id, sequence, suffix, reservation_status):
        response_status = {
            "reserved": "pending", "committed": "running", "held_unknown": "interrupted_unknown",
            "settled": "completed", "released": "failed",
        }[reservation_status]
        active = response_status == "running"
        started = now if response_status in {"running", "completed", "interrupted_unknown"} else None
        deadline = now + timedelta(seconds=30) if started else None
        message = ConversationMessage(
            id=uuid.uuid4(), goal_id=goal.id, actor_id=actor_id,
            client_request_id=uuid.uuid4(), sequence=sequence, content=suffix,
        )
        response = ConversationResponse(
            id=conversation_response_id(message.id), message_id=message.id, run_id=run.id,
            status=response_status, dossier={}, context_manifest={}, context_version=suffix,
            provider_request_id=f"response-{suffix}", started_at=started, deadline_at=deadline,
            finished_at=None if active or response_status == "pending" else now,
        )
        return message, response

    def investigation_for(response, actor_id, state):
        status = {
            "reserved": "pending", "committed": "running", "held_unknown": "interrupted_unknown",
            "settled": "completed", "released": "cancelled",
        }[state]
        started = now if status in {"running", "completed", "interrupted_unknown"} else None
        return ConversationInvestigation(
            id=uuid.uuid4(), response_id=response.id, goal_id=goal.id, actor_id=actor_id,
            context_version=response.context_version, status=status, objective=state, scope=[],
            input_manifest={"sources": [], "omissions": []}, provider_identity=f"provider-{response.context_version}",
            provider_request_id=f"request-{response.context_version}" if started else None,
            attempt_count=1 if started else 0,
            report={"findings": "done", "uncertainty": "", "sources": []} if status == "completed" else None,
            error={"code": "request_cancelled"} if status == "cancelled" else None,
            started_at=started, deadline_at=(now + timedelta(seconds=30) if started else None),
            finished_at=now if status not in {"pending", "running"} else None,
            cancelled_at=now if status == "cancelled" else None,
        )

    def reservation(model, parent, actor_id, state):
        reserved, settled, released = (
            chat_states if model is ConversationReservation else investigation_states
        )[state]
        fields = {
            "id": (conversation_reservation_id(parent.id) if model is ConversationReservation
                   else conversation_investigation_reservation_id(parent.id)),
            "goal_id": goal.id, "actor_id": actor_id, "ceiling_snapshot": 100,
            "reserved_tokens": reserved, "settled_tokens": settled, "released_tokens": released,
            "status": state, "committed_at": now if state in {"committed", "settled", "held_unknown"} else None,
            "settled_at": now if state == "settled" else None,
            "released_at": now if state in {"settled", "released"} else None,
        }
        fields["response_id" if model is ConversationReservation else "investigation_id"] = parent.id
        return model(**fields)

    chat_rows = [response_for(ANON_USER_ID, index, f"chat-{state}", state)
                 for index, state in enumerate(state_names, start=1)]
    investigation_rows = [response_for(
        ANON_USER_ID, index + len(state_names), f"investigation-{state}",
        "committed" if state == "reserved" else state,
    )
                          for index, state in enumerate(state_names, start=1)]
    other_chat = response_for(test_user.id, 11, "other-chat", "held_unknown")
    other_investigation_response = response_for(test_user.id, 12, "other-investigation", "committed")
    db_session.add_all([item for pair in (*chat_rows, *investigation_rows, other_chat, other_investigation_response) for item in pair])
    await db_session.flush()
    investigations = [investigation_for(response, ANON_USER_ID, state)
                      for (_, response), state in zip(investigation_rows, state_names)]
    other_investigation = investigation_for(other_investigation_response[1], test_user.id, "committed")
    db_session.add_all((*investigations, other_investigation))
    await db_session.flush()
    db_session.add_all([
        *(reservation(ConversationReservation, response, ANON_USER_ID, state)
          for (_, response), state in zip(chat_rows, state_names)),
        *(reservation(ConversationInvestigationReservation, investigation, ANON_USER_ID, state)
          for investigation, state in zip(investigations, state_names)),
        reservation(ConversationReservation, other_chat[1], test_user.id, "held_unknown"),
        reservation(ConversationInvestigationReservation, other_investigation, test_user.id, "committed"),
    ])
    await db_session.commit()
    monkeypatch.setattr(goals_router.conversation_service, "recover_goal", AsyncMock())
    monkeypatch.setattr(
        "huddleroom.routers.orchestration_goals.settings.orchestration_conversation_allowance_tokens", 100
    )

    response = await conversation_client.get(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation"
    )

    assert response.status_code == 200
    assert response.json()["allowance"] == {
        "enabled": True, "limit": 100, "used": 150, "remaining": 0,
    }


def test_openapi_keeps_conversation_as_the_only_investigation_surface():
    openapi = create_app().openapi()
    paths = openapi["paths"]
    conversation_path = "/api/v1/projects/{project_id}/orchestration/goals/{goal_id}/conversation"
    operations = paths[conversation_path]
    assert set(operations) == {"get", "post"}
    for operation in operations.values():
        assert {
            (parameter["name"], parameter["in"], parameter["required"])
            for parameter in operation["parameters"]
        } == {("project_id", "path", True), ("goal_id", "path", True)}
        assert not any(parameter["in"] == "query" for parameter in operation["parameters"])
    assert "requestBody" not in operations["get"]
    assert operations["post"]["requestBody"] == {
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/OrchestrationConversationSubmitRequest"}
            }
        },
        "required": True,
    }
    submit = openapi["components"]["schemas"]["OrchestrationConversationSubmitRequest"]
    assert set(submit["properties"]) == {"client_request_id", "content"}
    assert set(submit["required"]) == {"client_request_id", "content"}
    assert "investigation" not in submit["properties"]
    assert not any("investigation" in path for path in paths)


@pytest.mark.parametrize(
    ("allowance", "content", "status", "code"),
    [
        (0, "question", 409, "conversation_disabled"),
        (1, "question", 429, "conversation_exhausted"),
    ],
)
async def test_submit_maps_allowance_errors_to_stable_detail(
    conversation_client, conversation_goal_run, db_session, monkeypatch, allowance, content, status, code
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_allowance_tokens",
        allowance,
    )

    response = await conversation_client.post(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation",
        json={"client_request_id": str(uuid.uuid4()), "content": content},
    )

    assert response.status_code == status
    assert response.json()["detail"]["code"] == code


async def test_submit_closes_request_session_before_provider_dispatch(
    test_engine, conversation_goal_run, db_session, monkeypatch
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_allowance_tokens",
        10_000,
    )
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    request_session = None
    observed_transaction = None

    async def completion(**_kwargs):
        nonlocal observed_transaction
        observed_transaction = request_session.in_transaction()
        return {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(
        goals_router,
        "conversation_service",
        OrchestrationConversationService(factory, completion, orchestration_service=OrchestrationService()),
    )
    app = create_app()

    async def override_get_db():
        nonlocal request_session
        async with factory() as db:
            request_session = db
            yield db

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation",
            json={"client_request_id": str(uuid.uuid4()), "content": "question"},
        )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["answer"] == "done"
    assert observed_transaction is False


async def test_submit_does_not_recover_fresh_attempt_between_prepare_and_claim(
    conversation_client, conversation_goal_run, db_session, monkeypatch
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_allowance_tokens",
        10_000,
    )
    service = goals_router.conversation_service
    original_claim = service._claim
    calls = 0
    claim_started, allow_claim = asyncio.Event(), asyncio.Event()

    async def completion(**_kwargs):
        nonlocal calls
        calls += 1
        return {
            "choices": [{"message": {"content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    service._completion_fn = completion

    async def claim_after_recovery(goal_id, response_id):
        claim_started.set()
        await allow_claim.wait()
        return await original_claim(goal_id, response_id)

    monkeypatch.setattr(service, "_claim", claim_after_recovery)
    post = asyncio.create_task(
        conversation_client.post(
            f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation",
            json={"client_request_id": str(uuid.uuid4()), "content": "question"},
        )
    )
    await claim_started.wait()
    await service.recover_goal(goal.id)
    allow_claim.set()
    response = await post

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert calls == 1


async def test_completed_replay_survives_later_disable(
    conversation_client, conversation_goal_run, db_session, monkeypatch
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_allowance_tokens",
        10_000,
    )
    url = f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation"
    request_id = str(uuid.uuid4())

    first = await conversation_client.post(
        url, json={"client_request_id": request_id, "content": "question"}
    )
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_allowance_tokens",
        0,
    )
    replay = await conversation_client.post(
        url, json={"client_request_id": request_id, "content": "question"}
    )
    new = await conversation_client.post(
        url, json={"client_request_id": str(uuid.uuid4()), "content": "question"}
    )

    assert replay.status_code == 200
    assert replay.json()["response_id"] == first.json()["response_id"]
    assert new.status_code == 409
    assert new.json()["detail"]["code"] == "conversation_disabled"


async def test_submit_maps_idempotency_conflict_to_stable_detail(
    conversation_client, conversation_goal_run, db_session, monkeypatch
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_allowance_tokens",
        10_000,
    )
    request_id = str(uuid.uuid4())
    url = f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation"

    first = await conversation_client.post(url, json={"client_request_id": request_id, "content": "first"})
    second = await conversation_client.post(url, json={"client_request_id": request_id, "content": "second"})

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "idempotency_conflict"


async def test_history_keeps_existing_missing_goal_boundary(
    conversation_client, conversation_goal_run, db_session
):
    goal, _ = conversation_goal_run
    await db_session.commit()

    response = await conversation_client.get(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{uuid.uuid4()}/conversation"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Orchestration goal not found"


async def test_history_recovers_before_loading_steering_ledger(
    conversation_client, conversation_goal_run, db_session, monkeypatch
):
    goal, _ = conversation_goal_run
    await db_session.commit()
    calls = []
    original_ledger = goals_router._steering_ledger

    async def recover(goal_id):
        calls.append(("recover", goal_id))

    async def ledger(db, project_id, goal_id, actor_id):
        calls.append(("ledger", goal_id))
        return await original_ledger(db, project_id, goal_id, actor_id)

    monkeypatch.setattr(goals_router.conversation_service, "recover_goal", recover)
    monkeypatch.setattr(goals_router, "_steering_ledger", ledger)
    response = await conversation_client.get(
        f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation"
    )
    assert response.status_code == 200
    assert [kind for kind, _ in calls] == ["recover", "ledger"]


@pytest.mark.unsupported_mode
async def test_denied_conversation_requests_do_not_recover_or_dispatch(
    conversation_client, conversation_goal_run, test_project, test_engine, db_session, monkeypatch, tmp_path
):
    """Authorization rejects before the real recovery/Task3 path can mutate its committed attempt."""
    goal, _ = conversation_goal_run
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "risk.txt").write_text("gate ready\n", encoding="utf-8")
    project = await db_session.get(Project, goal.project_id)
    project.workspace_path = str(workspace)
    other_project = Project(
        name="Other project", description="", workspace_path=test_project.workspace_path, config={}
    )
    db_session.add(other_project)
    await db_session.commit()
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_allowance_tokens",
        50_000,
    )
    monkeypatch.setattr(
        "huddleroom.services.orchestration_conversation_service.settings.orchestration_conversation_investigation_enabled",
        True,
    )
    completion_ids, lookup_ids, dispatches, recoveries = [], [], [], []

    async def completion(**kwargs):
        completion_ids.append(kwargs["litellm_call_id"])
        content = (
            '{"findings":"missing required fields"}'
            if len(completion_ids) == 1
            else json.dumps({
                "findings": "Gate ready.", "uncertainty": "", "sources": ["risk.txt#L1-L1"],
            })
        )
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": len(completion_ids)},
        }

    async def lookup(provider_request_id):
        lookup_ids.append(provider_request_id)
        return {
            "choices": [{"message": {"content": None, "tool_calls": [{"function": {
                "name": "request_investigation",
                "arguments": json.dumps({
                    "objective": "Inspect release gate",
                    "requests": [{"operation": "read", "path": "risk.txt", "query": None}],
                }),
            }}]}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }

    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    service = OrchestrationConversationService(
        factory, completion, lookup, OrchestrationService()
    )
    original_recover, original_dispatch = service.recover_goal, service._investigations._dispatch

    async def recover(goal_id):
        recoveries.append(goal_id)
        return await original_recover(goal_id)

    async def dispatch(goal_id, row):
        dispatches.append((goal_id, row.id, row.provider_request_id, row.attempt_count))
        return await original_dispatch(goal_id, row)

    monkeypatch.setattr(service, "recover_goal", recover)
    monkeypatch.setattr(service._investigations, "_dispatch", dispatch)
    monkeypatch.setattr(goals_router, "conversation_service", service)
    prepared, created = await service._prepare(
        goal.project_id, goal.id, ANON_USER_ID, uuid.uuid4(), "Inspect release gate"
    )
    assert created is True
    claimed = await service._claim(goal.id, prepared.response.id)
    assert claimed is not None
    async with factory() as db:
        expired = await db.get(ConversationResponse, claimed.id)
        expired.deadline_at = _utcnow() - timedelta(seconds=1)
        await db.commit()

    def snapshot_value(value):
        if isinstance(value, datetime):
            return _utc_timestamp(value)
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, (dict, list)):
            return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False))
        return value

    def snapshot_rows(rows, fields):
        return tuple(
            tuple(snapshot_value(getattr(row, field)) for field in fields)
            for row in rows
        )

    async def committed_snapshot():
        async with factory() as db:
            messages = (await db.scalars(select(ConversationMessage).order_by(ConversationMessage.id))).all()
            responses = (await db.scalars(select(ConversationResponse).order_by(ConversationResponse.id))).all()
            reservations = (await db.scalars(select(ConversationReservation).order_by(ConversationReservation.id))).all()
            investigations = (await db.scalars(
                select(ConversationInvestigation).order_by(ConversationInvestigation.id)
            )).all()
            investigation_reservations = (await db.scalars(
                select(ConversationInvestigationReservation).order_by(ConversationInvestigationReservation.id)
            )).all()
            return (
                snapshot_rows(messages, (
                    "id", "goal_id", "actor_id", "client_request_id", "sequence", "content", "created_at",
                )),
                snapshot_rows(responses, (
                    "id", "message_id", "run_id", "status", "dossier", "context_manifest", "context_version",
                    "provider_request_id", "answer", "error", "started_at", "deadline_at", "finished_at",
                    "created_at", "updated_at",
                )),
                snapshot_rows(reservations, (
                    "id", "response_id", "goal_id", "actor_id", "ceiling_snapshot", "reserved_tokens",
                    "settled_tokens", "released_tokens", "status", "committed_at", "settled_at", "released_at",
                    "created_at", "updated_at",
                )),
                snapshot_rows(investigations, (
                    "id", "response_id", "goal_id", "actor_id", "context_version", "status", "objective", "scope",
                    "input_manifest", "provider_identity", "provider_request_id", "attempt_count", "repair_count",
                    "retry_count", "accumulated_tokens", "report", "error", "started_at", "deadline_at", "finished_at",
                    "cancelled_at", "created_at", "updated_at",
                )),
                snapshot_rows(investigation_reservations, (
                    "id", "investigation_id", "goal_id", "actor_id", "ceiling_snapshot", "reserved_tokens",
                    "settled_tokens", "released_tokens", "status", "committed_at", "settled_at", "released_at",
                    "created_at", "updated_at",
                )),
            )

    before = await committed_snapshot()
    allowed_url = f"/api/v1/projects/{goal.project_id}/orchestration/goals/{goal.id}/conversation"
    wrong_project_url = f"/api/v1/projects/{other_project.id}/orchestration/goals/{goal.id}/conversation"
    async def assert_denied(method, url, status):
        response = await getattr(conversation_client, method)(
            url,
            **({"json": {"client_request_id": str(uuid.uuid4()), "content": "question"}} if method == "post" else {}),
        )
        assert response.status_code == status
        if status == 404:
            assert response.json()["detail"] == "Orchestration goal not found"
        assert (recoveries, lookup_ids, dispatches, completion_ids) == ([], [], [], [])
        assert await committed_snapshot() == before

    await assert_denied("get", wrong_project_url, 404)
    await assert_denied("post", wrong_project_url, 404)

    from huddleroom.config import Settings
    import huddleroom.dependencies as dependencies

    original_dependency_settings = dependencies.settings
    monkeypatch.setattr("huddleroom.dependencies.settings", Settings(auth_enabled=True))
    await assert_denied("get", allowed_url, 401)
    await assert_denied("post", allowed_url, 401)
    monkeypatch.setattr("huddleroom.dependencies.settings", original_dependency_settings)

    allowed = await conversation_client.get(allowed_url)
    assert allowed.status_code == 200
    body = allowed.json()["items"][-1]
    assert body["status"] == "completed"
    assert body["answer"] == "Gate ready."
    assert body["investigation"]["report"] == {
        "findings": "Gate ready.", "uncertainty": "", "sources": ["risk.txt#L1-L1"],
    }
    assert recoveries == [goal.id]
    assert lookup_ids == [claimed.provider_request_id]
    async with factory() as db:
        response = await db.get(ConversationResponse, claimed.id)
        reservation = await db.get(ConversationReservation, conversation_reservation_id(claimed.id))
        investigation = await db.scalar(
            select(ConversationInvestigation).where(ConversationInvestigation.response_id == claimed.id)
        )
        investigation_reservation = await db.get(
            ConversationInvestigationReservation,
            conversation_investigation_reservation_id(investigation.id),
        )
        assert (response.status, response.answer, response.error) == ("completed", "Gate ready.", None)
        assert (reservation.status, reservation.settled_tokens, reservation.released_tokens) == (
            "settled", 5, reservation.reserved_tokens - 5,
        )
        assert (investigation.status, investigation.attempt_count, investigation.repair_count, investigation.retry_count) == (
            "completed", 2, 1, 0,
        )
        assert (investigation_reservation.status, investigation_reservation.settled_tokens) == ("settled", 7)
        assert investigation_reservation.released_tokens == investigation_reservation.reserved_tokens - 7
        assert dispatches == [(goal.id, investigation.id, conversation_investigation_provider_request_id(investigation.id, 1), 1)]
        assert completion_ids == [
            conversation_investigation_provider_request_id(investigation.id, 1),
            conversation_investigation_provider_request_id(investigation.id, 2),
        ]
