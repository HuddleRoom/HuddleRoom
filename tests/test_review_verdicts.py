from __future__ import annotations

from pathlib import Path
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select

from huddleroom.models.agent import Agent
from huddleroom.models.artifact import Artifact
from huddleroom.models.event_log import EventLog
from huddleroom.models.graph import GraphRun
from huddleroom.models.orchestration import OrchestrationEvidence
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.event_bus import BusEvent, emit_event_once
from huddleroom.services.graph_engine import GraphEngineService
from huddleroom.services.graph_service import GraphService
from tests.test_orchestration_evidence_ingestion import (
    _agent,
    _make_gate,
    _make_orchestrated_task,
    _make_run,
)

pytestmark = pytest.mark.usefixtures("safe_goal_analysis")


@pytest_asyncio.fixture(autouse=True)
async def _runnable_graph_project(db_session, test_project, tmp_path: Path):
    workspace = tmp_path / "graph-workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())
    await db_session.flush()


async def _complete_review(db_session, test_project, output: str):
    proto = await GraphService().load_from_yaml(db_session, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(
        name=f"reviewer-{uuid.uuid4()}",
        role="reviewer",
        provider="openai",
        model="gpt-4o-mini",
        adapter_type="api",
        capabilities=["code_review"],
        config={},
    )
    artifact = Artifact(
        project_id=test_project.id,
        name="verdict-pr",
        artifact_type="pull_request",
        url="http://example.com/pr/verdict",
        metadata_={"branch": "feature/verdict"},
    )
    task = Task(project_id=test_project.id, title="Review PR", status="in_progress")
    db_session.add_all([reviewer, artifact, task])
    await db_session.flush()
    run = GraphRun(
        graph_id=proto.id,
        project_id=test_project.id,
        linked_task_id=task.id,
        artifact_id=artifact.id,
        current_node="ready_for_review",
        status="active",
        actor_assignments={"reviewer": {"kind": "agent", "id": str(reviewer.id), "name": reviewer.name}},
        context={},
    )
    db_session.add(run)
    await db_session.flush()
    session = Session(
        agent_id=reviewer.id,
        task_id=task.id,
        project_id=test_project.id,
        graph_run_id=run.id,
        adapter_type="api",
        status="completed",
        output=output,
        metadata_={},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()
    await GraphEngineService().process_event(
        db_session,
        BusEvent(
            id=uuid.uuid4(),
            project_id=test_project.id,
            event_type="session.completed",
            payload={"session_id": str(session.id)},
            source="system",
        ),
    )
    await db_session.flush()
    await db_session.refresh(run)
    events = (
        await db_session.execute(
            select(EventLog.event_type).where(
                EventLog.project_id == test_project.id,
                EventLog.event_type.in_(["review.approved", "review.changes_requested"]),
            )
        )
    ).scalars().all()
    return run, events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        "I cannot approve until the tests are fixed.",
        "I won't approve this.",
        "Previously this was approved, but now it needs rework.",
        "VERDICT: APPROVE\nActually, on reflection I have more notes.",
    ],
)
async def test_prose_without_final_verdict_does_not_approve(db_session, test_project, output):
    run, events = await _complete_review(db_session, test_project, output)

    assert run.current_node == "ready_for_review"
    assert events == []


@pytest.mark.asyncio
async def test_final_line_approve_advances(db_session, test_project):
    run, events = await _complete_review(db_session, test_project, "Looks good.\n\nVERDICT: APPROVE\n")

    assert run.current_node == "approved"
    assert events == ["review.approved"]


@pytest.mark.asyncio
async def test_final_line_request_changes_awaits_revision(db_session, test_project):
    run, events = await _complete_review(db_session, test_project, "Needs work.\nverdict: request_changes")

    assert run.current_node == "awaiting_revision"
    assert events == ["review.changes_requested"]


@pytest.mark.asyncio
async def test_commented_outcome_records_no_review_evidence(db_session, test_project):
    service, _goal, run = await _make_run(db_session, test_project.id)
    gate = await _make_gate(db_session, run.id, gate_type="review_accepted")
    reviewer = _agent("reviewer", "reviewer", ["review"])
    db_session.add(reviewer)
    await db_session.flush()
    task = await _make_orchestrated_task(db_session, test_project.id, run.id, gate.id, reviewer.id, "review")
    session = Session(
        project_id=test_project.id,
        task_id=task.id,
        agent_id=reviewer.id,
        adapter_type="api",
        status="completed",
        input_context={},
        output="I cannot approve this.",
        metadata_={},
        origin="graph",
    )
    db_session.add(session)
    await db_session.flush()
    await emit_event_once(
        db_session,
        test_project.id,
        "review.approved",
        {
            "session_id": str(session.id),
            "task_id": str(task.id),
            "review_outcome": {"verdict": "commented", "reason": "verdict_missing"},
            "project_id": str(test_project.id),
        },
        dedup_key=f"commented-review:{session.id}",
    )

    await service.tick(db_session, run.id)

    rows = (
        await db_session.execute(select(OrchestrationEvidence).where(OrchestrationEvidence.run_id == run.id))
    ).scalars().all()
    assert [r for r in rows if r.source_type == "review"] == []
    await db_session.refresh(gate)
    assert gate.status != "accepted"
