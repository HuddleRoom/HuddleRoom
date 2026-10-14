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
        url=f"https://github.com/org/repo/commit/{'c' * 40}",  # pinned: only pinned content can be approved
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
    # the reviewer session is created the way the graph creates it, bound to the artifact's identity
    await GraphEngineService()._execute_on_enter(db_session, proto.definition["nodes"]["ready_for_review"], run)
    session = (await db_session.execute(select(Session).where(Session.graph_run_id == run.id))).scalar_one()
    session.status, session.output = "completed", output
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


# --- a graph approval binds to the content identity the reviewer was given ---

import hashlib  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402

from huddleroom.models.channel import Channel  # noqa: E402
from huddleroom.models.message import Message  # noqa: E402

COMMIT_URL = f"https://github.com/org/repo/commit/{'d' * 40}"


def _git_repo(workspace: Path) -> tuple[str, str]:
    """A workspace repository with two commits; returns (first sha, second sha)."""
    repo = workspace / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env)

    git("init", "-q")
    shas = []
    for text in ("one", "two"):
        (repo / "f.py").write_text(text)
        git("add", "f.py")
        git("commit", "-q", "-m", text)
        shas.append(git("rev-parse", "HEAD").stdout.strip())
    return shas[0], shas[1]


async def _open_review(db, project, **artifact_fields):
    """A code_review run driven to ready_for_review by real events; the graph creates the reviewer session."""
    await GraphService().load_from_yaml(db, Path("workspace/graphs/code_review.yaml"))
    reviewer = Agent(name=f"reviewer-{uuid.uuid4()}", role="reviewer", provider="openai", model="gpt-4o-mini",
                     adapter_type="api", capabilities=["code_review"], config={})
    merger = Agent(name=f"pm-{uuid.uuid4()}", role="pm", provider="openai", model="gpt-4o-mini",
                   adapter_type="api", capabilities=["code_merge"], config={})
    artifact = Artifact(project_id=project.id, name="identity-pr", **{
        "artifact_type": "pull_request", "metadata_": {"branch": "feature/identity"}, **artifact_fields,
    })
    db.add_all([reviewer, merger, artifact,
                Channel(project_id=project.id, name="general", channel_type="general", members=[])])
    await db.flush()
    for event_type in ("code.pr_opened", "test.passed"):
        await _event(db, project, event_type, artifact)
    run = (await db.execute(select(GraphRun).where(GraphRun.artifact_id == artifact.id))).scalar_one()
    assert run.current_node == "ready_for_review"
    return run, artifact, merger


async def _event(db, project, event_type, artifact):
    await GraphEngineService().process_event(db, BusEvent(
        id=uuid.uuid4(), project_id=project.id, event_type=event_type,
        payload={"artifact_id": str(artifact.id)}, source="system",
    ))
    await db.flush()


async def _review_sessions(db, run):
    return list((await db.execute(
        select(Session).where(Session.graph_run_id == run.id).order_by(Session.created_at, Session.id)
    )).scalars().all())


async def _answer(db, project, run, output="Looks right.\nVERDICT: APPROVE"):
    *_, session = await _review_sessions(db, run)
    session.status, session.output = "completed", output
    await db.flush()
    await GraphEngineService().process_event(db, BusEvent(
        id=uuid.uuid4(), project_id=project.id, event_type="session.completed",
        payload={"session_id": str(session.id)}, source="system",
    ))
    await db.flush()
    await db.refresh(run)
    return session


async def _approvals(db, project):
    return list((await db.execute(select(EventLog).where(
        EventLog.project_id == project.id, EventLog.event_type == "review.approved",
    ))).scalars().all())


async def _merger_messages(db, project, merger):
    channel = (await db.execute(select(Channel).where(
        Channel.project_id == project.id, Channel.name == f"dm-{merger.id}",
    ))).scalar_one_or_none()
    if channel is None:
        return []
    return [row.content for row in (await db.execute(select(Message).where(Message.channel_id == channel.id))).scalars()]


async def _task_description(db, session):
    return (await db.get(Task, session.task_id)).description


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.asyncio
async def test_graph_approval_of_edited_file_is_not_applied_and_review_is_requested_again(db_session, test_project):
    workspace = Path(test_project.workspace_path)
    (workspace / "app.py").write_text("v1")
    run, _artifact, merger = await _open_review(db_session, test_project, path="app.py")
    [first] = await _review_sessions(db_session, run)
    assert f"app.py (sha256 {_sha('v1')})" in await _task_description(db_session, first)

    (workspace / "app.py").write_text("v2")  # edited directly in the workspace after the review was dispatched
    await _answer(db_session, test_project, run)

    assert run.current_node == "ready_for_review" and await _approvals(db_session, test_project) == []
    assert run.context["review_identity_blocked"]["reason"] == "changed"
    _, second = await _review_sessions(db_session, run)  # a new review, bound to the new content
    assert f"app.py (sha256 {_sha('v2')})" in await _task_description(db_session, second)
    assert run.context["review_identity"]["identity"] == _sha("v2")

    await _answer(db_session, test_project, run)
    assert run.current_node == "approved" and len(await _approvals(db_session, test_project)) == 1
    assert "review_identity_blocked" not in run.context  # the later approval ends the earlier refusal
    [message] = await _merger_messages(db_session, test_project, merger)
    assert f"app.py (sha256 {_sha('v2')})" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", ["code.pr_merged", "code.pr_updated"])
async def test_graph_returns_to_review_when_content_changes_after_approval(db_session, test_project, event_type):
    workspace = Path(test_project.workspace_path)
    (workspace / "app.py").write_text("v1")
    run, artifact, _merger = await _open_review(db_session, test_project, path="app.py")
    await _answer(db_session, test_project, run)
    assert run.current_node == "approved"

    (workspace / "app.py").write_text("v2")
    await _event(db_session, test_project, event_type, artifact)

    await db_session.refresh(run)
    assert run.current_node == "ready_for_review" and run.status == "active"
    _, second = await _review_sessions(db_session, run)
    assert f"sha256 {_sha('v2')}" in await _task_description(db_session, second)


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{}, {"commit": "e" * 40}], ids=["bare_pr_url", "sha_only_in_metadata"])
async def test_graph_unpinned_pr_url_never_approves(db_session, test_project, metadata):
    run, _artifact, _merger = await _open_review(
        db_session, test_project, url="https://github.com/org/repo/pull/9", metadata_={"branch": "b", **metadata},
    )
    [session] = await _review_sessions(db_session, run)
    assert "No pinned content" in await _task_description(db_session, session)

    await _answer(db_session, test_project, run)

    assert run.current_node == "ready_for_review" and await _approvals(db_session, test_project) == []
    assert run.context["review_identity_blocked"]["reason"] == "artifact_identity_unpinned"
    assert len(await _review_sessions(db_session, run)) == 1  # no re-review loop and no producer revision


@pytest.mark.asyncio
@pytest.mark.parametrize("pin", ["commit_url", "local_repo_commit", "stored_plan"])
async def test_graph_pinned_content_approves_and_the_merger_gets_the_same_reference(db_session, test_project, pin):
    if pin == "commit_url":
        fields, reference = {"url": COMMIT_URL}, COMMIT_URL
    elif pin == "local_repo_commit":
        first, _second = _git_repo(Path(test_project.workspace_path))
        fields = {"url": "https://github.com/org/repo/pull/9", "metadata_": {"repo_path": "repo", "commit": first}}
        reference = f"repo at commit {first}"
    else:
        fields = {"artifact_type": "plan", "metadata_": {"schema_version": 1, "plan_kind": "work", "summary": "s",
                                                         "plan_items": [{"key": "a"}]}}
        reference = "stored plan content (sha256 "
    run, artifact, merger = await _open_review(db_session, test_project, **fields)
    [session] = await _review_sessions(db_session, run)
    assert reference in await _task_description(db_session, session)

    await _answer(db_session, test_project, run)

    assert run.current_node == "approved" and len(await _approvals(db_session, test_project)) == 1
    [message] = await _merger_messages(db_session, test_project, merger)
    assert reference in message
    await _event(db_session, test_project, "code.pr_merged", artifact)
    await db_session.refresh(run)
    assert run.current_node == "merged"


@pytest.mark.asyncio
async def test_graph_approval_applies_only_to_the_reviewed_local_commit(db_session, test_project):
    first, second = _git_repo(Path(test_project.workspace_path))
    run, artifact, _merger = await _open_review(
        db_session, test_project, url="https://github.com/org/repo/pull/9",
        metadata_={"repo_path": "repo", "commit": first},
    )
    await _answer(db_session, test_project, run)
    assert run.current_node == "approved"

    artifact.metadata_ = {**artifact.metadata_, "commit": second}  # the PR now points at another commit
    await db_session.flush()
    await _event(db_session, test_project, "code.pr_merged", artifact)

    await db_session.refresh(run)
    assert run.current_node == "ready_for_review"
    _, rereview = await _review_sessions(db_session, run)
    assert f"repo at commit {second}" in await _task_description(db_session, rereview)


async def _redeliver(db, project, session, *, with_outcome=False):
    """Replay a review session's completion; `with_outcome` replays a payload that already carries its verdict."""
    payload = {"session_id": str(session.id)}
    if with_outcome:
        payload["review_outcome"] = GraphEngineService()._parse_review_outcome(session.output)
    await GraphEngineService().process_event(db, BusEvent(
        id=uuid.uuid4(), project_id=project.id, event_type="session.completed", payload=payload, source="system",
    ))
    await db.flush()


@pytest.mark.asyncio
async def test_redelivered_approval_of_a_superseded_review_session_never_applies(db_session, test_project):
    workspace = Path(test_project.workspace_path)
    (workspace / "app.py").write_text("v1")
    service, orchestration_run, gate = await _review_gate(db_session, test_project)
    run, artifact, _merger = await _open_review(db_session, test_project, path="app.py")
    [first] = await _review_sessions(db_session, run)
    await _link_review(db_session, first, orchestration_run, gate)
    await _answer(db_session, test_project, run)  # approves v1
    (workspace / "app.py").write_text("v2")
    await _event(db_session, test_project, "code.pr_updated", artifact)  # back to review: a new session binds v2
    await db_session.refresh(run)
    assert run.current_node == "ready_for_review" and len(await _review_sessions(db_session, run)) == 2

    await _redeliver(db_session, test_project, first)
    await service.tick(db_session, orchestration_run.id)  # the v1 approval reaches the gate only now

    await db_session.refresh(run)
    await db_session.refresh(gate)
    assert run.current_node == "ready_for_review"  # nobody approved v2
    assert len(await _approvals(db_session, test_project)) == 1
    [evidence] = [row for row in (await db_session.execute(select(OrchestrationEvidence).where(
        OrchestrationEvidence.gate_id == gate.id))).scalars() if row.source_type == "review"]
    assert evidence.evidence_metadata["artifact_identity"][0]["identity"] == _sha("v1")  # bound to what S1 saw
    assert gate.status == "open" and gate.failure_reason == "Reviewed content changed after its review"


@pytest.mark.asyncio
async def test_redelivered_refused_approval_does_not_duplicate_reviews_or_merger_messages(db_session, test_project):
    workspace = Path(test_project.workspace_path)
    (workspace / "app.py").write_text("v1")
    run, _artifact, merger = await _open_review(db_session, test_project, path="app.py")
    (workspace / "app.py").write_text("v2")
    first = await _answer(db_session, test_project, run)  # refused: v1 changed, a second review is created
    await _redeliver(db_session, test_project, first, with_outcome=True)
    assert len(await _review_sessions(db_session, run)) == 2

    await _answer(db_session, test_project, run)  # the second review approves v2
    assert run.current_node == "approved"
    (workspace / "app.py").write_text("v3")
    _, second = await _review_sessions(db_session, run)
    await _redeliver(db_session, test_project, second, with_outcome=True)  # replayed while approved

    await db_session.refresh(run)
    assert run.current_node == "approved" and len(await _review_sessions(db_session, run)) == 2
    assert len(await _merger_messages(db_session, test_project, merger)) == 1


async def _review_gate(db, project):
    from huddleroom.models.orchestration import OrchestrationGate

    service, _goal, orchestration_run = await _make_run(db, project.id)
    gate = OrchestrationGate(run_id=orchestration_run.id, success_criterion_key="plan_item:review-pr",
                             gate_type="review_accepted",
                             required_evidence={"required_source_types": ["review"], "min_count": 1})
    db.add(gate)
    await db.flush()
    return service, orchestration_run, gate


async def _link_review(db, session, orchestration_run, gate):
    """The orchestrated review task a graph review session works on."""
    task = await db.get(Task, session.task_id)
    task.metadata_ = {**task.metadata_, "orchestration": {
        "run_id": str(orchestration_run.id), "plan_item_gate_id": str(gate.id), "work_function": "review",
    }}
    await db.flush()


@pytest.mark.asyncio
async def test_gate_is_not_accepted_on_content_a_later_review_session_was_bound_to(db_session, test_project):
    workspace = Path(test_project.workspace_path)
    (workspace / "app.py").write_text("v1")
    service, orchestration_run, gate = await _review_gate(db_session, test_project)
    run, artifact, _merger = await _open_review(db_session, test_project, path="app.py")
    [first] = await _review_sessions(db_session, run)
    await _link_review(db_session, first, orchestration_run, gate)
    await _answer(db_session, test_project, run)  # approves v1
    (workspace / "app.py").write_text("v2")
    await _event(db_session, test_project, "code.pr_updated", artifact)  # a second session now binds v2

    await service.tick(db_session, orchestration_run.id)  # the v1 approval is ingested only now

    await db_session.refresh(gate)
    assert gate.status == "open" and gate.failure_reason == "Reviewed content changed after its review"


@pytest.mark.asyncio
async def test_direct_edit_after_graph_approval_sends_the_graph_back_to_review(db_session, test_project):
    workspace = Path(test_project.workspace_path)
    (workspace / "app.py").write_text("v1")
    service, orchestration_run, gate = await _review_gate(db_session, test_project)
    run, _artifact, _merger = await _open_review(db_session, test_project, path="app.py")
    [first] = await _review_sessions(db_session, run)
    await _link_review(db_session, first, orchestration_run, gate)
    await _answer(db_session, test_project, run)
    assert run.current_node == "approved"
    (workspace / "app.py").write_text("v2")  # no PR event

    await service.tick(db_session, orchestration_run.id)
    await service.recover_run(db_session, orchestration_run.id, baseline_ready=True)

    await db_session.refresh(run)
    await db_session.refresh(gate)
    assert gate.status == "open" and run.current_node == "ready_for_review"
    _, second = await _review_sessions(db_session, run)
    assert f"sha256 {_sha('v2')}" in await _task_description(db_session, second)
    await _link_review(db_session, second, orchestration_run, gate)
    await _answer(db_session, test_project, run)
    await service.tick(db_session, orchestration_run.id)
    await db_session.refresh(gate)
    assert gate.status == "accepted"


@pytest.mark.asyncio
async def test_changed_graph_approval_asks_the_owner_when_its_graph_already_ended(db_session, test_project):
    workspace = Path(test_project.workspace_path)
    (workspace / "app.py").write_text("v1")
    service, orchestration_run, gate = await _review_gate(db_session, test_project)
    run, artifact, _merger = await _open_review(db_session, test_project, path="app.py")
    [first] = await _review_sessions(db_session, run)
    await _link_review(db_session, first, orchestration_run, gate)
    await _answer(db_session, test_project, run)
    await _event(db_session, test_project, "code.pr_merged", artifact)  # merged on v1
    await db_session.refresh(run)
    assert run.status == "completed"
    (workspace / "app.py").write_text("v2")  # edited before the gate consumed the approval

    await service.tick(db_session, orchestration_run.id)
    for _ in range(2):  # repeated passes ask once
        await service.recover_run(db_session, orchestration_run.id, baseline_ready=True)

    await db_session.refresh(gate)
    assert gate.status == "open"
    asks = [item for item in orchestration_run.active_blockers
            if item.get("kind") == "owner_escalation" and item.get("category") == "capability_unavailable"]
    assert [item["scope"] for item in asks] == [f"gate:{gate.id}:review:{run.id}"]


@pytest.mark.asyncio
async def test_task_update_cannot_replace_remove_or_inject_a_review_binding(db_session, test_project):
    from huddleroom.schemas.task import TaskUpdate
    from huddleroom.services.task_service import TaskService

    (Path(test_project.workspace_path) / "app.py").write_text("v1")
    run, _artifact, _merger = await _open_review(db_session, test_project, path="app.py")
    [session] = await _review_sessions(db_session, run)
    task = await db_session.get(Task, session.task_id)
    bound = task.metadata_["review_identity"]

    for metadata in ({"review_identity": {**bound, "identity": "0" * 64}, "note": "x"}, {"note": "y"}):
        await TaskService().update(db_session, test_project.id, task.id, TaskUpdate(metadata=metadata))
        assert task.metadata_["review_identity"] == bound and task.metadata_["note"] == metadata["note"]

    plain = Task(project_id=test_project.id, title="Plain", status="ready")
    db_session.add(plain)
    await db_session.flush()
    await TaskService().update(db_session, test_project.id, plain.id, TaskUpdate(metadata={"review_identity": bound}))
    assert "review_identity" not in plain.metadata_
