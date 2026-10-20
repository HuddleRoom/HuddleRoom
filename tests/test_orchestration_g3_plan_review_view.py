"""Focused G3 readable-plan projection and typed artifact-reference coverage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

from huddleroom.models.artifact import Artifact
from huddleroom.models.orchestration import OrchestrationGate, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.models.task import Task
from huddleroom.services.orchestration_needs_you_presenter import (
    NeedsYouReferenceRequest,
    resolve_needs_you_references,
)
from huddleroom.services.orchestration_plan_ingestion import plan_output_hash
from huddleroom.services.orchestration_plan_review import PlanReviewService


def _identity_hash(identity) -> str:
    encoded = json.dumps(identity.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


async def _candidate(db, project, agent, *, title="Pending candidate"):
    goal = OrchestrationGoal(project_id=project.id, objective="Review the current delivery plan")
    db.add(goal)
    await db.flush()
    run = OrchestrationRun(goal_id=goal.id, status="running", phase="ready")
    planner = Task(project_id=project.id, title="plan", status="done", assigned_to=agent.id, metadata_={})
    db.add_all([run, planner])
    await db.flush()
    item = {
        "id": "candidate-item", "title": title, "agent_id": str(agent.id), "work_function": "engineering",
        "scope": "bounded implementation", "deliverable": "tested change", "depends_on": [],
        "forbidden_work": ["live calls"], "success_evidence": ["focused tests"],
        "required_evidence": {"source": "reviewed"},
    }
    metadata = {
        "schema_version": 1, "plan_kind": "roadmap", "summary": title, "plan_items": [item],
        "planning_task_id": str(planner.id), "planning_version": 1, "winning_session_id": str(uuid4()),
        "plan_author_agent_id": str(agent.id),
    }
    digest = plan_output_hash({key: metadata[key] for key in ("schema_version", "plan_kind", "summary", "plan_items")})
    metadata["output_hash"] = digest
    artifact = Artifact(
        project_id=project.id, name=f"{title}.md", artifact_type="plan", linked_task_id=planner.id,
        created_by_agent=agent.id, content_hash=digest, metadata_=metadata,
    )
    db.add(artifact)
    await db.flush()
    run.plan_state = {"status": "pending_review", "plan_artifact_id": str(artifact.id)}
    review = await PlanReviewService().ensure(db, goal, run, artifact)
    await db.flush()
    return goal, run, artifact, review


@pytest.mark.asyncio
async def test_review_projection_keeps_accepted_outer_plan_and_stale_candidate_readable(
    db_session, test_project, test_agent,
):
    from huddleroom.routers.orchestration_goals import _plan_review_view, _plan_view

    goal, run, candidate, review = await _candidate(db_session, test_project, test_agent)
    accepted_metadata = dict(candidate.metadata_)
    accepted_metadata["summary"] = "Accepted historical plan"
    accepted_metadata["plan_items"] = [{**accepted_metadata["plan_items"][0], "id": "accepted-item", "title": "Accepted item"}]
    accepted_digest = plan_output_hash({key: accepted_metadata[key] for key in (
        "schema_version", "plan_kind", "summary", "plan_items",
    )})
    accepted_metadata["output_hash"] = accepted_digest
    accepted = Artifact(
        project_id=test_project.id, name="accepted.md", artifact_type="plan", linked_task_id=candidate.linked_task_id,
        created_by_agent=test_agent.id, content_hash=accepted_digest, metadata_=accepted_metadata,
    )
    binding_hash = _identity_hash(review.identity)
    gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="plan", gate_type="roadmap_plan_approval", status="open",
        required_evidence={"plan_review_identity_hash": binding_hash},
    )
    decision = OrchestrationAuthorityDecision(
        goal_id=goal.id, run_id=run.id, decision_key=f"roadmap_plan:{binding_hash}", title="Approve plan v1",
        authority="human", question="Approve?", options=["approve", "reject"],
        context=json.dumps({"plan_review_identity_hash": binding_hash}),
    )
    db_session.add_all([accepted, gate, decision])
    await db_session.flush()
    run.plan_state = {
        **run.plan_state,
        "accepted_artifact_id": str(accepted.id),
        "plan_artifact_id": str(candidate.id),
        "status": "pending_review",
    }
    await db_session.flush()

    projection = await _plan_view(db_session, goal, run, None)
    assert projection.artifact_id == accepted.id
    assert [item.key for item in projection.items] == ["accepted-item"]
    assert projection.review is not None
    assert projection.review.artifact_id == candidate.id
    assert projection.review.authority_decision_id == decision.id
    assert projection.review.authority_gate_id == gate.id
    assert projection.review.authority_status == "pending"

    export = Path(test_project.workspace_path) / review.export_path
    export.write_text("edited outside the review", encoding="utf-8")
    stale = await _plan_review_view(db_session, goal, run, candidate)
    assert stale is not None and stale.state == "stale"
    assert stale.reason == "plan export diverged"
    assert stale.markdown == review.markdown
    assert stale.authority_decision_id == decision.id and stale.authority_gate_id == gate.id


@pytest.mark.asyncio
async def test_review_handler_and_artifact_reference_require_current_bound_candidate(
    db_session, test_project, test_agent,
):
    from huddleroom.routers.orchestration_goals import get_plan_review

    goal, run, artifact, review = await _candidate(db_session, test_project, test_agent)
    unresolved = Artifact(project_id=test_project.id, name="unbound.md", artifact_type="plan", metadata_={})
    foreign = Artifact(project_id=uuid4(), name="foreign.md", artifact_type="plan", metadata_={})
    db_session.add_all([unresolved, foreign])
    await db_session.flush()

    wrong_goal_id = uuid4()
    references = await resolve_needs_you_references(
        db_session,
        project_id=test_project.id,
        requests=[
            NeedsYouReferenceRequest("artifact", artifact.id, goal.id, run.id),
            NeedsYouReferenceRequest("artifact", unresolved.id, goal.id, run.id),
            NeedsYouReferenceRequest("artifact", foreign.id, goal.id, run.id),
        ],
    )
    labels = {(reference.id, reference.goal_id): reference.label for reference in references}
    assert labels[(artifact.id, goal.id)] == "Plan candidate v1"
    assert labels[(unresolved.id, goal.id)] is None
    assert labels[(foreign.id, goal.id)] is None
    wrong_goal = await resolve_needs_you_references(
        db_session,
        project_id=test_project.id,
        requests=[NeedsYouReferenceRequest("artifact", artifact.id, wrong_goal_id, run.id)],
    )
    assert wrong_goal[0].label is None

    for artifact_id in (unresolved.id, foreign.id):
        with pytest.raises(HTTPException) as error:
            await get_plan_review(test_project.id, goal.id, artifact_id, None, db_session)
        assert error.value.status_code == 404

    response = await get_plan_review(test_project.id, goal.id, artifact.id, None, db_session)
    assert response.identity == review.identity and response.markdown == review.markdown
