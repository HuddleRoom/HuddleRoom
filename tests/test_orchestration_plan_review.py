import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from huddleroom.services.orchestration_plan_ingestion import plan_output_hash
from huddleroom.services.orchestration_plan_review import PlanReviewService


class _Db:
    def __init__(self, project, agents=None):
        self.project = project
        self.agents = agents or {}

    async def get(self, model, value):
        if value == self.project.id:
            return self.project
        return self.agents.get(str(value))


@pytest.mark.asyncio
async def test_g3_owner_name_resolves_string_agent_id_from_database(db_session, test_agent):
    artifact = SimpleNamespace(metadata_={"plan_items": [{"agent_id": str(test_agent.id)}]})

    names = await PlanReviewService._owner_names(db_session, artifact)

    assert names == {str(test_agent.id): test_agent.name}


def _candidate(workspace):
    project_id, goal_id, run_id, task_id, artifact_id, agent_id, session_id = (uuid4() for _ in range(7))
    metadata = {
        "schema_version": 1,
        "plan_kind": "delivery",
        "summary": "Deliver the retained MVP.",
        "plan_items": [{
            "id": "implement", "title": "Implement review", "agent_id": str(agent_id),
            "work_function": "engineering", "scope": "review binding", "deliverable": "service",
            "depends_on": ["design"], "forbidden_work": ["live calls"], "success_evidence": ["tests"],
            "required_evidence": {"source": "source-verified"},
        }],
        "planning_task_id": str(task_id), "planning_version": 2, "winning_session_id": str(session_id),
        "plan_author_agent_id": str(agent_id),
    }
    digest = plan_output_hash({key: metadata[key] for key in ("schema_version", "plan_kind", "summary", "plan_items")})
    metadata["output_hash"] = digest
    project = SimpleNamespace(id=project_id, workspace_path=str(workspace))
    goal = SimpleNamespace(id=goal_id, project_id=project_id)
    run = SimpleNamespace(id=run_id, plan_state={})
    artifact = SimpleNamespace(
        id=artifact_id, path=None, linked_task_id=task_id, created_by_agent=agent_id,
        content_hash=digest, metadata_=metadata,
    )
    return _Db(project), goal, run, artifact


def _ingested_provenance(artifact):
    metadata = artifact.metadata_
    return {
        "artifact_id": str(artifact.id),
        "planning_task_id": metadata["planning_task_id"],
        "planning_version": metadata["planning_version"],
        "winning_session_id": metadata["winning_session_id"],
        "plan_author_agent_id": metadata["plan_author_agent_id"],
        "canonical_content_hash": artifact.content_hash,
    }


@pytest.mark.asyncio
async def test_g3_projection_preserves_db_identity_and_full_content(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    record = await PlanReviewService().ensure(db, goal, run, artifact)

    assert record.state == "ready"
    assert artifact.path is None
    assert record.identity.canonical_content_hash == artifact.content_hash == artifact.metadata_["output_hash"]
    assert "Deliver the retained MVP." in record.markdown
    assert "Implement review" in record.markdown
    assert "engineering" in record.markdown and "tests" in record.markdown and "source-verified" in record.markdown
    assert (tmp_path / record.export_path).read_text() == record.markdown


@pytest.mark.asyncio
async def test_g3_projection_has_readable_sections_and_canonical_json_appendix(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)

    record = await PlanReviewService().ensure(db, goal, run, artifact)

    assert "## Summary" in record.markdown
    assert "## Plan items" in record.markdown
    assert "```json" in record.markdown
    assert record.markdown.index("```json") > record.markdown.index("## Plan items")


@pytest.mark.asyncio
async def test_g3_projection_keeps_named_and_unresolved_owners_readable_and_stable(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    agent_id = artifact.metadata_["plan_items"][0]["agent_id"]
    db.agents[agent_id] = SimpleNamespace(name="Named owner")
    service = PlanReviewService()

    first = await service.ensure(db, goal, run, artifact)
    db.agents.clear()
    replay = await service.ensure(db, goal, run, artifact)

    assert f"Named owner ({agent_id})" in first.markdown
    assert '- Forbidden work: ["live calls"]' in first.markdown
    assert replay.markdown == first.markdown

    db, goal, run, artifact = _candidate(tmp_path)
    unresolved = await PlanReviewService().ensure(db, goal, run, artifact)
    assert f"Unresolved agent ({artifact.metadata_['plan_items'][0]['agent_id']})" in unresolved.markdown
    artifact.metadata_["plan_items"][0]["agent_id"] = None
    artifact.metadata_["output_hash"] = artifact.content_hash = plan_output_hash({
        key: artifact.metadata_[key] for key in ("schema_version", "plan_kind", "summary", "plan_items")
    })
    no_owner = await PlanReviewService().ensure(db, goal, SimpleNamespace(id=uuid4(), plan_state={}), artifact)
    assert "Unassigned agent (blocked)" in no_owner.markdown


@pytest.mark.asyncio
async def test_g3_projection_rejects_unsafe_or_divergent_exports(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    service = PlanReviewService()
    first = await service.ensure(db, goal, run, artifact)
    path = tmp_path / first.export_path
    path.write_text("edited", encoding="utf-8")

    stale = await service.check(db, goal, run, artifact, boundary="review")
    assert stale.state == "stale" and stale.reason == "plan export diverged"


@pytest.mark.asyncio
async def test_g3_observed_mismatch_stays_stale_after_bytes_return(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    service = PlanReviewService()
    first = await service.ensure(db, goal, run, artifact)
    path = tmp_path / first.export_path
    path.write_text("edited", encoding="utf-8")
    await service.check(db, goal, run, artifact, boundary="approval")
    path.write_text(first.markdown, encoding="utf-8")

    result = await service.check(db, goal, run, artifact, expected=first.identity, boundary="consumption")
    assert result.state == "stale" and result.reason == "plan export diverged"


@pytest.mark.asyncio
async def test_g3_projection_rejects_symlink_export(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    service = PlanReviewService()
    identity, markdown = service._identity(goal, run, artifact, review_cycle=1)
    path = tmp_path / "plans" / f"goal-{goal.id}" / f"run-{run.id}"
    path.mkdir(parents=True)
    target = path / "plan-v2-review-1.md"
    target.symlink_to(tmp_path / "outside.md")

    record = await service.ensure(db, goal, run, artifact)
    assert identity.rendered_content_hash == record.identity.rendered_content_hash
    assert record.state == "stale" and record.reason == "plan export is uninspectable"
    assert artifact.path is None


@pytest.mark.asyncio
async def test_g3_canonical_metadata_change_stales_existing_projection(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    service = PlanReviewService()
    await service.ensure(db, goal, run, artifact)
    artifact.metadata_["summary"] = "tampered"
    digest = plan_output_hash({key: artifact.metadata_[key] for key in ("schema_version", "plan_kind", "summary", "plan_items")})
    artifact.metadata_["output_hash"] = digest
    artifact.content_hash = digest

    record = await service.check(db, goal, run, artifact, boundary="approval")
    assert record.state == "stale" and record.reason == "canonical plan content changed"


@pytest.mark.asyncio
async def test_g3_ingested_provenance_mismatch_stales_existing_anchor(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    run.plan_state["plan_provenance"] = _ingested_provenance(artifact)
    service = PlanReviewService()
    first = await service.ensure(db, goal, run, artifact)
    run.plan_state["plan_provenance"]["winning_session_id"] = str(uuid4())

    record = await service.check(db, goal, run, artifact, expected=first.identity, boundary="approval")

    assert record.state == "stale"
    assert run.plan_state["plan_reviews"][str(artifact.id)]["state"] == "stale"


@pytest.mark.asyncio
async def test_g3_invalid_first_provenance_never_exports(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    artifact.metadata_["plan_author_agent_id"] = str(uuid4())

    try:
        record = await PlanReviewService().ensure(db, goal, run, artifact)
    except ValueError:
        record = None

    assert record is None or record.state == "stale"
    assert not (tmp_path / "plans").exists()


@pytest.mark.asyncio
async def test_g3_identical_replay_preserves_existing_anchor(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    service = PlanReviewService()
    first = await service.ensure(db, goal, run, artifact)
    before = dict(run.plan_state["plan_reviews"])

    replay = await service.ensure(db, goal, run, artifact)

    assert replay == first
    assert run.plan_state["plan_reviews"] == before
    assert (tmp_path / first.export_path).read_text(encoding="utf-8") == first.markdown


@pytest.mark.asyncio
async def test_g3_projection_renders_full_envelope_without_markdown_spoofing(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    item = artifact.metadata_["plan_items"][0]
    item.update({
        "title": "# not a heading", "inputs": ["input-marker"], "required_capabilities": ["cap-marker"],
        "success_criterion_keys": ["criterion-marker"], "artifact_ids": [str(uuid4())],
        "adopt_task_id": str(uuid4()), "priority": 7, "why_runnable": "runnable-marker",
        "agent_id": None, "owner_type": "human",
        "human_prerequisite": {"question": "human-marker", "owner_user_id": str(uuid4())},
        "allocation": {"max_tokens": "1"}, "staging_boundary": {"identifier": "stage-marker"},
        "objective": "objective-marker", "success_criteria": [{"key": "criteria-marker"}],
        "constraints": {"constraint": "constraint-marker"},
    })
    artifact.metadata_["output_hash"] = artifact.content_hash = plan_output_hash({
        key: artifact.metadata_[key] for key in ("schema_version", "plan_kind", "summary", "plan_items")
    })

    record = await PlanReviewService().ensure(db, goal, run, artifact)

    for marker in ("input-marker", "cap-marker", "criterion-marker", "runnable-marker", "human-marker",
                   "stage-marker", "objective-marker", "criteria-marker", "constraint-marker", '"priority": 7'):
        assert marker in record.markdown
    assert "Forbidden work" in record.markdown and "live calls" in record.markdown
    assert "Human prerequisite (" in record.markdown
    assert "### not a heading" not in record.markdown


@pytest.mark.asyncio
async def test_g3_projection_refuses_ancestor_symlink_before_writing_outside_workspace(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    outside = tmp_path.parent / "outside-plans"
    outside.mkdir()
    (tmp_path / "plans").symlink_to(outside, target_is_directory=True)

    record = await PlanReviewService().ensure(db, goal, run, artifact)

    assert record.state == "stale" and record.reason == "plan export path is unsafe"
    assert not list(outside.iterdir())


@pytest.mark.asyncio
async def test_g3_ancestor_swap_during_publication_writes_no_external_copy(tmp_path, monkeypatch):
    db, goal, run, artifact = _candidate(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    original_link = os.link

    def swap_ancestor(source, target, **kwargs):
        plans = tmp_path / "plans"
        plans.rename(tmp_path / "plans-before-swap")
        plans.symlink_to(outside, target_is_directory=True)
        return original_link(source, target, **kwargs)

    monkeypatch.setattr("huddleroom.services.orchestration_plan_review.os.link", swap_ancestor)
    service = PlanReviewService()
    first = await service.ensure(db, goal, run, artifact)
    record = await service.check(db, goal, run, artifact, expected=first.identity, boundary="approval")

    assert record.state == "stale"
    assert not list(outside.iterdir())


@pytest.mark.asyncio
async def test_g3_interrupted_publication_leaves_no_partial_export(tmp_path, monkeypatch):
    db, goal, run, artifact = _candidate(tmp_path)

    def interrupt(*_args, **_kwargs):
        raise OSError("simulated publication interruption")

    monkeypatch.setattr("huddleroom.services.orchestration_plan_review.os.link", interrupt)
    record = await PlanReviewService().ensure(db, goal, run, artifact)

    assert record.state == "stale"
    assert not list((tmp_path / "plans").rglob(".plan-review-*"))
    assert not list((tmp_path / "plans").rglob("plan-v*-review-*.md"))


@pytest.mark.asyncio
async def test_g3_provenance_change_stales_existing_projection(tmp_path):
    db, goal, run, artifact = _candidate(tmp_path)
    service = PlanReviewService()
    await service.ensure(db, goal, run, artifact)
    run.plan_state = {
        **run.plan_state,
        "plan_provenance": {
            "artifact_id": str(artifact.id), "planning_task_id": str(artifact.linked_task_id),
            "planning_version": 2, "winning_session_id": artifact.metadata_["winning_session_id"],
            "plan_author_agent_id": artifact.metadata_["plan_author_agent_id"],
            "canonical_content_hash": artifact.content_hash,
        },
    }
    artifact.metadata_["winning_session_id"] = str(uuid4())

    record = await service.check(db, goal, run, artifact, boundary="approval")

    assert record.state == "stale" and record.reason == "canonical plan provenance changed"
