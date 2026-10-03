import pytest
import pytest_asyncio
import uuid
from pathlib import Path

@pytest.mark.asyncio
async def test_load_yaml_protocol(db_session, test_project):
    from huddleroom.services.protocol_service import ProtocolService
    from huddleroom.models.protocol import Protocol
    from sqlalchemy import select

    svc = ProtocolService()
    yaml_path = Path("workspace/protocols/code_review.yaml")
    proto = await svc.load_from_yaml(db_session, yaml_path, project_id=None)

    assert proto.name == "code_review"
    assert proto.version == "1.0"
    assert len(proto.triggers) >= 1
    assert proto.triggers[0]["event_type"] == "code.pr_opened"
    assert proto.escalation_chain == "standard_dev_escalation"

@pytest.mark.asyncio
async def test_list_protocols(db_session, test_project):
    from huddleroom.services.protocol_service import ProtocolService

    svc = ProtocolService()
    yaml_path = Path("workspace/protocols/bug_fix.yaml")
    await svc.load_from_yaml(db_session, yaml_path, project_id=None)

    protocols = await svc.list(db_session, project_id=None)
    names = [p.name for p in protocols]
    assert "bug_fix" in names


@pytest.mark.asyncio
async def test_create_artifact(db_session, test_project):
    from huddleroom.services.artifact_service import ArtifactService
    svc = ArtifactService()
    artifact = await svc.create(
        db_session,
        project_id=test_project.id,
        name="feature-branch-pr",
        artifact_type="pull_request",
        url="https://github.com/org/repo/pull/42",
        metadata={"branch": "feature/auth", "pr_id": "42"},
    )
    assert artifact.id is not None
    assert artifact.content_hash is None


@pytest.mark.asyncio
async def test_update_artifact_hash(db_session, tmp_path, test_project):
    from huddleroom.services.artifact_service import ArtifactService
    svc = ArtifactService()

    f = tmp_path / "spec.yaml"
    f.write_text("openapi: 3.0.0\ninfo:\n  title: Test API")

    artifact = await svc.create(
        db_session, project_id=test_project.id,
        name="api-spec", artifact_type="api_spec",
        path=str(f), metadata={},
    )
    result = await svc.update_content_hash(db_session, artifact)
    assert result.content_hash is not None
    old_hash = result.content_hash

    f.write_text("openapi: 3.0.0\ninfo:\n  title: Modified API")
    result2 = await svc.update_content_hash(db_session, artifact)
    assert result2.content_hash != old_hash
    assert result2.previous_hash == old_hash


@pytest.mark.asyncio
async def test_add_watcher(db_session, test_project, test_agent):
    from huddleroom.services.artifact_service import ArtifactService
    svc = ArtifactService()
    artifact = await svc.create(
        db_session, project_id=test_project.id,
        name="pr-1", artifact_type="pull_request", url="http://x", metadata={},
    )
    watcher = await svc.add_watcher(
        db_session, artifact_id=artifact.id,
        watcher_kind="agent", watcher_id=test_agent.id,
        event_filter=["artifact.content_changed"],
    )
    assert watcher.id is not None

    watchers = await svc.list_watchers(db_session, artifact.id)
    assert len(watchers) == 1


@pytest.mark.asyncio
async def test_load_escalation_chain(db_session):
    from huddleroom.services.escalation_service import EscalationChainService
    from pathlib import Path

    svc = EscalationChainService()
    chain = await svc.load_from_yaml(
        db_session, Path("workspace/escalation_chains/standard_dev_escalation.yaml")
    )
    assert chain.name == "standard_dev_escalation"
    assert len(chain.steps) == 3
    assert chain.steps[0]["step"] == 1
    assert chain.steps[-1]["action"] == "human_notify"
