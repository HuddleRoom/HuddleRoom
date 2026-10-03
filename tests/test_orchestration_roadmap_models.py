import uuid

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import DropConstraint

from huddleroom.models.artifact import Artifact
from huddleroom.models.orchestration import (
    OrchestrationGoal,
    OrchestrationGate,
    OrchestrationRoadmapItem,
    OrchestrationRoadmapVersion,
    OrchestrationRun,
)
from huddleroom.models.task import Task


@pytest.mark.unsupported_mode
def test_deferred_roadmap_version_fk_has_postgresql_drop_name():
    constraint = next(
        foreign_key.constraint
        for foreign_key in OrchestrationGoal.__table__.foreign_keys
        if foreign_key.parent.name == "roadmap_version_id"
    )

    ddl = str(DropConstraint(constraint).compile(dialect=postgresql.dialect()))

    assert ddl == "ALTER TABLE orchestration_goals DROP CONSTRAINT fk_orch_goals_roadmap_version_id"


async def _goal(db, project_id, **lineage):
    goal = OrchestrationGoal(
        project_id=project_id,
        objective="Deliver roadmap",
        original_request="Deliver roadmap",
        success_criteria=[],
        constraints={},
        budget={"caps": {"max_tokens": 1000}},
        goal_type="roadmap",
        **lineage,
    )
    db.add(goal)
    await db.flush()
    return goal


@pytest.mark.asyncio
async def test_child_lineage_must_be_all_null_or_all_present(db_session, test_project):
    parent = await _goal(db_session, test_project.id)
    child = OrchestrationGoal(
        project_id=test_project.id,
        objective="Child",
        original_request="Child",
        success_criteria=[],
        constraints={},
        budget={},
        goal_type="outcome",
        parent_goal_id=parent.id,
    )
    db_session.add(child)

    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_roadmap_version_per_goal_is_unique(db_session, test_project):
    parent = await _goal(db_session, test_project.id)
    run = OrchestrationRun(goal_id=parent.id, phase="authorized")
    artifact = Artifact(
        project_id=test_project.id,
        name="Roadmap V1",
        artifact_type="plan",
        metadata_={"plan_items": []},
    )
    db_session.add_all((run, artifact))
    await db_session.flush()
    version = OrchestrationRoadmapVersion(
        goal_id=parent.id,
        run_id=run.id,
        version=1,
        plan_artifact_id=artifact.id,
        snapshot={"schema_version": 1, "items": []},
        fingerprint="a" * 64,
        approval_reference={"kind": "human", "id": str(uuid.uuid4())},
    )
    db_session.add(version)
    await db_session.flush()
    db_session.add(
        OrchestrationRoadmapVersion(
            goal_id=parent.id,
            run_id=run.id,
            version=1,
            plan_artifact_id=artifact.id,
            snapshot={"schema_version": 1, "items": []},
            fingerprint="b" * 64,
            approval_reference={"kind": "human", "id": str(uuid.uuid4())},
        )
    )

    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_roadmap_item_key_is_unique_per_goal(db_session, test_project):
    parent = await _goal(db_session, test_project.id)
    run = OrchestrationRun(goal_id=parent.id, phase="authorized")
    artifact = Artifact(project_id=test_project.id, name="Roadmap V1", artifact_type="plan", metadata_={})
    db_session.add_all((run, artifact))
    await db_session.flush()
    version = OrchestrationRoadmapVersion(
        goal_id=parent.id,
        run_id=run.id,
        version=1,
        plan_artifact_id=artifact.id,
        snapshot={"schema_version": 1, "items": []},
        fingerprint="a" * 64,
        approval_reference={"kind": "human", "id": str(uuid.uuid4())},
    )
    first_task = Task(project_id=test_project.id, title="First")
    first_gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="build", gate_type="item", required_evidence={}
    )
    db_session.add_all((version, first_task, first_gate))
    await db_session.flush()
    db_session.add(
        OrchestrationRoadmapItem(
            goal_id=parent.id,
            first_version_id=version.id,
            item_key="build",
            unit_type="task",
            item_snapshot={},
            task_id=first_task.id,
            gate_id=first_gate.id,
        )
    )
    await db_session.flush()
    second_task = Task(project_id=test_project.id, title="Second")
    second_gate = OrchestrationGate(
        run_id=run.id, success_criterion_key="build", gate_type="item", required_evidence={}
    )
    db_session.add_all((second_task, second_gate))
    await db_session.flush()
    db_session.add(
        OrchestrationRoadmapItem(
            goal_id=parent.id,
            first_version_id=version.id,
            item_key="build",
            unit_type="task",
            item_snapshot={},
            task_id=second_task.id,
            gate_id=second_gate.id,
        )
    )

    with pytest.raises(IntegrityError):
        await db_session.flush()


@pytest.mark.asyncio
async def test_roadmap_version_lineage_is_immutable(db_session, test_project):
    parent = await _goal(db_session, test_project.id)
    run = OrchestrationRun(goal_id=parent.id, phase="authorized")
    artifact = Artifact(project_id=test_project.id, name="Roadmap V1", artifact_type="plan", metadata_={})
    db_session.add_all((run, artifact))
    await db_session.flush()
    version = OrchestrationRoadmapVersion(
        goal_id=parent.id,
        run_id=run.id,
        version=1,
        plan_artifact_id=artifact.id,
        snapshot={"schema_version": 1, "items": []},
        fingerprint="a" * 64,
        approval_reference={"kind": "human", "id": str(uuid.uuid4())},
    )
    db_session.add(version)
    await db_session.flush()

    with pytest.raises(ValueError, match="roadmap lineage is immutable"):
        version.snapshot = {"schema_version": 2, "items": []}
