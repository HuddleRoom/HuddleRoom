import uuid
from types import SimpleNamespace

from huddleroom.schemas.orchestration import OrchestrationDelegationContract
from huddleroom.services.orchestration_service import OrchestrationService


def test_delegation_description_renders_criterion_inputs_with_descriptions():
    goal = SimpleNamespace(
        objective="Ship it",
        orchestrator_context={},
        success_criteria=[{"key": "criterion_1", "id": "c1", "description": "Login works"}, "bad-entry"],
    )
    contract = OrchestrationDelegationContract(
        goal_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        action_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
        work_function="build",
        scope="s",
        deliverable="d",
        inputs=["criterion:criterion_1", "criterion:unknown", "plain input"],
    )

    text = OrchestrationService._delegation_task_description(goal, contract)

    assert "- criterion:criterion_1 — Login works" in text
    assert "- criterion:unknown\n" in text + "\n"
    assert "- plain input" in text
    assert contract.inputs == ["criterion:criterion_1", "criterion:unknown", "plain input"]
