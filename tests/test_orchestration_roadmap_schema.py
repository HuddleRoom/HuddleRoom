import pytest
from fastapi import HTTPException

from huddleroom.services.orchestration_roadmap_service import parse_roadmap_items


def _task(key, depends_on=()):
    return {
        "item_key": key,
        "unit_type": "task",
        "title": key,
        "depends_on": list(depends_on),
        "mutates_shared_state": True,
        "staging_boundary": {"type": "git_worktree", "identifier": f"roadmap/{key}", "reversible": True},
        "work_function": "implementation",
        "scope": f"Implement {key}",
        "deliverable": f"Verified {key}",
    }


def test_roadmap_dag_normalizes_dependencies():
    items = parse_roadmap_items([_task("a"), _task("b", [" a ", "a"])], {"max_tokens"})

    assert items[1].depends_on == ["a"]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([_task("a"), _task("a")], "Duplicate roadmap item_key 'a'"),
        ([_task("a", ["missing"])], "depends on unknown item 'missing'"),
        ([_task("a", ["a"])], "cannot depend on itself"),
        ([_task("a", ["b"]), _task("b", ["a"])], "Roadmap dependency cycle detected"),
    ],
)
def test_roadmap_dag_rejects_invalid_graph(items, message):
    with pytest.raises(HTTPException, match=message):
        parse_roadmap_items(items, {"max_tokens"})


def test_goal_item_rejects_unmeasured_budget_dimension():
    raw = {
        "item_key": "child",
        "unit_type": "goal",
        "title": "Child",
        "depends_on": [],
        "mutates_shared_state": False,
        "objective": "Deliver bounded child",
        "success_criteria": [{"key": "done"}],
        "constraints": {},
        "allocation": {"max_cost_usd": "1"},
    }

    with pytest.raises(HTTPException, match="Unsupported measured budget dimension"):
        parse_roadmap_items([raw], {"max_cost_usd"})
