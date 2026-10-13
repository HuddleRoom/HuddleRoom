import pytest
from fastapi import HTTPException

from huddleroom.services.orchestration_roadmap_service import parse_roadmap_items


@pytest.mark.parametrize("allocation", [{"max_tokens": 0}, {"max_tokens": "0", "max_turns": 0}])
def test_child_allocation_must_reserve_nonzero_capacity(allocation):
    child = {"item_key": "child", "unit_type": "goal", "title": "Child", "objective": "Deliver child",
             "success_criteria": [{"key": "done"}], "allocation": allocation}
    with pytest.raises(HTTPException, match="nonzero|positive|greater than zero"):
        parse_roadmap_items([child], set(allocation))


@pytest.mark.parametrize("field,value", [("scope", "Implement"), ("deliverable", "Candidate"),
                                        ("work_function", "implementation")])
def test_goal_item_rejects_task_fields_even_with_valid_allocation(field, value):
    child = {"item_key": "child", "unit_type": "goal", "title": "Child", "objective": "Deliver child",
             "success_criteria": [{"key": "done"}], "allocation": {"max_tokens": 1}, field: value}
    with pytest.raises(HTTPException, match="goal item cannot define task-only fields"):
        parse_roadmap_items([child], {"max_tokens"})


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
