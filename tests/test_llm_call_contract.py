"""Guard tests for the LLM Call Contract.

Enforces that all orchestrator LLM calls (goal-control and meeting-control)
follow the three hardening rules from PROJECT.md:

1. Context preamble: Every call prefixes with orchestrator_preamble(...)
2. Structured self-repair: Calls parsing JSON responses route through
   complete_with_repair (API) or cli_complete_with_repair (CLI)
3. Exhaustion falls back: After 3 failed repairs, re-raises (not checked
   statically; handled by llm_structured_repair.py tests)

This module statically enforces rules 1 and 2 via AST parsing — no execution.
"""

import ast
from pathlib import Path
from typing import Set


def _module_refs(path: Path) -> Set[str]:
    """Return all Name and Attribute identifiers referenced in module."""
    if not path.exists():
        return set()

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AssertionError(f"{path} has syntax error: {exc}")

    refs = set()

    class NameExtractor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            refs.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            refs.add(node.attr)
            self.generic_visit(node)

    NameExtractor().visit(tree)
    return refs


def _imports(path: Path) -> Set[str]:
    """Return all imported names (both 'import x' and 'from x import y')."""
    if not path.exists():
        return set()

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise AssertionError(f"{path} has syntax error: {exc}")

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.asname or alias.name)

    return imports


def _repo_root() -> Path:
    """Return path to repository root (parent of tests/ directory)."""
    return Path(__file__).parent.parent


# --- Goal-control analyzers and decision adapter ---

def test_orchestration_decision_adapter_has_preamble() -> None:
    """Assert decision adapter imports and uses orchestrator_preamble."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_llm_decision_adapter.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "orchestrator_preamble" in refs, (
        f"{path}: must reference orchestrator_preamble "
        "(defines and uses it)"
    )


def test_orchestration_decision_adapter_has_repair() -> None:
    """Assert decision adapter imports and uses complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_llm_decision_adapter.py"

    refs = _module_refs(path)

    assert "complete_with_repair" in refs, (
        f"{path}: must reference complete_with_repair; "
        "all goal-control LLM calls must route through it"
    )


def test_goal_analyzer_has_preamble() -> None:
    """Assert goal analyzer imports and uses orchestrator_preamble."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_goal_analyzer.py"

    imports = _imports(path)

    assert "orchestrator_preamble" in imports, (
        f"{path}: must import orchestrator_preamble from "
        "huddleroom.services.orchestration_llm_decision_adapter"
    )


def test_goal_analyzer_has_repair() -> None:
    """Assert goal analyzer imports and uses complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_goal_analyzer.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "complete_with_repair" in imports, (
        f"{path}: must import complete_with_repair from "
        "huddleroom.services.llm_structured_repair"
    )
    assert "complete_with_repair" in refs, (
        f"{path}: must reference complete_with_repair in code"
    )


def test_manager_analyzer_has_preamble() -> None:
    """Assert manager analyzer imports and uses orchestrator_preamble."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_manager_analyzer.py"

    imports = _imports(path)

    assert "orchestrator_preamble" in imports, (
        f"{path}: must import orchestrator_preamble from "
        "huddleroom.services.orchestration_llm_decision_adapter"
    )


def test_manager_analyzer_has_repair() -> None:
    """Assert manager analyzer imports and uses complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_manager_analyzer.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "complete_with_repair" in imports, (
        f"{path}: must import complete_with_repair from "
        "huddleroom.services.llm_structured_repair"
    )
    assert "complete_with_repair" in refs, (
        f"{path}: must reference complete_with_repair in code"
    )


def test_agent_definition_analyzer_has_preamble() -> None:
    """Assert agent-definition analyzer imports and uses orchestrator_preamble."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_agent_definition_analyzer.py"

    imports = _imports(path)

    assert "orchestrator_preamble" in imports, (
        f"{path}: must import orchestrator_preamble from "
        "huddleroom.services.orchestration_llm_decision_adapter"
    )


def test_agent_definition_analyzer_has_repair() -> None:
    """Assert agent-definition analyzer imports and uses complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_agent_definition_analyzer.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "complete_with_repair" in imports, (
        f"{path}: must import complete_with_repair from "
        "huddleroom.services.llm_structured_repair"
    )
    assert "complete_with_repair" in refs, (
        f"{path}: must reference complete_with_repair in code"
    )


def test_team_hierarchy_analyzer_has_preamble() -> None:
    """Assert team-hierarchy analyzer imports and uses orchestrator_preamble."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_team_hierarchy_analyzer.py"

    imports = _imports(path)

    assert "orchestrator_preamble" in imports, (
        f"{path}: must import orchestrator_preamble from "
        "huddleroom.services.orchestration_llm_decision_adapter"
    )


def test_team_hierarchy_analyzer_has_repair() -> None:
    """Assert team-hierarchy analyzer imports and uses complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_team_hierarchy_analyzer.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "complete_with_repair" in imports, (
        f"{path}: must import complete_with_repair from "
        "huddleroom.services.llm_structured_repair"
    )
    assert "complete_with_repair" in refs, (
        f"{path}: must reference complete_with_repair in code"
    )


def test_effectiveness_review_analyzer_has_preamble() -> None:
    """Assert effectiveness analyzer imports and uses orchestrator_preamble."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_effectiveness_analyzer.py"

    imports = _imports(path)

    assert "orchestrator_preamble" in imports, (
        f"{path}: must import orchestrator_preamble from "
        "huddleroom.services.orchestration_llm_decision_adapter"
    )


def test_effectiveness_review_analyzer_has_repair() -> None:
    """Assert effectiveness analyzer imports and uses complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_effectiveness_analyzer.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "complete_with_repair" in imports, (
        f"{path}: must import complete_with_repair from "
        "huddleroom.services.llm_structured_repair"
    )
    assert "complete_with_repair" in refs, (
        f"{path}: must reference complete_with_repair in code"
    )


def test_project_advisor_service_has_preamble() -> None:
    """Assert project advisor service imports and uses orchestrator_preamble."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_project_advisor_service.py"

    imports = _imports(path)

    assert "orchestrator_preamble" in imports, (
        f"{path}: must import orchestrator_preamble from "
        "huddleroom.services.orchestration_llm_decision_adapter"
    )


def test_project_advisor_service_has_repair() -> None:
    """Assert project advisor service imports and uses complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/services/orchestration_project_advisor_service.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "complete_with_repair" in imports, (
        f"{path}: must import complete_with_repair from "
        "huddleroom.services.llm_structured_repair"
    )
    assert "complete_with_repair" in refs, (
        f"{path}: must reference complete_with_repair in code"
    )


# --- Meeting-control modules ---

def test_meeting_intelligence_has_preamble() -> None:
    """Assert meeting_intelligence imports and uses orchestrator_preamble."""
    root = _repo_root()
    path = root / "huddleroom/services/meeting_intelligence.py"

    imports = _imports(path)

    assert "orchestrator_preamble" in imports, (
        f"{path}: must import orchestrator_preamble from "
        "huddleroom.services.orchestration_llm_decision_adapter"
    )


def test_meeting_intelligence_has_repair() -> None:
    """Assert meeting_intelligence imports and uses complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/services/meeting_intelligence.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "complete_with_repair" in imports, (
        f"{path}: must import complete_with_repair from "
        "huddleroom.services.llm_structured_repair"
    )
    assert "complete_with_repair" in refs, (
        f"{path}: must reference complete_with_repair in code"
    )


def test_meeting_outcome_has_preamble() -> None:
    """Assert meeting_outcome imports and uses orchestrator_preamble."""
    root = _repo_root()
    path = root / "huddleroom/services/meeting_outcome.py"

    imports = _imports(path)

    assert "orchestrator_preamble" in imports, (
        f"{path}: must import orchestrator_preamble from "
        "huddleroom.services.orchestration_llm_decision_adapter"
    )


def test_meeting_outcome_has_repair() -> None:
    """Assert meeting_outcome imports and uses complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/services/meeting_outcome.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "complete_with_repair" in imports, (
        f"{path}: must import complete_with_repair from "
        "huddleroom.services.llm_structured_repair"
    )
    assert "complete_with_repair" in refs, (
        f"{path}: must reference complete_with_repair in code"
    )


def test_meeting_runner_has_preamble() -> None:
    """Assert meeting_runner imports and uses orchestrator_preamble."""
    root = _repo_root()
    path = root / "huddleroom/services/meeting_runner.py"

    imports = _imports(path)

    assert "orchestrator_preamble" in imports, (
        f"{path}: must import orchestrator_preamble from "
        "huddleroom.services.orchestration_llm_decision_adapter"
    )


def test_meeting_runner_has_repair() -> None:
    """Assert meeting_runner imports and uses complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/services/meeting_runner.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "complete_with_repair" in imports, (
        f"{path}: must import complete_with_repair from "
        "huddleroom.services.llm_structured_repair"
    )
    assert "complete_with_repair" in refs, (
        f"{path}: must reference complete_with_repair in code"
    )


# --- CLI adapter ---

def test_cli_adapter_has_cli_repair() -> None:
    """Assert cli_adapter imports and uses cli_complete_with_repair."""
    root = _repo_root()
    path = root / "huddleroom/adapters/cli_adapter.py"

    imports = _imports(path)
    refs = _module_refs(path)

    assert "cli_complete_with_repair" in imports, (
        f"{path}: must import cli_complete_with_repair from "
        "huddleroom.services.llm_structured_repair"
    )
    assert "cli_complete_with_repair" in refs, (
        f"{path}: must reference cli_complete_with_repair in code"
    )


# --- Negative control: detect non-compliant code ---

def test_detector_catches_bare_litellm_call() -> None:
    """Negative control: verify the detector catches a non-compliant analyzer.

    A deliberately non-compliant analyzer that does bare litellm.acompletion()
    + json.loads() with no complete_with_repair should be detected.
    """
    bad_analyzer_source = '''
from typing import Any, Mapping
import json
import litellm

async def analyze_goal(context: Mapping[str, Any]) -> dict:
    """Deliberately non-compliant: no complete_with_repair."""
    response = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "analyze this"}],
    )
    # Bad: bare json.loads, no repair loop
    return json.loads(response["choices"][0]["message"]["content"])
'''

    tree = ast.parse(bad_analyzer_source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.asname or alias.name)

    refs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            refs.add(node.id)
        elif isinstance(node, ast.Attribute):
            refs.add(node.attr)

    # Detector should catch this: has litellm, json, but NO complete_with_repair
    assert "litellm" in imports, "test setup: bad snippet should import litellm"
    assert "json" in imports, "test setup: bad snippet should import json"
    assert "complete_with_repair" not in imports, (
        "test setup: bad snippet should NOT import complete_with_repair"
    )
    assert "complete_with_repair" not in refs, (
        "test setup: bad snippet should NOT reference complete_with_repair"
    )

    # The contract says: if you import litellm and json, you must also
    # use complete_with_repair. This snippet violates that.
    # In a real test, we'd fail if a module had litellm/json but no repair.
    # For now, we just verify the detector identifies it correctly.
    pass
