import pytest
import pytest_asyncio
import uuid
from pathlib import Path

@pytest.mark.asyncio
async def test_load_yaml_graph(db_session, test_project):
    from huddleroom.services.graph_service import GraphService
    from huddleroom.models.graph import Graph
    from sqlalchemy import select

    svc = GraphService()
    yaml_path = Path("workspace/graphs/code_review.yaml")
    graph = await svc.load_from_yaml(db_session, yaml_path, project_id=None)

    assert graph.name == "code_review"
    assert graph.version == "1.0"
    assert len(graph.triggers) >= 1
    assert graph.triggers[0]["event_type"] == "code.pr_opened"
    assert graph.escalation_chain == "standard_dev_escalation"

@pytest.mark.asyncio
async def test_list_graphs(db_session, test_project):
    from huddleroom.services.graph_service import GraphService

    svc = GraphService()
    yaml_path = Path("workspace/graphs/bug_fix.yaml")
    await svc.load_from_yaml(db_session, yaml_path, project_id=None)

    graphs = await svc.list(db_session, project_id=None)
    names = [p.name for p in graphs]
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


_LEGACY_YAMLS = {
    "states": "name: legacy\nstates:\n  a: {}\ninitial_state: a\nterminal_states: {success: [a]}\n",
    "node_transitions": (
        "name: legacy\nstart_node: a\nnodes:\n  a:\n    transitions:\n      - to: a\n"
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", list(_LEGACY_YAMLS))
async def test_load_yaml_rejects_legacy_keys(db_session, tmp_path, case):
    from huddleroom.services.graph_service import GraphService

    path = tmp_path / "legacy.yaml"
    path.write_text(_LEGACY_YAMLS[case])
    with pytest.raises(ValueError, match="legacy"):
        await GraphService().load_from_yaml(db_session, path, project_id=None)


@pytest.mark.asyncio
async def test_load_all_ignores_legacy_protocols_dir_with_warning(db_session, tmp_path, caplog):
    from huddleroom.services.graph_service import GraphService

    (tmp_path / "protocols").mkdir()
    (tmp_path / "protocols" / "old.yaml").write_text(_LEGACY_YAMLS["states"])
    with caplog.at_level("WARNING"):
        loaded = await GraphService().load_all_from_workspace(db_session, tmp_path)
    assert loaded == []
    assert "workspace/protocols/ is no longer supported" in caplog.text


@pytest.mark.asyncio
async def test_load_all_resilience_continues_after_file_failure(db_session, tmp_path, caplog):
    """Test that loader continues after a malformed file and loads valid files."""
    from huddleroom.services.graph_service import GraphService

    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()

    # Create a valid graph
    valid_yaml = """
name: valid_graph
start_node: start
nodes:
  start:
    edges: []
terminal_nodes:
  success: [start]
"""
    (graphs_dir / "01_valid.yaml").write_text(valid_yaml)

    # Create a legacy graph that will fail validation
    legacy_yaml = """
name: legacy_graph
states:
  start: {}
initial_state: start
terminal_states:
  success: [start]
"""
    (graphs_dir / "02_legacy.yaml").write_text(legacy_yaml)

    with caplog.at_level("ERROR"):
        loaded = await GraphService().load_all_from_workspace(db_session, tmp_path)

    # Should load only the valid graph
    assert len(loaded) == 1
    assert loaded[0].name == "valid_graph"
    # Logger should have recorded the failure
    assert "Failed to load graph" in caplog.text


@pytest.mark.asyncio
async def test_validate_rejects_legacy_action_type_update_protocol_context(db_session, tmp_path):
    """Test validation rejects update_protocol_context action type."""
    legacy_def = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "on_enter": [
                    {"action_type": "update_protocol_context", "context_key": "x"}
                ],
                "edges": []
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    from huddleroom.services.graph_service import validate_graph_definition
    with pytest.raises(ValueError, match="update_protocol_context.*renamed to.*update_graph_context"):
        validate_graph_definition(legacy_def)


@pytest.mark.asyncio
async def test_validate_rejects_legacy_trigger_events(db_session):
    """Test validation rejects legacy trigger event types."""
    from huddleroom.services.graph_service import validate_graph_definition

    legacy_def = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "edges": [
                    {
                        "name": "e1",
                        "to": "a",
                        "trigger_event": "protocol.instance_started"
                    }
                ]
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    with pytest.raises(ValueError, match="protocol.instance_started.*renamed to.*graph.run_started"):
        validate_graph_definition(legacy_def)


@pytest.mark.asyncio
async def test_validate_rejects_legacy_orchestration_trigger_events(db_session):
    """Test validation rejects legacy orchestration.protocol_* trigger events."""
    from huddleroom.services.graph_service import validate_graph_definition

    legacy_def = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "edges": [
                    {
                        "name": "e1",
                        "to": "a",
                        "trigger_event": "orchestration.protocol_started"
                    }
                ]
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    with pytest.raises(ValueError, match="orchestration.protocol_started.*renamed to.*orchestration.graph_started"):
        validate_graph_definition(legacy_def)


@pytest.mark.asyncio
async def test_validate_rejects_legacy_action_in_edge(db_session):
    """Test validation rejects legacy action types in edge actions."""
    from huddleroom.services.graph_service import validate_graph_definition

    legacy_def = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "edges": [
                    {
                        "name": "e1",
                        "to": "a",
                        "trigger_event": "test.event",
                        "actions": [
                            {"action_type": "update_protocol_context"}
                        ]
                    }
                ]
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    with pytest.raises(ValueError, match="update_protocol_context.*renamed to.*update_graph_context"):
        validate_graph_definition(legacy_def)


@pytest.mark.asyncio
async def test_validate_rejects_legacy_event_type_in_triggers(db_session):
    """Test validation rejects legacy event_type in top-level triggers."""
    from huddleroom.services.graph_service import validate_graph_definition

    legacy_def = {
        "name": "bad",
        "start_node": "a",
        "nodes": {"a": {"edges": []}},
        "terminal_nodes": {"success": ["a"]},
        "triggers": [
            {"event_type": "protocol.instance_started"}
        ]
    }

    with pytest.raises(ValueError, match="protocol.instance_started.*renamed to.*graph.run_started"):
        validate_graph_definition(legacy_def)


@pytest.mark.asyncio
async def test_validate_rejects_non_string_action_type(db_session):
    """Test validation rejects non-string action_type."""
    from huddleroom.services.graph_service import validate_graph_definition

    legacy_def = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "on_enter": [
                    {"action_type": None}
                ],
                "edges": []
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    with pytest.raises(ValueError, match="action_type must be a string"):
        validate_graph_definition(legacy_def)


@pytest.mark.asyncio
async def test_validate_rejects_non_string_trigger_event(db_session):
    """Test validation rejects non-string trigger_event."""
    from huddleroom.services.graph_service import validate_graph_definition

    legacy_def = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "edges": [
                    {
                        "name": "e1",
                        "to": "a",
                        "trigger_event": 123
                    }
                ]
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    with pytest.raises(ValueError, match="trigger_event must be a string"):
        validate_graph_definition(legacy_def)


@pytest.mark.asyncio
async def test_validate_rejects_legacy_template_tokens(db_session):
    """Test validation rejects legacy template token patterns."""
    from huddleroom.services.graph_service import validate_graph_definition

    # Test {{protocol_ pattern
    legacy_def = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "on_enter": [
                    {"action_type": "post_message", "template": "{{protocol_name}}"}
                ],
                "edges": []
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    with pytest.raises(ValueError, match="legacy 'protocol_' prefix"):
        validate_graph_definition(legacy_def)

    # Test /protocol-instances/ pattern
    legacy_def2 = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "on_enter": [
                    {"action_type": "post_message", "url": "/dashboard/protocol-instances/123"}
                ],
                "edges": []
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    with pytest.raises(ValueError, match="'/protocol-instances/' has been renamed"):
        validate_graph_definition(legacy_def2)


@pytest.mark.asyncio
async def test_validate_rejects_spaced_legacy_template_tokens(db_session):
    """Test validation rejects spaced variants like {{ protocol_ and {{ current_state."""
    from huddleroom.services.graph_service import validate_graph_definition

    # Test {{ protocol_ (with space)
    legacy_def = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "on_enter": [
                    {"action_type": "post_message", "template": "{{ protocol_name }}"}
                ],
                "edges": []
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    with pytest.raises(ValueError, match="legacy 'protocol_' prefix"):
        validate_graph_definition(legacy_def)

    # Test {{ current_state (with space)
    legacy_def2 = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "on_enter": [
                    {"action_type": "post_message", "template": "Current: {{ current_state }}"}
                ],
                "edges": []
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    with pytest.raises(ValueError, match="current_state.*renamed to.*current_node"):
        validate_graph_definition(legacy_def2)


@pytest.mark.asyncio
async def test_validate_accepts_description_with_legacy_tokens(db_session):
    """Test validation skips description fields (free text documentation)."""
    from huddleroom.services.graph_service import validate_graph_definition

    # Description field can mention legacy syntax as documentation
    good_def = {
        "name": "ok",
        "description": "This graph was formerly using {{protocol_name}} and /protocol-instances/; now uses graphs",
        "start_node": "a",
        "nodes": {
            "a": {
                "description": "Legacy terms {{current_state}} mentioned in docs are OK",
                "edges": []
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    # Should not raise
    validate_graph_definition(good_def)


@pytest.mark.asyncio
async def test_validate_accepts_false_positive_current_stated(db_session):
    """Test validation doesn't flag {{current_stated}} (word boundary prevents false positive)."""
    from huddleroom.services.graph_service import validate_graph_definition

    # {{current_stated}} should NOT match the word boundary check
    good_def = {
        "name": "ok",
        "start_node": "a",
        "nodes": {
            "a": {
                "on_enter": [
                    {"action_type": "post_message", "template": "Value: {{current_stated}}"}
                ],
                "edges": []
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    # Should not raise
    validate_graph_definition(good_def)


@pytest.mark.asyncio
async def test_validate_rejects_legacy_timeout_action(db_session):
    """Test validation rejects legacy timeout.action."""
    from huddleroom.services.graph_service import validate_graph_definition

    legacy_def = {
        "name": "bad",
        "start_node": "a",
        "nodes": {
            "a": {
                "timeout": {
                    "duration": "1h",
                    "action": "start_protocol"
                },
                "edges": []
            }
        },
        "terminal_nodes": {"success": ["a"]}
    }

    with pytest.raises(ValueError, match="start_protocol.*renamed to.*start_graph"):
        validate_graph_definition(legacy_def)


@pytest.mark.asyncio
async def test_load_all_resilience_with_flush_error(db_session, tmp_path, monkeypatch):
    """Test that loader continues after a file causes a flush error and db stays usable."""
    from huddleroom.services.graph_service import GraphService
    from unittest.mock import AsyncMock

    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()

    # Create a valid graph
    (graphs_dir / "01_valid.yaml").write_text("""
name: valid1
start_node: a
nodes:
  a:
    edges: []
terminal_nodes:
  success: [a]
""")

    # Create another valid graph
    (graphs_dir / "03_valid.yaml").write_text("""
name: valid2
start_node: b
nodes:
  b:
    edges: []
terminal_nodes:
  success: [b]
""")

    svc = GraphService()
    original_load = svc.load_from_yaml

    # Track which files we've attempted
    attempted = []

    async def patched_load_from_yaml(db, path, project_id=None):
        attempted.append(path.name)
        result = await original_load(db, path, project_id=project_id)
        # Simulate flush error on the second file attempt (valid2)
        if "03_" in path.name:
            # Add to db, then force flush error by modifying to invalid state
            raise RuntimeError("Simulated flush error")
        return result

    monkeypatch.setattr(svc, "load_from_yaml", patched_load_from_yaml)

    # Load all - should get 1 valid, skip 1 with error
    loaded = await svc.load_all_from_workspace(db_session, tmp_path)

    assert len(loaded) == 1
    assert loaded[0].name == "valid1"
    # Verify both files were attempted
    assert "01_valid.yaml" in attempted
    assert "03_valid.yaml" in attempted

    # Verify db session is still usable by doing a commit
    await db_session.commit()
