from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pytest
from unittest.mock import AsyncMock

from tests.live import live_test_protocols


def _protocol_test_ids() -> list[str]:
    return [test_id for test_id, _ in live_test_protocols.PROTOCOL_TESTS]



def test_protocol_live_defaults_keep_live_agent_tests_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["live_test_protocols.py"])

    args = live_test_protocols.parse_args()

    assert args.tests == "PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10"
    assert args.reuse_project is False


def test_protocol_live_parse_args_accepts_project_reuse_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["live_test_protocols.py", "--reuse-project"])

    args = live_test_protocols.parse_args()

    assert args.reuse_project is True


def test_build_project_name_uses_stable_base_with_run_suffix(monkeypatch: pytest.MonkeyPatch):
    fixed_now = datetime(2026, 5, 20, 10, 30, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(live_test_protocols, "datetime", type("FrozenDateTime", (), {"now": staticmethod(lambda tz=None: fixed_now)}))
    monkeypatch.setattr(live_test_protocols.uuid, "uuid4", lambda: "12345678-1234-5678-9abc-def012345678")

    project_name = live_test_protocols.build_project_name()

    assert project_name == "huddleroom-protocol-live-test-20260520t103045z-12345678"


def test_build_project_name_reuse_mode_returns_stable_name():
    assert live_test_protocols.build_project_name(reuse_project=True) == live_test_protocols.PROJECT_NAME


def test_setup_fixtures_uses_generated_project_name_by_default(monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []

    class FakeClient:
        def get_or_create_project(self, name: str) -> dict:
            seen.append(name)
            return {"id": "project-1", "name": name}

        def get_or_create_agent(self, defn: dict) -> dict:
            return {"id": f"{defn['name']}-id"}

    monkeypatch.setattr(live_test_protocols, "build_project_name", lambda reuse_project=False: "huddleroom-protocol-live-test-run")

    project, agents = live_test_protocols.setup_fixtures(FakeClient())

    assert project == {"id": "project-1", "name": "huddleroom-protocol-live-test-run"}
    assert seen == ["huddleroom-protocol-live-test-run"]
    assert set(agents) == {agent["name"] for agent in live_test_protocols.PROTO_AGENTS}


def test_setup_fixtures_can_reuse_stable_project_name():
    seen: list[str] = []

    class FakeClient:
        def get_or_create_project(self, name: str) -> dict:
            seen.append(name)
            return {"id": "project-1", "name": name}

        def get_or_create_agent(self, defn: dict) -> dict:
            return {"id": f"{defn['name']}-id"}

    live_test_protocols.setup_fixtures(FakeClient(), reuse_project=True)

    assert seen == [live_test_protocols.PROJECT_NAME]


def test_rally_client_paged_get_items_consumes_all_pages():
    client = live_test_protocols.RallyClient("http://example.test")
    responses = [
        {"items": [{"id": "first"}], "next_cursor": "cursor-1"},
        {"items": [{"id": "second"}], "next_cursor": None},
    ]
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_get(path: str, **params: object) -> dict:
        calls.append((path, dict(params)))
        return responses.pop(0)

    client._get = fake_get  # type: ignore[method-assign]

    items = client._paged_get_items("/api/v1/sessions", project_id="project-1")

    assert items == [{"id": "first"}, {"id": "second"}]
    assert calls == [
        ("/api/v1/sessions", {"project_id": "project-1", "limit": 500}),
        ("/api/v1/sessions", {"project_id": "project-1", "limit": 500, "cursor": "cursor-1"}),
    ]


def test_get_or_create_project_reuses_match_beyond_first_page():
    client = live_test_protocols.RallyClient("http://example.test")
    seen: list[tuple[str, int]] = []

    def fake_paged_get_items(path: str, limit: int = live_test_protocols.DEFAULT_PAGE_LIMIT, **params: object) -> list[dict]:
        seen.append((path, limit))
        assert params == {}
        return [
            {"id": "project-1", "name": "unrelated-project"},
            {"id": "project-2", "name": live_test_protocols.PROJECT_NAME},
        ]

    client._paged_get_items = fake_paged_get_items  # type: ignore[method-assign]
    client._post = lambda *_args, **_kwargs: pytest.fail("project should be reused, not created")  # type: ignore[method-assign]

    project = client.get_or_create_project(live_test_protocols.PROJECT_NAME)

    assert project == {"id": "project-2", "name": live_test_protocols.PROJECT_NAME}
    assert seen == [("/api/v1/projects", 200)]


def test_wait_for_state_raises_diagnostic_timeout():
    client = live_test_protocols.RallyClient("http://example.test")
    client.get_instance = lambda project_id, instance_id: {  # type: ignore[method-assign]
        "id": instance_id,
        "current_state": "opened",
        "status": "active",
    }

    with pytest.raises(TimeoutError, match="expected state 'ready_for_review'"):
        live_test_protocols.wait_for_state(
            client,
            "project-1",
            "instance-1",
            "ready_for_review",
            timeout=0.01,
            interval=0,
        )


def test_wait_for_message_polls_until_protocol_message_is_visible():
    client = live_test_protocols.RallyClient("http://example.test")
    responses = [
        [{"id": "older", "metadata": {}}],
        [{"id": "target", "metadata": {"protocol_instance_id": "instance-1"}, "content": "ready"}],
    ]

    client.list_messages = lambda channel_id, limit=500: responses.pop(0)  # type: ignore[method-assign]

    message = live_test_protocols.wait_for_message(
        client,
        "channel-1",
        lambda item: item.get("metadata", {}).get("protocol_instance_id") == "instance-1",
        timeout=0.01,
        interval=0,
    )

    assert message["id"] == "target"


def test_start_server_forces_auth_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch):
    server_log_path = tmp_path / "server.log"
    seen: dict[str, object] = {}

    class FakeProc:
        returncode = 0

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, cwd, env, stdout, stderr):
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        seen["env"] = env
        seen["stdout"] = stdout
        seen["stderr"] = stderr
        return FakeProc()

    monkeypatch.setattr(live_test_protocols, "run_migrations", lambda project_root: None)
    monkeypatch.setattr(live_test_protocols, "check_server", lambda base_url: True)
    monkeypatch.setattr(live_test_protocols, "project_venv_executable", lambda *_: Path("/mock/huddleroom"))
    monkeypatch.setattr(live_test_protocols.subprocess, "Popen", fake_popen)

    proc, log_file = live_test_protocols.start_server(
        "http://localhost:8001",
        8001,
        live_test_protocols.PROJECT_ROOT,
        server_log_path,
    )

    try:
        assert isinstance(proc, FakeProc)
        assert seen["cmd"] == ["/mock/huddleroom", "serve", "--port", "8001"]
        assert seen["env"]["HUDDLEROOM_AUTH_ENABLED"] == "false"
        assert seen["env"]["HUDDLEROOM_API_BASE_URL"] == "http://localhost:8001"
    finally:
        log_file.close()


def test_check_provider_ready_requires_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    error = live_test_protocols.check_provider_ready("openai")

    assert error == "OPENAI_API_KEY is not set; OpenAI-backed PL tests cannot run"


def test_check_provider_ready_runs_openai_probe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_probe = AsyncMock()
    monkeypatch.setattr(live_test_protocols, "_probe_openai_readiness", mock_probe)

    error = live_test_protocols.check_provider_ready("openai")

    assert error is None
    mock_probe.assert_awaited_once_with()


def test_check_provider_ready_reports_unsupported_provider_action(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    error = live_test_protocols.check_provider_ready("anthropic")

    assert error == (
        "Provider 'anthropic' readiness probe is not implemented for PL tests; "
        "switch the live reviewer to OpenAI or skip PL* tests"
    )


@pytest.mark.parametrize(
    "exception_class,exception_kwargs,expected_error",
    [
        (
            live_test_protocols.litellm.AuthenticationError,
            {
                "message": "invalid key",
                "llm_provider": "openai",
                "model": live_test_protocols.OPENAI_READINESS_MODEL,
            },
            (
                "OpenAI probe failed authentication for model "
                f"{live_test_protocols.OPENAI_READINESS_MODEL}; check OPENAI_API_KEY and model access"
            ),
        ),
        (
            live_test_protocols.litellm.BadRequestError,
            {
                "message": "model not available to this project",
                "model": live_test_protocols.OPENAI_READINESS_MODEL,
                "llm_provider": "openai",
                "response": None,
            },
            (
                "OpenAI probe could not use model "
                f"{live_test_protocols.OPENAI_READINESS_MODEL}: "
                "model not available to this project"
            ),
        ),
        (
            live_test_protocols.litellm.APIConnectionError,
            {
                "message": "network down",
                "llm_provider": "openai",
                "model": live_test_protocols.OPENAI_READINESS_MODEL,
            },
            "OpenAI probe failed to reach provider: network down",
        ),
    ],
)
def test_check_provider_ready_reports_openai_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    exception_class,
    exception_kwargs,
    expected_error,
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        live_test_protocols,
        "_probe_openai_readiness",
        AsyncMock(side_effect=exception_class(**exception_kwargs)),
    )

    error = live_test_protocols.check_provider_ready("openai")

    assert error == expected_error


def test_build_synthetic_pr_metadata_uses_scenario_shape_instead_of_fixed_patch_copy():
    metadata = live_test_protocols.build_synthetic_pr_metadata("PL3 Payments Hotfix 321")

    assert metadata["branch"] == "hotfix/test/pl3-payments-hotfix-321"
    assert metadata["base_branch"] == "main"
    assert metadata["repository"] == live_test_protocols.DEFAULT_PR_REPOSITORY
    assert metadata["repository_url"].endswith(live_test_protocols.DEFAULT_PR_REPOSITORY)
    assert metadata["pr_number"] == 321
    assert metadata["pr_url"].endswith("/pull/321")
    assert metadata["title"] == "PL3 Payments Hotfix 321"
    assert metadata["summary"].startswith("Synthetic hotfix pull request for the payments scenario")
    assert len(metadata["change_summary"]) == 3
    assert all(isinstance(item, str) and item for item in metadata["change_summary"])
    assert len(metadata["test_plan"]) == 2
    assert len(metadata["risk_notes"]) == 2
    assert any("payments" in item.lower() for item in metadata["change_summary"] + metadata["risk_notes"])


def test_setup_pr_creates_synthetic_pull_request_metadata(monkeypatch: pytest.MonkeyPatch):
    task_calls: list[dict] = []
    artifact_calls: list[dict] = []

    class FakeClient:
        def create_task(self, project_id: str, title: str, description: str, metadata: dict) -> dict:
            task_calls.append(
                {
                    "project_id": project_id,
                    "title": title,
                    "description": description,
                    "metadata": metadata,
                }
            )
            return {"id": "task-1"}

        def create_artifact(
            self,
            project_id: str,
            name: str,
            artifact_type: str,
            metadata: dict,
            linked_task_id: str | None = None,
        ) -> dict:
            artifact_calls.append(
                {
                    "project_id": project_id,
                    "name": name,
                    "artifact_type": artifact_type,
                    "metadata": metadata,
                    "linked_task_id": linked_task_id,
                }
            )
            return {"id": "artifact-1", "metadata": metadata}

    monkeypatch.setattr(live_test_protocols.time, "sleep", lambda _: None)

    task, artifact = live_test_protocols.setup_pr(FakeClient(), "project-1", "PC1-TEST-123")

    assert task == {"id": "task-1"}
    assert artifact["id"] == "artifact-1"
    assert task_calls == [
        {
            "project_id": "project-1",
            "title": "Implement: PC1-TEST-123",
            "description": "Test task for protocol",
            "metadata": {},
        }
    ]
    assert artifact_calls[0]["artifact_type"] == "pull_request"
    assert artifact_calls[0]["linked_task_id"] == "task-1"
    assert artifact_calls[0]["name"] == "PC1-TEST-123"
    metadata = artifact_calls[0]["metadata"]
    assert metadata["base_branch"] == "main"
    assert metadata["repository"] == live_test_protocols.DEFAULT_PR_REPOSITORY
    assert metadata["pr_number"] == 123
    assert metadata["pr_url"].endswith("/pull/123")
    assert metadata["branch"].endswith("/pc1-test-123")
    assert metadata["title"] == "PC1-TEST-123"
    assert len(metadata["change_summary"]) == 3
    assert len(metadata["test_plan"]) == 2
    assert len(metadata["risk_notes"]) == 2
