import json
import os
import ssl
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner


def _response(status_code, payload):
    return SimpleNamespace(status_code=status_code, json=lambda: payload)


def test_verify_onecli_returns_only_safe_resolved_agent_metadata(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    responses = {
        ("GET", "http://management/v1/health"): _response(200, {}),
        ("GET", "http://gateway/healthz"): _response(200, {}),
        ("GET", "http://management/v1/agents"): _response(
            200, [{"id": "agent-id", "identifier": "gateway-agent", "name": "Gateway", "accessToken": "never-show"}]
        ),
        ("GET", "http://management/v1/agents/agent-id/effective-credentials"): _response(
            200, {"agentId": "agent-id", "mode": "selective", "secrets": [], "connections": []}
        ),
    }

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def request(self, method, url, **kwargs): return responses[(method, url)]

    monkeypatch.setattr(onecli.httpx, "Client", Client)
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    config = Settings(_env_file=None, onecli_agent="gateway-agent", onecli_management_url="http://management", onecli_gateway_url="http://gateway")

    assert onecli.verify_onecli(config) == {"id": "agent-id", "identifier": "gateway-agent", "name": "Gateway"}


def test_verify_onecli_rejects_bad_effective_credential_schema(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_request", lambda *_args, **_kwargs: {"id": "unexpected"})
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    with pytest.raises(onecli.OneCliError, match="management service returned an unsupported response"):
        onecli.verify_onecli(Settings(_env_file=None, onecli_agent="gateway"))


def test_settings_reject_onecli_urls_with_credentials_or_paths():
    from pydantic import ValidationError
    from huddleroom.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, onecli_management_url="https://key@example.test/path")


def test_setup_onecli_never_saves_secret(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    monkeypatch.setattr(onecli, "verify_onecli", lambda *_args, **_kwargs: {"id": "agent-id", "identifier": "gateway", "name": "Gateway"})
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_request", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_agents", lambda *_args: [{"id": "agent-id", "identifier": "gateway", "name": "Gateway"}])
    monkeypatch.setattr(onecli, "_grants", lambda *_args: {"agentId": "agent-id", "mode": "grants", "connections": [], "secrets": []})
    monkeypatch.setattr(onecli, "_credentials", lambda *_args: [])
    monkeypatch.setattr(onecli.click, "prompt", lambda label, **kwargs: "gateway" if label == "OneCLI gateway agent" else "skip")
    monkeypatch.setattr(onecli.click, "confirm", lambda *_args, **_kwargs: False)
    updates = onecli.setup_onecli(Settings(_env_file=None, onecli_agent="gateway"))

    assert updates == {
        "credential_mode": "onecli",
        "onecli_agent": "gateway",
        "onecli_management_url": "http://127.0.0.1:10256",
        "onecli_gateway_url": "http://127.0.0.1:10255",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "agentId": "agent-id",
            "mode": "selective",
            "secrets": [{"id": "secret-id", "name": "OpenAI", "host": "api.openai.com", "status": "usable"}],
            "connections": [],
        },
        {
            "agentId": "agent-id",
            "mode": "selective",
            "secrets": [
                {
                    "kind": "secret",
                    "id": "secret-id",
                    "name": "OpenAI",
                    "host": "api.openai.com",
                    "status": "usable",
                    "provenance": "not-an-array",
                }
            ],
            "connections": [],
        },
    ],
)
def test_effective_credentials_fail_closed_when_required_secret_fields_are_malformed(monkeypatch, payload):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    monkeypatch.setattr(onecli, "_request", lambda *_args, **_kwargs: payload)

    with pytest.raises(onecli.OneCliError, match="unsupported response"):
        onecli._effective(Settings(_env_file=None), "agent-id", None)


def test_grants_fail_closed_when_secret_entries_are_not_validated(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    monkeypatch.setattr(
        onecli,
        "_request",
        lambda *_args, **_kwargs: {"agentId": "agent-id", "mode": "grants", "connections": [], "secrets": [{}]},
    )

    with pytest.raises(onecli.OneCliError, match="unsupported response"):
        onecli._grants(Settings(_env_file=None), "agent-id", None)


def test_management_errors_hide_server_error_and_keep_auth_private(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    sentinel = "management-server-error-with-secret"
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {"timeout": 5.0, "trust_env": False, "follow_redirects": False}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return _response(500, {"error": sentinel})

    monkeypatch.setattr(onecli.httpx, "Client", Client)
    monkeypatch.setattr(onecli, "_api_key", lambda: "management-auth-token")

    with pytest.raises(onecli.OneCliError) as error:
        onecli._request(Settings(_env_file=None, onecli_management_url="http://management"), "GET", "/agents")

    assert sentinel not in str(error.value)
    assert "management-auth-token" not in str(error.value)
    assert calls == [
        (
            "GET",
            "http://management/v1/agents",
            {"headers": {"Accept": "application/json", "Authorization": "Bearer management-auth-token"}, "json": None},
        )
    ]


def test_create_secret_posts_only_contract_body_and_never_echoes_preview(monkeypatch, capsys):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    secret = "sk-hidden-key"
    captured = {}

    def request(_config, method, path, **kwargs):
        captured.update(method=method, path=path, **kwargs)
        return {"id": "secret-id", "preview": f"first-{secret}-last"}

    prompts = iter(("  OpenAI for HuddleRoom  ", secret))
    monkeypatch.setattr(onecli, "_request", request)
    monkeypatch.setattr(onecli.click, "prompt", lambda *_args, **_kwargs: next(prompts))
    onecli._RETAINED_RESOURCE_IDS.set(())

    assert onecli._create_secret(Settings(_env_file=None), "openai", "project-id") == "secret-id"
    assert captured == {
        "method": "POST",
        "path": "/secrets",
        "body": {
            "name": "OpenAI for HuddleRoom",
            "type": "openai",
            "valueSource": "inline",
            "value": secret,
            "hostPattern": "api.openai.com",
        },
        "project_id": "project-id",
        "expected_status": 201,
    }
    assert secret not in capsys.readouterr().out


def test_create_secret_rejects_non_created_success_response(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, _method, _url, **_kwargs):
            return _response(
                200,
                {"id": "secret-created-with-wrong-status", "preview": "secret-preview-must-not-appear"},
            )

    prompts = iter(("OpenAI", "sk-hidden-key"))
    monkeypatch.setattr(onecli.httpx, "Client", Client)
    monkeypatch.setattr(onecli, "_api_key", lambda: None)
    monkeypatch.setattr(onecli.click, "prompt", lambda *_args, **_kwargs: next(prompts))
    onecli._RETAINED_RESOURCE_IDS.set(())

    with pytest.raises(onecli.OneCliError) as error:
        onecli._create_secret(Settings(_env_file=None), "openai", None)
    assert onecli.retained_resource_ids() == ()
    message = str(error.value)
    assert "http://127.0.0.1:10256" in message
    assert "POST /secrets" in message
    assert "HTTP 200" in message
    assert "inventory" in message.lower()
    assert "secret-preview-must-not-appear" not in message


def test_attach_secret_is_bodyless_and_does_not_replace_existing_grants(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    requests = []
    original_grants = {
        "agentId": "agent-id",
        "mode": "grants",
        "connections": [],
        "secrets": [{"secretId": "existing-id", "name": "Existing", "type": "openai", "scope": "agent"}],
    }
    monkeypatch.setattr(onecli, "_grants", lambda *_args: original_grants)
    monkeypatch.setattr(onecli, "_request", lambda *args, **kwargs: requests.append((args, kwargs)) or original_grants)

    onecli._attach_secret(Settings(_env_file=None), "agent-id", "new-id", "project-id")

    assert len(requests) == 1
    args, kwargs = requests[0]
    assert args[1:] == ("PUT", "/agents/agent-id/grants/secrets/new-id")
    assert kwargs == {"project_id": "project-id"}
    assert original_grants["secrets"] == [
        {"secretId": "existing-id", "name": "Existing", "type": "openai", "scope": "agent"}
    ]


def test_direct_setup_never_imports_or_probes_onecli(monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.onecli as onecli

    updates = {}
    monkeypatch.setattr(cli, "_update_config", lambda values: updates.update(values))
    monkeypatch.setattr(
        onecli,
        "setup_onecli",
        lambda *_args: pytest.fail("direct setup must not call OneCLI"),
    )

    cli.setup.callback(
        provider="skip",
        credential_mode="direct",
        onecli_agent=None,
        onecli_management_url=None,
        onecli_gateway_url=None,
        orchestration_model="openai/model",
        database_path="state.db",
        workspace_dir="workspace",
    )

    assert updates["credential_mode"] == "direct"
    assert updates["database_url"] == "sqlite+aiosqlite:///state.db"


def test_onecli_local_save_failure_reports_retained_ids_without_secret(monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.onecli as onecli

    onecli._RETAINED_RESOURCE_IDS.set(())

    def setup(_config, **_kwargs):
        onecli._retain("created-agent-id")
        return {
            "credential_mode": "onecli",
            "onecli_agent": "gateway",
            "onecli_management_url": "http://management",
            "onecli_gateway_url": "http://gateway",
        }

    monkeypatch.setattr(onecli, "setup_onecli", setup)
    monkeypatch.setattr(onecli, "verify_onecli", lambda *_args: {"id": "created-agent-id"})
    monkeypatch.setattr(onecli, "_credentials", lambda *_args: [])
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)
    monkeypatch.setattr(onecli, "report_onecli_readiness", lambda *_args: None)
    monkeypatch.setattr(
        cli,
        "_update_config",
        lambda _updates: (_ for _ in ()).throw(click.ClickException("Could not save setup values. No changes were written.")),
    )

    result = CliRunner().invoke(
        cli.main,
        [
            "setup",
            "--credential-mode",
            "onecli",
            "--onecli-agent",
            "gateway",
            "--orchestration-model",
            "openai/model",
            "--database-path",
            "state.db",
            "--workspace-dir",
            "workspace",
        ],
    )

    assert result.exit_code != 0
    assert "created-agent-id" in result.output


def test_explicit_direct_provider_persists_mode(monkeypatch):
    import huddleroom.cli as cli

    updates = {}
    monkeypatch.setattr(cli, "_update_config", lambda values: updates.update(values))
    monkeypatch.setattr(cli.click, "prompt", lambda *_args, **_kwargs: "provider-secret")

    cli.setup.callback(
        provider="openai",
        credential_mode="direct",
        onecli_agent=None,
        onecli_management_url=None,
        onecli_gateway_url=None,
        orchestration_model="openai/model",
        database_path="state.db",
        workspace_dir="workspace",
    )

    assert updates["credential_mode"] == "direct"


def test_explicit_onecli_agent_does_not_prompt_for_identifier(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_request", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)
    monkeypatch.setattr(onecli, "_agents", lambda *_args: [{"id": "agent-id", "identifier": "fixed", "name": "Gateway"}])
    monkeypatch.setattr(onecli, "_grants", lambda *_args: {"agentId": "agent-id", "mode": "grants", "connections": [], "secrets": []})
    monkeypatch.setattr(onecli, "_credentials", lambda *_args: [])
    monkeypatch.setattr(onecli.click, "prompt", lambda *_args, **_kwargs: pytest.fail("must not prompt for explicit agent"))
    monkeypatch.setattr(onecli.click, "confirm", lambda *_args, **_kwargs: False)

    updates = onecli.setup_onecli(Settings(_env_file=None, onecli_agent="fixed"), prompt_agent=False)

    assert updates["onecli_agent"] == "fixed"


def test_new_secret_without_auto_grant_offers_additive_attachment(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    attachments = []
    credentials = iter(([], [{"id": "new-secret", "type": "openai", "status": "missing"}], [{"id": "new-secret", "type": "openai", "status": "usable"}]))
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_request", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)
    monkeypatch.setattr(onecli, "_agents", lambda *_args: [{"id": "agent-id", "identifier": "gateway", "name": "Gateway"}])
    monkeypatch.setattr(onecli, "_grants", lambda *_args: {"agentId": "agent-id", "mode": "grants", "connections": [], "secrets": []})
    monkeypatch.setattr(onecli, "_credentials", lambda *_args: next(credentials))
    monkeypatch.setattr(onecli, "_create_secret", lambda *_args: "new-secret")
    monkeypatch.setattr(onecli, "_attach_secret", lambda *_args: attachments.append(_args[2]))
    monkeypatch.setattr(onecli.click, "prompt", lambda *_args, **_kwargs: "gateway")
    answers = iter((True, True, False))
    monkeypatch.setattr(onecli.click, "confirm", lambda *_args, **_kwargs: next(answers))

    onecli.setup_onecli(Settings(_env_file=None, onecli_agent="gateway"))

    assert attachments == ["new-secret"]


def test_granted_blocked_secret_is_reported_without_reattachment(monkeypatch, capsys):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    attachments = []
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_request", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)
    monkeypatch.setattr(onecli, "_agents", lambda *_args: [{"id": "agent-id", "identifier": "gateway", "name": "Gateway"}])
    monkeypatch.setattr(onecli, "_grants", lambda *_args: {"agentId": "agent-id", "mode": "grants", "connections": [], "secrets": [{"secretId": "blocked", "name": "OpenAI", "type": "openai", "scope": "agent"}]})
    monkeypatch.setattr(onecli, "_credentials", lambda *_args: [{"id": "blocked", "type": "openai", "status": "blocked"}])
    monkeypatch.setattr(onecli, "_attach_secret", lambda *_args: attachments.append(True))
    monkeypatch.setattr(onecli.click, "prompt", lambda *_args, **_kwargs: "gateway")
    monkeypatch.setattr(onecli.click, "confirm", lambda *_args, **_kwargs: False)

    onecli.setup_onecli(Settings(_env_file=None, onecli_agent="gateway"))

    assert attachments == []
    assert "policy was not changed" in capsys.readouterr().out


def test_old_cli_capability_probe_fails_without_echoing_command_output(monkeypatch):
    import huddleroom.onecli as onecli

    sentinel = "old-cli-output-must-stay-private"
    monkeypatch.setattr(
        onecli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=f"--id {sentinel}", stderr=""),
    )

    with pytest.raises(onecli.OneCliError) as error:
        onecli._check_cli_capabilities()

    assert "unsupported" in str(error.value).lower()
    assert sentinel not in str(error.value)


def test_missing_cli_binary_fails_with_upgrade_guidance(monkeypatch):
    import huddleroom.onecli as onecli

    monkeypatch.setattr(
        onecli.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("onecli-not-installed-private-path")),
    )

    with pytest.raises(onecli.OneCliError, match="unavailable or too old") as error:
        onecli._check_cli_capabilities()

    assert "onecli-not-installed-private-path" not in str(error.value)


def test_management_denial_hides_response_body(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    sentinel = "authorization-body-must-stay-private"

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, *_args, **_kwargs):
            return _response(403, {"error": sentinel})

    monkeypatch.setattr(onecli.httpx, "Client", Client)

    with pytest.raises(onecli.OneCliError, match="log in or request permission") as error:
        onecli._request(Settings(_env_file=None), "GET", "/health")

    assert sentinel not in str(error.value)


def test_management_outage_hides_transport_details(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, *_args, **_kwargs):
            raise onecli.httpx.ConnectError("management-private-address")

    monkeypatch.setattr(onecli.httpx, "Client", Client)

    with pytest.raises(onecli.OneCliError, match="management service is unavailable") as error:
        onecli._request(Settings(_env_file=None), "GET", "/health")

    assert "management-private-address" not in str(error.value)


def test_verify_onecli_resolves_project_slug_and_scopes_agent_requests(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    calls = []

    def request(_config, method, path, **kwargs):
        calls.append((method, path, kwargs.get("project_id")))
        if path == "/health":
            return {}
        if path == "/projects":
            return [{"id": "project-id", "slug": "team"}]
        if path == "/agents":
            return [{"id": "agent-id", "identifier": "gateway", "name": "Gateway"}]
        if path == "/agents/agent-id/effective-credentials":
            return {"agentId": "agent-id", "mode": "selective", "secrets": [], "connections": []}
        pytest.fail(f"unexpected OneCLI request {method} {path}")

    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_configured_project", lambda: "team")
    monkeypatch.setattr(onecli, "_request", request)
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)

    resolved = onecli.verify_onecli(Settings(_env_file=None, onecli_agent="gateway"))

    assert resolved == {"id": "agent-id", "identifier": "gateway", "name": "Gateway"}
    assert calls == [
        ("GET", "/health", None),
        ("GET", "/projects", None),
        ("GET", "/agents", "project-id"),
        ("GET", "/agents/agent-id/effective-credentials", "project-id"),
    ]


def test_gateway_offline_is_reported_without_endpoint_details(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, *_args, **_kwargs):
            raise onecli.httpx.ConnectError("offline-gateway-private-host")

    monkeypatch.setattr(onecli.httpx, "Client", Client)

    with pytest.raises(onecli.OneCliError, match="gateway is unavailable") as error:
        onecli._gateway_health(Settings(_env_file=None, onecli_gateway_url="http://gateway"))

    assert "offline-gateway-private-host" not in str(error.value)


def test_partial_agent_create_retains_only_id_and_hides_access_token(monkeypatch, capsys):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    token = "access-token-must-never-print"
    onecli._RETAINED_RESOURCE_IDS.set(())
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)
    monkeypatch.setattr(onecli, "_agents", lambda *_args: [])
    monkeypatch.setattr(onecli.click, "prompt", lambda *_args, **_kwargs: "gateway")
    monkeypatch.setattr(onecli.click, "confirm", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        onecli,
        "_request",
        lambda _config, method, path, **_kwargs: (
            {} if path == "/health" else {"id": "created-agent", "identifier": "gateway", "name": "Gateway", "accessToken": token}
        ),
    )
    monkeypatch.setattr(
        onecli,
        "_grants",
        lambda *_args: (_ for _ in ()).throw(onecli.OneCliError("safe next-request failure")),
    )

    with pytest.raises(onecli.OneCliError, match="safe next-request failure"):
        onecli.setup_onecli(Settings(_env_file=None), prompt_agent=True)

    output = capsys.readouterr().out
    assert onecli.retained_resource_ids() == ("created-agent",)
    assert "created-agent" in output
    assert token not in output


def test_rerun_reuses_exact_existing_agent_without_creation(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_request", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)
    monkeypatch.setattr(onecli, "_agents", lambda *_args: [{"id": "agent-id", "identifier": "gateway", "name": "Gateway"}])
    monkeypatch.setattr(onecli, "_create_agent", lambda *_args: pytest.fail("rerun must reuse the exact existing identifier"))
    monkeypatch.setattr(onecli, "_grants", lambda *_args: {"agentId": "agent-id", "mode": "grants", "connections": [], "secrets": []})
    monkeypatch.setattr(onecli, "_credentials", lambda *_args: [])
    monkeypatch.setattr(onecli.click, "confirm", lambda *_args, **_kwargs: False)

    updates = onecli.setup_onecli(Settings(_env_file=None, onecli_agent="gateway"), prompt_agent=False)

    assert updates["onecli_agent"] == "gateway"


def test_onecli_setup_abort_never_reaches_local_config_write(monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.onecli as onecli

    writes = []
    monkeypatch.setattr(onecli, "setup_onecli", lambda *_args, **_kwargs: (_ for _ in ()).throw(click.Abort()))
    monkeypatch.setattr(cli, "_update_config", lambda values: writes.append(values))

    result = CliRunner().invoke(
        cli.main,
        [
            "setup",
            "--credential-mode",
            "onecli",
            "--onecli-agent",
            "gateway",
            "--orchestration-model",
            "openai/model",
            "--database-path",
            "state.db",
            "--workspace-dir",
            "workspace",
        ],
    )

    assert result.exit_code != 0
    assert writes == []


@pytest.mark.parametrize(("environment", "project"), [(None, "production"), ("dev", "development")])
def test_configured_project_reads_native_json_config(monkeypatch, tmp_path, environment, project):
    import huddleroom.onecli as onecli

    onecli_directory = tmp_path / ".onecli"
    onecli_directory.mkdir()
    (onecli_directory / "config.json").write_text('{"project": "production"}')
    (onecli_directory / "config-dev.json").write_text('{"project": "development"}')
    monkeypatch.setattr(onecli.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("ONECLI_PROJECT", raising=False)
    if environment is None:
        monkeypatch.delenv("ONECLI_ENV", raising=False)
    else:
        monkeypatch.setenv("ONECLI_ENV", environment)

    assert onecli._configured_project() == project


def test_configured_project_environment_overrides_native_json(monkeypatch, tmp_path):
    import huddleroom.onecli as onecli

    (tmp_path / ".onecli").mkdir()
    (tmp_path / ".onecli" / "config.json").write_text('{"project": "file-project"}')
    monkeypatch.setattr(onecli.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("ONECLI_PROJECT", "environment-project")

    assert onecli._configured_project() == "environment-project"


def test_default_embedding_readiness_requires_openai_even_with_anthropic(capsys):
    from huddleroom.config import Settings
    from huddleroom.onecli import report_onecli_readiness

    report_onecli_readiness(Settings(_env_file=None), [{"id": "anthropic", "type": "anthropic", "status": "usable"}])

    assert "Embeddings: missing usable OpenAI credential." in capsys.readouterr().out


def test_default_embedding_readiness_recognizes_usable_openai(capsys):
    from huddleroom.config import Settings
    from huddleroom.onecli import report_onecli_readiness

    report_onecli_readiness(Settings(_env_file=None), [{"id": "openai", "type": "openai", "status": "usable"}])

    assert "Embeddings: credential access known." in capsys.readouterr().out


def test_credentials_distinguish_metadata_only_and_oauth_manual(monkeypatch, capsys):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    monkeypatch.setattr(
        onecli,
        "_request",
        lambda *_args, **_kwargs: [
            {"id": "metadata-only", "name": "OpenAI key", "type": "openai", "hostPattern": "api.openai.com", "metadata": {"authMode": "api-key"}},
            {"id": "oauth", "name": "OpenAI OAuth", "type": "openai", "hostPattern": "api.openai.com", "metadata": {"authMode": "oauth"}},
        ],
    )
    monkeypatch.setattr(onecli, "_effective", lambda *_args: {"secrets": [], "connections": []})

    credentials = onecli._credentials(Settings(_env_file=None), "agent-id", None)

    assert credentials == [
        {"id": "metadata-only", "type": "openai", "status": "present but inaccessible"},
        {"id": "oauth", "name": "OpenAI OAuth", "type": "openai", "status": "manual/unsupported", "manual": True},
    ]


def test_oauth_only_stays_manual_and_does_not_block_api_key_choice(monkeypatch, capsys):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    attachments = []
    created = []
    oauth = {"id": "oauth", "name": "OpenAI OAuth", "type": "openai", "status": "manual/unsupported", "manual": True}
    new_key = {"id": "new-key", "type": "openai", "status": "present but inaccessible"}
    credential_sets = iter(([oauth], [oauth, new_key], [oauth, new_key]))
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_request", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)
    monkeypatch.setattr(onecli, "_agents", lambda *_args: [{"id": "agent", "identifier": "gateway", "name": "Gateway"}])
    monkeypatch.setattr(onecli, "_grants", lambda *_args: {"agentId": "agent", "mode": "grants", "connections": [], "secrets": []})
    monkeypatch.setattr(onecli, "_credentials", lambda *_args: next(credential_sets))
    monkeypatch.setattr(onecli, "_create_secret", lambda *_args: created.append("new-key") or "new-key")
    monkeypatch.setattr(onecli, "_attach_secret", lambda *_args: attachments.append(_args[2]))
    monkeypatch.setattr(onecli.click, "prompt", lambda *_args, **_kwargs: "gateway")
    answers = iter((True, True, False))
    monkeypatch.setattr(onecli.click, "confirm", lambda *_args, **_kwargs: next(answers))

    onecli.setup_onecli(Settings(_env_file=None, onecli_agent="gateway"))

    assert created == ["new-key"]
    assert attachments == ["new-key"]
    assert "OpenAI OAuth: manual/unsupported" in capsys.readouterr().out


def test_oauth_does_not_shadow_usable_api_key_status(monkeypatch, capsys):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    oauth = {"id": "oauth", "name": "OpenAI OAuth", "type": "openai", "status": "manual/unsupported", "manual": True}
    usable = {"id": "api-key", "type": "openai", "status": "usable"}
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_request", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)
    monkeypatch.setattr(onecli, "_agents", lambda *_args: [{"id": "agent", "identifier": "gateway", "name": "Gateway"}])
    monkeypatch.setattr(onecli, "_grants", lambda *_args: {"agentId": "agent", "mode": "grants", "connections": [], "secrets": []})
    monkeypatch.setattr(onecli, "_credentials", lambda *_args: [oauth, usable])
    monkeypatch.setattr(onecli.click, "prompt", lambda *_args, **_kwargs: "gateway")
    monkeypatch.setattr(onecli.click, "confirm", lambda *_args, **_kwargs: False)

    onecli.setup_onecli(Settings(_env_file=None, onecli_agent="gateway"))

    output = capsys.readouterr().out
    assert "Openai credential: usable." in output
    assert "OpenAI OAuth: manual/unsupported" in output


def test_onecli_environment_replaces_provider_secrets_and_keeps_gateway_routing(monkeypatch):
    from huddleroom.config import Settings
    from huddleroom.onecli import apply_onecli_environment

    monkeypatch.setenv("HTTP_PROXY", "http://gateway:10255")
    monkeypatch.setenv("HTTPS_PROXY", "http://gateway:10255")
    monkeypatch.setenv("ONECLI_GATEWAY", "true")
    monkeypatch.setenv("HUDDLEROOM_ONECLI_AGENT", "gateway")
    config = Settings(_env_file=None, credential_mode="onecli", onecli_agent="gateway", onecli_gateway_url="http://gateway:10255")
    env = {"OPENAI_API_KEY": "direct-secret", "ANTHROPIC_API_KEY": "direct-anthropic", "NO_PROXY": "api.openai.com"}

    apply_onecli_environment(env, config)

    assert env["OPENAI_API_KEY"] != "direct-secret"
    assert env["ANTHROPIC_API_KEY"] != "direct-anthropic"
    assert env["HTTP_PROXY"] == "http://gateway:10255"
    assert env["HTTPS_PROXY"] == "http://gateway:10255"
    assert "api.openai.com" not in env["NO_PROXY"]


def test_wrapped_context_rejects_other_agent_proxy(monkeypatch, tmp_path):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    ca_file = tmp_path / "gateway-ca.pem"
    ca_file.write_text("certificate")
    config = Settings(_env_file=None, credential_mode="onecli", onecli_agent="gateway", onecli_gateway_url="http://gateway:10255")
    monkeypatch.setenv("ONECLI_GATEWAY", "true")
    monkeypatch.setenv("HUDDLEROOM_ONECLI_AGENT", "gateway")
    monkeypatch.setenv("HUDDLEROOM_ONECLI_GATEWAY_URL", "http://gateway:10255")
    monkeypatch.setenv("HUDDLEROOM_ONECLI_MANAGEMENT_URL", config.onecli_management_url)
    monkeypatch.setenv("HTTP_PROXY", "http://wrong-token@gateway:10255")
    monkeypatch.setenv("HTTPS_PROXY", "http://wrong-token@gateway:10255")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))
    monkeypatch.setattr(onecli, "_container_config", lambda *_args: {"env": {"HTTP_PROXY": "http://right-token@gateway:10255", "HTTPS_PROXY": "http://right-token@gateway:10255"}, "caCertificate": "pem", "caCertificateContainerPath": str(ca_file)})

    with pytest.raises(onecli.OneCliError, match="context"):
        onecli.validate_onecli_context(config)


def test_wrapped_context_rejects_proxy_with_matching_user_but_wrong_password(monkeypatch, tmp_path):
    """A proxy username alone does not identify the selected native wrapper."""
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    ca_file = tmp_path / "gateway-ca.pem"
    ca_file.write_text("certificate")
    config = Settings(_env_file=None, credential_mode="onecli", onecli_agent="gateway", onecli_gateway_url="http://gateway:10255")
    monkeypatch.setenv("ONECLI_GATEWAY", "true")
    monkeypatch.setenv("HUDDLEROOM_ONECLI_AGENT", "gateway")
    monkeypatch.setenv("HUDDLEROOM_ONECLI_GATEWAY_URL", "http://gateway:10255")
    monkeypatch.setenv("HUDDLEROOM_ONECLI_MANAGEMENT_URL", config.onecli_management_url)
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(key, "http://agent:wrong-token@gateway:10255")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))
    monkeypatch.setattr(
        onecli,
        "_container_config",
        lambda *_args: {
            "env": {"HTTP_PROXY": "http://agent:right-token@gateway:10255"},
            "caCertificate": "pem",
            "caCertificateContainerPath": str(ca_file),
        },
    )

    with pytest.raises(onecli.OneCliError, match="context"):
        onecli.validate_onecli_context(config)


def test_wrapped_context_rejects_missing_or_unusable_tls_bundle(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    config = Settings(_env_file=None, credential_mode="onecli", onecli_agent="gateway", onecli_gateway_url="http://gateway:10255")
    for key, value in {
        "ONECLI_GATEWAY": "true",
        "HUDDLEROOM_ONECLI_AGENT": "gateway",
        "HUDDLEROOM_ONECLI_GATEWAY_URL": "http://gateway:10255",
        "HUDDLEROOM_ONECLI_MANAGEMENT_URL": config.onecli_management_url,
        "HTTP_PROXY": "http://agent:token@gateway:10255",
        "http_proxy": "http://agent:token@gateway:10255",
        "HTTPS_PROXY": "http://agent:token@gateway:10255",
        "https_proxy": "http://agent:token@gateway:10255",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("NODE_EXTRA_CA_CERTS", raising=False)
    monkeypatch.setattr(
        onecli,
        "_container_config",
        lambda *_args: {
            "env": {"HTTP_PROXY": "http://agent:token@gateway:10255"},
            "caCertificate": "pem",
            "caCertificateContainerPath": "/missing/gateway-ca.pem",
        },
    )

    with pytest.raises(onecli.OneCliError, match="usable TLS trust"):
        onecli.validate_onecli_context(config)


def test_onecli_environment_keeps_only_loopback_bypass_in_both_casings(monkeypatch):
    from huddleroom.config import Settings
    from huddleroom.onecli import apply_onecli_environment

    monkeypatch.setenv("NO_PROXY", "api.openai.com,localhost,example.test")
    monkeypatch.setenv("no_proxy", "api.anthropic.com,127.0.0.1")
    env = {"NO_PROXY": "attacker.invalid", "no_proxy": "api.openai.com"}

    apply_onecli_environment(env, Settings(_env_file=None, credential_mode="onecli", onecli_agent="gateway"))

    assert env["NO_PROXY"] == "localhost"
    assert env["no_proxy"] == "127.0.0.1"


@pytest.mark.parametrize(
    ("gateway_url", "native_gateway"),
    [
        ("http://gateway.example:10255", "gateway.example"),
        ("http://[::1]:10255", "[::1]"),
    ],
)
def test_launch_onecli_execs_native_wrapper_with_current_interpreter_and_env_auth(
    monkeypatch, gateway_url, native_gateway,
):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    captured = {}
    config = Settings(
        _env_file=None, credential_mode="onecli", onecli_agent="gateway",
        onecli_management_url="http://management.example:10256", onecli_gateway_url=gateway_url,
    )
    monkeypatch.setenv("ONECLI_API_KEY", "environment-only-management-key")
    monkeypatch.setattr(onecli, "verify_onecli", lambda *_args: {"id": "id", "identifier": "gateway", "name": "Gateway"})
    monkeypatch.setattr(onecli, "_container_config", lambda *_args: {"env": {}, "caCertificate": "pem", "caCertificateContainerPath": "/unused"})
    monkeypatch.setattr(onecli.os, "execvpe", lambda executable, argv, env: captured.update(executable=executable, argv=argv, env=env))

    onecli.launch_onecli(config, "0.0.0.0", 8123, True)

    assert captured["executable"] == "onecli"
    assert captured["argv"] == [
        "onecli", "run", "--agent", "gateway", "--gateway", native_gateway, "--", sys.executable,
        "-m", "huddleroom.cli", "serve", "--host", "0.0.0.0", "--port", "8123", "--reload",
    ]
    assert captured["env"]["ONECLI_API_KEY"] == "environment-only-management-key"
    assert captured["env"]["ONECLI_API_HOST"] == "http://management.example:10256"


@pytest.mark.parametrize(
    ("gateway_url", "runtime_proxy", "normalized_proxy"),
    [
        (
            "http://127.0.0.1:10255",
            "http://agent:token@127.0.0.1:10255",
            "http://agent:token@127.0.0.1:10255",
        ),
        (
            "http://127.0.0.1:10255",
            "http://agent:token@gateway:10255",
            "http://agent:token@127.0.0.1:10255",
        ),
        (
            "http://127.0.0.1:18443",
            "http://agent:token@127.0.0.1:10255",
            "http://agent:token@127.0.0.1:18443",
        ),
    ],
)
def test_wrapped_context_accepts_server_or_native_gateway_and_normalizes_routing(
    monkeypatch, tmp_path, gateway_url, runtime_proxy, normalized_proxy,
):
    """The server's internal gateway hostname is valid only for the selected wrapper."""
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    ca_file = tmp_path / "gateway-ca.pem"
    system_bundle = Path(ssl.get_default_verify_paths().cafile)
    ca_file.write_bytes(system_bundle.read_bytes())
    gateway_certificate = ca_file.read_text(encoding="utf-8")
    config = Settings(
        _env_file=None,
        credential_mode="onecli",
        onecli_agent="gateway",
        onecli_gateway_url=gateway_url,
    )
    server_proxy = "http://agent:token@gateway:10255"
    for key, value in {
        "ONECLI_GATEWAY": "true", "HUDDLEROOM_ONECLI_AGENT": "gateway",
        "HUDDLEROOM_ONECLI_GATEWAY_URL": gateway_url,
        "HUDDLEROOM_ONECLI_MANAGEMENT_URL": config.onecli_management_url,
        "HTTP_PROXY": runtime_proxy, "http_proxy": runtime_proxy,
        "HTTPS_PROXY": runtime_proxy, "https_proxy": runtime_proxy,
        "NO_PROXY": "localhost", "no_proxy": "127.0.0.1",
        **{key: str(ca_file) for key in onecli._CA_ENV_KEYS},
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        onecli,
        "_container_config",
        lambda *_args: {
            "env": {"HTTP_PROXY": server_proxy},
            "caCertificate": gateway_certificate,
            "caCertificateContainerPath": "/tmp/onecli-gateway-ca.pem",
        },
    )

    onecli.validate_onecli_context(config)

    assert os.environ["NO_PROXY"] == onecli._SAFE_NO_PROXY
    assert os.environ["no_proxy"] == onecli._SAFE_NO_PROXY
    assert {os.environ[key] for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")} == {
        normalized_proxy
    }


@pytest.mark.parametrize(
    ("runtime_proxy", "error"),
    [
        ("http://agent:token@other-gateway:10255", "selected gateway"),
        ("http://agent:token@127.0.0.1:10256", "selected gateway"),
        ("http://agent:token@127.0.0.1", "selected gateway"),
        ("http://agent:other-token@gateway:10255", "selected agent"),
    ],
)
def test_wrapped_context_rejects_unselected_gateway_destination(monkeypatch, tmp_path, runtime_proxy, error):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    ca_file = tmp_path / "gateway-ca.pem"
    ca_file.write_text("certificate")
    config = Settings(
        _env_file=None,
        credential_mode="onecli",
        onecli_agent="gateway",
        onecli_gateway_url="http://127.0.0.1:10255",
    )
    for key, value in {
        "ONECLI_GATEWAY": "true",
        "HUDDLEROOM_ONECLI_AGENT": "gateway",
        "HUDDLEROOM_ONECLI_GATEWAY_URL": "http://127.0.0.1:10255",
        "HUDDLEROOM_ONECLI_MANAGEMENT_URL": config.onecli_management_url,
        "HTTP_PROXY": runtime_proxy,
        "http_proxy": runtime_proxy,
        "HTTPS_PROXY": runtime_proxy,
        "https_proxy": runtime_proxy,
        "SSL_CERT_FILE": str(ca_file),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        onecli,
        "_container_config",
        lambda *_args: {
            "env": {"HTTP_PROXY": "http://agent:token@gateway:10255"},
            "caCertificate": "pem",
            "caCertificateContainerPath": str(ca_file),
        },
    )

    with pytest.raises(onecli.OneCliError, match=error):
        onecli.validate_onecli_context(config)


def test_wrapped_context_uses_one_canonical_local_gateway_ca_bundle(monkeypatch, tmp_path):
    """All trust consumers must resolve one local bundle, never the container path."""
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    canonical_bundle = tmp_path / "local-gateway-ca.pem"
    canonical_bundle.write_bytes(Path(ssl.get_default_verify_paths().cafile).read_bytes())
    configured_bundle = tmp_path / "configured-gateway-ca.pem"
    configured_bundle.symlink_to(canonical_bundle)
    gateway_certificate = canonical_bundle.read_text(encoding="utf-8")
    config = Settings(_env_file=None, credential_mode="onecli", onecli_agent="gateway", onecli_gateway_url="http://gateway:10255")
    proxy = "http://agent:token@gateway:10255"
    for key, value in {
        "ONECLI_GATEWAY": "true", "HUDDLEROOM_ONECLI_AGENT": "gateway",
        "HUDDLEROOM_ONECLI_GATEWAY_URL": "http://gateway:10255",
        "HUDDLEROOM_ONECLI_MANAGEMENT_URL": config.onecli_management_url,
        "HTTP_PROXY": proxy, "http_proxy": proxy, "HTTPS_PROXY": proxy, "https_proxy": proxy,
        "NO_PROXY": "localhost", "no_proxy": "127.0.0.1",
        **{key: str(configured_bundle) for key in onecli._CA_ENV_KEYS},
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        onecli,
        "_container_config",
        lambda *_args: {
            "env": {"HTTP_PROXY": proxy},
            "caCertificate": gateway_certificate,
            "caCertificateContainerPath": "/run/onecli/gateway-ca.pem",
        },
    )

    onecli.validate_onecli_context(config)

    assert Path(os.environ["SSL_CERT_FILE"]).resolve() == canonical_bundle
    assert {
        Path(os.environ[key]).resolve() for key in onecli._CA_ENV_KEYS
    } == {canonical_bundle}


def test_management_url_uses_native_environment_when_huddleroom_did_not_set_one(monkeypatch):
    """An implicit HuddleRoom default must not hide OneCLI's configured API host."""
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    monkeypatch.setenv("ONECLI_API_HOST", "http://native-onecli.test:10254")

    assert onecli.resolve_management_url(Settings(_env_file=None, credential_mode="onecli")) == (
        "http://native-onecli.test:10254"
    )


def test_management_url_reads_native_dev_config_when_environment_is_absent(monkeypatch, tmp_path):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    onecli_directory = tmp_path / ".onecli"
    onecli_directory.mkdir()
    (onecli_directory / "config-dev.json").write_text('{"api-host": "http://native-file.test:10254"}')
    monkeypatch.setattr(onecli.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("ONECLI_API_HOST", "")
    monkeypatch.setenv("ONECLI_ENV", "dev")

    assert onecli.resolve_management_url(Settings(_env_file=None, credential_mode="onecli")) == (
        "http://native-file.test:10254"
    )


@pytest.mark.parametrize("source", ["init", "environment", "toml"])
def test_management_url_preserves_explicit_huddleroom_value_over_native_onecli(monkeypatch, tmp_path, source):
    from huddleroom import config as config_module
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    configured = "http://huddleroom-override.test:18443"
    monkeypatch.setenv("ONECLI_API_HOST", "http://native-onecli.test:10254")
    if source == "init":
        settings = Settings(
            _env_file=None,
            credential_mode="onecli",
            onecli_management_url=configured,
        )
    elif source == "environment":
        monkeypatch.setenv("HUDDLEROOM_ONECLI_MANAGEMENT_URL", configured)
        settings = Settings(_env_file=None, credential_mode="onecli")
    else:
        config_file = tmp_path / "config.toml"
        config_file.write_text(f'onecli_management_url = "{configured}"\n')
        monkeypatch.setattr(config_module, "DEFAULT_CONFIG_FILE", config_file)
        settings = Settings(_env_file=None, credential_mode="onecli")

    assert onecli.resolve_management_url(settings) == configured


def test_management_url_rejects_malformed_native_origin_without_echoing_it(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    invalid_origin = "https://key@native-onecli.test/path?leak=1"
    monkeypatch.setenv("ONECLI_API_HOST", invalid_origin)

    with pytest.raises(onecli.OneCliError) as error:
        onecli.resolve_management_url(Settings(_env_file=None, credential_mode="onecli"))

    assert invalid_origin not in str(error.value)
    assert "ONECLI_API_HOST" in str(error.value)


@pytest.mark.parametrize("api_host", [None, 10254, ["http://native-onecli.test:10254"]])
def test_management_url_rejects_nonstring_native_file_origin_without_echoing_it(monkeypatch, tmp_path, api_host):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    onecli_directory = tmp_path / ".onecli"
    onecli_directory.mkdir()
    (onecli_directory / "config.json").write_text(json.dumps({"api-host": api_host}))
    monkeypatch.setattr(onecli.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("ONECLI_API_HOST", raising=False)

    with pytest.raises(onecli.OneCliError) as error:
        onecli.resolve_management_url(Settings(_env_file=None, credential_mode="onecli"))

    message = str(error.value)
    assert "config.json api-host" in message
    assert str(api_host) not in message


@pytest.mark.parametrize("contents", ["{not-json secret-malformed-config", "[\"secret-nonmapping-config\"]"])
def test_management_url_rejects_unusable_existing_native_config_without_echoing_contents(monkeypatch, tmp_path, contents):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    onecli_directory = tmp_path / ".onecli"
    onecli_directory.mkdir()
    (onecli_directory / "config.json").write_text(contents)
    monkeypatch.setattr(onecli.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("ONECLI_API_HOST", raising=False)

    with pytest.raises(onecli.OneCliError) as error:
        onecli.resolve_management_url(Settings(_env_file=None, credential_mode="onecli"))

    message = str(error.value)
    assert "config.json" in message
    assert "secret-" not in message


def test_management_url_rejects_unreadable_existing_native_config_without_echoing_os_error(monkeypatch, tmp_path):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    onecli_directory = tmp_path / ".onecli"
    onecli_directory.mkdir()
    config_file = onecli_directory / "config.json"
    config_file.write_text('{"api-host": "http://native-onecli.test:10254"}')
    original_read_text = onecli.Path.read_text
    monkeypatch.setattr(onecli.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("ONECLI_API_HOST", raising=False)

    def unreadable(path, *args, **kwargs):
        if path == config_file:
            raise PermissionError("private-native-config-path")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(onecli.Path, "read_text", unreadable)

    with pytest.raises(onecli.OneCliError) as error:
        onecli.resolve_management_url(Settings(_env_file=None, credential_mode="onecli"))

    assert "config.json" in str(error.value)
    assert "private-native-config-path" not in str(error.value)


@pytest.mark.parametrize("contents", [None, "{}"])
def test_management_url_uses_cloud_only_when_native_config_is_absent_or_has_no_api_host(monkeypatch, tmp_path, contents):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    onecli_directory = tmp_path / ".onecli"
    onecli_directory.mkdir()
    if contents is not None:
        (onecli_directory / "config.json").write_text(contents)
    monkeypatch.setattr(onecli.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("ONECLI_API_HOST", raising=False)

    assert onecli.resolve_management_url(Settings(_env_file=None, credential_mode="onecli")) == "https://api.onecli.sh"


def test_direct_setup_never_resolves_or_probes_native_onecli_management(monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.onecli as onecli

    updates = {}
    monkeypatch.setattr(cli, "_update_config", lambda values: updates.update(values))
    monkeypatch.setattr(
        onecli,
        "resolve_management_url",
        lambda *_args: pytest.fail("direct setup must not read native OneCLI configuration"),
        raising=False,
    )

    cli.setup.callback(
        provider="skip",
        credential_mode="direct",
        onecli_agent=None,
        onecli_management_url=None,
        onecli_gateway_url=None,
        orchestration_model="openai/model",
        database_path="state.db",
        workspace_dir="workspace",
    )

    assert updates["credential_mode"] == "direct"


def test_management_outage_names_selected_origin_without_transport_details(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, *_args, **_kwargs):
            raise onecli.httpx.ConnectError("private-connect-detail?token=must-not-appear")

    monkeypatch.setattr(onecli.httpx, "Client", Client)
    config = Settings(_env_file=None, onecli_management_url="http://127.0.0.1:10254")

    with pytest.raises(onecli.OneCliError) as error:
        onecli._request(config, "GET", "/health")

    message = str(error.value)
    assert "http://127.0.0.1:10254" in message
    assert "unavailable" in message
    assert "private-connect-detail" not in message
    assert "must-not-appear" not in message


def test_management_denial_names_selected_origin_and_status_without_response_body(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, *_args, **_kwargs):
            return _response(401, {"error": "body-token-must-not-appear"})

    monkeypatch.setattr(onecli.httpx, "Client", Client)
    monkeypatch.setattr(onecli, "_api_key", lambda: "management-auth-must-not-appear")
    config = Settings(_env_file=None, onecli_management_url="http://127.0.0.1:10254")

    with pytest.raises(onecli.OneCliError) as error:
        onecli._request(config, "GET", "/health")

    message = str(error.value)
    assert "http://127.0.0.1:10254" in message
    assert "401" in message
    assert "body-token-must-not-appear" not in message
    assert "management-auth-must-not-appear" not in message


def test_missing_required_grants_endpoint_requests_management_upgrade_without_echoing_body(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, *_args, **_kwargs):
            return _response(404, {"error": "old-server-secret-body"})

    monkeypatch.setattr(onecli.httpx, "Client", Client)
    config = Settings(_env_file=None, onecli_management_url="http://127.0.0.1:10254")

    with pytest.raises(onecli.OneCliError) as error:
        onecli._request(config, "GET", "/agents/agent-id/grants")

    message = str(error.value).lower()
    assert "http://127.0.0.1:10254" in message
    assert "upgrade" in message
    assert "old-server-secret-body" not in message


def test_setup_reports_known_incompatible_server_version_for_missing_grants(monkeypatch):
    """A pre-grants server version makes a required-endpoint 404 actionable."""
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    private_body = "grants-error-body-must-stay-private"
    responses = {
        ("GET", "http://management/v1/health"): _response(200, {"version": "1.41.0"}),
        ("GET", "http://management/v1/agents"): _response(
            200, [{"id": "agent-id", "identifier": "gateway", "name": "Gateway"}]
        ),
        ("GET", "http://management/v1/agents/agent-id/grants"): _response(404, {"error": private_body}),
    }

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, method, url, **_kwargs):
            return responses[(method, url)]

    monkeypatch.setattr(onecli.httpx, "Client", Client)
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)

    with pytest.raises(onecli.OneCliError) as error:
        onecli.setup_onecli(
            Settings(
                _env_file=None,
                onecli_agent="gateway",
                onecli_management_url="http://management",
                onecli_gateway_url="http://gateway",
            ),
            prompt_agent=False,
        )

    message = str(error.value)
    assert "version error" in message.lower()
    assert "incompatible" in message.lower()
    assert "detected 1.41.0" in message
    assert "requires 1.44.0 or later" in message
    assert private_body not in message


def test_verify_reports_known_incompatible_server_version_for_missing_effective_credentials(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    private_body = "effective-error-body-must-stay-private"
    responses = {
        ("GET", "http://management/v1/health"): _response(200, {"version": "1.41.0"}),
        ("GET", "http://management/v1/agents"): _response(
            200, [{"id": "agent-id", "identifier": "gateway", "name": "Gateway"}]
        ),
        ("GET", "http://management/v1/agents/agent-id/effective-credentials"): _response(404, {"error": private_body}),
    }

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, method, url, **_kwargs):
            return responses[(method, url)]

    monkeypatch.setattr(onecli.httpx, "Client", Client)
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)

    with pytest.raises(onecli.OneCliError) as error:
        onecli.verify_onecli(
            Settings(
                _env_file=None,
                onecli_agent="gateway",
                onecli_management_url="http://management",
                onecli_gateway_url="http://gateway",
            )
        )

    message = str(error.value)
    assert "version error" in message.lower()
    assert "incompatible" in message.lower()
    assert "detected 1.41.0" in message
    assert "requires 1.44.0 or later" in message
    assert private_body not in message


def test_setup_reports_known_incompatible_server_version_for_effective_credentials(monkeypatch):
    """Setup reaches effective access through _credentials after grants succeed."""
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    responses = {
        ("GET", "http://management/v1/health"): _response(200, {"version": "1.41.0"}),
        ("GET", "http://management/v1/agents"): _response(
            200, [{"id": "agent-id", "identifier": "gateway", "name": "Gateway"}]
        ),
        ("GET", "http://management/v1/agents/agent-id/grants"): _response(
            200, {"agentId": "agent-id", "mode": "grants", "connections": [], "secrets": []}
        ),
        ("GET", "http://management/v1/secrets"): _response(200, []),
        ("GET", "http://management/v1/agents/agent-id/effective-credentials"): _response(
            404, {"error": "credential-error-body-must-stay-private"}
        ),
    }

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, method, url, **_kwargs):
            return responses[(method, url)]

    monkeypatch.setattr(onecli.httpx, "Client", Client)
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)

    with pytest.raises(onecli.OneCliError) as error:
        onecli.setup_onecli(
            Settings(
                _env_file=None,
                onecli_agent="gateway",
                onecli_management_url="http://management",
                onecli_gateway_url="http://gateway",
            ),
            prompt_agent=False,
        )

    message = str(error.value)
    assert "version error" in message.lower()
    assert "incompatible" in message.lower()
    assert "detected 1.41.0" in message
    assert "requires 1.44.0 or later" in message
    assert "credential-error-body-must-stay-private" not in message


@pytest.mark.parametrize(
    ("health", "reports_current_server"),
    [
        ({}, False),
        ({"version": "1.41.0 private-version-must-not-appear"}, False),
        ({"version": ["1.41.0"]}, False),
        ({"version": "1.44.0"}, True),
    ],
)
def test_missing_required_api_reports_only_safe_version_details(monkeypatch, health, reports_current_server):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    responses = {
        ("GET", "http://management/v1/health"): _response(200, health),
        ("GET", "http://management/v1/agents"): _response(
            200, [{"id": "agent-id", "identifier": "gateway", "name": "Gateway"}]
        ),
        ("GET", "http://management/v1/agents/agent-id/grants"): _response(404, {"error": "private-404-body"}),
    }

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, method, url, **_kwargs):
            return responses[(method, url)]

    monkeypatch.setattr(onecli.httpx, "Client", Client)
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)

    with pytest.raises(onecli.OneCliError) as error:
        onecli.setup_onecli(
            Settings(
                _env_file=None,
                onecli_agent="gateway",
                onecli_management_url="http://management",
                onecli_gateway_url="http://gateway",
            ),
            prompt_agent=False,
        )

    message = str(error.value)
    assert "upgrade" in message.lower()
    assert "incompatible" not in message.lower()
    if reports_current_server:
        assert "api compatibility error" in message.lower()
        assert "detected server 1.44.0" in message
        assert "api host" in message.lower()
        assert "deployment" in message.lower()
        assert "version error" not in message.lower()
        assert "requires 1.44.0" not in message
    else:
        assert "detected" not in message.lower()
        assert "1.44.0" not in message
    assert "private-version-must-not-appear" not in message
    assert "private-404-body" not in message


def test_setup_accepts_old_server_when_required_apis_are_available(monkeypatch):
    """Endpoint capability succeeds even when an older server reports a backport version."""
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    responses = {
        ("GET", "http://management/v1/health"): _response(200, {"version": "1.41.0"}),
        ("GET", "http://management/v1/agents"): _response(
            200, [{"id": "agent-id", "identifier": "gateway", "name": "Gateway"}]
        ),
        ("GET", "http://management/v1/agents/agent-id/grants"): _response(
            200, {"agentId": "agent-id", "mode": "grants", "connections": [], "secrets": []}
        ),
        ("GET", "http://management/v1/secrets"): _response(200, []),
        ("GET", "http://management/v1/agents/agent-id/effective-credentials"): _response(
            200, {"agentId": "agent-id", "mode": "selective", "secrets": [], "connections": []}
        ),
    }

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, method, url, **_kwargs):
            return responses[(method, url)]

    monkeypatch.setattr(onecli.httpx, "Client", Client)
    monkeypatch.setattr(onecli, "_check_cli_capabilities", lambda: None)
    monkeypatch.setattr(onecli, "_gateway_health", lambda *_args: None)
    monkeypatch.setattr(onecli, "_project_id", lambda *_args: None)
    monkeypatch.setattr(onecli.click, "confirm", lambda *_args, **_kwargs: False)

    updates = onecli.setup_onecli(
        Settings(
            _env_file=None,
            onecli_agent="gateway",
            onecli_management_url="http://management",
            onecli_gateway_url="http://gateway",
        ),
        prompt_agent=False,
    )

    assert updates["onecli_agent"] == "gateway"


def test_gateway_outage_names_selected_origin_without_transport_details(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, *_args, **_kwargs):
            raise onecli.httpx.ConnectError("private-gateway-detail?token=must-not-appear")

    monkeypatch.setattr(onecli.httpx, "Client", Client)

    with pytest.raises(onecli.OneCliError) as error:
        onecli._gateway_health(Settings(_env_file=None, onecli_gateway_url="http://127.0.0.1:10255"))

    message = str(error.value)
    assert "http://127.0.0.1:10255" in message
    assert "unavailable" in message
    assert "private-gateway-detail" not in message
    assert "must-not-appear" not in message


def test_gateway_non_success_names_selected_origin_and_status_without_response_body(monkeypatch):
    from huddleroom.config import Settings
    import huddleroom.onecli as onecli

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def request(self, *_args, **_kwargs):
            return _response(503, {"error": "gateway-response-secret"})

    monkeypatch.setattr(onecli.httpx, "Client", Client)

    with pytest.raises(onecli.OneCliError) as error:
        onecli._gateway_health(Settings(_env_file=None, onecli_gateway_url="http://127.0.0.1:10255"))

    message = str(error.value)
    assert "http://127.0.0.1:10255" in message
    assert "503" in message
    assert "gateway-response-secret" not in message


@pytest.mark.asyncio
async def test_mocked_litellm_completion_and_embedding_see_onecli_placeholders(monkeypatch, tmp_path):
    """The SDK boundary sees placeholders, never values that would bypass OneCLI."""
    import litellm
    from huddleroom.config import load_provider_env

    dotenv = tmp_path / ".env"
    toml = tmp_path / "config.toml"
    dotenv.write_text("OPENAI_API_KEY=dotenv-openai-sentinel\nANTHROPIC_API_KEY=dotenv-anthropic-sentinel\n")
    toml.write_text('OPENAI_API_KEY = "toml-openai-sentinel"\nANTHROPIC_API_KEY = "toml-anthropic-sentinel"\n')
    monkeypatch.setenv("OPENAI_API_KEY", "process-openai-sentinel")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "process-anthropic-sentinel")
    load_provider_env(dotenv, toml, credential_mode="onecli")
    observed = []

    async def completion(**_kwargs):
        observed.append((os.environ.get("OPENAI_API_KEY"), os.environ.get("ANTHROPIC_API_KEY")))
        return SimpleNamespace(choices=[])

    async def embedding(**_kwargs):
        observed.append((os.environ.get("OPENAI_API_KEY"), os.environ.get("ANTHROPIC_API_KEY")))
        return SimpleNamespace(data=[{"embedding": [0.0]}])

    monkeypatch.setattr(litellm, "acompletion", completion)
    monkeypatch.setattr(litellm, "aembedding", embedding)
    await litellm.acompletion(model="openai/test", messages=[])
    await litellm.aembedding(model="text-embedding-3-small", input=["test"])

    assert observed == [
        ("onecli-openai-placeholder", "onecli-anthropic-placeholder"),
        ("onecli-openai-placeholder", "onecli-anthropic-placeholder"),
    ]
