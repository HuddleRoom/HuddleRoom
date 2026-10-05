"""Small, fail-closed OneCLI setup transport.

Only metadata crosses this boundary; provider values and OneCLI auth remain in
memory and are deliberately never returned or persisted by HuddleRoom.
"""

from __future__ import annotations

import os
import json
import re
import ssl
import subprocess
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import click
import httpx

from huddleroom.config import Settings

_TIMEOUT = 5.0
_RECIPES = {
    "openai": {"host": "api.openai.com", "prefix": "sk-"},
    "anthropic": {"host": "api.anthropic.com", "prefix": "sk-ant-api"},
}
_RETAINED_RESOURCE_IDS: ContextVar[tuple[str, ...]] = ContextVar("onecli_retained_resource_ids", default=())
_PROVIDER_PLACEHOLDERS = {
    "OPENAI_API_KEY": "onecli-openai-placeholder",
    "ANTHROPIC_API_KEY": "onecli-anthropic-placeholder",
}
_CONTEXT_KEYS = (
    "ONECLI_GATEWAY", "ONECLI_GATEWAY_SKILL_PATH", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO", "DENO_CERT",
    "NODE_USE_ENV_PROXY", "NODE_EXTRA_CA_CERTS", "NODE_OPTIONS",
    "HUDDLEROOM_ONECLI_AGENT", "HUDDLEROOM_ONECLI_GATEWAY_URL", "HUDDLEROOM_ONECLI_MANAGEMENT_URL",
)
_PROTECTED_ENV_KEYS = frozenset(_CONTEXT_KEYS + (
    "NO_PROXY", "no_proxy", "ONECLI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OR_API_KEY",
    "OLLAMA_API_BASE", "OPENROUTER_API_BASE", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
))
_CA_ENV_KEYS = (
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO", "DENO_CERT", "NODE_EXTRA_CA_CERTS",
)
_SAFE_NO_PROXY = "127.0.0.1,localhost,::1"
_PEM_CERTIFICATE = re.compile(r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.DOTALL)


def _container_config(config: Settings) -> dict[str, Any]:
    value = _request(config, "GET", f"/container-config?agent={config.onecli_agent}")
    if not isinstance(value, dict) or not isinstance(value.get("env"), dict):
        raise OneCliError("OneCLI returned an unsupported runtime context.")
    if not isinstance(value.get("caCertificate"), str) or not isinstance(value.get("caCertificateContainerPath"), str):
        raise OneCliError("OneCLI returned an unsupported runtime context.")
    return value


def _gateway_proxy(value: str, config: Settings) -> tuple[str, str | None, str | None]:
    parsed = urlsplit(value)
    target = urlsplit(config.onecli_gateway_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname != target.hostname or parsed.port != target.port:
        raise OneCliError("OneCLI runtime context does not use the selected gateway.")
    return f"{parsed.hostname}:{parsed.port}", (
        unquote(parsed.username) if parsed.username is not None else None
    ), (unquote(parsed.password) if parsed.password is not None else None)


def _safe_no_proxy(value: str | None) -> str:
    entries = [item.strip() for item in (value or "").split(",")]
    return ",".join(item for item in entries if item in {"127.0.0.1", "localhost", "::1"})


def apply_onecli_environment(env: dict[str, str], config: Settings) -> None:
    """Apply the already-validated parent wrapper context to an execution child."""
    if config.credential_mode != "onecli":
        return
    trusted = {key: os.environ[key] for key in _CONTEXT_KEYS if key in os.environ}
    for key in _PROTECTED_ENV_KEYS:
        env.pop(key, None)
    proxy = trusted.get("HTTP_PROXY") or trusted.get("http_proxy")
    secure_proxy = trusted.get("HTTPS_PROXY") or trusted.get("https_proxy") or proxy
    if proxy and secure_proxy:
        env.update({
            "HTTP_PROXY": proxy, "http_proxy": proxy,
            "HTTPS_PROXY": secure_proxy, "https_proxy": secure_proxy,
        })
    for key, value in trusted.items():
        if key not in {"HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"}:
            env[key] = value
    env.update(_PROVIDER_PLACEHOLDERS)
    env["NO_PROXY"] = _safe_no_proxy(os.environ.get("NO_PROXY"))
    env["no_proxy"] = _safe_no_proxy(os.environ.get("no_proxy"))


def validate_onecli_context(config: Settings) -> None:
    """Fail before startup unless this process is the selected native wrapper child."""
    markers = {
        "HUDDLEROOM_ONECLI_AGENT": config.onecli_agent or "",
        "HUDDLEROOM_ONECLI_GATEWAY_URL": config.onecli_gateway_url,
        "HUDDLEROOM_ONECLI_MANAGEMENT_URL": config.onecli_management_url,
    }
    if os.environ.get("ONECLI_GATEWAY") != "true" or any(os.environ.get(key) != value for key, value in markers.items()):
        raise OneCliError("OneCLI runtime context is missing or does not match the selected agent.")
    allowed_bypass = {"127.0.0.1", "localhost", "::1"}
    for key in ("NO_PROXY", "no_proxy"):
        entries = [entry.strip() for entry in os.environ.get(key, "").split(",") if entry.strip()]
        if any(entry not in allowed_bypass for entry in entries):
            raise OneCliError("OneCLI runtime context has an unsafe proxy bypass.")
    os.environ["NO_PROXY"] = _SAFE_NO_PROXY
    os.environ["no_proxy"] = _SAFE_NO_PROXY
    container = _container_config(config)
    expected = container["env"]
    expected_proxy = expected.get("HTTP_PROXY") or expected.get("http_proxy")
    if not isinstance(expected_proxy, str):
        raise OneCliError("OneCLI returned an unsupported runtime context.")
    _, expected_user, expected_password = _gateway_proxy(expected_proxy, config)
    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = os.environ.get(key)
        if not value:
            raise OneCliError("OneCLI runtime context is missing proxy routing.")
        _, actual_user, actual_password = _gateway_proxy(value, config)
        if (actual_user, actual_password) != (expected_user, expected_password):
            raise OneCliError("OneCLI runtime context does not match the selected agent.")
    ca_certificate = container["caCertificate"]
    try:
        canonical_ca = Path(os.environ["SSL_CERT_FILE"]).expanduser().resolve(strict=True)
        valid_ca = canonical_ca.read_text(encoding="utf-8")
        ssl.create_default_context(cafile=str(canonical_ca))
    except (KeyError, OSError, ssl.SSLError):
        raise OneCliError("OneCLI runtime context is missing usable TLS trust.") from None
    if not valid_ca or not ca_certificate.strip():
        raise OneCliError("OneCLI runtime context is missing usable TLS trust.")
    try:
        gateway_certificates = {ssl.PEM_cert_to_DER_cert(item) for item in _PEM_CERTIFICATE.findall(ca_certificate)}
        bundle_certificates = {ssl.PEM_cert_to_DER_cert(item) for item in _PEM_CERTIFICATE.findall(valid_ca)}
    except ValueError:
        raise OneCliError("OneCLI runtime context is missing usable TLS trust.") from None
    if not gateway_certificates or not gateway_certificates.intersection(bundle_certificates):
        raise OneCliError("OneCLI runtime context is missing usable TLS trust.")
    for key in _CA_ENV_KEYS:
        try:
            if Path(os.environ[key]).expanduser().resolve(strict=True) != canonical_ca:
                raise ValueError
        except (KeyError, OSError, ValueError):
            raise OneCliError("OneCLI runtime context is missing usable TLS trust.") from None


def launch_onecli(config: Settings, host: str, port: int, reload: bool) -> None:
    """Run one native wrapper, preserving this interpreter and its exit status."""
    agent = verify_onecli(config)
    _container_config(config)
    env = os.environ.copy()
    for key in _PROTECTED_ENV_KEYS - {"ONECLI_API_KEY"}:
        if key in _PROVIDER_PLACEHOLDERS:
            continue
        env.pop(key, None)
    env.update(_PROVIDER_PLACEHOLDERS)
    env.update({
        "ONECLI_API_HOST": config.onecli_management_url,
        "HUDDLEROOM_ONECLI_AGENT": agent["identifier"],
        "HUDDLEROOM_ONECLI_GATEWAY_URL": config.onecli_gateway_url,
        "HUDDLEROOM_ONECLI_MANAGEMENT_URL": config.onecli_management_url,
    })
    gateway = urlsplit(config.onecli_gateway_url).netloc
    command = ["onecli", "run", "--agent", agent["identifier"], "--gateway", gateway, "--", os.sys.executable,
               "-m", "huddleroom.cli", "serve", "--host", host, "--port", str(port)]
    if reload:
        command.append("--reload")
    try:
        os.execvpe(command[0], command, env)
    except OSError:
        raise OneCliError("OneCLI runtime could not be started; run huddleroom setup after checking OneCLI.") from None


class OneCliError(click.ClickException):
    """A user-safe OneCLI setup failure."""


def retained_resource_ids() -> tuple[str, ...]:
    """Return non-secret IDs created during this setup attempt only."""
    return _RETAINED_RESOURCE_IDS.get()


def _retain(resource_id: str) -> None:
    _RETAINED_RESOURCE_IDS.set((*_RETAINED_RESOURCE_IDS.get(), resource_id))


def _check_cli_capabilities() -> None:
    commands = (
        ("onecli", "agents", "credentials", "--help"),
        ("onecli", "agents", "grants", "list", "--help"),
        ("onecli", "agents", "grants", "attach-secret", "--help"),
        ("onecli", "run", "--help"),
    )
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=_TIMEOUT, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise OneCliError("OneCLI is unavailable or too old; install a supported OneCLI CLI.") from None
        output = (result.stdout + result.stderr).lower()
        required = {
            ("onecli", "agents", "credentials", "--help"): ("--id",),
            ("onecli", "agents", "grants", "list", "--help"): ("--id",),
            ("onecli", "agents", "grants", "attach-secret", "--help"): ("--id", "--secret-id"),
            ("onecli", "run", "--help"): ("--agent", "--gateway"),
        }[command]
        if result.returncode != 0 or any(option not in output for option in required):
            raise OneCliError("OneCLI is unsupported; install a CLI with agent credentials and additive grants.")


def _api_key() -> str | None:
    key = os.environ.get("ONECLI_API_KEY")
    if key:
        return key
    directory = "credentials-dev" if os.environ.get("ONECLI_ENV") == "dev" else "credentials"
    try:
        value = (Path.home() / ".onecli" / directory / "api-key").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _configured_project() -> str | None:
    if project := os.environ.get("ONECLI_PROJECT"):
        return project
    try:
        filename = "config-dev.json" if os.environ.get("ONECLI_ENV") == "dev" else "config.json"
        value = json.loads((Path.home() / ".onecli" / filename).read_text(encoding="utf-8")).get("project")
        return value if isinstance(value, str) and value else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _request(
    config: Settings,
    method: str,
    path: str,
    *,
    body: dict | None = None,
    project_id: str | None = None,
    expected_status: int | None = None,
) -> Any:
    headers: dict[str, str] = {"Accept": "application/json"}
    if key := _api_key():
        headers["Authorization"] = f"Bearer {key}"
    if project_id:
        headers["X-Project-Id"] = project_id
    try:
        with httpx.Client(timeout=_TIMEOUT, trust_env=False, follow_redirects=False) as client:
            response = client.request(method, f"{config.onecli_management_url}/v1{path}", headers=headers, json=body)
    except httpx.HTTPError:
        raise OneCliError("OneCLI management service is unavailable.") from None
    if response.status_code in {401, 403}:
        raise OneCliError("OneCLI management access was denied; log in or request permission.")
    if expected_status is not None and response.status_code != expected_status:
        if method == "POST":
            raise OneCliError(
                "OneCLI did not confirm credential creation. Check OneCLI inventory before retrying; no local changes were written."
            )
        raise OneCliError("OneCLI management service returned an unsupported response.")
    if not 200 <= response.status_code < 300:
        raise OneCliError("OneCLI management request failed. Check the service and selected project.")
    try:
        return response.json()
    except ValueError:
        raise OneCliError("OneCLI management service returned an unsupported response.") from None


def _gateway_health(config: Settings) -> None:
    try:
        with httpx.Client(timeout=_TIMEOUT, trust_env=False, follow_redirects=False) as client:
            response = client.request("GET", f"{config.onecli_gateway_url}/healthz")
    except httpx.HTTPError:
        raise OneCliError("OneCLI gateway is unavailable.") from None
    if not 200 <= response.status_code < 300:
        raise OneCliError("OneCLI gateway is unavailable.")


def _project_id(config: Settings) -> str | None:
    slug = _configured_project()
    if not slug:
        return None
    projects = _request(config, "GET", "/projects")
    if not isinstance(projects, list):
        raise OneCliError("OneCLI management service returned an unsupported response.")
    for project in projects:
        if isinstance(project, dict) and project.get("slug") == slug and isinstance(project.get("id"), str):
            return project["id"]
    raise OneCliError("The configured OneCLI project could not be found.")


def _agents(config: Settings, project_id: str | None) -> list[dict[str, Any]]:
    value = _request(config, "GET", "/agents", project_id=project_id)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise OneCliError("OneCLI management service returned an unsupported response.")
    return value


def _effective(config: Settings, agent_id: str, project_id: str | None) -> dict[str, Any]:
    value = _request(config, "GET", f"/agents/{agent_id}/effective-credentials", project_id=project_id)
    if (
        not isinstance(value, dict)
        or value.get("agentId") != agent_id
        or not isinstance(value.get("mode"), str)
        or not value["mode"]
        or not isinstance(value.get("secrets"), list)
        or not isinstance(value.get("connections"), list)
    ):
        raise OneCliError("OneCLI management service returned an unsupported response.")
    for secret in value["secrets"]:
        if (
            not isinstance(secret, dict)
            or secret.get("kind") != "secret"
            or not all(isinstance(secret.get(field), str) and secret[field] for field in ("id", "name", "host", "status"))
            or not isinstance(secret.get("provenance"), list)
        ):
            raise OneCliError("OneCLI management service returned an unsupported response.")
        if secret["status"] not in {"usable", "limited", "blocked", "unknown"}:
            raise OneCliError("OneCLI management service returned an unsupported response.")
    return value


def verify_onecli(config: Settings, *, require_agent: bool = True) -> dict[str, str]:
    """Check the supported CLI/service contract and resolve the selected gateway agent."""
    _check_cli_capabilities()
    health = _request(config, "GET", "/health")
    if not isinstance(health, dict):
        raise OneCliError("OneCLI management service returned an unsupported response.")
    _gateway_health(config)
    project_id = _project_id(config)
    if not config.onecli_agent:
        if require_agent:
            raise OneCliError("OneCLI mode requires a gateway agent identifier.")
        return {}
    found = next((agent for agent in _agents(config, project_id) if agent.get("identifier") == config.onecli_agent), None)
    if not found or not all(isinstance(found.get(field), str) for field in ("id", "identifier", "name")):
        raise OneCliError("The selected OneCLI gateway agent was not found.")
    _effective(config, found["id"], project_id)
    return {field: found[field] for field in ("id", "identifier", "name")}


def _credentials(config: Settings, agent_id: str, project_id: str | None) -> list[dict[str, Any]]:
    metadata = _request(config, "GET", "/secrets", project_id=project_id)
    if not isinstance(metadata, list) or any(not isinstance(item, dict) for item in metadata):
        raise OneCliError("OneCLI management service returned an unsupported response.")
    effective = _effective(config, agent_id, project_id)
    statuses = {(item["id"], item["host"]): item["status"] for item in effective["secrets"]}
    results = []
    for item in metadata:
        secret_id, secret_type, host, name = item.get("id"), item.get("type"), item.get("hostPattern"), item.get("name")
        auth_mode = item.get("metadata", {}).get("authMode") if isinstance(item.get("metadata"), dict) else None
        if isinstance(secret_id, str) and secret_type in _RECIPES and host == _RECIPES[secret_type]["host"] and auth_mode == "api-key":
            results.append(
                {
                    "id": secret_id,
                    "type": secret_type,
                    "status": statuses.get((secret_id, host), "present but inaccessible"),
                }
            )
        elif isinstance(secret_id, str):
            results.append(
                {
                    "id": secret_id,
                    "name": name if isinstance(name, str) else "OneCLI credential",
                    "type": secret_type if isinstance(secret_type, str) else "unknown",
                    "status": "manual/unsupported",
                    "manual": True,
                }
            )
    return results


def _create_agent(config: Settings, identifier: str, project_id: str | None) -> dict[str, str]:
    name = click.prompt("OneCLI gateway agent name", default="HuddleRoom gateway")
    value = _request(config, "POST", "/agents", body={"name": name, "identifier": identifier}, project_id=project_id)
    if not isinstance(value, dict) or not all(isinstance(value.get(field), str) for field in ("id", "identifier", "name")):
        raise OneCliError("OneCLI management service returned an unsupported response.")
    _retain(value["id"])
    click.echo(f"Created OneCLI gateway agent {value['id']}. Retain this ID if setup stops before saving.")
    return {field: value[field] for field in ("id", "identifier", "name")}


def _grants(config: Settings, agent_id: str, project_id: str | None) -> dict[str, Any]:
    value = _request(config, "GET", f"/agents/{agent_id}/grants", project_id=project_id)
    if (
        not isinstance(value, dict)
        or value.get("agentId") != agent_id
        or value.get("mode") != "grants"
        or not isinstance(value.get("connections"), list)
        or not isinstance(value.get("secrets"), list)
    ):
        raise OneCliError("OneCLI management service returned an unsupported response.")
    for secret in value["secrets"]:
        if not isinstance(secret, dict) or not all(
            isinstance(secret.get(field), str) and secret[field] for field in ("secretId", "name", "type", "scope")
        ):
            raise OneCliError("OneCLI management service returned an unsupported response.")
    return value


def _attach_secret(config: Settings, agent_id: str, secret_id: str, project_id: str | None) -> None:
    # The endpoint is explicitly additive and accepts no body.  Re-read
    # effective access below; a grant can still be blocked by policy.
    _grants(config, agent_id, project_id)
    _request(config, "PUT", f"/agents/{agent_id}/grants/secrets/{secret_id}", project_id=project_id)


def _create_secret(config: Settings, provider: str, project_id: str | None) -> str:
    recipe = _RECIPES[provider]
    name = click.prompt(f"Name for new {provider.title()} credential", default=f"HuddleRoom {provider.title()}").strip()
    value = click.prompt(f"{provider.title()} API key", hide_input=True, confirmation_prompt=True).strip()
    if not 1 <= len(name) <= 255 or not 1 <= len(value) <= 10000 or not value.startswith(recipe["prefix"]):
        raise OneCliError("The credential name or API key format is invalid.")
    if provider == "anthropic" and value.startswith("sk-ant-oat"):
        raise OneCliError("Anthropic OAuth credentials must be configured manually in OneCLI.")
    created = _request(
        config,
        "POST",
        "/secrets",
        body={"name": name, "type": provider, "valueSource": "inline", "value": value, "hostPattern": recipe["host"]},
        project_id=project_id,
        expected_status=201,
    )
    if not isinstance(created, dict) or not isinstance(created.get("id"), str):
        raise OneCliError("OneCLI management service returned an unsupported response.")
    _retain(created["id"])
    # Do not expose creation previews: they include secret characters.
    click.echo(f"Created OneCLI credential {created['id']}. Retain this ID if setup stops before saving.")
    return created["id"]


def _provider_for_model(model: str | None) -> str | None:
    if not model:
        return None
    if model == "text-embedding-3-small":
        return "openai"
    prefix = model.split("/", 1)[0].lower()
    return prefix if prefix in _RECIPES else None


def report_onecli_readiness(config: Settings, credentials: list[dict[str, Any]]) -> None:
    usable = {item["type"] for item in credentials if item["status"] == "usable"}

    def report(operation: str, model: str | None) -> None:
        provider = _provider_for_model(model)
        if provider is None:
            state = "manual/unknown" if model else "uses orchestration fallback"
        elif provider in usable:
            state = "credential access known"
        else:
            provider_name = "OpenAI" if provider == "openai" else provider.title()
            state = f"missing usable {provider_name} credential"
        click.echo(f"{operation}: {state}.")

    report("Orchestration model", config.orchestration_model)
    report("Meeting/control-plane model", config.meeting_control_model or config.orchestration_model)
    click.echo("API agents: model-specific/manual until an API-agent model is selected.")
    report("Embeddings", config.embedding_model)


def setup_onecli(config: Settings, *, prompt_agent: bool = True) -> dict[str, str]:
    """Interactively choose a gateway identity and report conservative provider access."""
    _RETAINED_RESOURCE_IDS.set(())
    _check_cli_capabilities()
    _request(config, "GET", "/health")
    _gateway_health(config)
    project_id = _project_id(config)
    identifier = (
        click.prompt("OneCLI gateway agent", default=config.onecli_agent or "huddleroom")
        if prompt_agent
        else config.onecli_agent
    )
    if not identifier:
        raise OneCliError("OneCLI mode requires a gateway agent identifier.")
    agents = _agents(config, project_id)
    selected = next((item for item in agents if item.get("identifier") == identifier), None)
    if selected is None:
        if not click.confirm(f"Create OneCLI gateway agent '{identifier}'?", default=True):
            raise click.Abort()
        selected = _create_agent(config, identifier, project_id)
    if not all(isinstance(selected.get(field), str) for field in ("id", "identifier", "name")):
        raise OneCliError("OneCLI management service returned an unsupported response.")
    grants = _grants(config, selected["id"], project_id)
    granted_ids = {item["secretId"] for item in grants["secrets"]}
    available = _credentials(config, selected["id"], project_id)
    for provider in _RECIPES:
        status = next(
            (item["status"] for item in available if item["type"] == provider and not item.get("manual")),
            "missing",
        )
        click.echo(f"{provider.title()} credential: {status}.")
    for item in available:
        if item.get("manual"):
            click.echo(f"{item['name']}: manual/unsupported; configure it in OneCLI.")
    for item in available:
        if item.get("manual"):
            continue
        if item["status"] == "usable":
            continue
        if item["id"] in granted_ids:
            click.echo(f"{item['type'].title()} credential {item['id']} is granted but {item['status']}; policy was not changed.")
            continue
        if click.confirm(f"Attach existing {item['type'].title()} credential {item['id']} additively?", default=False):
            _attach_secret(config, selected["id"], item["id"], project_id)
            available = _credentials(config, selected["id"], project_id)
            grants = _grants(config, selected["id"], project_id)
            granted_ids = {entry["secretId"] for entry in grants["secrets"]}
    for provider in _RECIPES:
        if any(item["type"] == provider and not item.get("manual") for item in available):
            continue
        if click.confirm(f"Add a new {provider.title()} API-key credential to OneCLI?", default=False):
            secret_id = _create_secret(config, provider, project_id)
            # Server settings may auto-attach.  Never retry POST after an
            # uncertain result; inspect inventory/effective access instead.
            available = _credentials(config, selected["id"], project_id)
            grants = _grants(config, selected["id"], project_id)
            granted_ids = {entry["secretId"] for entry in grants["secrets"]}
            if secret_id not in granted_ids and click.confirm(
                f"Attach credential {secret_id} additively?", default=False
            ):
                _attach_secret(config, selected["id"], secret_id, project_id)
                available = _credentials(config, selected["id"], project_id)
    click.echo("Credential access is representative; path or method policy may still deny model or embedding requests.")
    return {
        "credential_mode": "onecli",
        "onecli_agent": selected["identifier"],
        "onecli_management_url": config.onecli_management_url,
        "onecli_gateway_url": config.onecli_gateway_url,
    }
