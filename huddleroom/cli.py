import json
import os
import pathlib
import re
import shutil
import sysconfig
import tempfile
import tomllib
from typing import TYPE_CHECKING

import click

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": "OLLAMA_API_BASE",
}
_ASSIGNMENT = re.compile(
    r"^(?P<before>\s*)(?P<key>[A-Za-z0-9_-]+)(?P<equals>\s*=\s*)(?P<value>.*?)(?P<newline>\r?\n)?$"
)

if TYPE_CHECKING:
    from huddleroom.config import Settings


def _migration_config_path() -> pathlib.Path:
    migrations = pathlib.Path(sysconfig.get_path("data")) / "huddleroom" / "migrations"
    config = migrations / "alembic.ini"
    if config.is_file() and (migrations / "versions").is_dir():
        return config
    return _PROJECT_ROOT / "alembic.ini"


def _run_migrations(config: "Settings") -> None:
    """Create local runtime directories and bring the configured database to head."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy.engine import make_url

    database_url = make_url(config.database_url)
    database = database_url.database
    if database and database != ":memory:":
        if database.startswith("file:"):
            if not (database.startswith("file::memory:") or database_url.query.get("mode") == "memory"):
                pathlib.Path(database[5:]).expanduser().parent.mkdir(parents=True, exist_ok=True)
        else:
            pathlib.Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(config.workspace_dir).expanduser().mkdir(parents=True, exist_ok=True)

    config_path = _migration_config_path().resolve()
    migration_config = Config(str(config_path))
    script_location = config_path.parent if config_path.parent.name == "migrations" else config_path.parent / "alembic"
    migration_config.set_main_option("script_location", str(script_location))
    command.upgrade(migration_config, "head")


def _prepare_database(config: "Settings | None" = None) -> "Settings":
    try:
        from huddleroom.config import settings as default_settings, validate_supported_settings
    except Exception:
        raise click.ClickException(
            "Could not prepare the local database. Run 'huddleroom setup', check "
            "~/.huddleroom/config.toml, or set HUDDLEROOM_ environment variables."
        ) from None
    try:
        if config is None:
            config = default_settings
        validate_supported_settings(config)
        _run_migrations(config)
    except click.ClickException:
        raise
    except Exception:
        raise click.ClickException(
            "Could not prepare the local database. Run 'huddleroom setup', check "
            "~/.huddleroom/config.toml, or set HUDDLEROOM_ environment variables."
        ) from None
    return config


@click.group()
def main():
    """HuddleRoom — Autonomous Agent Workspace Platform"""


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _inline_comment(value: str) -> str:
    quoted = False
    escaped = False
    for index, character in enumerate(value):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif character == "#" and not quoted:
            comment_start = index
            while comment_start and value[comment_start - 1] in " \t":
                comment_start -= 1
            return value[comment_start:]
    return ""


def _update_config(updates: dict[str, str | None]) -> None:
    """Validate a flat TOML candidate, then atomically replace only setup keys."""
    from huddleroom.config import DEFAULT_CONFIG_FILE, Settings, validate_supported_settings

    config_file = DEFAULT_CONFIG_FILE
    try:
        if config_file.exists():
            with config_file.open("r", encoding="utf-8", newline="") as config:
                original = config.read()
        else:
            original = ""
        existing = tomllib.loads(original) if original else {}
        if not isinstance(existing, dict):
            raise ValueError
    except Exception:
        raise click.ClickException("Invalid setup values. No changes were written.") from None

    remaining = dict(updates)
    source_newline = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines(keepends=True)
    rewritten: list[str] = []
    for line in lines:
        match = _ASSIGNMENT.match(line)
        if not match or match["key"] not in remaining:
            rewritten.append(line)
            continue
        if match["value"].lstrip().startswith(('"""', "'''")):
            raise click.ClickException(
                f"Could not safely update {config_file}. Please manually edit it. No changes were written."
            )
        comment = _inline_comment(match["value"])
        newline = match["newline"] or source_newline
        value = remaining.pop(match["key"])
        if value is None:
            continue
        rewritten.append(
            f'{match["before"]}{match["key"]}{match["equals"]}'
            f'{_toml_string(value)}{comment}{newline}'
        )
    if set(remaining).intersection(existing):
        raise click.ClickException(
            f"Could not safely update {config_file}. Please manually edit it. No changes were written."
        )
    if remaining:
        if rewritten and not rewritten[-1].endswith(("\n", "\r")):
            rewritten.append(source_newline)
        rewritten.extend(
            f"{key} = {_toml_string(value)}{source_newline}" for key, value in remaining.items() if value is not None
        )

    rendered = "".join(rewritten)
    try:
        rendered_values = tomllib.loads(rendered)
        if any(rendered_values.get(key) != value for key, value in updates.items()):
            raise ValueError
        candidate = {
            key: field.get_default(call_default_factory=True) for key, field in Settings.model_fields.items()
        }
        candidate.update({key: value for key, value in rendered_values.items() if key in candidate})
        validate_supported_settings(Settings(_env_file=None, **candidate))
    except Exception:
        raise click.ClickException("Invalid setup values. No changes were written.") from None

    temporary_name: str | None = None
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=config_file.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
            if os.name == "posix":
                os.chmod(temporary_name, 0o600)
            temporary.write(rendered)
        os.replace(temporary_name, config_file)
    except OSError:
        raise click.ClickException("Could not save setup values. No changes were written.") from None
    finally:
        if temporary_name:
            pathlib.Path(temporary_name).unlink(missing_ok=True)


def _database_path(database_url: str) -> str:
    prefix = "sqlite+aiosqlite:///"
    return database_url[len(prefix):] if database_url.startswith(prefix) else database_url


def _orchestration_backend_availability() -> dict[str, tuple[bool, bool]]:
    """Report PATH installation and backend availability separately."""
    from huddleroom.services.orchestration_completion import is_orchestration_backend_supported

    return {
        "api": (True, True),
        "claude": (shutil.which("claude") is not None, is_orchestration_backend_supported("claude")),
        "codex": (shutil.which("codex") is not None, is_orchestration_backend_supported("codex")),
    }


def _supported_orchestration_efforts(backend: str, model: str | None) -> frozenset[str]:
    try:
        from huddleroom.services.orchestration_completion import supported_orchestration_efforts

        return supported_orchestration_efforts(backend, model)
    except Exception as error:
        raise click.ClickException(str(error)) from None


def _resolve_orchestration_effort(
    backend: str,
    model: str | None,
    explicit_effort: str | None,
    saved_effort: str | None,
    *,
    interactive: bool,
) -> tuple[str | None, bool]:
    """Return the validated effort and whether setup must remove a saved override."""
    if explicit_effort == "default":
        return None, True
    supported = _supported_orchestration_efforts(backend, model)
    if explicit_effort is not None:
        if explicit_effort in supported:
            return explicit_effort, False
        raise click.ClickException(f"{explicit_effort!r} is not supported by the {backend} orchestration backend.")
    if interactive:
        choices = ["default", *sorted(supported)]
        effort = click.prompt(
            "Orchestration effort (default keeps the backend default)",
            type=click.Choice(choices),
            default=saved_effort if saved_effort in supported else "default",
        )
        return (None, True) if effort == "default" else (effort, False)
    if saved_effort in supported:
        return saved_effort, False
    if not interactive:
        if saved_effort is not None:
            raise click.ClickException(
                f"Saved orchestration effort {saved_effort!r} is not supported by the selected backend/model. "
                "Pass --orchestration-effort default to remove it. No changes were written."
            )
        return None, False


@main.command()
@click.option("--provider", type=click.Choice([*_PROVIDER_KEYS, "skip"], case_sensitive=False))
@click.option("--credential-mode", type=click.Choice(["direct", "onecli"], case_sensitive=False))
@click.option("--onecli-agent")
@click.option("--onecli-management-url")
@click.option("--onecli-gateway-url")
@click.option("--orchestration-backend", type=click.Choice(["api", "claude", "codex"], case_sensitive=False))
@click.option("--orchestration-effort")
@click.option("--orchestration-model")
@click.option("--orchestration-cli-model", help="Model for the claude/codex CLI backend ('default' removes it).")
@click.option("--database-path")
@click.option("--workspace-dir")
def setup(
    provider: str | None,
    credential_mode: str | None,
    onecli_agent: str | None,
    onecli_management_url: str | None,
    onecli_gateway_url: str | None,
    orchestration_backend: str | None = None,
    orchestration_effort: str | None = None,
    orchestration_model: str | None = None,
    orchestration_cli_model: str | None = None,
    database_path: str | None = None,
    workspace_dir: str | None = None,
):
    """Save local direct or OneCLI gateway settings without exposing secrets."""
    try:
        from huddleroom.config import DEFAULT_CONFIG_FILE, settings, validate_cli_model
    except Exception:
        config_file = pathlib.Path.home() / ".huddleroom" / "config.toml"
        raise click.ClickException(
            f"Could not read {config_file}. Please manually repair it before running huddleroom setup."
        ) from None

    availability = _orchestration_backend_availability()
    backend_was_prompted = orchestration_backend is None and provider is None and credential_mode is None
    if orchestration_backend is None:
        default_backend = settings.orchestration_backend if availability[settings.orchestration_backend][0] else "api"
        if backend_was_prompted:
            choices = [backend for backend, (installed, _supported) in availability.items() if installed]
            if not availability[settings.orchestration_backend][0]:
                click.echo(f"Configured {settings.orchestration_backend} orchestration backend is unavailable.")
            orchestration_backend = click.prompt(
                "Orchestration backend", type=click.Choice(choices), default=default_backend
            )
        else:
            orchestration_backend = "api"
    orchestration_backend = orchestration_backend.lower()
    if orchestration_backend == "api" and orchestration_cli_model is not None:
        raise click.ClickException(
            "--orchestration-cli-model only applies to claude/codex. No changes were written."
        )
    if orchestration_backend != "api" and any(
        value is not None
        for value in (provider, credential_mode, onecli_agent, onecli_management_url, onecli_gateway_url, orchestration_model)
    ):
        raise click.ClickException(
            "CLI orchestration setup cannot be combined with provider, credential, or orchestration-model options. "
            "Configure those separately. No changes were written."
        )
    installed, _supported = availability[orchestration_backend]
    if not installed:
        raise click.ClickException(
            f"The {orchestration_backend} CLI is not installed. Install it, then rerun setup. No changes were written."
        )
    if orchestration_backend != "api":
        # A saved model belongs to the backend it was saved for; switching backends drops it.
        switched = orchestration_backend != settings.orchestration_backend
        saved_model = None if switched else settings.orchestration_cli_model
        cli_model, update_cli_model = saved_model, switched
        if orchestration_cli_model is None and backend_was_prompted:
            orchestration_cli_model = click.prompt(
                f"Orchestration CLI model (Enter keeps {saved_model}, type default to use the CLI default)"
                if saved_model else "Orchestration CLI model (blank = CLI default)",
                default=saved_model or "",
                show_default=False,
            )
        if orchestration_cli_model is not None:
            try:
                cli_model = None if orchestration_cli_model.strip().lower() == "default" else validate_cli_model(orchestration_cli_model)
            except ValueError as error:
                raise click.ClickException(f"Invalid orchestration CLI model: {error}. No changes were written.") from None
            update_cli_model = True
        orchestration_effort, remove_effort = _resolve_orchestration_effort(
            orchestration_backend,
            cli_model,
            orchestration_effort,
            settings.orchestration_effort,
            interactive=backend_was_prompted,
        )
        database_path = database_path or click.prompt("Database path", default=_database_path(settings.database_url))
        workspace_dir = workspace_dir or click.prompt("Workspace directory", default=settings.workspace_dir)
        updates: dict[str, str | None] = {
            "orchestration_backend": orchestration_backend,
            "database_url": f"sqlite+aiosqlite:///{database_path}",
            "workspace_dir": workspace_dir,
        }
        if orchestration_effort is not None or remove_effort:
            updates["orchestration_effort"] = orchestration_effort
        if update_cli_model:
            updates["orchestration_cli_model"] = cli_model
        _update_config(updates)
        click.echo(f"Saved setup values to {DEFAULT_CONFIG_FILE}.")
        click.echo(f"Orchestration CLI model: {cli_model or 'CLI default'}.")
        click.echo(f"{orchestration_backend.title()} was found on PATH; authentication is not verified.")
        click.echo("Embeddings and API agents may still need separately configured provider credentials.")
        return

    # Existing non-interactive --provider invocations are direct setup for
    # compatibility.  Interactive setup asks mode before every other prompt.
    credential_mode_explicit = credential_mode is not None
    mode_was_prompted = credential_mode is None and provider is None
    if credential_mode is None:
        credential_mode = "direct" if provider is not None else click.prompt(
            "Credential mode", type=click.Choice(["direct", "onecli"]), default=settings.credential_mode
        )
    credential_mode = credential_mode.lower()
    onecli_updates: dict[str, str] = {}
    if credential_mode == "onecli":
        from huddleroom.config import Settings
        from huddleroom.onecli import resolve_management_url, setup_onecli

        candidate_values = dict(
            _env_file=None,
            credential_mode="onecli",
            onecli_agent=onecli_agent if onecli_agent is not None else settings.onecli_agent,
        )
        if onecli_management_url is not None:
            candidate_values["onecli_management_url"] = onecli_management_url
        elif settings.setting_was_supplied("onecli_management_url"):
            candidate_values["onecli_management_url"] = settings.onecli_management_url
        if onecli_gateway_url is not None:
            candidate_values["onecli_gateway_url"] = onecli_gateway_url
        elif settings.setting_was_supplied("onecli_gateway_url"):
            candidate_values["onecli_gateway_url"] = settings.onecli_gateway_url
        candidate = Settings(**candidate_values)
        candidate = candidate.model_copy(update={"onecli_management_url": resolve_management_url(candidate)})
        onecli_updates = setup_onecli(candidate, prompt_agent=onecli_agent is None)
        provider = "skip"
    else:
        provider = provider or click.prompt("Provider", type=click.Choice([*_PROVIDER_KEYS, "skip"]), default="skip")
        provider = provider.lower()
    orchestration_model = orchestration_model or click.prompt(
        "Orchestration model", default=settings.orchestration_model
    )
    orchestration_effort, remove_effort = _resolve_orchestration_effort(
        "api", orchestration_model, orchestration_effort, settings.orchestration_effort, interactive=backend_was_prompted
    )
    database_path = database_path or click.prompt("Database path", default=_database_path(settings.database_url))
    workspace_dir = workspace_dir or click.prompt("Workspace directory", default=settings.workspace_dir)
    updates: dict[str, str | None] = {
        "orchestration_model": orchestration_model,
        "database_url": f"sqlite+aiosqlite:///{database_path}",
        "workspace_dir": workspace_dir,
    }
    if backend_was_prompted or orchestration_backend != settings.orchestration_backend:
        updates["orchestration_backend"] = orchestration_backend
    if orchestration_effort is not None or remove_effort:
        updates["orchestration_effort"] = orchestration_effort
    updates.update(onecli_updates)
    if credential_mode == "direct" and (provider is not None) and provider != "skip":
        secret = click.prompt(f"{provider.title()} credential (leave blank to keep existing)", default="", hide_input=True, show_default=False)
        if secret:
            updates[_PROVIDER_KEYS[provider]] = secret
    if credential_mode == "direct" and (credential_mode_explicit or mode_was_prompted or provider != "skip"):
        updates["credential_mode"] = "direct"
    elif credential_mode == "direct" and (provider is None or provider == "skip"):
        # Only an explicit choice writes this marker; old --provider setup
        # stays byte-compatible while still behaving as direct setup.
        if credential_mode_explicit or any(value is not None for value in (onecli_agent, onecli_management_url, onecli_gateway_url)):
            updates["credential_mode"] = "direct"
    if credential_mode == "onecli":
        from huddleroom.config import Settings
        from huddleroom.onecli import _credentials, _project_id, report_onecli_readiness, verify_onecli

        final_config = settings.model_copy(update={**onecli_updates, "orchestration_model": orchestration_model})
        agent = verify_onecli(final_config)
        report_onecli_readiness(final_config, _credentials(final_config, agent["id"], _project_id(final_config)))
    try:
        _update_config(updates)
    except click.ClickException:
        if credential_mode == "onecli":
            from huddleroom.onecli import retained_resource_ids
            if identifiers := retained_resource_ids():
                click.echo("OneCLI resources were retained after the local save failure: " + ", ".join(identifiers), err=True)
        raise
    click.echo(f"Saved setup values to {DEFAULT_CONFIG_FILE}.")
    if credential_mode == "direct" and provider == "skip":
        click.echo(
            "Authenticated CLI agents can execute without an API key. API-backed orchestration, meeting control, "
            "embeddings, and API agents need provider credentials when used."
        )


@main.command()
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start HuddleRoom server (API + task runner + scheduler)."""
    from huddleroom.config import settings as configured_settings
    try:
        from huddleroom.services.orchestration_completion import validate_orchestration_backend

        validate_orchestration_backend(configured_settings)
    except Exception as error:
        raise click.ClickException(str(error)) from None
    if configured_settings.credential_mode == "onecli":
        from huddleroom.onecli import launch_onecli, resolve_management_url, validate_onecli_context
        configured_settings = configured_settings.model_copy(update={
            "onecli_management_url": resolve_management_url(configured_settings)
        })
        if os.environ.get("ONECLI_GATEWAY") == "true":
            validate_onecli_context(configured_settings)
        else:
            launch_onecli(configured_settings, host, port, reload)
    settings = _prepare_database(configured_settings)

    click.echo(f"Starting HuddleRoom on http://{host}:{port}")
    click.echo("Database: " + _get_db_url_display())
    click.echo("Dashboard: " + f"http://{host}:{port}/dashboard")
    click.echo("")

    import uvicorn

    uvicorn.run(
        "huddleroom.main:app",
        host=host,
        port=port,
        reload=reload,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "fmt": "%(levelprefix)s %(message)s",
                    "use_colors": None,
                    "()": "uvicorn.logging.DefaultFormatter",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "huddleroom": {
                    "handlers": ["default"],
                    "level": "DEBUG" if settings.debug else "INFO",
                    "propagate": False,
                },
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
            },
        },
    )


def _get_db_url_display() -> str:
    from huddleroom.config import settings
    url: str = settings.database_url
    if "sqlite" in url:
        return url.replace("sqlite+aiosqlite:///", "SQLite: ")  # pylint: disable=no-member
    return url.rsplit("@", maxsplit=1)[-1] if "@" in url else url  # pylint: disable=no-member


@main.command()
def init_db():
    """Initialize database tables (run migrations)."""
    _prepare_database()


if __name__ == "__main__":
    main()
