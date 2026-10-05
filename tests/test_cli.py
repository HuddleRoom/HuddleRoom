import importlib
import os
import pathlib
import sqlite3
import stat
import subprocess
import sys

import pytest
from click.testing import CliRunner


PROVIDERS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": "OLLAMA_API_BASE",
}


@pytest.fixture
def setup_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    for key in PROVIDERS.values():
        monkeypatch.delenv(key, raising=False)
    sys.modules.pop("huddleroom.config", None)
    return home / ".huddleroom" / "config.toml"


def _invoke(args, input_text=""):
    cli = importlib.import_module("huddleroom.cli")
    return CliRunner().invoke(cli.main, ["setup", *args], input=input_text)


@pytest.mark.parametrize(("provider", "key"), PROVIDERS.items())
def test_setup_writes_only_the_selected_provider_secret(setup_home, provider, key):
    result = _invoke(
        ["--provider", provider, "--orchestration-model", "openai/gpt-test", "--database-path", "data.db", "--workspace-dir", "work"],
        "provider-secret\n",
    )

    assert result.exit_code == 0, result.output
    saved = setup_home.read_text()
    assert f'{key} = "provider-secret"' in saved
    assert "OPENAI_API_KEY" not in saved if key != "OPENAI_API_KEY" else True
    assert "database_url = \"sqlite+aiosqlite:///data.db\"" in saved
    assert 'workspace_dir = "work"' in saved
    assert "provider-secret" not in result.output


def test_setup_skip_writes_durable_settings_without_provider_marker(setup_home):
    result = _invoke(
        ["--provider", "skip", "--orchestration-model", "anthropic/test", "--database-path", "state.db", "--workspace-dir", "workspace"]
    )

    assert result.exit_code == 0, result.output
    saved = setup_home.read_text()
    assert "database_url" in saved
    assert "workspace_dir" in saved
    assert "API_KEY" not in saved
    assert "credential_mode" not in saved
    assert "authenticated cli agents" in result.output.lower()
    assert "API-backed orchestration" in result.output


def test_setup_explicit_missing_cli_leaves_config_unchanged(setup_home, monkeypatch):
    setup_home.parent.mkdir()
    original = 'custom = "keep"\n'
    setup_home.write_text(original)
    cli = importlib.import_module("huddleroom.cli")
    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)

    result = _invoke(
        ["--orchestration-backend", "claude", "--database-path", "state.db", "--workspace-dir", "workspace"]
    )

    assert result.exit_code != 0
    assert setup_home.read_text() == original
    assert "claude" in result.output.lower()


def test_setup_rejects_cli_backend_with_provider_flags_without_writing(setup_home, monkeypatch):
    setup_home.parent.mkdir()
    original = 'custom = "keep"\n'
    setup_home.write_text(original)
    cli = importlib.import_module("huddleroom.cli")
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/bin/{name}" if name == "claude" else None)

    result = _invoke(
        ["--orchestration-backend", "claude", "--provider", "openai", "--database-path", "state.db", "--workspace-dir", "workspace"]
    )

    assert result.exit_code != 0
    assert setup_home.read_text() == original
    assert "provider" in result.output.lower()


def test_setup_cli_backend_preserves_api_model_and_saves_default_effort_removal(setup_home, monkeypatch):
    setup_home.parent.mkdir()
    setup_home.write_text('orchestration_model = "openai/saved"\norchestration_effort = "high"\n')
    cli = importlib.import_module("huddleroom.cli")
    completion = importlib.import_module("huddleroom.services.orchestration_completion")
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/bin/{name}" if name == "codex" else None)
    monkeypatch.setattr(completion, "is_orchestration_backend_supported", lambda _backend: True)

    result = _invoke(
        [
            "--orchestration-backend", "codex", "--orchestration-effort", "default",
            "--database-path", "state.db", "--workspace-dir", "workspace",
        ]
    )

    assert result.exit_code == 0, result.output
    saved = setup_home.read_text()
    assert 'orchestration_backend = "codex"' in saved
    assert "orchestration_effort" not in saved
    assert 'orchestration_model = "openai/saved"' in saved


def test_setup_rejects_installed_but_unsupported_cli_without_writing(setup_home, monkeypatch):
    setup_home.parent.mkdir()
    original = 'custom = "keep"\n'
    setup_home.write_text(original)
    cli = importlib.import_module("huddleroom.cli")
    completion = importlib.import_module("huddleroom.services.orchestration_completion")
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/bin/{name}" if name == "codex" else None)
    monkeypatch.setattr(completion, "is_orchestration_backend_supported", lambda _backend: False)

    result = _invoke(["--orchestration-backend", "codex", "--database-path", "state.db", "--workspace-dir", "workspace"])

    assert result.exit_code != 0
    assert setup_home.read_text() == original
    assert "installed" in result.output.lower()
    assert "unsupported" in result.output.lower()


def test_setup_switch_preserves_compatible_saved_effort(setup_home, monkeypatch):
    setup_home.parent.mkdir()
    setup_home.write_text('orchestration_effort = "high"\n')
    cli = importlib.import_module("huddleroom.cli")
    completion = importlib.import_module("huddleroom.services.orchestration_completion")
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/bin/{name}" if name == "codex" else None)
    monkeypatch.setattr(completion, "is_orchestration_backend_supported", lambda _backend: True)
    monkeypatch.setattr(completion, "supported_orchestration_efforts", lambda *_args: frozenset({"high"}))

    result = _invoke(["--orchestration-backend", "codex", "--database-path", "state.db", "--workspace-dir", "workspace"])

    assert result.exit_code == 0, result.output
    assert 'orchestration_effort = "high"' in setup_home.read_text()


def test_setup_rejects_incompatible_saved_effort_on_noninteractive_switch(setup_home, monkeypatch):
    setup_home.parent.mkdir()
    original = 'orchestration_effort = "none"\n'
    setup_home.write_text(original)
    cli = importlib.import_module("huddleroom.cli")
    completion = importlib.import_module("huddleroom.services.orchestration_completion")
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/bin/{name}" if name == "codex" else None)
    monkeypatch.setattr(completion, "is_orchestration_backend_supported", lambda _backend: True)
    monkeypatch.setattr(completion, "supported_orchestration_efforts", lambda *_args: frozenset({"high"}))

    result = _invoke(["--orchestration-backend", "codex", "--database-path", "state.db", "--workspace-dir", "workspace"])

    assert result.exit_code != 0
    assert setup_home.read_text() == original
    assert "default" in result.output.lower()


def test_interactive_setup_reprompts_for_a_compatible_saved_effort(monkeypatch):
    import huddleroom.cli as cli

    monkeypatch.setattr(cli, "_supported_orchestration_efforts", lambda *_args: frozenset({"high"}))
    seen = {}
    monkeypatch.setattr(cli.click, "prompt", lambda *_args, **kwargs: seen.setdefault("default", kwargs["default"]) and "default")

    assert cli._resolve_orchestration_effort("api", "model", None, "high", interactive=True) == (None, True)
    assert seen["default"] == "high"


def test_setup_keeps_blank_existing_secret_and_preserves_unrelated_toml(setup_home):
    setup_home.parent.mkdir()
    original = '# keep this comment\ncustom = "line\\nvalue" # keep inline\nOPENAI_API_KEY = "old-secret" # keep secret comment\n'
    setup_home.write_text(original)

    result = _invoke(
        ["--provider", "openai", "--orchestration-model", "openai/new", "--database-path", "next.db", "--workspace-dir", "next-work"],
        "\n",
    )

    assert result.exit_code == 0, result.output
    saved = setup_home.read_text()
    assert '# keep this comment\ncustom = "line\\nvalue" # keep inline\n' in saved
    assert 'OPENAI_API_KEY = "old-secret" # keep secret comment' in saved
    assert 'orchestration_model = "openai/new"' in saved


def test_setup_rewrites_owned_values_without_losing_inline_comments(setup_home):
    setup_home.parent.mkdir()
    setup_home.write_text(
        'database_url = "sqlite+aiosqlite:///old.db" # database note\n'
        'workspace_dir = "old-work" # workspace note\n'
        'OPENAI_API_KEY = "old-secret" # credential note\n'
    )

    result = _invoke(
        ["--provider", "openai", "--orchestration-model", "openai/new", "--database-path", "next.db", "--workspace-dir", "next-work"],
        "new-secret\n",
    )

    assert result.exit_code == 0, result.output
    saved = setup_home.read_text()
    assert 'database_url = "sqlite+aiosqlite:///next.db" # database note' in saved
    assert 'workspace_dir = "next-work" # workspace note' in saved
    assert 'OPENAI_API_KEY = "new-secret" # credential note' in saved
    assert "new-secret" not in result.output


def test_setup_refuses_quoted_owned_keys_without_writing(setup_home):
    setup_home.parent.mkdir()
    original = '"workspace_dir" = "old-work"\n'
    setup_home.write_text(original)

    result = _invoke(
        ["--provider", "skip", "--orchestration-model", "openai/test", "--database-path", "state.db", "--workspace-dir", "work"]
    )

    assert result.exit_code != 0
    assert setup_home.read_text() == original
    assert "manually edit" in result.output.lower()


def test_setup_refuses_multiline_owned_values_without_writing(setup_home):
    setup_home.parent.mkdir()
    original = 'workspace_dir = """old\nwork"""\n'
    setup_home.write_text(original)

    result = _invoke(
        ["--provider", "skip", "--orchestration-model", "openai/test", "--database-path", "state.db", "--workspace-dir", "work"]
    )

    assert result.exit_code != 0
    assert setup_home.read_text() == original
    assert "manually edit" in result.output.lower()


def test_setup_preserves_unrelated_crlf_bytes(setup_home):
    setup_home.parent.mkdir()
    original = b'# preserve\r\ncustom = "value"\r\nworkspace_dir = "old-work"\r\n'
    setup_home.write_bytes(original)

    result = _invoke(
        ["--provider", "skip", "--orchestration-model", "openai/test", "--database-path", "state.db", "--workspace-dir", "work"]
    )

    saved = setup_home.read_bytes()
    assert result.exit_code == 0, result.output
    assert b'# preserve\r\ncustom = "value"\r\n' in saved
    assert b"\n" not in saved.replace(b"\r\n", b"")


def test_setup_rejects_invalid_candidate_even_when_environment_shadows_it(setup_home, monkeypatch):
    setup_home.parent.mkdir()
    original = 'auth_enabled = true # sentinel-secret\n'
    setup_home.write_text(original)
    monkeypatch.setenv("HUDDLEROOM_AUTH_ENABLED", "false")

    result = _invoke(["--provider", "skip", "--orchestration-model", "openai/test", "--database-path", "state.db", "--workspace-dir", "work"])

    assert result.exit_code != 0
    assert setup_home.read_text() == original
    assert "sentinel-secret" not in result.output
    assert "No changes were written" in result.output


def test_setup_validation_ignores_unrelated_ambient_settings(setup_home, monkeypatch):
    monkeypatch.setenv("HUDDLEROOM_AUTH_ENABLED", "true")

    result = _invoke(
        ["--provider", "skip", "--orchestration-model", "openai/test", "--database-path", "state.db", "--workspace-dir", "work"]
    )

    assert result.exit_code == 0, result.output


def test_setup_reports_malformed_existing_toml_without_secret_or_write(setup_home):
    setup_home.parent.mkdir()
    original = 'OPENAI_API_KEY = "sentinel-secret"\nnot valid toml = [\n'
    setup_home.write_text(original)

    result = _invoke(["--provider", "skip", "--orchestration-model", "openai/test", "--database-path", "state.db", "--workspace-dir", "work"])

    assert result.exit_code != 0
    assert setup_home.read_text() == original
    assert "manually repair" in result.output.lower()
    assert str(setup_home) in result.output
    assert "sentinel-secret" not in result.output


def test_setup_abort_leaves_no_file(setup_home):
    result = _invoke([], "\x03")

    assert result.exit_code != 0
    assert not setup_home.exists()


def test_setup_writes_owner_only_permissions(setup_home):
    result = _invoke(["--provider", "skip", "--orchestration-model", "openai/test", "--database-path", "state.db", "--workspace-dir", "work"])

    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(os.stat(setup_home).st_mode) == 0o600


def test_setup_does_not_report_failure_after_atomic_replace(setup_home, monkeypatch):
    cli = importlib.import_module("huddleroom.cli")
    original_chmod = cli.os.chmod

    def chmod(path, mode):
        if pathlib.Path(path) == setup_home:
            raise OSError("after replace")
        original_chmod(path, mode)

    monkeypatch.setattr(cli.os, "chmod", chmod)
    result = CliRunner().invoke(
        cli.main,
        ["setup", "--provider", "skip", "--orchestration-model", "openai/test", "--database-path", "state.db", "--workspace-dir", "work"],
    )

    assert result.exit_code == 0, result.output
    assert setup_home.exists()


def test_setup_does_not_copy_environment_secret_or_replace_existing_toml_secret(setup_home, monkeypatch):
    setup_home.parent.mkdir()
    setup_home.write_text('OPENAI_API_KEY = "toml-secret"\n')
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    result = _invoke(
        ["--provider", "openai", "--orchestration-model", "openai/test", "--database-path", "state.db", "--workspace-dir", "work"],
        "\n",
    )

    assert result.exit_code == 0, result.output
    saved = setup_home.read_text()
    assert 'OPENAI_API_KEY = "toml-secret"' in saved
    assert "environment-secret" not in saved


def test_setup_replace_failure_keeps_original_file_and_hides_secret(setup_home, monkeypatch):
    setup_home.parent.mkdir()
    original = 'OPENAI_API_KEY = "old-secret"\n'
    setup_home.write_text(original)
    cli = importlib.import_module("huddleroom.cli")
    monkeypatch.setattr(cli.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("sentinel-secret")))

    result = CliRunner().invoke(
        cli.main,
        ["setup", "--provider", "openai", "--orchestration-model", "openai/test", "--database-path", "state.db", "--workspace-dir", "work"],
        input="new-secret\n",
    )

    assert result.exit_code != 0
    assert setup_home.read_text() == original
    assert "new-secret" not in result.output
    assert "sentinel-secret" not in result.output


def test_run_migrations_creates_schema_and_is_idempotent(tmp_path, monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.config as config

    database = tmp_path / "state" / "huddleroom.db"
    configured = config.Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{database}",
        workspace_dir=str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(config, "settings", configured)
    monkeypatch.chdir(tmp_path)

    cli._run_migrations(configured)
    with sqlite3.connect(database) as connection:
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'projects'"
        ).fetchone()

    cli._run_migrations(configured)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == version
    assert (tmp_path / "workspace").is_dir()


def test_run_migrations_keeps_existing_llm_logger_enabled(tmp_path, monkeypatch):
    import logging
    import huddleroom.cli as cli
    import huddleroom.config as config

    configured = config.Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'huddleroom.db'}",
    )
    logger = logging.getLogger("huddleroom.llm")
    monkeypatch.setattr(config, "settings", configured)
    monkeypatch.setattr(logger, "disabled", False)

    cli._run_migrations(configured)

    assert logger.disabled is False


def test_run_migrations_creates_parent_for_disk_sqlite_uri(tmp_path, monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.config as config

    database = tmp_path / "nested" / "state.db"
    configured = config.Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///file:{database}?uri=true",
    )
    monkeypatch.setattr(config, "settings", configured)

    cli._run_migrations(configured)

    assert database.is_file()


def test_serve_migrates_before_banner_and_uvicorn(monkeypatch, tmp_path):
    import huddleroom.cli as cli
    import huddleroom.config as config

    configured = config.Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'huddleroom.db'}",
    )
    calls = []
    monkeypatch.setattr(config, "settings", configured)
    monkeypatch.setattr(cli, "_run_migrations", lambda settings: calls.append(settings))
    monkeypatch.setitem(sys.modules, "uvicorn", type("Uvicorn", (), {"run": lambda *_args, **_kwargs: calls.append("uvicorn")}))

    result = CliRunner().invoke(cli.main, ["serve"])

    assert result.exit_code == 0, result.output
    assert calls == [configured, "uvicorn"]
    assert result.output.startswith("Starting HuddleRoom")


def test_serve_reports_startup_failure_before_uvicorn_without_credentials(monkeypatch, tmp_path):
    import huddleroom.cli as cli
    import huddleroom.config as config

    configured = config.Settings(
        _env_file=None,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'huddleroom.db'}",
    )
    run = lambda *_args, **_kwargs: pytest.fail("Uvicorn must not start")
    monkeypatch.setattr(config, "settings", configured)
    monkeypatch.setattr(cli, "_run_migrations", lambda _settings: (_ for _ in ()).throw(OSError("private path")))
    monkeypatch.setitem(sys.modules, "uvicorn", type("Uvicorn", (), {"run": run}))

    result = CliRunner().invoke(cli.main, ["serve"])

    assert result.exit_code != 0
    assert "huddleroom setup" in result.output
    assert "~/.huddleroom/config.toml" in result.output
    assert "HUDDLEROOM_" in result.output
    assert "private path" not in result.output


def test_serve_preflights_orchestration_backend_before_onecli_or_database(monkeypatch, tmp_path):
    import huddleroom.cli as cli
    import huddleroom.config as config
    import huddleroom.onecli as onecli
    import huddleroom.services.orchestration_completion as completion

    configured = config.Settings(
        _env_file=None,
        orchestration_backend="codex",
        credential_mode="onecli",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'huddleroom.db'}",
    )
    monkeypatch.setattr(config, "settings", configured)
    monkeypatch.setattr(
        completion,
        "validate_orchestration_backend",
        lambda _config: (_ for _ in ()).throw(RuntimeError("unsupported backend")),
    )
    monkeypatch.setattr(onecli, "launch_onecli", lambda *_args: pytest.fail("OneCLI must not launch"))
    monkeypatch.setattr(cli, "_prepare_database", lambda *_args: pytest.fail("database must not be prepared"))

    result = CliRunner().invoke(cli.main, ["serve"])

    assert result.exit_code != 0
    assert "unsupported backend" in result.output


def test_serve_sanitizes_invalid_configuration_before_uvicorn(monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.config as config

    started = []
    monkeypatch.setattr(config, "settings", config.Settings(_env_file=None, auth_enabled=True))
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        type("Uvicorn", (), {"run": lambda *_args, **_kwargs: started.append(True)}),
    )

    result = CliRunner().invoke(cli.main, ["serve"])

    assert result.exit_code != 0
    assert not started
    assert "huddleroom setup" in result.output
    assert "~/.huddleroom/config.toml" in result.output
    assert "HUDDLEROOM_" in result.output


def test_onecli_serve_rejects_invalid_wrapped_context_before_migrations(monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.config as config
    import huddleroom.onecli as onecli

    configured = config.Settings(_env_file=None, credential_mode="onecli", onecli_agent="gateway")
    monkeypatch.setattr(config, "settings", configured)
    monkeypatch.setenv("ONECLI_GATEWAY", "true")
    monkeypatch.setattr(onecli, "validate_onecli_context", lambda *_args: (_ for _ in ()).throw(onecli.OneCliError("safe context failure")))
    monkeypatch.setattr(cli, "_prepare_database", lambda: pytest.fail("migration must not run"))

    result = CliRunner().invoke(cli.main, ["serve"])

    assert result.exit_code != 0
    assert "safe context failure" in result.output


def test_onecli_wrapped_serve_validates_then_migrates_without_rewrapping(monkeypatch, tmp_path):
    import huddleroom.cli as cli
    import huddleroom.config as config
    import huddleroom.onecli as onecli

    configured = config.Settings(
        _env_file=None, credential_mode="onecli", onecli_agent="gateway",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'huddleroom.db'}",
    )
    calls = []
    monkeypatch.setattr(config, "settings", configured)
    monkeypatch.setenv("ONECLI_GATEWAY", "true")
    monkeypatch.setattr(onecli, "validate_onecli_context", lambda value: calls.append(("validate", value)))
    monkeypatch.setattr(onecli, "launch_onecli", lambda *_args: pytest.fail("wrapped child must not rewrap"))
    monkeypatch.setattr(cli, "_run_migrations", lambda value: calls.append(("migrate", value)))
    monkeypatch.setitem(sys.modules, "uvicorn", type("Uvicorn", (), {"run": lambda *_args, **_kwargs: calls.append(("uvicorn", None))}))

    result = CliRunner().invoke(cli.main, ["serve"])

    assert result.exit_code == 0, result.output
    assert [entry[0] for entry in calls] == ["validate", "migrate", "uvicorn"]
    validated = calls[0][1]
    migrated = calls[1][1]
    assert validated is migrated
    assert validated.onecli_management_url == onecli.resolve_management_url(configured)


def test_run_migrations_upgrades_an_old_database_without_losing_sentinel(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    import huddleroom.cli as cli
    import huddleroom.config as config

    database = tmp_path / "huddleroom.db"
    configured = config.Settings(_env_file=None, database_url=f"sqlite+aiosqlite:///{database}")
    monkeypatch.setattr(config, "settings", configured)
    cli._run_migrations(configured)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES ('preserve me')")
    migration_config = Config(str(cli._migration_config_path()))
    migration_config.set_main_option("script_location", str(cli._migration_config_path().parent / "alembic"))
    command.downgrade(migration_config, "045")

    cli._run_migrations(configured)

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT value FROM sentinel").fetchone() == ("preserve me",)
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == ("047",)


def test_concurrent_first_migrations_leave_a_usable_database(tmp_path):
    database = tmp_path / "huddleroom.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sentinel (value TEXT)")
        connection.execute("INSERT INTO sentinel VALUES ('preserve me')")
    environment = os.environ | {
        "HOME": str(tmp_path / "home"),
        "HUDDLEROOM_DATABASE_URL": f"sqlite+aiosqlite:///{database}",
        "HUDDLEROOM_WORKSPACE_DIR": str(tmp_path / "workspace"),
    }
    command = [sys.executable, "-m", "huddleroom.cli", "init-db"]
    first = subprocess.Popen(
        command, cwd=pathlib.Path(__file__).parents[1], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    second = subprocess.Popen(
        command, cwd=pathlib.Path(__file__).parents[1], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    outcomes = [process.communicate(timeout=30) for process in (first, second)]
    results = [process.returncode for process in (first, second)]

    assert 0 in results
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == ("047",)
        assert connection.execute("SELECT value FROM sentinel").fetchone() == ("preserve me",)
    for returncode, output in zip(results, outcomes):
        if returncode:
            message = "".join(output)
            assert "huddleroom setup" in message
            assert "~/.huddleroom/config.toml" in message
            assert "HUDDLEROOM_" in message
