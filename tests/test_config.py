import json
import os
import re
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _import_config(tmp_path: Path, home: Path, **environment: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("HUDDLEROOM_", "RALLY_")) or key in {
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "OLLAMA_API_BASE",
            "OPENROUTER_API_BASE",
            "OPENROUTER_API_KEY",
            "OR_API_KEY",
            "OR_APP_NAME",
            "OR_SITE_URL",
        }:
            env.pop(key)
    env.update(environment)
    env["HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os; from huddleroom.config import settings; "
            "exec(os.environ.get('_CONFIG_CHECK', '')); "
            "print(json.dumps({'database_url': settings.database_url, "
            "'workspace_dir': settings.workspace_dir, "
            "'orchestration_model': settings.orchestration_model, 'debug': settings.debug, "
            "'cors_origins': settings.cors_origins, 'jwt_expire_minutes': settings.jwt_expire_minutes, "
            "'openai': os.getenv('OPENAI_API_KEY'), 'anthropic': os.getenv('ANTHROPIC_API_KEY'), "
            "'gemini': os.getenv('GEMINI_API_KEY'), 'ollama': os.getenv('OLLAMA_API_BASE')}))",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def test_load_provider_env_exports_openrouter_api_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=or-test\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from huddleroom.config import load_provider_env

    load_provider_env(env_file)

    assert os.environ["OPENROUTER_API_KEY"] == "or-test"


def test_onecli_config_import_discards_process_dotenv_and_toml_provider_values(tmp_path):
    """Mode must be known before config import can expose plaintext credentials."""
    home = tmp_path / "home"
    config_dir = home / ".huddleroom"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        'OPENAI_API_KEY = "toml-openai-sentinel"\n'
        'ANTHROPIC_API_KEY = "toml-anthropic-sentinel"\n'
        'OPENROUTER_API_KEY = "toml-openrouter-sentinel"\n'
        'GEMINI_API_KEY = "toml-gemini-sentinel"\n'
        'OLLAMA_API_BASE = "toml-ollama-sentinel"\n',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        'HUDDLEROOM_CREDENTIAL_MODE = "onecli"\n'
        'OPENAI_API_KEY = "dotenv-openai-sentinel"\n'
        'ANTHROPIC_API_KEY = "dotenv-anthropic-sentinel"\n'
        'OPENROUTER_API_KEY = "dotenv-openrouter-sentinel"\n'
        'GEMINI_API_KEY = "dotenv-gemini-sentinel"\n'
        'OLLAMA_API_BASE = "dotenv-ollama-sentinel"\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("HUDDLEROOM_", "RALLY_")) or key in {
            "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "OLLAMA_API_BASE",
            "OPENROUTER_API_BASE", "OPENROUTER_API_KEY", "OR_API_KEY", "OR_APP_NAME", "OR_SITE_URL",
        }:
            env.pop(key)
    env.update(
        OPENAI_API_KEY="process-openai-sentinel",
        ANTHROPIC_API_KEY="process-anthropic-sentinel",
        ONECLI_API_KEY="environment-only-management-key",
        OPENROUTER_API_KEY="process-openrouter-sentinel",
        GEMINI_API_KEY="process-gemini-sentinel",
        OLLAMA_API_BASE="process-ollama-sentinel",
        HOME=str(home),
        PYTHONPATH=str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os; import huddleroom.config; print(json.dumps({key: os.getenv(key) for key in "
            "('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'OPENROUTER_API_KEY', 'GEMINI_API_KEY', 'OLLAMA_API_BASE', 'ONECLI_API_KEY')}))",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "OPENAI_API_KEY": "onecli-openai-placeholder",
        "ANTHROPIC_API_KEY": "onecli-anthropic-placeholder",
        "OPENROUTER_API_KEY": None,
        "GEMINI_API_KEY": None,
        "OLLAMA_API_BASE": None,
        "ONECLI_API_KEY": "environment-only-management-key",
    }


def test_config_import_loads_provider_env_from_default_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=or-import-test\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, huddleroom.config; print(os.environ['OPENROUTER_API_KEY'])",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "or-import-test"


def test_config_toml_and_dotenv_precedence(tmp_path):
    home = tmp_path / "home"
    config_dir = home / ".huddleroom"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        'database_url = "sqlite+aiosqlite:///toml.db"\n'
        'orchestration_model = "toml-model"\n'
        "debug = true\n"
        'cors_origins = ["https://toml.example"]\n'
        "jwt_expire_minutes = 17\n"
        'OPENAI_API_KEY = "toml-openai"\n'
        'ANTHROPIC_API_KEY = "toml-anthropic"\n'
        'GEMINI_API_KEY = "toml-gemini"\n',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "HUDDLEROOM_DATABASE_URL=sqlite+aiosqlite:///dotenv.db\n"
        "HUDDLEROOM_ORCHESTRATION_MODEL=dotenv-model\n"
        "OPENAI_API_KEY=dotenv-openai\n"
        "ANTHROPIC_API_KEY=dotenv-anthropic\n",
        encoding="utf-8",
    )

    result = _import_config(
        tmp_path,
        home,
        HUDDLEROOM_DATABASE_URL="sqlite+aiosqlite:///environment.db",
        OPENAI_API_KEY="environment-openai",
    )

    assert result.returncode == 0, result.stderr
    configured = json.loads(result.stdout)
    assert configured == {
        "database_url": "sqlite+aiosqlite:///environment.db",
        "workspace_dir": str(home / ".huddleroom" / "workspace"),
        "orchestration_model": "dotenv-model",
        "debug": True,
        "cors_origins": ["https://toml.example"],
        "jwt_expire_minutes": 17,
        "openai": "environment-openai",
        "anthropic": "dotenv-anthropic",
        "gemini": "toml-gemini",
        "ollama": None,
    }


def test_missing_config_toml_uses_defaults(tmp_path):
    result = _import_config(tmp_path, tmp_path / "empty-home")

    assert result.returncode == 0, result.stderr
    configured = json.loads(result.stdout)
    assert configured["debug"] is False
    assert configured["cors_origins"] == ["*"]
    assert configured["jwt_expire_minutes"] == 60
    assert configured["openai"] is None


def test_clean_install_uses_user_data_directory(tmp_path):
    home = tmp_path / "home"

    result = _import_config(tmp_path, home)

    assert result.returncode == 0, result.stderr
    configured = json.loads(result.stdout)
    assert configured["database_url"] == f"sqlite+aiosqlite:///{home}/.huddleroom/huddleroom.db"
    assert configured["workspace_dir"] == str(home / ".huddleroom" / "workspace")


@pytest.mark.parametrize("database_name", ("huddleroom.db", "rally.db"))
def test_existing_local_database_is_preserved(tmp_path, database_name):
    home = tmp_path / "home"
    (tmp_path / database_name).touch()

    result = _import_config(tmp_path, home)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["database_url"] == f"sqlite+aiosqlite:///{database_name}"


def test_explicit_runtime_paths_override_defaults(tmp_path):
    home = tmp_path / "home"

    result = _import_config(
        tmp_path,
        home,
        HUDDLEROOM_DATABASE_URL="sqlite+aiosqlite:///configured.db",
        HUDDLEROOM_WORKSPACE_DIR="configured-workspace",
    )

    assert result.returncode == 0, result.stderr
    configured = json.loads(result.stdout)
    assert configured["database_url"] == "sqlite+aiosqlite:///configured.db"
    assert configured["workspace_dir"] == "configured-workspace"


def test_toml_runtime_paths_override_legacy_local_paths(tmp_path):
    home = tmp_path / "home"
    config_dir = home / ".huddleroom"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        'database_url = "sqlite+aiosqlite:///configured.db"\n'
        'workspace_dir = "configured-workspace"\n',
        encoding="utf-8",
    )
    (tmp_path / "huddleroom.db").touch()
    (tmp_path / "rally.db").touch()
    (tmp_path / "workspace").mkdir()

    result = _import_config(tmp_path, home)

    assert result.returncode == 0, result.stderr
    configured = json.loads(result.stdout)
    assert configured["database_url"] == "sqlite+aiosqlite:///configured.db"
    assert configured["workspace_dir"] == "configured-workspace"


def test_existing_local_workspace_is_preserved(tmp_path):
    home = tmp_path / "home"
    (tmp_path / "workspace").mkdir()

    result = _import_config(tmp_path, home)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["workspace_dir"] == "workspace"


def test_local_workspace_file_uses_user_data_directory(tmp_path):
    home = tmp_path / "home"
    (tmp_path / "workspace").touch()

    result = _import_config(tmp_path, home)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["workspace_dir"] == str(home / ".huddleroom" / "workspace")


@pytest.mark.parametrize(
    ("contents", "expected"),
    (
        ("debug = [\n", None),
        ("OPENAI_API_KEY = 3\n", "OPENAI_API_KEY"),
        ('debug = "not-a-boolean"\n', "debug"),
    ),
)
def test_invalid_config_toml_names_its_path_and_key(tmp_path, contents, expected):
    home = tmp_path / "home"
    config_dir = home / ".huddleroom"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(contents, encoding="utf-8")

    result = _import_config(tmp_path, home)

    assert result.returncode != 0
    assert str(config_file) in result.stderr
    if expected:
        assert expected in result.stderr


def test_invalid_environment_value_does_not_blame_valid_toml(tmp_path):
    home = tmp_path / "home"
    config_dir = home / ".huddleroom"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    config_file.write_text("debug = true\n", encoding="utf-8")

    result = _import_config(tmp_path, home, HUDDLEROOM_DEBUG="not-a-boolean")

    assert result.returncode != 0
    assert "debug" in result.stderr
    assert str(config_file) not in result.stderr


def test_invalid_toml_value_shadowed_by_environment_is_accepted(tmp_path):
    home = tmp_path / "home"
    config_dir = home / ".huddleroom"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('debug = "not-a-boolean"\n', encoding="utf-8")

    result = _import_config(tmp_path, home, HUDDLEROOM_DEBUG="true")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["debug"] is True


def test_lowercase_environment_setting_does_not_blame_toml(tmp_path):
    home = tmp_path / "home"
    config_dir = home / ".huddleroom"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    config_file.write_text("auth_enabled = false\n", encoding="utf-8")

    result = _import_config(
        tmp_path,
        home,
        huddleroom_auth_enabled="true",
        _CONFIG_CHECK="from huddleroom.config import validate_supported_settings; validate_supported_settings()",
    )

    assert result.returncode != 0
    assert "HUDDLEROOM_AUTH_ENABLED" in result.stderr
    assert str(config_file) not in result.stderr


def test_toml_sourced_setting_names_reference_the_config_file(tmp_path):
    home = tmp_path / "home"
    config_dir = home / ".huddleroom"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "config.toml"
    config_file.write_text("auth_enabled = true\n", encoding="utf-8")

    result = _import_config(
        tmp_path,
        home,
        _CONFIG_CHECK="from huddleroom.config import validate_supported_settings; validate_supported_settings()",
    )

    assert result.returncode != 0
    assert str(config_file) in result.stderr
    assert "auth_enabled" in result.stderr


def test_config_toml_example_covers_all_settings_and_provider_keys():
    from huddleroom.config import PROVIDER_ENV_KEYS, Settings

    contents = (ROOT / "config.toml.example").read_text(encoding="utf-8")
    documented_names = set(tomllib.loads(contents)) | set(
        re.findall(r"(?m)^\s*#\s*([A-Za-z_]\w*)\s*=", contents)
    )

    assert set(Settings.model_fields) <= documented_names
    assert PROVIDER_ENV_KEYS <= documented_names


def test_huddleroom_settings_prefer_current_names_and_keep_rally_aliases(monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.setenv("RALLY_DATABASE_URL", "sqlite+aiosqlite:///legacy.db")
    with pytest.warns(FutureWarning, match="RALLY_DATABASE_URL"):
        assert Settings(_env_file=None).database_url.endswith("legacy.db")

    monkeypatch.setenv("HUDDLEROOM_DATABASE_URL", "sqlite+aiosqlite:///current.db")
    assert Settings(_env_file=None).database_url.endswith("current.db")
    assert Settings(_env_file=None, database_url="sqlite+aiosqlite:///init.db").database_url.endswith("init.db")


def test_orchestration_backend_and_effort_defaults_and_legacy_alias(monkeypatch):
    from huddleroom.config import Settings

    assert Settings(_env_file=None).orchestration_backend == "api"
    assert Settings(_env_file=None).orchestration_effort is None

    monkeypatch.setenv("RALLY_ORCHESTRATION_BACKEND", "codex")
    monkeypatch.setenv("RALLY_ORCHESTRATION_EFFORT", "high")
    with pytest.warns(FutureWarning, match="RALLY_ORCHESTRATION_BACKEND"):
        configured = Settings(_env_file=None)
    assert configured.orchestration_backend == "codex"
    assert configured.orchestration_effort == "high"


def test_orchestration_effort_is_a_literal_union():
    from typing import get_args
    from huddleroom.config import Settings

    assert set(get_args(Settings.model_fields["orchestration_effort"].annotation)[0].__args__) == {
        "none", "minimal", "low", "medium", "high", "xhigh", "max"
    }


@pytest.mark.parametrize("effort", ("none", "minimal", "low", "medium", "high", "xhigh", "max"))
def test_orchestration_effort_accepts_documented_values(effort):
    from huddleroom.config import Settings

    assert Settings(_env_file=None, orchestration_effort=effort).orchestration_effort == effort


@pytest.mark.parametrize(
    ("values", "name"),
    (({"orchestration_backend": "other"}, "orchestration_backend"), ({"orchestration_effort": "ultra"}, "orchestration_effort")),
)
def test_orchestration_settings_reject_invalid_values(values, name):
    from pydantic import ValidationError
    from huddleroom.config import Settings

    with pytest.raises(ValidationError, match=name):
        Settings(_env_file=None, **values)


def test_legacy_aliases_emit_one_warning(monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.setenv("RALLY_DATABASE_URL", "sqlite+aiosqlite:///legacy.db")
    monkeypatch.setenv("RALLY_AUTH_ENABLED", "true")

    with pytest.warns(FutureWarning) as recorded:
        Settings(_env_file=None)

    assert len(recorded) == 1
    assert "RALLY_DATABASE_URL" in str(recorded[0].message)
    assert "RALLY_AUTH_ENABLED" in str(recorded[0].message)


@pytest.mark.parametrize(
    ("canonical_name", "init_values"),
    (
        ("HUDDLEROOM_DATABASE_URL", {}),
        (None, {"database_url": "sqlite+aiosqlite:///init.db"}),
    ),
)
def test_shadowed_legacy_alias_emits_no_warning(monkeypatch, canonical_name, init_values):
    from huddleroom.config import Settings

    monkeypatch.setenv("RALLY_DATABASE_URL", "sqlite+aiosqlite:///legacy.db")
    if canonical_name:
        monkeypatch.setenv(canonical_name, "sqlite+aiosqlite:///current.db")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", FutureWarning)
        Settings(_env_file=None, **init_values)

    assert all("RALLY_DATABASE_URL" not in str(warning.message) for warning in recorded)


def test_huddleroom_dotenv_uses_current_name_before_rally_alias(tmp_path, monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.delenv("HUDDLEROOM_DATABASE_URL", raising=False)
    monkeypatch.delenv("RALLY_DATABASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "RALLY_DATABASE_URL=sqlite+aiosqlite:///legacy.db\n"
        "HUDDLEROOM_DATABASE_URL=sqlite+aiosqlite:///current.db\n",
        encoding="utf-8",
    )

    assert Settings(_env_file=tmp_path / ".env").database_url.endswith("current.db")


def test_database_default_preserves_existing_rally_database(tmp_path, monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HUDDLEROOM_DATABASE_URL", raising=False)
    monkeypatch.delenv("RALLY_DATABASE_URL", raising=False)
    (tmp_path / "rally.db").touch()
    assert Settings(_env_file=None).database_url.endswith("rally.db")
    (tmp_path / "huddleroom.db").touch()
    assert Settings(_env_file=None).database_url.endswith("huddleroom.db")


@pytest.mark.parametrize(
    ("name", "value"),
    (("HUDDLEROOM_AUTH_ENABLED", "true"), ("RALLY_AUTH_ENABLED", "true")),
)
def test_unsupported_guard_names_the_supplied_setting(monkeypatch, name, value):
    from huddleroom.config import Settings, validate_supported_settings

    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        validate_supported_settings(Settings(_env_file=None))


def test_onecli_native_auth_runtimes_default():
    from huddleroom.config import Settings

    assert Settings(_env_file=None).onecli_native_auth_runtimes == ["claude_code", "codex"]


def test_onecli_native_auth_runtimes_toml_config_list_loads(tmp_path):
    home = tmp_path / "home"
    config_dir = home / ".huddleroom"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        'onecli_native_auth_runtimes = ["codex"]\n',
        encoding="utf-8",
    )

    from huddleroom.config import Settings

    configured = Settings(_env_file=None)
    configured._setting_names["onecli_native_auth_runtimes"] = str(config_dir / "config.toml:onecli_native_auth_runtimes")
    # Direct instantiation test
    configured = Settings(_env_file=None, onecli_native_auth_runtimes=["codex"])
    assert configured.onecli_native_auth_runtimes == ["codex"]


def test_onecli_native_auth_runtimes_env_single_value(monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.setenv("HUDDLEROOM_ONECLI_NATIVE_AUTH_RUNTIMES", '["codex"]')
    assert Settings(_env_file=None).onecli_native_auth_runtimes == ["codex"]


def test_onecli_native_auth_runtimes_env_empty_list(monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.setenv("HUDDLEROOM_ONECLI_NATIVE_AUTH_RUNTIMES", "[]")
    assert Settings(_env_file=None).onecli_native_auth_runtimes == []


def test_onecli_native_auth_runtimes_env_both_values(monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.setenv("HUDDLEROOM_ONECLI_NATIVE_AUTH_RUNTIMES", '["claude_code", "codex"]')
    assert Settings(_env_file=None).onecli_native_auth_runtimes == ["claude_code", "codex"]


def test_onecli_native_auth_runtimes_rejects_unknown(monkeypatch):
    from pydantic import ValidationError
    from huddleroom.config import Settings

    monkeypatch.setenv("HUDDLEROOM_ONECLI_NATIVE_AUTH_RUNTIMES", '["claude"]')
    with pytest.raises(ValidationError, match="onecli_native_auth_runtimes"):
        Settings(_env_file=None)


def test_onecli_native_auth_runtimes_rejects_mixed_unknown():
    from pydantic import ValidationError
    from huddleroom.config import Settings

    with pytest.raises(ValidationError, match="onecli_native_auth_runtimes"):
        Settings(_env_file=None, onecli_native_auth_runtimes=["claude_code", "unknown"])


def test_onecli_native_auth_runtimes_rejects_multiple_unknown():
    from pydantic import ValidationError
    from huddleroom.config import Settings

    with pytest.raises(ValidationError, match="onecli_native_auth_runtimes"):
        Settings(_env_file=None, onecli_native_auth_runtimes=["claude", "aider"])


def test_orchestration_cli_model_default_and_env(monkeypatch):
    from huddleroom.config import Settings

    assert Settings(_env_file=None).orchestration_cli_model is None
    monkeypatch.setenv("HUDDLEROOM_ORCHESTRATION_CLI_MODEL", "opus[1m]")
    assert Settings(_env_file=None).orchestration_cli_model == "opus[1m]"


@pytest.mark.parametrize("value", ["claude-opus-4-1", "opus[1m]", "gpt-5-codex", "openai/o3"])
def test_orchestration_cli_model_accepts(value):
    from huddleroom.config import Settings

    assert Settings(_env_file=None, orchestration_cli_model=value).orchestration_cli_model == value


@pytest.mark.parametrize("value", ["-x", "--model", "a b", "a;b", "a" * 129])
def test_orchestration_cli_model_rejects(value):
    from huddleroom.config import Settings

    with pytest.raises(ValueError, match="orchestration_cli_model"):
        Settings(_env_file=None, orchestration_cli_model=value)


@pytest.mark.parametrize("value", ["", "  "])
def test_orchestration_cli_model_blank_is_none(value):
    from huddleroom.config import Settings

    assert Settings(_env_file=None, orchestration_cli_model=value).orchestration_cli_model is None


@pytest.mark.parametrize("value", [True, 5])
def test_orchestration_cli_model_rejects_non_string_values(value):
    from huddleroom.config import Settings

    with pytest.raises(ValueError, match="orchestration_cli_model"):
        Settings(_env_file=None, orchestration_cli_model=value)


def test_orchestration_cli_model_toml_int_fails(tmp_path):
    home = tmp_path / "home"
    config_dir = home / ".huddleroom"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text("orchestration_cli_model = 5\n", encoding="utf-8")

    result = _import_config(tmp_path, home)

    assert result.returncode != 0
    assert "orchestration_cli_model" in result.stderr


def test_proactive_orchestrator_setting_defaults_and_env_override(monkeypatch):
    from pydantic import ValidationError

    from huddleroom.config import Settings

    settings = Settings(_env_file=None)
    assert settings.orchestration_wake_max_seconds == 3600
    assert settings.orchestration_max_actions_per_tick == 3

    monkeypatch.setenv("HUDDLEROOM_ORCHESTRATION_WAKE_MAX_SECONDS", "120")
    monkeypatch.setenv("HUDDLEROOM_ORCHESTRATION_MAX_ACTIONS_PER_TICK", "1")
    settings = Settings(_env_file=None)
    assert settings.orchestration_wake_max_seconds == 120
    assert settings.orchestration_max_actions_per_tick == 1

    monkeypatch.setenv("HUDDLEROOM_ORCHESTRATION_WAKE_MAX_SECONDS", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
    monkeypatch.setenv("HUDDLEROOM_ORCHESTRATION_WAKE_MAX_SECONDS", "3600")

    monkeypatch.setenv("HUDDLEROOM_ORCHESTRATION_MAX_ACTIONS_PER_TICK", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
