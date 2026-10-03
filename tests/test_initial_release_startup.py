"""Startup boundary checks for the SQLite-only initial release."""

import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from click.testing import CliRunner


ROOT = Path(__file__).resolve().parents[1]
_SETTINGS_ENV = tuple(
    f"{prefix}_{name}"
    for prefix in ("RALLY", "HUDDLEROOM")
    for name in ("DATABASE_URL", "REDIS_URL", "AUTH_ENABLED")
)


def test_default_pytest_excludes_unsupported_modes():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = config["tool"]["pytest"]["ini_options"]

    assert "unsupported_mode:" in "\n".join(pytest_options["markers"])
    assert "not unsupported_mode" in pytest_options["addopts"]


def _environment(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in _SETTINGS_ENV:
        env.pop(name, None)
    env.update(
        RALLY_DATABASE_URL="sqlite+aiosqlite:///huddleroom.db",
        RALLY_REDIS_URL="",
        RALLY_AUTH_ENABLED="false",
    )
    env.update(overrides)
    return env


def _import_process(module: str, **settings: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=ROOT,
        env=_environment(**settings),
        capture_output=True,
        text=True,
        check=False,
    )


def _cli_process(*args: str, **settings: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "huddleroom.cli", *args],
        cwd=ROOT,
        env=_environment(**settings),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("config", "setting_name"),
    (
        ({"database_url": "postgresql+asyncpg://localhost/rally"}, "HUDDLEROOM_DATABASE_URL"),
        ({"database_url": "sqliteevil:///huddleroom.db"}, "HUDDLEROOM_DATABASE_URL"),
        ({"database_url": "sqlite:///huddleroom.db"}, "HUDDLEROOM_DATABASE_URL"),
        ({"database_url": "sqlite+missing:///huddleroom.db"}, "HUDDLEROOM_DATABASE_URL"),
        ({"database_url": "sqlite+aiosqlite://localhost/huddleroom.db"}, "HUDDLEROOM_DATABASE_URL"),
        ({"database_url": "not-a-url"}, "HUDDLEROOM_DATABASE_URL"),
        ({"redis_url": "redis://localhost:6379/0"}, "HUDDLEROOM_REDIS_URL"),
        ({"auth_enabled": True}, "HUDDLEROOM_AUTH_ENABLED"),
    ),
)
def test_validate_supported_settings_rejects_each_unsupported_mode(config, setting_name):
    from huddleroom.config import Settings, validate_supported_settings

    with pytest.raises(RuntimeError, match=setting_name):
        validate_supported_settings(Settings(_env_file=None, **config))


def test_validate_supported_settings_accepts_default_and_explicit_sqlite(monkeypatch, tmp_path):
    from huddleroom.config import Settings, validate_supported_settings

    for name in _SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)

    validate_supported_settings(Settings(_env_file=None))
    validate_supported_settings(
        Settings(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'huddleroom.db'}")
    )


@pytest.mark.parametrize(
    "settings, setting_name",
    (
        ({"RALLY_DATABASE_URL": "postgresql+asyncpg://localhost/rally"}, "RALLY_DATABASE_URL"),
        ({"RALLY_DATABASE_URL": "sqliteevil:///huddleroom.db"}, "RALLY_DATABASE_URL"),
        ({"RALLY_DATABASE_URL": "sqlite:///huddleroom.db"}, "RALLY_DATABASE_URL"),
        ({"RALLY_DATABASE_URL": "sqlite+missing:///huddleroom.db"}, "RALLY_DATABASE_URL"),
        ({"RALLY_DATABASE_URL": "sqlite+aiosqlite://localhost/huddleroom.db"}, "RALLY_DATABASE_URL"),
        ({"RALLY_DATABASE_URL": "not-a-url"}, "RALLY_DATABASE_URL"),
        ({"RALLY_REDIS_URL": "redis://localhost:6379/0"}, "RALLY_REDIS_URL"),
        ({"RALLY_AUTH_ENABLED": "true"}, "RALLY_AUTH_ENABLED"),
    ),
)
def test_server_import_rejects_unsupported_settings_before_startup(settings, setting_name):
    result = _import_process("huddleroom.main", **settings)

    assert result.returncode != 0
    assert setting_name in result.stderr


def test_server_import_accepts_explicit_sqlite(tmp_path):
    result = _import_process("huddleroom.main", RALLY_DATABASE_URL=f"sqlite+aiosqlite:///{tmp_path / 'huddleroom.db'}")

    assert result.returncode == 0, result.stderr


async def test_lifespan_rejects_before_opening_database(monkeypatch):
    import huddleroom.main as main

    init_db = AsyncMock()
    monkeypatch.setattr(main, "init_db", init_db)
    monkeypatch.setattr(main.settings, "auth_enabled", True)

    with pytest.raises(RuntimeError, match="HUDDLEROOM_AUTH_ENABLED"):
        async with main.lifespan(None):
            pass

    init_db.assert_not_awaited()


def test_cli_rejects_before_calling_uvicorn(monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.config as config

    run = Mock()
    monkeypatch.setattr(config, "settings", config.Settings(_env_file=None, auth_enabled=True))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))

    result = CliRunner().invoke(cli.main, ["serve"])

    assert isinstance(result.exception, RuntimeError)
    assert "HUDDLEROOM_AUTH_ENABLED" in str(result.exception)
    run.assert_not_called()


def test_init_db_rejects_before_calling_alembic(monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.config as config

    upgrade = Mock()
    monkeypatch.setattr(config, "settings", config.Settings(_env_file=None, auth_enabled=True))
    monkeypatch.setattr("alembic.command.upgrade", upgrade)

    result = CliRunner().invoke(cli.main, ["init-db"])

    assert isinstance(result.exception, RuntimeError)
    assert "HUDDLEROOM_AUTH_ENABLED" in str(result.exception)
    upgrade.assert_not_called()


def test_init_db_runs_local_alembic_upgrade(monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.config as config

    upgrade = Mock()
    migration_config = ROOT / "alembic.ini"
    monkeypatch.setattr(config, "settings", config.Settings(_env_file=None))
    monkeypatch.setattr(cli, "_migration_config_path", lambda: migration_config)
    monkeypatch.setattr("alembic.command.upgrade", upgrade)

    result = CliRunner().invoke(cli.main, ["init-db"])

    assert result.exit_code == 0, result.output
    config_arg, revision = upgrade.call_args.args
    assert config_arg.config_file_name == str(migration_config)
    assert revision == "head"


def test_cli_shows_legacy_setting_future_warning():
    result = _cli_process("init-db", RALLY_AUTH_ENABLED="true")

    assert result.returncode != 0
    assert "FutureWarning" in result.stderr
    assert "RALLY_AUTH_ENABLED" in result.stderr


def test_cli_defaults_bind_to_localhost_on_port_8000(monkeypatch):
    import huddleroom.cli as cli
    import huddleroom.config as config

    run = Mock()
    monkeypatch.setattr(config, "settings", config.Settings(_env_file=None))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))

    result = CliRunner().invoke(cli.main, ["serve"])

    assert result.exit_code == 0, result.output
    assert run.call_args.args == ("huddleroom.main:app",)
    assert run.call_args.kwargs["host"] == "127.0.0.1"
    assert run.call_args.kwargs["port"] == 8000
    assert run.call_args.kwargs["reload"] is False


@pytest.mark.parametrize("command", ("worker", "beat"))
def test_celery_entry_points_reject_default_settings(command):
    if importlib.util.find_spec("celery") is None:
        pytest.skip("Celery optional dependency is not installed")

    result = subprocess.run(
        [sys.executable, "-m", "celery", "-A", "huddleroom.workers.celery_app", command, "--loglevel=WARNING"],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert "Celery workers and beat are not yet supported" in result.stderr


@pytest.mark.parametrize("command", ("worker", "beat"))
def test_celery_entry_points_reject_unsupported_settings(command):
    if importlib.util.find_spec("celery") is None:
        pytest.skip("Celery optional dependency is not installed")

    result = subprocess.run(
        [sys.executable, "-m", "celery", "-A", "huddleroom.workers.celery_app", command, "--loglevel=WARNING"],
        cwd=ROOT,
        env=_environment(RALLY_AUTH_ENABLED="true"),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert "RALLY_AUTH_ENABLED" in result.stderr


