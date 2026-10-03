import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_load_provider_env_exports_openrouter_api_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=or-test\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from huddleroom.config import load_provider_env

    load_provider_env(env_file)

    assert os.environ["OPENROUTER_API_KEY"] == "or-test"


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


def test_huddleroom_settings_prefer_current_names_and_keep_rally_aliases(monkeypatch):
    from huddleroom.config import Settings

    monkeypatch.setenv("RALLY_DATABASE_URL", "sqlite+aiosqlite:///legacy.db")
    with pytest.warns(FutureWarning, match="RALLY_DATABASE_URL"):
        assert Settings(_env_file=None).database_url.endswith("legacy.db")

    monkeypatch.setenv("HUDDLEROOM_DATABASE_URL", "sqlite+aiosqlite:///current.db")
    assert Settings(_env_file=None).database_url.endswith("current.db")
    assert Settings(_env_file=None, database_url="sqlite+aiosqlite:///init.db").database_url.endswith("init.db")


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
