from pathlib import Path
import os
import re
import tomllib
import warnings
from typing import Any, Literal
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import Field, PrivateAttr, ValidationError, field_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    SettingsConfigDict,
    SettingsError,
    TomlConfigSettingsSource,
)
from sqlalchemy.exc import ArgumentError
from sqlalchemy.engine import make_url


PROVIDER_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_BASE",
    "OPENROUTER_API_KEY",
    "OR_API_KEY",
    "OR_APP_NAME",
    "OR_SITE_URL",
    "OLLAMA_API_BASE",
}
ONECLI_NATIVE_AUTH_RUNTIMES = frozenset(("claude_code", "codex"))
OrchestrationEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
DEFAULT_CONFIG_FILE = Path.home() / ".huddleroom" / "config.toml"
DEFAULT_DATA_DIR = Path.home() / ".huddleroom"


def validate_onecli_origin(value: str) -> str:
    """Return a safe OneCLI HTTP(S) origin or reject it."""
    if not isinstance(value, str):
        raise ValueError("must have a usable HTTP(S) authority")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("must have a usable HTTP(S) authority") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or port is not None and not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("must be an HTTP(S) origin without credentials, path, query, or fragment")
    return value.rstrip("/")


_CLI_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@\[\]-]{0,127}$")


def validate_cli_model(value: str | None) -> str | None:
    """Return a safe CLI model name (blank -> None) or reject it."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("must be a string")
    value = value.strip()
    if not value:
        return None
    if not _CLI_MODEL_RE.fullmatch(value):
        raise ValueError("must be 1-128 chars of letters, digits and . _ : / @ [ ] -, not starting with '-'")
    return value


def _toml_values(config_file: Path) -> dict[str, Any]:
    if not config_file.is_file():
        return {}
    try:
        with config_file.open("rb") as file:
            return tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid TOML configuration in {config_file}: {error}") from error


def load_provider_env(
    env_file: str | os.PathLike = ".env", config_file: str | os.PathLike | None = None, *, credential_mode: str = "direct"
) -> None:
    config_file = DEFAULT_CONFIG_FILE if config_file is None else Path(config_file)
    toml = _toml_values(config_file)
    dotenv = dotenv_values(env_file)
    if credential_mode == "onecli":
        for key in PROVIDER_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update({"OPENAI_API_KEY": "onecli-openai-placeholder", "ANTHROPIC_API_KEY": "onecli-anthropic-placeholder"})
        return
    for key in PROVIDER_ENV_KEYS:
        if key in toml and not isinstance(toml[key], str):
            raise ValueError(f"Invalid provider value in {config_file}:{key}; expected a string.")
        if key not in os.environ:
            value = dotenv.get(key)
            if value is None:
                value = toml.get(key)
            if value is not None:
                os.environ[key] = value

def _default_database_url() -> str:
    if Path("huddleroom.db").exists():
        return "sqlite+aiosqlite:///huddleroom.db"
    if Path("rally.db").exists():
        return "sqlite+aiosqlite:///rally.db"
    return f"sqlite+aiosqlite:///{DEFAULT_DATA_DIR / 'huddleroom.db'}"


def _default_workspace_dir() -> str:
    if Path("workspace").is_dir():
        return "workspace"
    return str(DEFAULT_DATA_DIR / "workspace")


class _CompatibilitySettingsSource(EnvSettingsSource):
    """Read the current prefix first, with Rally aliases for one release."""

    def get_field_value(self, field, field_name):
        value_is_complex = self._extract_field_info(field, field_name)[0][2]
        for prefix in ("HUDDLEROOM_", "RALLY_"):
            name = self._apply_case_sensitive(f"{prefix}{field_name.upper()}")
            value = self.env_vars.get(name)
            if value is not None:
                return value, field_name, value_is_complex
        return None, field_name, value_is_complex


class _CompatibilityDotEnvSettingsSource(DotEnvSettingsSource, _CompatibilitySettingsSource):
    pass


class _TomlSettingsSource(TomlConfigSettingsSource):
    def _read_file(self, file_path):
        try:
            return super()._read_file(file_path)
        except tomllib.TOMLDecodeError as error:
            raise SettingsError(f"Invalid TOML configuration in {file_path}: {error}") from error


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    database_url: str = Field(default_factory=_default_database_url)
    redis_url: str | None = None   # only used with [postgres] extras
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_key_prefix: str = "rly_"
    cors_origins: list[str] = ["*"]
    debug: bool = False
    log_file: str | None = None
    auth_enabled: bool = False
    api_base_url: str = "http://localhost:8000"
    embedding_model: str = "text-embedding-3-small"
    workspace_dir: str = Field(default_factory=_default_workspace_dir)
    orchestration_model: str = "openai/gpt-6.1-sol"
    orchestration_backend: Literal["api", "claude", "codex"] = "api"
    orchestration_cli_model: str | None = None  # --model/-m for the claude/codex CLI; orchestration_model is API-only
    orchestration_effort: OrchestrationEffort | None = None
    orchestration_conversation_allowance_tokens: int = Field(default=50000, ge=0)
    # -1 = unlimited (default), 0 = disabled, >0 = lifetime token cap
    orchestration_advisor_allowance_tokens: int = Field(default=-1, ge=-1)
    orchestration_conversation_investigation_enabled: bool = True
    orchestration_conversation_steering_enabled: bool = True
    orchestration_checkpoint_max_questions: int = Field(default=5, ge=1)
    orchestration_reconcile_interval_seconds: int = Field(default=30, ge=1)
    orchestration_event_coalesce_seconds: int = Field(default=1, ge=1)
    orchestration_semantic_progress_seconds: int = Field(default=120, ge=1)
    orchestration_sweep_goal_limit: int = Field(default=100, ge=1)
    orchestration_sweep_seconds_limit: int = Field(default=5, ge=1)
    orchestration_wake_max_seconds: int = Field(default=3600, ge=1)
    orchestration_max_actions_per_tick: int = Field(default=3, ge=1)
    effectiveness_recovery_threshold: int = Field(default=2, ge=1)
    effectiveness_failed_session_threshold: int = Field(default=3, ge=1)
    effectiveness_inactivity_hours: int = Field(default=24, ge=1)
    meeting_control_model: str | None = None
    credential_mode: Literal["direct", "onecli"] = "direct"
    onecli_agent: str | None = None
    onecli_management_url: str = "http://127.0.0.1:10256"
    onecli_gateway_url: str = "http://127.0.0.1:10255"
    onecli_native_auth_runtimes: list[str] = Field(default=["claude_code", "codex"])
    _setting_names: dict[str, str] = PrivateAttr(default_factory=dict)

    @field_validator("onecli_management_url", "onecli_gateway_url")
    @classmethod
    def validate_onecli_url(cls, value: str) -> str:
        return validate_onecli_origin(value)

    @field_validator("orchestration_cli_model", mode="before")
    @classmethod
    def validate_orchestration_cli_model(cls, value):
        return validate_cli_model(value)

    @field_validator("onecli_native_auth_runtimes")
    @classmethod
    def validate_onecli_native_auth_runtimes(cls, value: list[str]) -> list[str]:
        unknown = [r for r in value if r not in ONECLI_NATIVE_AUTH_RUNTIMES]
        if unknown:
            raise ValueError(f"onecli_native_auth_runtimes: unknown {unknown}; allowed: {', '.join(sorted(ONECLI_NATIVE_AUTH_RUNTIMES))}")
        return value

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        return (
            init_settings,
            _CompatibilitySettingsSource(settings_cls),
            _CompatibilityDotEnvSettingsSource(
                settings_cls,
                env_file=dotenv_settings.env_file,
                env_file_encoding=dotenv_settings.env_file_encoding,
            ),
            _TomlSettingsSource(settings_cls, toml_file=DEFAULT_CONFIG_FILE),
            file_secret_settings,
        )

    def __init__(self, **values):
        try:
            super().__init__(**values)
        except ValidationError as error:
            supplied_names = self._supplied_setting_names(values)
            invalid_toml_names = sorted({
                supplied_names[item["loc"][0]]
                for item in error.errors(include_input=False)
                if item["loc"]
                and supplied_names.get(item["loc"][0], "").startswith(f"{DEFAULT_CONFIG_FILE}:")
            })
            if invalid_toml_names:
                raise ValueError(f"Invalid configuration value for {', '.join(invalid_toml_names)}.") from error
            raise
        self._setting_names = self._supplied_setting_names(values)
        legacy_names = [name for name in self._setting_names.values() if name.startswith("RALLY_")]
        if legacy_names:
            warnings.warn(
                f"{', '.join(legacy_names)} {'is' if len(legacy_names) == 1 else 'are'} deprecated; "
                "use HUDDLEROOM_ names.",
                FutureWarning,
                stacklevel=2,
            )

    @staticmethod
    def _supplied_setting_names(values: dict) -> dict[str, str]:
        names = {field: f"HUDDLEROOM_{field.upper()}" for field in values}
        env_file = values.get("_env_file", ".env")
        dotenv = {} if env_file is None else dotenv_values(env_file)
        sources = [{key.upper(): value for key, value in source.items()} for source in (os.environ, dotenv)]
        toml = _toml_values(DEFAULT_CONFIG_FILE)
        for field in Settings.model_fields:  # pylint: disable=not-an-iterable
            if field in names:
                continue
            for source in sources:
                for prefix in ("HUDDLEROOM_", "RALLY_"):
                    name = f"{prefix}{field.upper()}"
                    if source.get(name) is not None:
                        names[field] = name
                        break
                if field in names:
                    break
            if field not in names and field in toml:
                names[field] = f"{DEFAULT_CONFIG_FILE}:{field}"
        return names

    def setting_name(self, field: str) -> str:
        return self._setting_names.get(field, f"HUDDLEROOM_{field.upper()}")

    def setting_was_supplied(self, field: str) -> bool:
        """Whether a configuration source set this field rather than its default."""
        return field in self._setting_names

    @property
    def is_sqlite(self) -> bool:
        try:
            return make_url(self.database_url).get_backend_name() == "sqlite"
        except ArgumentError:
            return False

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql") or self.database_url.startswith("postgres")  # pylint: disable=no-member

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_dir)


settings = Settings()
# Resolve the mode from validated configuration before importing any plaintext
# provider values from TOML or dotenv files.
load_provider_env(credential_mode=settings.credential_mode)


def validate_supported_settings(config: Settings = settings) -> None:
    """Reject settings outside the initial local-release support boundary."""
    try:
        database_url = make_url(config.database_url)
        is_supported_sqlite = (
            database_url.drivername == "sqlite+aiosqlite"
            and database_url.username is None
            and database_url.password is None
            and database_url.host is None
            and database_url.port is None
        )
    except ArgumentError:
        is_supported_sqlite = False
    if not is_supported_sqlite:
        raise RuntimeError(
            f"{config.setting_name('database_url')} database is not yet supported; use SQLite with sqlite+aiosqlite."
        )
    if config.redis_url:
        raise RuntimeError(f"{config.setting_name('redis_url')} is not yet supported; unset it.")
    if config.auth_enabled:
        raise RuntimeError(f"{config.setting_name('auth_enabled')}=true is not yet supported; set it to false.")
