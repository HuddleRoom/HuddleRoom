from pathlib import Path
import os
import warnings
from dotenv import dotenv_values
from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings, DotEnvSettingsSource, EnvSettingsSource, SettingsConfigDict
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
}


def load_provider_env(env_file: str | os.PathLike = ".env") -> None:
    for key, value in dotenv_values(env_file).items():
        if key in PROVIDER_ENV_KEYS and value is not None:
            os.environ.setdefault(key, value)


load_provider_env()


def _default_database_url() -> str:
    if not Path("huddleroom.db").exists() and Path("rally.db").exists():
        return "sqlite+aiosqlite:///rally.db"
    return "sqlite+aiosqlite:///huddleroom.db"


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
    workspace_dir: str = "workspace"
    orchestration_model: str = "openai/gpt-4o-mini"
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
    effectiveness_recovery_threshold: int = Field(default=2, ge=1)
    effectiveness_failed_session_threshold: int = Field(default=3, ge=1)
    effectiveness_inactivity_hours: int = Field(default=24, ge=1)
    meeting_control_model: str | None = None
    _setting_names: dict[str, str] = PrivateAttr(default_factory=dict)

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
            file_secret_settings,
        )

    def __init__(self, **values):
        super().__init__(**values)
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
        for field in Settings.model_fields:  # pylint: disable=not-an-iterable
            if field in names:
                continue
            for source in (os.environ, dotenv):
                for prefix in ("HUDDLEROOM_", "RALLY_"):
                    name = f"{prefix}{field.upper()}"
                    if source.get(name) is not None:
                        names[field] = name
                        break
                if field in names:
                    break
        return names

    def setting_name(self, field: str) -> str:
        return self._setting_names.get(field, f"HUDDLEROOM_{field.upper()}")

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
