import pathlib
import sysconfig

import click

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def _migration_config_path() -> pathlib.Path:
    migrations = pathlib.Path(sysconfig.get_path("data")) / "huddleroom" / "migrations"
    config = migrations / "alembic.ini"
    if config.is_file() and (migrations / "versions").is_dir():
        return config
    return _PROJECT_ROOT / "alembic.ini"


@click.group()
def main():
    """HuddleRoom — Autonomous Agent Workspace Platform"""


@main.command()
@click.option("--host", default="127.0.0.1", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start HuddleRoom server (API + task runner + scheduler)."""
    import uvicorn

    from huddleroom.config import settings, validate_supported_settings

    validate_supported_settings(settings)
    click.echo(f"Starting HuddleRoom on http://{host}:{port}")
    click.echo("Database: " + _get_db_url_display())
    click.echo("Dashboard: " + f"http://{host}:{port}/dashboard")
    click.echo("")

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
    from alembic import command
    from alembic.config import Config

    from huddleroom.config import settings, validate_supported_settings

    validate_supported_settings(settings)
    command.upgrade(Config(str(_migration_config_path())), "head")


if __name__ == "__main__":
    main()
