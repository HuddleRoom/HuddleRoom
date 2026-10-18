"""Server environment for running orchestration through the codex CLI."""

from tests.live_acceptance.fixtures import ORCH_CLI_MODEL, ORCH_EFFORT


def server_env_for_cli(orch_model: str = ORCH_CLI_MODEL, effort: str = ORCH_EFFORT) -> dict[str, str]:
    """Pass as Supervisor(env_overrides=...). Confirm in dry run that the model and effort are accepted by the CLI."""
    return {
        "HUDDLEROOM_ORCHESTRATION_BACKEND": "codex",
        "HUDDLEROOM_ORCHESTRATION_CLI_MODEL": orch_model,
        "HUDDLEROOM_ORCHESTRATION_EFFORT": effort,
    }
