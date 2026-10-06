import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tarfile
import time
import tomllib
import venv
import zipfile
from configparser import ConfigParser
from contextlib import contextmanager
from email import message_from_string
from collections.abc import Iterator
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_ENVIRONMENT = {
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OLLAMA_API_BASE",
    "OPENAI_API_KEY",
    "OPENROUTER_API_BASE",
    "OPENROUTER_API_KEY",
    "OR_API_KEY",
    "OR_APP_NAME",
    "OR_SITE_URL",
}
RUNTIME_CREDENTIAL_ENVIRONMENT = {
    "ONECLI_API_KEY",
    "GITHUB_TOKEN",
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITLAB_TOKEN",
}
BUILD_TIMEOUT = 180
COMMAND_TIMEOUT = 120
SERVER_TIMEOUT = 30
TEST_DIST_DIR_ENVIRONMENT = "HUDDLEROOM_TEST_DIST_DIR"
PRIVATE_DISTRIBUTION_PATHS = (".agents/", ".claude/", ".codex/", "features/", "bugs.db")


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
    redact: tuple[Path, ...] = (),
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, cwd=cwd, env=env, check=True, text=True, capture_output=True, timeout=timeout
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        stdout = _sanitize_output(error.stdout or "", redact)
        stderr = _sanitize_output(error.stderr or "", redact)
        display_command = _sanitize_output(" ".join(command), redact)
        raise AssertionError(f"Command failed: {display_command}\nstdout:\n{stdout}\nstderr:\n{stderr}") from error


def _build_artifacts(output: Path) -> tuple[Path, Path]:
    _run([sys.executable, "-m", "build", "--outdir", str(output)], cwd=ROOT, timeout=BUILD_TIMEOUT)
    return next(output.glob("*.whl")), next(output.glob("*.tar.gz"))


def _supplied_artifacts() -> tuple[Path, Path] | None:
    value = os.environ.get(TEST_DIST_DIR_ENVIRONMENT)
    if value is None:
        return None
    output = Path(value).resolve()
    assert output.is_dir(), f"{TEST_DIST_DIR_ENVIRONMENT} is not a directory: {output}"
    wheels = list(output.glob("*.whl"))
    sdists = list(output.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected one wheel in {output}, found {wheels}"
    assert len(sdists) == 1, f"expected one sdist in {output}, found {sdists}"
    return wheels[0], sdists[0]


def _distribution_artifacts(output: Path) -> tuple[Path, Path]:
    return _supplied_artifacts() or _build_artifacts(output)


def _build_dashboard_if_needed() -> None:
    if _supplied_artifacts() is not None:
        return
    if shutil.which("npm") is None:
        raise RuntimeError("npm is required to build the dashboard before testing the distribution")
    _run(["make", "build-frontend"], cwd=ROOT, timeout=BUILD_TIMEOUT)


def _expected_static_files() -> set[str]:
    static = ROOT / "huddleroom" / "static"
    return {
        str(path.relative_to(ROOT)).replace(os.sep, "/")
        for directory in ("dev-dashboard", "dashboard")
        for path in (static / directory).glob("**/*")
        if path.is_file()
    }


def _assert_static_contents(names: set[str], expected: set[str]) -> None:
    for path in expected:
        assert path in names, path


def _assert_migration_contents(names: set[str], config_path: str, scripts_prefix: str) -> None:
    assert config_path in names
    for filename in ("env.py", "script.py.mako"):
        assert f"{scripts_prefix}/{filename}" in names
    for migration in (ROOT / "alembic" / "versions").glob("*.py"):
        assert f"{scripts_prefix}/versions/{migration.name}" in names


def _archive_names(sdist: Path) -> set[str]:
    with tarfile.open(sdist) as archive:
        prefix = archive.getnames()[0].split("/", 1)[0] + "/"
        return {name.removeprefix(prefix) for name in archive.getnames()}


def _project_metadata() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        return tomllib.load(project_file)["project"]


def _assert_no_private_distribution_files(names: set[str]) -> None:
    for name in names:
        assert not name.startswith(PRIVATE_DISTRIBUTION_PATHS), name
        assert not Path(name).name.startswith(".env"), name


def _assert_distribution_metadata(wheel: Path, sdist: Path) -> None:
    project = _project_metadata()
    expected_name = project["name"]
    expected_version = project["version"]
    assert expected_name == "huddleroom"
    assert expected_version == "0.1.0a1"
    assert project["description"]
    assert project["readme"] == "README.md"
    assert project["urls"].get("Repository")

    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = message_from_string(archive.read(metadata_name).decode())
        entries_name = next(name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt"))
        entries = ConfigParser()
        entries.read_string(archive.read(entries_name).decode())
    assert metadata["Name"] == expected_name
    assert metadata["Version"] == expected_version
    assert metadata["Summary"] == project["description"]
    assert dict(entries["console_scripts"]) == {"huddleroom": "huddleroom.cli:main"}

    with tarfile.open(sdist) as archive:
        pyproject = next(member for member in archive.getmembers() if member.name.endswith("/pyproject.toml"))
        sdist_project = tomllib.loads(archive.extractfile(pyproject).read().decode())["project"]
    assert sdist_project["name"] == expected_name
    assert sdist_project["version"] == expected_version


def _isolated_environment(home: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in list(environment):
        normalized = name.upper()
        if (
            normalized.startswith(("HUDDLEROOM_", "RALLY_"))
            or normalized in PROVIDER_ENVIRONMENT | RUNTIME_CREDENTIAL_ENVIRONMENT
            or normalized == "PYTHONPATH"
        ):
            environment.pop(name)
    environment["HOME"] = str(home)
    return environment


def _sanitize_output(output: str, redact: tuple[Path, ...]) -> str:
    for path in redact:
        output = output.replace(str(path), f"<{path.name}>")
    output = re.sub(r"(?i)(https?://)[^/\s@]*@", r"\1<redacted>@", output)
    output = re.sub(r"(?i)\bbearer\s+\S+", "Bearer <redacted>", output)
    output = re.sub(r"(?i)\binvalid\s+api\s+key\s+\S+", "invalid API key <redacted>", output)
    output = re.sub(
        r"\b(?:sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,})\b",
        "<redacted token>",
        output,
    )
    return re.sub(
        r"(?im)^.*(?:api[_-]?key|authorization|credential|password|secret|token)\s*[=:].*$",
        "<redacted sensitive output>",
        output,
    )


def _pipx_environment(home: Path, root: Path, python: Path) -> dict[str, str]:
    environment = _isolated_environment(home)
    for name in list(environment):
        normalized = name.upper()
        if normalized.startswith("PIPX_") or normalized in {
            "GIT_DIR",
            "GIT_WORK_TREE",
            "PYTHONHOME",
            "VIRTUAL_ENV",
        }:
            environment.pop(name)
    environment.update(
        {
            "PIPX_HOME": str(root / "pipx-home"),
            "PIPX_BIN_DIR": str(root / "pipx-bin"),
            "PIPX_MAN_DIR": str(root / "pipx-man"),
            "PIPX_DEFAULT_PYTHON": str(python),
            "XDG_CACHE_HOME": str(root / "cache"),
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_DATA_HOME": str(root / "data"),
        }
    )
    return environment


def _pipx() -> Path:
    executable = shutil.which("pipx")
    if executable is None:
        raise RuntimeError("pipx is required for the clean-install acceptance test")
    return Path(executable).resolve()


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _request(url: str) -> tuple[int, str]:
    with urlopen(url, timeout=2) as response:  # noqa: S310 -- test-only loopback URL
        return response.status, response.read().decode()


def _assert_server_ready(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + SERVER_TIMEOUT
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"server exited before readiness with status {process.returncode}")
        try:
            status, body = _request(health_url)
            if status == 200 and json.loads(body) == {"status": "ok"}:
                break
        except (URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.1)
    else:
        raise AssertionError(f"server did not become healthy within {SERVER_TIMEOUT} seconds")

    dashboard_status, dashboard = _request(f"http://127.0.0.1:{port}/dashboard/")
    assert dashboard_status == 200
    assert "<html" in dashboard.lower()
    asset = re.search(r"(?:src|href)=[\"']([^\"']*assets/[^\"']+)[\"']", dashboard)
    assert asset, "dashboard does not reference a built asset"
    asset_path = asset.group(1)
    if not asset_path.startswith("/"):
        asset_path = f"/dashboard/{asset_path.removeprefix('./')}"
    asset_status, _ = _request(f"http://127.0.0.1:{port}{asset_path}")
    assert asset_status == 200


def _stop_server(process: subprocess.Popen[str], redact: tuple[Path, ...]) -> str:
    was_running = process.poll() is None
    if was_running:
        process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=COMMAND_TIMEOUT)
        raise AssertionError(
            "server did not stop within the timeout\n"
            f"stdout:\n{_sanitize_output(stdout, redact)}\n"
            f"stderr:\n{_sanitize_output(stderr, redact)}"
        ) from None
    logs = (
        f"stdout:\n{_sanitize_output(stdout, redact)}\n"
        f"stderr:\n{_sanitize_output(stderr, redact)}"
    )
    if not was_running:
        raise AssertionError(f"server exited before shutdown with status {process.returncode}\n{logs}")
    if process.returncode == 0:
        return logs
    if process.returncode == -signal.SIGTERM and "Finished server process" in logs:
        return logs
    raise AssertionError(f"server stopped with status {process.returncode}\n{logs}")


@contextmanager
def _running_server(
    command: list[str], *, cwd: Path, env: dict[str, str], redact: tuple[Path, ...]
) -> Iterator[int]:
    port = _free_loopback_port()
    process = subprocess.Popen(
        [*command, "serve", "--host", "127.0.0.1", "--port", str(port)],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _assert_server_ready(port, process)
        yield port
    except BaseException as error:
        logs = _stop_server(process, redact)
        raise AssertionError(f"installed server failed\n{logs}") from error
    else:
        _stop_server(process, redact)


def _wheel_version(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
    match = re.search(r"^Version: (.+)$", metadata, re.MULTILINE)
    assert match, wheel
    return match.group(1)


def _next_release_version(version: str) -> str:
    match = re.fullmatch(r"(\d+(?:\.\d+)*)(?:a(\d+))?", version)
    assert match, f"cannot derive a higher local release from {version!r}"
    if match.group(2) is not None:
        return f"{match.group(1)}a{int(match.group(2)) + 1}"
    components = match.group(1).split(".")
    components[-1] = str(int(components[-1]) + 1)
    return ".".join(components)


def _build_upgraded_wheel(sdist: Path, output: Path) -> tuple[Path, str]:
    unpacked = output / "unpacked"
    unpacked.mkdir(parents=True)
    with tarfile.open(sdist) as archive:
        archive.extractall(unpacked, filter="data")
    source_tree = next(unpacked.iterdir())
    pyproject = source_tree / "pyproject.toml"
    rendered = pyproject.read_text()
    version_match = re.search(r'(?m)^version = "([^"]+)"$', rendered)
    assert version_match, pyproject
    upgraded_version = _next_release_version(version_match.group(1))
    pyproject.write_text(
        rendered[:version_match.start(1)] + upgraded_version + rendered[version_match.end(1):]
    )
    wheel_output = output / "wheel"
    wheel_output.mkdir()
    _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_output)],
        cwd=source_tree,
        timeout=BUILD_TIMEOUT,
    )
    wheel = next(wheel_output.glob("*.whl"))
    assert _wheel_version(wheel) == upgraded_version
    return wheel, upgraded_version


def _expected_head() -> str:
    latest = max(
        (ROOT / "alembic" / "versions").glob("*.py"), key=lambda path: int(path.name.split("_", 1)[0])
    )
    match = re.search(r'^revision\s*=\s*["\']([^"\']+)["\']', latest.read_text(), re.MULTILINE)
    assert match, latest
    return match.group(1)


def _assert_initialized_database(home: Path) -> None:
    database = home / ".huddleroom" / "huddleroom.db"
    assert database.is_file()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (_expected_head(),)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'users'"
        ).fetchone() == ("users",)


def _install_and_exercise(wheel: Path, root: Path) -> None:
    install_root = root / "installed"
    venv.create(install_root, with_pip=True, symlinks=True)
    home = root / "home"
    home.mkdir()
    empty_cwd = root / "empty"
    empty_cwd.mkdir()
    environment = _isolated_environment(home)
    _run(
        [str(install_root / "bin" / "python"), "-m", "pip", "install", str(wheel)],
        cwd=empty_cwd,
        env=environment,
        timeout=BUILD_TIMEOUT,
    )
    help_result = _run(
        [str(install_root / "bin" / "huddleroom"), "--help"],
        cwd=empty_cwd,
        env=environment,
        timeout=COMMAND_TIMEOUT,
    )
    assert "Usage:" in help_result.stdout
    assert not (install_root / "bin" / "rally").exists()
    _run(
        [str(install_root / "bin" / "huddleroom"), "init-db"],
        cwd=empty_cwd,
        env=environment,
        timeout=COMMAND_TIMEOUT,
    )
    _assert_initialized_database(home)


def _assert_configured_cli_fails_closed(
    command: Path, root: Path, empty_cwd: Path, backend: str, *, cli_is_on_path: bool
) -> None:
    home = root / "home"
    database = root / "state" / "huddleroom.db"
    config_directory = home / ".huddleroom"
    config_directory.mkdir(parents=True)
    (config_directory / "config.toml").write_text(
        f'orchestration_backend = "{backend}"\ndatabase_url = "sqlite+aiosqlite:///{database}"\n'
    )
    environment = _isolated_environment(home)
    path_directory = root / ("installed-cli" if cli_is_on_path else "missing-cli")
    path_directory.mkdir()
    marker = root / "cli-was-run"
    if cli_is_on_path:
        executable = path_directory / backend
        executable.write_text('#!/bin/sh\nprintf invoked > "$HUDDLEROOM_TEST_CLI_MARKER"\n')
        executable.chmod(0o755)
        environment["HUDDLEROOM_TEST_CLI_MARKER"] = str(marker)
    environment["PATH"] = str(path_directory)

    result = subprocess.run(
        [str(command), "serve"], cwd=empty_cwd, env=environment, text=True, capture_output=True,
        timeout=COMMAND_TIMEOUT,
    )

    assert result.returncode != 0
    assert f"The {backend} orchestration backend is unsupported" in result.stdout + result.stderr
    assert not marker.exists()
    assert not database.exists()


def test_isolated_environment_removes_runtime_credentials_case_insensitively(monkeypatch, tmp_path):
    for name in RUNTIME_CREDENTIAL_ENVIRONMENT | {"OPENAI_API_KEY"}:
        monkeypatch.setenv(name.lower(), "must-not-reach-an-installed-command")
    monkeypatch.setenv("PIP_INDEX_URL", "https://index.example.test/simple")

    environment = _isolated_environment(tmp_path)

    assert not {
        name.upper() for name in environment
    }.intersection(RUNTIME_CREDENTIAL_ENVIRONMENT | PROVIDER_ENVIRONMENT)
    assert environment["PIP_INDEX_URL"] == "https://index.example.test/simple"


def test_sanitize_output_redacts_retained_routing_credentials_and_tokens(tmp_path):
    output = "\n".join(
        (
            "https://package-user:package-password@index.example.test/simple",
            "Authorization: Bearer bearer-value",
            "invalid API key invalid-value",
            "tokens sk-12345678 AIza12345678 ghp_12345678 github_pat_12345678",
            str(tmp_path),
        )
    )

    sanitized = _sanitize_output(output, (tmp_path,))

    for secret in (
        "package-user",
        "package-password",
        "bearer-value",
        "invalid-value",
        "sk-12345678",
        "AIza12345678",
        "ghp_12345678",
        "github_pat_12345678",
        str(tmp_path),
    ):
        assert secret not in sanitized
    assert "index.example.test/simple" in sanitized


def test_stop_server_rejects_nonzero_exit_and_redacts_logs():
    class FailedServer:
        returncode = 3

        def poll(self):
            return self.returncode

        def communicate(self, timeout):
            assert timeout == COMMAND_TIMEOUT
            return "Bearer should-not-appear", "invalid API key also-hidden"

    try:
        _stop_server(FailedServer(), ())
    except AssertionError as error:
        message = str(error)
    else:
        raise AssertionError("a nonzero server exit must fail the acceptance harness")

    assert "status 3" in message
    assert "should-not-appear" not in message
    assert "also-hidden" not in message


def test_stop_server_accepts_harness_initiated_graceful_sigterm():
    class GracefulServer:
        returncode = -signal.SIGTERM
        terminated = False

        def poll(self):
            return None if not self.terminated else self.returncode

        def terminate(self):
            self.terminated = True

        def communicate(self, timeout):
            assert timeout == COMMAND_TIMEOUT
            return "", "Finished server process [123]"

    process = GracefulServer()

    assert "Finished server process" in _stop_server(process, ())
    assert process.terminated


def test_stop_server_rejects_preexisting_exit_without_terminating():
    class ExitedServer:
        returncode = 1
        terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def communicate(self, timeout):
            assert timeout == COMMAND_TIMEOUT
            return "", "Bearer should-not-appear"

    process = ExitedServer()

    try:
        _stop_server(process, ())
    except AssertionError as error:
        message = str(error)
    else:
        raise AssertionError("an already-exited server must fail the acceptance harness")

    assert not process.terminated
    assert "before shutdown" in message
    assert "should-not-appear" not in message


def test_stop_server_rejects_sigterm_without_graceful_shutdown_marker():
    class UnconfirmedServer:
        returncode = -signal.SIGTERM
        terminated = False

        def poll(self):
            return None if not self.terminated else self.returncode

        def terminate(self):
            self.terminated = True

        def communicate(self, timeout):
            assert timeout == COMMAND_TIMEOUT
            return "", "stopped"

    try:
        _stop_server(UnconfirmedServer(), ())
    except AssertionError as error:
        message = str(error)
    else:
        raise AssertionError("SIGTERM without graceful shutdown evidence must fail")

    assert f"status {-signal.SIGTERM}" in message


def test_next_release_version_supports_stable_and_alpha_versions():
    assert _next_release_version("0.1.0") == "0.1.1"
    assert _next_release_version("0.1.0a1") == "0.1.0a2"


def test_distribution_contains_runtime_files_and_installs(tmp_path):
    _build_dashboard_if_needed()
    expected_static = _expected_static_files()

    build_output = tmp_path / "dist"
    build_output.mkdir()
    wheel, sdist = _distribution_artifacts(build_output)
    _assert_distribution_metadata(wheel, sdist)

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    _assert_no_private_distribution_files(wheel_names)
    _assert_static_contents(wheel_names, expected_static)
    migration_config = next(
        name for name in wheel_names if name.endswith("/data/huddleroom/migrations/alembic.ini")
    )
    _assert_migration_contents(wheel_names, migration_config, migration_config.removesuffix("/alembic.ini"))

    sdist_names = _archive_names(sdist)
    _assert_no_private_distribution_files(sdist_names)
    _assert_static_contents(sdist_names, expected_static)
    _assert_migration_contents(sdist_names, "huddleroom/migrations/alembic.ini", "alembic")
    _install_and_exercise(wheel, tmp_path / "wheel-install")

    unpacked = tmp_path / "unpacked"
    with tarfile.open(sdist) as archive:
        archive.extractall(unpacked, filter="data")
    source_tree = next(unpacked.iterdir())
    from_sdist = tmp_path / "from-sdist"
    from_sdist.mkdir()
    _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(from_sdist)],
        cwd=source_tree,
        timeout=BUILD_TIMEOUT,
    )
    rebuilt_wheel = next(from_sdist.glob("*.whl"))
    with zipfile.ZipFile(rebuilt_wheel) as archive:
        sdist_wheel_names = set(archive.namelist())
    _assert_static_contents(sdist_wheel_names, expected_static)
    migration_config = next(
        name for name in sdist_wheel_names if name.endswith("/data/huddleroom/migrations/alembic.ini")
    )
    _assert_migration_contents(sdist_wheel_names, migration_config, migration_config.removesuffix("/alembic.ini"))
    _install_and_exercise(rebuilt_wheel, tmp_path / "sdist-wheel-install")


def test_distribution_clean_pipx_install_serves_restarts_and_upgrades(tmp_path):
    _build_dashboard_if_needed()
    build_output = tmp_path / "dist"
    build_output.mkdir()
    wheel, sdist = _distribution_artifacts(build_output)
    upgraded_wheel, upgraded_version = _build_upgraded_wheel(sdist, tmp_path / "upgrade")

    root = tmp_path / "end-user"
    home = root / "home"
    empty_cwd = root / "empty-cwd"
    bootstrap = root / "bootstrap"
    home.mkdir(parents=True)
    empty_cwd.mkdir(parents=True)
    assert not (empty_cwd / ".env").exists()
    venv.create(bootstrap, with_pip=True, symlinks=True)
    bootstrap_python = bootstrap / "bin" / "python"
    environment = _pipx_environment(home, root, bootstrap_python)
    redact = (ROOT, root, home)
    pipx = _pipx()
    pipx_version = _run(
        [str(pipx), "--version"], cwd=empty_cwd, env=environment, timeout=COMMAND_TIMEOUT, redact=redact
    )
    assert pipx_version.stdout.strip()

    _run(
        [str(pipx), "install", "--python", str(bootstrap_python), str(wheel)],
        cwd=empty_cwd,
        env=environment,
        timeout=BUILD_TIMEOUT,
        redact=redact,
    )
    installed = Path(environment["PIPX_BIN_DIR"]) / "huddleroom"
    installed_python = Path(environment["PIPX_HOME"]) / "venvs" / "huddleroom" / "bin" / "python"
    assert installed.is_file()
    assert installed_python.is_file()

    database = root / "state" / "huddleroom.db"
    workspace = root / "workspace"
    model = "openai/acceptance-placeholder"
    _run(
        [
            str(installed),
            "setup",
            "--provider",
            "skip",
            "--orchestration-model",
            model,
            "--database-path",
            str(database),
            "--workspace-dir",
            str(workspace),
        ],
        cwd=empty_cwd,
        env=environment,
        timeout=COMMAND_TIMEOUT,
        redact=redact,
    )
    with (home / ".huddleroom" / "config.toml").open("rb") as config_file:
        config = tomllib.load(config_file)
    assert config["database_url"] == f"sqlite+aiosqlite:///{database}"
    assert config["workspace_dir"] == str(workspace)
    assert config["orchestration_model"] == model
    assert "setup_complete" not in config
    assert not PROVIDER_ENVIRONMENT.intersection(config)
    effective_settings = _run(
        [
            str(installed_python),
            "-c",
            (
                "import json; from huddleroom.config import settings; "
                "print(json.dumps({'database_url': settings.database_url, "
                "'workspace_dir': settings.workspace_dir, "
                "'orchestration_model': settings.orchestration_model, "
                "'orchestration_backend': settings.orchestration_backend, "
                "'credential_mode': settings.credential_mode}))"
            ),
        ],
        cwd=empty_cwd,
        env=environment,
        timeout=COMMAND_TIMEOUT,
        redact=redact,
    )
    assert json.loads(effective_settings.stdout) == {
        "database_url": f"sqlite+aiosqlite:///{database}",
        "workspace_dir": str(workspace),
        "orchestration_model": model,
        "orchestration_backend": "api",
        "credential_mode": "direct",
    }

    with _running_server([str(installed)], cwd=empty_cwd, env=environment, redact=redact):
        pass
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE acceptance_sentinel (value TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO acceptance_sentinel VALUES ('preserved')")
    with _running_server([str(installed)], cwd=empty_cwd, env=environment, redact=redact):
        pass

    for backend in ("claude", "codex"):
        _assert_configured_cli_fails_closed(
            installed, root / f"missing-{backend}", empty_cwd, backend, cli_is_on_path=False
        )
        _assert_configured_cli_fails_closed(
            installed, root / f"installed-{backend}", empty_cwd, backend, cli_is_on_path=True
        )

    _run(
        [str(pipx), "install", "--force", "--python", str(bootstrap_python), str(upgraded_wheel)],
        cwd=empty_cwd,
        env=environment,
        timeout=BUILD_TIMEOUT,
        redact=redact,
    )
    installed_version = _run(
        [str(pipx), "runpip", "huddleroom", "show", "huddleroom"],
        cwd=empty_cwd,
        env=environment,
        timeout=COMMAND_TIMEOUT,
        redact=redact,
    )
    assert f"Version: {upgraded_version}" in installed_version.stdout
    _run(
        [str(installed), "init-db"],
        cwd=empty_cwd,
        env=environment,
        timeout=COMMAND_TIMEOUT,
        redact=redact,
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (_expected_head(),)
        assert connection.execute("SELECT value FROM acceptance_sentinel").fetchone() == ("preserved",)
    with _running_server([str(installed)], cwd=empty_cwd, env=environment, redact=redact):
        pass
