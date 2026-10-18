import os
import pwd
import subprocess
from pathlib import Path

import pytest

from tests.live_acceptance.run_dir import (
    EVIDENCE_ROOT_ENV,
    IsolationError,
    assert_isolated,
    assert_isolated_url,
    build_run_dir,
    init_workspace_repo,
    real_home,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def evidence_root(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir()
    monkeypatch.setenv(EVIDENCE_ROOT_ENV, str(root))
    return root


@pytest.fixture
def run_dir(evidence_root):
    return build_run_dir(None, "selfcheck")


def _real_home_from_passwd() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


@pytest.mark.live
def test_accepts_db_and_workspace_inside_run_dir(run_dir):
    assert_isolated(run_dir, run_dir.db, run_dir.workspace)


@pytest.mark.live
def test_rejects_rally_test_db_name(run_dir):
    with pytest.raises(IsolationError):
        assert_isolated(run_dir, run_dir.root / "rally_test.db", run_dir.workspace)


@pytest.mark.live
def test_rejects_mimlo_path(run_dir):
    with pytest.raises(IsolationError):
        assert_isolated(run_dir, run_dir.root / "mimlo-huddleroom.db 2", run_dir.workspace)


@pytest.mark.live
@pytest.mark.parametrize("name", ["Rally.db", "MIMLO-HuddleRoom.db"])
def test_rejects_case_variant_names(run_dir, name):
    with pytest.raises(IsolationError):
        assert_isolated(run_dir, run_dir.root / name, run_dir.workspace)


@pytest.mark.live
@pytest.mark.parametrize("name", ["rally_test.db-wal", "rally_test.db-shm", "mimlo-huddleroom.db-wal"])
def test_rejects_sidecar_names(run_dir, name):
    with pytest.raises(IsolationError):
        assert_isolated(run_dir, run_dir.root / name, run_dir.workspace)


@pytest.mark.live
def test_rejects_symlinked_db_pointing_outside(run_dir, tmp_path):
    outside = tmp_path / "outside.db"
    outside.write_bytes(b"")
    link = run_dir.root / "app.db"
    link.symlink_to(outside)
    with pytest.raises(IsolationError):
        assert_isolated(run_dir, link, run_dir.workspace)


@pytest.mark.live
def test_rejects_symlinked_workspace(run_dir, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = run_dir.root / "ws-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(IsolationError):
        assert_isolated(run_dir, run_dir.db, link)


@pytest.mark.live
def test_rejects_workspace_outside_run_dir(run_dir, tmp_path):
    with pytest.raises(IsolationError):
        assert_isolated(run_dir, run_dir.db, tmp_path / "elsewhere")


@pytest.mark.live
def test_rejects_relative_paths(run_dir):
    with pytest.raises(IsolationError):
        assert_isolated(run_dir, Path("app.db"), run_dir.workspace)
    with pytest.raises(IsolationError):
        assert_isolated(run_dir, run_dir.db, Path("workspace"))


@pytest.mark.live
def test_rejects_home_database(run_dir, monkeypatch, tmp_path):
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    home_db = _real_home_from_passwd() / ".huddleroom" / "huddleroom.db"
    assert real_home() == _real_home_from_passwd()
    with pytest.raises(IsolationError):
        assert_isolated(run_dir, home_db, run_dir.workspace)


@pytest.mark.live
def test_rejects_run_root_inside_home_huddleroom(tmp_path, monkeypatch):
    monkeypatch.setenv(EVIDENCE_ROOT_ENV, str(_real_home_from_passwd() / ".huddleroom" / "evidence"))
    with pytest.raises(IsolationError):
        build_run_dir(None, "selfcheck")


@pytest.mark.live
def test_rejects_run_root_inside_repo(tmp_path):
    target = REPO_ROOT / "tmp-live-evidence-should-not-exist"
    with pytest.raises(IsolationError):
        build_run_dir(target, "selfcheck")
    assert not target.exists()


@pytest.mark.live
def test_missing_evidence_root_env_raises(monkeypatch):
    monkeypatch.delenv(EVIDENCE_ROOT_ENV, raising=False)
    with pytest.raises(ValueError):
        build_run_dir(None, "selfcheck")


@pytest.mark.live
@pytest.mark.parametrize("label", ["", "a/b", "a\\b", "..", "../x", "a..b", "a b"])
def test_rejects_bad_labels(evidence_root, label):
    with pytest.raises(ValueError):
        build_run_dir(evidence_root, label)


@pytest.mark.live
def test_same_label_gives_distinct_run_dirs(evidence_root):
    first = build_run_dir(evidence_root, "same")
    second = build_run_dir(evidence_root, "same")
    assert first.root != second.root


@pytest.mark.live
@pytest.mark.parametrize(
    "url",
    [
        "postgresql://user:pw@localhost/huddleroom",
        "sqlite://",
        "sqlite:///:memory:",
        "sqlite:///relative.db",
    ],
)
def test_url_guard_rejects(run_dir, url):
    with pytest.raises(IsolationError):
        assert_isolated_url(run_dir, url)


@pytest.mark.live
def test_url_guard_rejects_uri_mode(run_dir):
    url = f"sqlite:///{run_dir.db}?uri=true"
    with pytest.raises(IsolationError):
        assert_isolated_url(run_dir, url)


@pytest.mark.live
def test_url_guard_rejects_home_database(run_dir):
    home_db = _real_home_from_passwd() / ".huddleroom" / "huddleroom.db"
    with pytest.raises(IsolationError):
        assert_isolated_url(run_dir, f"sqlite+aiosqlite:///{home_db}")


@pytest.mark.live
def test_url_guard_accepts_run_dir_database(run_dir):
    assert_isolated_url(run_dir, f"sqlite+aiosqlite:///{run_dir.db}")


@pytest.mark.live
def test_init_workspace_repo_makes_single_commit(run_dir):
    repo = run_dir.workspace / "repo"
    init_workspace_repo(repo, run_dir)
    assert (repo / ".git").is_dir()
    log = subprocess.run(
        ["git", "-C", str(repo), "log", "--oneline"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    assert len(log) == 1


@pytest.mark.live
def test_init_workspace_repo_rejects_path_outside_run_dir(run_dir, tmp_path):
    with pytest.raises(IsolationError):
        init_workspace_repo(tmp_path / "ws", run_dir)
