import builtins
import json
from pathlib import Path

import pytest

import tests.live_acceptance.auth_dir_watch as adw
from tests.live_acceptance.auth_dir_watch import classify, diff, listing, write_listing


@pytest.fixture
def fake_home(tmp_path):
    home = tmp_path / "fake_home"
    (home / ".codex").mkdir(parents=True)
    (home / ".claude").mkdir()
    (home / ".codex" / "config.toml").write_text("model = 'a'\n")
    (home / ".codex" / "auth.json").write_text('{"t": "1"}')
    (home / ".claude" / "settings.json").write_text("{}")
    (home / ".claude.json").write_text("{}")
    return home


def _paths(home: Path) -> list[Path]:
    return [home / ".codex", home / ".claude", home / ".claude.json"]


@pytest.mark.live
def test_history_added_is_other(fake_home):
    before = listing(_paths(fake_home))
    (fake_home / ".codex" / "history.jsonl").write_text("line\n")
    c = classify(diff(before, listing(_paths(fake_home))))
    assert c["other_changed"] == [".codex/history.jsonl"]
    assert c["config_changed"] == [] and c["auth_refreshed"] == []


@pytest.mark.live
def test_config_modified_is_config_changed(fake_home):
    before = listing(_paths(fake_home))
    (fake_home / ".codex" / "config.toml").write_text("model = 'changed-longer'\n")
    c = classify(diff(before, listing(_paths(fake_home))))
    assert c["config_changed"] == [".codex/config.toml"]
    assert c["other_changed"] == []


@pytest.mark.live
def test_auth_modified_is_auth_refreshed(fake_home):
    before = listing(_paths(fake_home))
    (fake_home / ".codex" / "auth.json").write_text('{"t": "2222222"}')
    c = classify(diff(before, listing(_paths(fake_home))))
    assert c["auth_refreshed"] == [".codex/auth.json"]
    assert c["config_changed"] == []


@pytest.mark.live
def test_claude_json_change_is_state_only(fake_home):
    before = listing(_paths(fake_home))
    (fake_home / ".claude.json").write_text('{"numStartups": 42}')
    c = classify(diff(before, listing(_paths(fake_home))))
    assert c["state_changed"] == [".claude.json"]
    assert c["config_changed"] == []
    assert c["other_changed"] == []


@pytest.mark.live
def test_removed_and_added_detected(fake_home):
    before = listing(_paths(fake_home))
    (fake_home / ".claude" / "settings.json").unlink()
    (fake_home / ".claude" / "session.log").write_text("x")
    d = diff(before, listing(_paths(fake_home)))
    assert d["removed"] == [".claude/settings.json"]
    assert d["added"] == [".claude/session.log"]
    c = classify(d)
    assert c["config_changed"] == [".claude/settings.json"]
    assert c["other_changed"] == [".claude/session.log"]


@pytest.mark.live
def test_nested_config_is_other_changed(fake_home):
    (fake_home / ".codex" / "sessions" / "x").mkdir(parents=True)
    (fake_home / ".codex" / "tools").mkdir()
    (fake_home / ".codex" / "tools" / "config.toml").write_text("a")
    before = listing(_paths(fake_home))
    (fake_home / ".codex" / "tools" / "config.toml").write_text("bbbbbbbb")
    c = classify(diff(before, listing(_paths(fake_home))))
    assert c["config_changed"] == []
    assert c["other_changed"] == [".codex/tools/config.toml"]


@pytest.mark.live
def test_nested_sessions_config_is_other_changed(fake_home):
    d = fake_home / ".codex" / "sessions" / "x"
    d.mkdir(parents=True)
    (d / "config.toml").write_text("a")
    before = listing(_paths(fake_home))
    (d / "config.toml").write_text("bbbbbbbb")
    c = classify(diff(before, listing(_paths(fake_home))))
    assert c["config_changed"] == []
    assert c["other_changed"] == [".codex/sessions/"]


@pytest.mark.live
def test_pruned_tree_summarized_and_change_detected(fake_home):
    sessions = fake_home / ".codex" / "sessions"
    sessions.mkdir()
    for i in range(500):
        (sessions / f"s{i}.jsonl").write_text("x")
    snap = listing(_paths(fake_home))
    summary_key = ".codex/sessions/"
    assert summary_key in snap
    assert snap[summary_key]["count"] == 500
    assert not any(k.startswith(".codex/sessions/s") for k in snap)

    (sessions / "new.jsonl").write_text("y")
    after = listing(_paths(fake_home))
    assert after[summary_key]["count"] == 501
    assert diff(snap, after)["changed"] == [summary_key]


@pytest.mark.live
def test_vanished_entry_recorded_as_error(fake_home, monkeypatch):
    real = adw._entry_stat

    def _flaky(entry):
        if entry.name == "history.jsonl":
            raise FileNotFoundError(2, "gone")
        return real(entry)

    (fake_home / ".codex" / "history.jsonl").write_text("line\n")
    monkeypatch.setattr(adw, "_entry_stat", _flaky)
    snap = listing(_paths(fake_home))
    assert snap[".codex/history.jsonl"] == {"error": "ENOENT"}
    assert ".codex/config.toml" in snap


@pytest.mark.live
def test_symlink_pointing_out_not_followed(fake_home, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside")
    (fake_home / ".codex" / "linked").symlink_to(outside, target_is_directory=True)
    snap = listing(_paths(fake_home))
    assert snap[".codex/linked"]["symlink"] == str(outside)
    assert not any("secret.txt" in k for k in snap)


@pytest.mark.live
def test_symlink_retarget_detected(fake_home, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    link = fake_home / ".codex" / "link"
    link.symlink_to(a, target_is_directory=True)
    before = listing(_paths(fake_home))
    link.unlink()
    link.symlink_to(b, target_is_directory=True)
    after = listing(_paths(fake_home))
    assert after[".codex/link"]["symlink"] == str(b)
    assert diff(before, after)["changed"] == [".codex/link"]


@pytest.mark.live
def test_symlinked_root_raises(fake_home, tmp_path):
    real_dir = tmp_path / "real_codex"
    real_dir.mkdir()
    root_link = tmp_path / "rootlink"
    root_link.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ValueError):
        listing([root_link])


@pytest.mark.live
def test_listing_never_opens_files(fake_home, monkeypatch):
    def _fail(*a, **k):
        raise AssertionError("listing must not open files")

    monkeypatch.setattr(builtins, "open", _fail)
    snap = listing(_paths(fake_home))
    assert snap[".codex/config.toml"]["size"] == len("model = 'a'\n")


@pytest.mark.live
def test_write_listing_round_trip(fake_home, tmp_path):
    snap = listing(_paths(fake_home))
    out = tmp_path / "listing.json"
    write_listing(out, snap)
    assert json.loads(out.read_text()) == snap
