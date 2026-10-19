import io
import json
import subprocess
import time
import sqlite3
import tarfile
import uuid

import httpx
import pytest

from tests.live_acceptance import oracle
from tests.live_acceptance.dry_run import _sha256, _snapshot_dir
from tests.live_acceptance.run_dir import RunDir
from tests.live_acceptance.scenario import LiveScenario, RestoreError, find_path_columns, restore_snapshot

pytestmark = pytest.mark.live


def _bare_run_dir(root):
    return RunDir(root, root / "app.db", root / "workspace", root / "snapshots", root / "shim",
                  root / "proxy.jsonl", root / "spend.jsonl", root / "config.json")


def test_artifact_root_defaults_and_can_be_overridden(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    default = LiveScenario("default", None, tmp_path, 1)
    default.run_dir = _bare_run_dir(run_root)
    default._set_artifact_root()
    default.mark("default")
    assert default.artifact_root == run_root.resolve()
    assert (run_root / "timeline.md").exists()

    artifacts = tmp_path / "artifacts"
    redirected = LiveScenario("redirected", None, tmp_path, 1, artifact_root=artifacts)
    redirected.run_dir = _bare_run_dir(run_root)
    redirected._set_artifact_root()
    redirected.mark("redirected")
    assert redirected.artifact_root == artifacts.resolve()
    assert "redirected" in (artifacts / "timeline.md").read_text()
    assert "redirected" not in (run_root / "timeline.md").read_text()


@pytest.fixture
def fakes(tmp_path):
    out = {}
    for tool in ("codex", "claude"):
        p = tmp_path / f"real-{tool}"
        p.write_text("#!/bin/sh\necho \"Logged in using ChatGPT\" >&2\necho fake 0.0\n")
        p.chmod(0o755)
        out[tool] = p
    return out


@pytest.fixture
def snapshot(tmp_path, fakes):
    """Seed a run with a project, snapshot it, return (snapshot_dir, old_project_id, old_root)."""
    with LiveScenario("seed", None, tmp_path / "ev-a", 1000, real_paths=fakes) as sc:
        pid = sc.ids["project_id"]
        old_root = sc.run_dir.root
        (sc.run_dir.workspace / "marker.txt").write_text("hi")
        with open(sc.run_dir.workspace / ".git" / "config", "a") as f:
            f.write(f'[remote "x"]\n\turl = {old_root}/workspace\n')
        snap = _snapshot_dir(sc.run_dir, "snap-x")
        sc.set_verdict(True)
    return old_root / "snapshots" / "snap-x", pid, old_root


def test_restore_and_scenario(tmp_path, fakes, snapshot):
    snap, pid, old_root = snapshot
    assert str(old_root) in sqlite3.connect(snap / "app.db").execute(
        "select workspace_path from projects").fetchone()[0]
    with LiveScenario("restored", snap, tmp_path / "ev-b", 1000, real_paths=fakes,
                      hold_spec={"codex": [1]}) as sc:
        rd = sc.run_dir
        assert rd.root != old_root and (rd.workspace / "marker.txt").read_text() == "hi"
        assert find_path_columns(rd.db, [str(old_root)]) == []
        assert httpx.get(f"{sc.base_url}/health").status_code == 200
        project = httpx.get(f"{sc.base_url}/api/v1/projects/{pid}").json()
        assert project["workspace_path"] == str(rd.workspace.resolve())
        assert [p["id"] for p in sc.oracle.event_log(pid)] is not None
        assert (rd.shim / "gates" / "codex.nth.1").exists()
        sc.mark("checked")
        assert sc.oracle_dump("probe").exists()
        assert (rd.snapshots / "00-before.db").exists()
        sc.set_verdict(True)
    assert (rd.snapshots / "90-after.db").exists()
    assert "checked" in (rd.root / "timeline.md").read_text()
    assert json.loads((rd.root / "verdict.json").read_text())["passed"] is True
    assert all(not h.live_pids() for h in sc.shims.values())
    with pytest.raises(httpx.HTTPError):
        httpx.get(f"{sc.base_url}/health", timeout=1)
    assert sc.findings["ledger_reaped"] is not None
    assert not sc.errors


def test_scenario_accepts_g2g4_roster(tmp_path, fakes):
    with LiveScenario("g2g4", None, tmp_path / "ev", 1000, real_paths=fakes, roster="g2g4") as sc:
        agents = httpx.get(f"{sc.base_url}/api/v1/agents").json()["items"]
        assert {agent["name"] for agent in agents} == {
            "live-manager", "live-writer", "live-analyst", "live-reviewer",
        }
        sc.set_verdict(True)


def test_tampered_snapshot_refused(tmp_path, snapshot):
    snap, _, _ = snapshot
    with open(snap / "app.db", "ab") as f:
        f.write(b"x")
    with pytest.raises(RestoreError):
        restore_snapshot(snap, tmp_path / "ev-c", "bad")


def _refresh(snap):
    m = json.loads((snap / "manifest.json").read_text())
    m["db_file_sha256"] = _sha256(snap / "app.db")
    m["db_canonical_sha256"] = oracle._canonical_hash(snap / "app.db")
    m["workspace_tar_sha256"] = _sha256(snap / "workspace.tar.gz")
    (snap / "manifest.json").write_text(json.dumps(m))


def _edit(snap, sql, *args):
    conn = sqlite3.connect(snap / "app.db")
    conn.execute(sql, args)
    conn.commit()
    conn.close()
    _refresh(snap)


def _add_run(snap, status, paused):
    gid, rid = uuid.uuid4().hex, uuid.uuid4().hex
    conn = sqlite3.connect(snap / "app.db")
    pid = conn.execute("select id from projects").fetchone()[0]
    now = "2026-01-01 00:00:00"
    conn.execute("insert into orchestration_goals (id, project_id, objective, created_at, updated_at, original_request)"
                 " values (?,?,?,?,?,?)", (gid, pid, "o", now, now, "o"))
    conn.execute("insert into orchestration_runs (id, goal_id, status, pause_requested, started_at, created_at,"
                 " updated_at) values (?,?,?,?,?,?,?)", (rid, gid, status, int(paused), now, now, now))
    conn.commit()
    conn.close()
    _refresh(snap)


def test_git_config_rewritten_and_00_restored(tmp_path, snapshot):
    snap, _, old_root = snapshot
    rd = restore_snapshot(snap, tmp_path / "ev-g", "git")
    cfg = (rd.workspace / ".git" / "config").read_text()
    assert str(old_root) not in cfg and f"{rd.root.resolve()}/workspace" in cfg
    assert (rd.snapshots / "00-restored.db").exists()


def test_boundary_aware(tmp_path, snapshot):
    snap, _, old_root = snapshot
    _edit(snap, "update projects set description = ?", f"{old_root}-b/keep")  # sibling dir, not the root
    rd = restore_snapshot(snap, tmp_path / "ev-h", "bound")
    assert f"{old_root}-b/keep" == sqlite3.connect(rd.db).execute("select description from projects").fetchone()[0]


def test_leftover_old_root_fails(tmp_path, snapshot):
    snap, _, old_root = snapshot
    _edit(snap, "update projects set description = ?", f"see {old_root}/x")
    with pytest.raises(RestoreError, match="projects"):
        restore_snapshot(snap, tmp_path / "ev-d", "left")


def test_no_matching_project_path_fails(tmp_path, snapshot):
    snap, _, _ = snapshot
    _edit(snap, "update projects set workspace_path = '/elsewhere'")
    with pytest.raises(RestoreError, match="no projects"):
        restore_snapshot(snap, tmp_path / "ev-e", "nomatch")


def test_paused_gate(tmp_path, snapshot):
    snap, _, _ = snapshot
    _add_run(snap, "paused", True)
    restore_snapshot(snap, tmp_path / "ev-p", "paused")
    _add_run(snap, "running", False)
    with pytest.raises(RestoreError, match="not paused"):
        restore_snapshot(snap, tmp_path / "ev-q", "running")


def test_manifest_missing_key(tmp_path, snapshot):
    snap, _, _ = snapshot
    m = json.loads((snap / "manifest.json").read_text())
    del m["db_canonical_sha256"]
    (snap / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(RestoreError, match="manifest missing"):
        restore_snapshot(snap, tmp_path / "ev-m", "mf")


def test_canonical_mismatch(tmp_path, snapshot):
    snap, _, _ = snapshot
    m = json.loads((snap / "manifest.json").read_text())
    m["db_canonical_sha256"] = "0" * 64
    (snap / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(RestoreError, match="canonical"):
        restore_snapshot(snap, tmp_path / "ev-n", "canon")


def test_tar_member_outside_workspace(tmp_path, snapshot):
    snap, _, _ = snapshot
    with tarfile.open(tmp_path / "t.tar.gz", "w:gz") as tf:
        info = tarfile.TarInfo("workspace/ok")
        tf.addfile(info, io.BytesIO(b""))
        info = tarfile.TarInfo("evil")
        tf.addfile(info, io.BytesIO(b""))
    (tmp_path / "t.tar.gz").replace(snap / "workspace.tar.gz")
    _refresh(snap)
    with pytest.raises(RestoreError, match="outside workspace"):
        restore_snapshot(snap, tmp_path / "ev-t", "tar")


def test_error_path_findings(tmp_path, fakes, snapshot, monkeypatch):
    snap, _, _ = snapshot
    _add_run(snap, "paused", True)
    monkeypatch.setenv("LIVE_TEST_TOKEN", "supersecretvalue123")
    with pytest.raises(ValueError):
        with LiveScenario("err", snap, tmp_path / "ev-x", 1000, real_paths=fakes) as sc:
            sc.findings["leak"] = "supersecretvalue123"
            raise ValueError("boom")
    rd = sc.run_dir
    assert "pause_error_path" in sc.findings
    assert (rd.snapshots / "oracle-end.json").exists()
    f = (rd.root / "findings.json").read_text()
    assert "supersecretvalue123" not in f and "secret-looking" in f and '"timeout_hints"' in f
    assert json.loads((rd.root / "verdict.json").read_text())["passed"] is False
    assert sc.findings["campaign_ledger"]["path"] == str(tmp_path / "ev-x" / "campaign-spend.jsonl")
    assert "campaign_ledger" in sc.findings


def test_runs_table_missing(tmp_path, snapshot):
    snap, _, _ = snapshot
    _edit(snap, "drop table orchestration_runs")
    with pytest.raises(RestoreError, match="orchestration_runs"):
        restore_snapshot(snap, tmp_path / "ev-r", "norun")


def test_campaign_cap_refuses_call(tmp_path, fakes):
    marker = tmp_path / "ran"
    fakes["codex"].write_text(f'#!/bin/sh\n[ "$1" = login ] && {{ echo "Logged in using ChatGPT" >&2; exit 0; }}\ntouch {marker}\n')
    with LiveScenario("cap", None, tmp_path / "ev-c2", 100_000, real_paths=fakes, campaign_cap=100) as sc:
        r = subprocess.run([str(sc.shims["codex"].path), "exec", "x"], input=b"hello", capture_output=True)
        assert r.returncode == 75
        assert "campaign" in (sc.run_dir.shim / "shim.log").read_text()
        sc.set_verdict(True)
    assert not marker.exists()


def test_campaign_support_mandatory(tmp_path, fakes, monkeypatch):
    import inspect
    from tests.live_acceptance import scenario
    monkeypatch.setattr(scenario.inspect, "signature", lambda f: inspect.Signature())
    with pytest.raises(RuntimeError, match="campaign"):
        with LiveScenario("nocamp", None, tmp_path / "ev-n2", 1000, real_paths=fakes):
            pass


def test_teardown_kills_server_first_and_held_call_never_runs(tmp_path, fakes):
    marker = tmp_path / "ran"
    fakes["codex"].write_text(f'#!/bin/sh\n[ "$1" = login ] && {{ echo "Logged in using ChatGPT" >&2; exit 0; }}\ntouch {marker}\n')
    with LiveScenario("order", None, tmp_path / "ev-o", 100_000, real_paths=fakes, hold_spec={"codex": [1]}) as sc:
        p = subprocess.Popen([str(sc.shims["codex"].path), "exec", "x"], stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        p.stdin.write(b"hi")
        p.stdin.close()
        assert sc.shims["codex"].wait_started(15)
        time.sleep(0.5)
        assert sc.shims["codex"].live_pids()
        sc.set_verdict(True)
    p.wait(timeout=15)
    assert not marker.exists()
    assert all(not h.live_pids() for h in sc.shims.values()) and not sc.errors
    lines = [l for l in (sc.run_dir.root / "timeline.md").read_text().splitlines() if "teardown:" in l]
    assert [l.split("teardown: ")[1] for l in lines] == ["server killed", "deny", "gates released"]
    conn = sqlite3.connect(sc.run_dir.db)
    assert conn.execute("select count(*) from orchestration_runs where active_blockers != '[]'").fetchone()[0] == 0


def test_take_snapshot_clean_and_dirty(tmp_path, fakes, snapshot):
    snap, _, _ = snapshot
    _add_run(snap, "paused", True)
    with LiveScenario("snapc", snap, tmp_path / "ev-s", 1000, real_paths=fakes) as sc:
        m = sc.take_snapshot("snap-ok")
        assert m["clean"] is True and (sc.run_dir.snapshots / "snap-ok" / "manifest.json").exists()
        assert json.loads(sc.run_dir.config.read_text())["snapshots"]["snap-ok"] == sc.snapshots["snap-ok"]
        conn = sqlite3.connect(sc.run_dir.db, timeout=30)
        conn.execute("update orchestration_runs set active_blockers = '[\"x_analyzer_error\"]'")
        conn.commit()
        conn.close()
        with pytest.raises(RuntimeError, match="dirty"):
            sc.take_snapshot("snap-bad")
        assert sc.take_snapshot("snap-bad2", require_clean=False)["clean"] is False
        sc.set_verdict(True)
