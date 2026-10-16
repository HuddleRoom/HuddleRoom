"""SPR230/SPR231 migration preserves runtime state while adding the scalar fence."""
import os
import sqlite3
import subprocess
import sys
from tests.test_migration_049_goal_planning_hierarchy import _goal, _run


def test_051_backfills_pause_and_preserves_baseline_and_terminal_rows(tmp_path):
    path = tmp_path / "control.db"
    env = {**os.environ, "RALLY_DATABASE_URL": f"sqlite+aiosqlite:///{path}"}
    def migrate(revision):
        result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", revision],
                                env=env, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
    migrate("050")
    with sqlite3.connect(path) as conn:
        rows = {}
        for status in ("running", "paused", "completed", "cancelled"):
            goal = _goal(conn, status="paused" if status == "paused" else "active")
            rows[status] = _run(conn, goal, status=status, baseline_authorized=1)
        conn.commit()
    migrate("051")
    with sqlite3.connect(path) as conn:
        for status, run_id in rows.items():
            row = conn.execute("SELECT status, pause_requested, control_version, baseline_authorized "
                               "FROM orchestration_runs WHERE id=?", (run_id,)).fetchone()
            assert row == (status, int(status == "paused"), 0, 1)
