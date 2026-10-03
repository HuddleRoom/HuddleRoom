from pathlib import Path

import pytest
from fastapi import HTTPException

from huddleroom.services.project_service import ProjectService


@pytest.mark.asyncio
async def test_require_runnable_project_returns_verified_canonical_workspace(db_session, test_project, tmp_path: Path):
    """Removing the canonical-path check would allow a changed symlink target to launch."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_project.workspace_path = str(workspace.resolve())

    assert await ProjectService().require_runnable_project(db_session, test_project.id) == workspace.resolve()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "workspace_path", "reason"),
    [
        ("archived", None, "project_inactive"),
        ("active", None, "workspace_unset"),
        ("active", "relative/workspace", "workspace_invalid"),
    ],
)
async def test_require_runnable_project_rejects_non_runnable_project(
    db_session, test_project, status: str, workspace_path: str | None, reason: str
):
    """Removing each guard branch would permit an unsafe execution launch."""
    test_project.status = status
    test_project.workspace_path = workspace_path

    with pytest.raises(HTTPException) as exc_info:
        await ProjectService().require_runnable_project(db_session, test_project.id)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {"code": "project_not_runnable", "reason": reason}


@pytest.mark.asyncio
async def test_require_runnable_project_rejects_workspace_that_no_longer_exists(db_session, test_project, tmp_path: Path):
    """Removing strict re-resolution would launch in a deleted workspace."""
    test_project.workspace_path = str(tmp_path / "missing")

    with pytest.raises(HTTPException) as exc_info:
        await ProjectService().require_runnable_project(db_session, test_project.id)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {"code": "project_not_runnable", "reason": "workspace_unavailable"}


@pytest.mark.asyncio
async def test_require_runnable_project_rejects_symlink_with_changed_target(db_session, test_project, tmp_path: Path):
    """Removing path equality would silently launch in the symlink's new target."""
    first_target = tmp_path / "first"
    second_target = tmp_path / "second"
    first_target.mkdir()
    second_target.mkdir()
    workspace = tmp_path / "workspace"
    workspace.symlink_to(first_target, target_is_directory=True)
    test_project.workspace_path = str(workspace)
    workspace.unlink()
    workspace.symlink_to(second_target, target_is_directory=True)

    with pytest.raises(HTTPException) as exc_info:
        await ProjectService().require_runnable_project(db_session, test_project.id)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == {"code": "project_not_runnable", "reason": "workspace_invalid"}
