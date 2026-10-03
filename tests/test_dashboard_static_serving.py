import os
import pytest

_DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "huddleroom", "static", "dashboard")


def _dashboard_built() -> bool:
    return os.path.isfile(os.path.join(_DASHBOARD_DIR, "index.html"))


@pytest.mark.asyncio
async def test_dashboard_serves_index(client):
    if not _dashboard_built():
        pytest.skip("dashboard build output not present; run `make build-frontend`")
    resp = await client.get("/dashboard/")
    assert resp.status_code == 200
    assert "root" in resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [
    "/dashboard/tasks",
    "/dashboard/meetings/some-uuid",
])
async def test_dashboard_spa_fallback(client, path):
    if not _dashboard_built():
        pytest.skip("dashboard build output not present; run `make build-frontend`")
    resp = await client.get(path)
    assert resp.status_code == 200
    assert "root" in resp.text


@pytest.mark.asyncio
async def test_dashboard_missing_asset_returns_404(client):
    if not _dashboard_built():
        pytest.skip("dashboard build output not present; run `make build-frontend`")
    resp = await client.get("/dashboard/assets/nonexistent.js")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dev_dashboard_serves(client):
    dev_dir = os.path.join(os.path.dirname(__file__), "..", "huddleroom", "static", "dev-dashboard")
    if not os.path.isfile(os.path.join(dev_dir, "index.html")):
        pytest.skip("dev-dashboard/index.html not present")
    resp = await client.get("/dev-dashboard/")
    assert resp.status_code == 200
