import pytest
from httpx import AsyncClient, ASGITransport
from core.server import create_app

@pytest.fixture
def app():
    return create_app()

@pytest.mark.asyncio
async def test_health(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

@pytest.mark.asyncio
async def test_list_models(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        r = await client.get("/models")
        assert r.status_code == 200
        assert "models" in r.json()

@pytest.mark.asyncio
async def test_infer_unknown_model(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        r = await client.post("/infer", json={
            "model_name": "does_not_exist",
            "inputs": {"data": [1.0]},
        })
        assert r.status_code == 200
        assert r.json()["status"] == "failed"

@pytest.mark.asyncio
async def test_queue_stats(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        r = await client.get("/queue/stats")
        assert r.status_code == 200
        assert "queue_size" in r.json()

@pytest.mark.asyncio
async def test_shadow_summary(app):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        r = await client.get("/shadow/summary")
        assert r.status_code == 200