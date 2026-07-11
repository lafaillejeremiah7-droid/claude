"""FastAPI app tests using the TestClient (Req 1, 2, 12, 17)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dashboard.backend.app import create_app


@pytest.fixture
def client(live_bot_dir):
    env = {
        "BOT_DIR": str(live_bot_dir),
        "DASHBOARD_MODE": "paper",
        "TL_EMAIL": "a@b.com",
        "TL_PASSWORD": "hunter2",
        "TL_SERVER": "AQUA",
        "TL_ENVIRONMENT": "live",
    }
    app = create_app(env=env)
    with TestClient(app) as c:
        yield c


def test_health_returns_mode(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["mode"] == "paper"
    assert "uptime_s" in body
    # no secrets anywhere in health
    assert "hunter2" not in res.text


def test_snapshot_returns_json_without_secrets(client):
    res = client.get("/api/snapshot")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/json")
    text = res.text
    assert "hunter2" not in text
    assert "TL_PASSWORD" not in text
    assert "a@b.com" not in text
    body = res.json()
    assert "account" in body and "pnl" in body and "feed" in body


def test_credential_retrieval_is_rejected(client):
    for path in [
        "/api/snapshot?field=TL_PASSWORD",
        "/api/secret",
        "/api/snapshot?show=access_token",
        "/api/credentials",
    ]:
        res = client.get(path)
        assert res.status_code == 400, path
        assert "hunter2" not in res.text
        assert "access_token" not in res.text.lower() or "never exposed" in res.text


def test_root_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "BTC POLYMARKET TERMINAL" in res.text
    assert "/api/stream" in res.text  # subscribes to SSE


def test_reports_endpoint(client):
    res = client.get("/api/reports")
    assert res.status_code == 200
    body = res.json()
    assert body["daily"]["pnl_usd"] == 142.30
    assert body["weekly"]["improvements"]


def test_stream_route_registered(client):
    # The SSE route exists and is declared as an event-stream producer.
    paths = {getattr(r, "path", None) for r in client.app.routes}
    assert "/api/stream" in paths


def test_sse_event_stream_emits_snapshot_then_stops(live_bot_dir):
    """The SSE generator pushes an initial snapshot and terminates on disconnect."""
    import asyncio
    import json

    from dashboard.backend.app import sse_event_stream
    from dashboard.backend.readers import FileReader
    from dashboard.backend.store import SnapshotStore

    store = SnapshotStore(reader=FileReader(bot_dir=live_bot_dir, mode="live"), env={})

    class FakeRequest:
        async def is_disconnected(self):
            return True  # stop after the initial push

    async def collect():
        frames = []
        async def nosleep(_):  # never actually sleep in the test
            return None
        gen = sse_event_stream(store, FakeRequest(), sleep=nosleep)
        async for frame in gen:
            frames.append(frame)
            if len(frames) >= 3:
                break
        return frames

    frames = asyncio.run(collect())
    assert frames, "expected at least one SSE frame"
    assert frames[0].startswith("event: snapshot")
    payload = frames[0].split("data: ", 1)[1].strip()
    data = json.loads(payload)
    assert "account" in data and "pnl" in data
    # No secret leaked in the SSE frame.
    assert "hunter2" not in frames[0]
