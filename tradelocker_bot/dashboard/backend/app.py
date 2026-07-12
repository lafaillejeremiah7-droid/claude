"""FastAPI application for the read-only trading dashboard.

Endpoints:
  GET /            -> serves the self-contained frontend (index.html)
  GET /api/snapshot-> current DashboardSnapshot JSON (no secrets)
  GET /api/stream  -> SSE stream: pushes the snapshot on content-hash change +
                      a heartbeat every ~10s
  GET /api/health  -> {status, mode, uptime_s} (no secrets)
  GET /api/reports -> latest daily/weekly/monthly report payloads

A background async poller refreshes the store (files <= 2s). Any request that
tries to read credentials/tokens is rejected (4xx) with no secret in the body.

Run:  uvicorn tradelocker_bot.dashboard.backend.app:app --port 8080
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)

from .readers import resolve_bot_dir, resolve_mode
from .security import CREDENTIAL_FIELDS, load_credentials
from .store import SnapshotStore

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover
    _load_dotenv = None

UTC = timezone.utc

FILE_POLL_INTERVAL = 2.0
HEARTBEAT_INTERVAL = 10.0

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"


async def sse_event_stream(
    store: "SnapshotStore",
    request,
    heartbeat_interval: float = HEARTBEAT_INTERVAL,
    sleep=asyncio.sleep,
    clock=time.monotonic,
):
    """Yield SSE frames: the snapshot on content-hash change + periodic heartbeat.

    Factored out of the route so it is directly testable: the loop terminates as
    soon as ``request.is_disconnected()`` is true (Req 12.2, 12 heartbeat).
    """
    last_beat = clock()
    # Push the current snapshot immediately on connect.
    snap = store.get()
    last_hash = store.content_hash
    yield f"event: snapshot\ndata: {json.dumps(snap, default=str)}\n\n"
    while True:
        if await request.is_disconnected():
            break
        current = store.content_hash
        if current is not None and current != last_hash:
            last_hash = current
            yield f"event: snapshot\ndata: {json.dumps(store.get(), default=str)}\n\n"
        now = clock()
        if now - last_beat >= heartbeat_interval:
            last_beat = now
            ts = datetime.now(UTC).isoformat()
            yield f": heartbeat {ts}\n\n"
        await sleep(1.0)

# Substrings that indicate an attempt to retrieve secrets (Req 2.4).
_SECRET_QUERY_TOKENS = tuple(f.lower() for f in CREDENTIAL_FIELDS) + (
    "password",
    "token",
    "secret",
    "credential",
)


def _secret_values_from_env(env: Optional[dict] = None) -> list:
    creds = load_credentials(env)
    return [v for v in creds.values() if isinstance(v, str) and v]


def _requests_secret(request: Request) -> bool:
    """True if the request path/query appears to ask for a secret value."""
    q = str(request.url.query or "").lower()
    path = request.url.path.lower()
    haystack = f"{path}?{q}"
    return any(tok in haystack for tok in _SECRET_QUERY_TOKENS)


def create_app(env: Optional[dict] = None) -> FastAPI:
    env = env if env is not None else os.environ

    # Load credentials from the bot's .env file so the dashboard can use the
    # TradeLocker API without requiring the user to duplicate env vars (Issue 2).
    # Safe: if the file doesn't exist or dotenv is unavailable, we silently skip.
    if _load_dotenv is not None:
        bot_dir = resolve_bot_dir(env)
        dotenv_path = bot_dir / ".env"
        if dotenv_path.is_file():
            _load_dotenv(dotenv_path, override=False)
            # If env is os.environ, the vars are now available globally.
            # If env is a custom dict (tests), we merge loaded vars into it
            # only when it's a mutable mapping referencing os.environ.
            if env is os.environ:
                pass  # load_dotenv already populated os.environ
            else:
                # For custom env dicts (e.g. tests), don't modify — they control
                # their own credential presence.
                pass

    store = SnapshotStore(
        secret_values=_secret_values_from_env(env),
        env=env,
    )
    app = FastAPI(title="TradeLocker Dashboard", docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.mode = resolve_mode(env)
    app.state.started_at = time.monotonic()
    app.state._poller_task = None

    async def _poll_loop():
        while True:
            try:
                await asyncio.to_thread(store.refresh)
            except Exception:
                # Degrade, never crash the poller (Req 1.8, 13.5).
                pass
            await asyncio.sleep(FILE_POLL_INTERVAL)

    @app.on_event("startup")
    async def _startup():
        app.state._poller_task = asyncio.create_task(_poll_loop())

    @app.on_event("shutdown")
    async def _shutdown():
        task = app.state._poller_task
        if task:
            task.cancel()

    # -- secret-retrieval rejection (applies before route handlers) -----
    @app.middleware("http")
    async def _reject_secret_requests(request: Request, call_next):
        if _requests_secret(request):
            return JSONResponse(
                status_code=400,
                content={"error": "credentials are never exposed"},
            )
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        try:
            return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
        except OSError:
            return HTMLResponse("<h1>Dashboard frontend not found</h1>", status_code=500)

    SIGNAL_TERMINAL_HTML = FRONTEND_DIR / "signal_terminal.html"

    @app.get("/terminal", response_class=HTMLResponse)
    async def signal_terminal():
        """Serve the XAUUSD ASWP signal terminal dashboard (dark theme)."""
        try:
            return HTMLResponse(SIGNAL_TERMINAL_HTML.read_text(encoding="utf-8"))
        except OSError:
            return HTMLResponse("<h1>Signal terminal not found</h1>", status_code=500)

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "mode": app.state.mode,
            "uptime_s": round(time.monotonic() - app.state.started_at, 3),
        }

    @app.get("/api/snapshot")
    async def snapshot():
        return JSONResponse(store.get())

    @app.get("/api/reports")
    async def reports():
        snap = store.get()
        return JSONResponse(snap.get("reports", {}))

    @app.get("/api/stream")
    async def stream(request: Request):
        return StreamingResponse(
            sse_event_stream(store, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
