"""
FastAPI routes for the Dashboard Backend.

Endpoints:
  GET /api/health       – Health check
  GET /api/state        – Full current state snapshot
  GET /api/stream       – SSE event stream (real-time updates)

All endpoints are read-only GET (Req 1.1).
No credentials or tokens are exposed (Req 2.3, 2.4).
"""

from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

router = APIRouter()


def _get_stream_manager(request: Request):
    """Get the stream manager from app state."""
    from dashboard_backend.main import stream_manager
    return stream_manager


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "service": "dashboard_backend",
    }


@router.get("/state")
async def get_state(request: Request) -> JSONResponse:
    """
    Get the full current state snapshot.
    Returns all computed metrics for the dashboard panels.

    This is a polling fallback; prefer /api/stream for real-time updates.
    """
    mgr = _get_stream_manager(request)
    if mgr is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Service not ready", "timestamp": time.time()},
        )

    state = await mgr.get_current_state()

    # Req 2.3: Ensure no secrets leak into response
    _sanitize_response(state)

    return JSONResponse(content=state)


@router.get("/stream")
async def event_stream(request: Request):
    """
    Server-Sent Events stream for real-time dashboard updates.

    The client connects once and receives JSON updates whenever
    file data or API data changes. Keepalive sent every 30s.

    Reconnection: client should reconnect within 5s on disconnect (Req 12.5, 12.6).
    """
    mgr = _get_stream_manager(request)
    if mgr is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Service not ready"},
        )

    async def event_generator():
        async for data in mgr.subscribe():
            # Check if client disconnected
            if await request.is_disconnected():
                break
            yield {
                "event": "update",
                "data": data,
                "retry": int(mgr._settings.sse_reconnect_interval_seconds * 1000),
            }

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.get("/instruments")
async def get_instruments(request: Request) -> Dict[str, Any]:
    """Get configured instruments list (Req 15.5)."""
    mgr = _get_stream_manager(request)
    if mgr is None:
        return {"instruments": [], "error": "Service not ready"}

    instruments = mgr._settings.instrument_list
    if not instruments:
        return {
            "instruments": [],
            "error": "No instruments configured for monitoring",
        }

    return {"instruments": instruments}


def _sanitize_response(state: Dict[str, Any]) -> None:
    """
    Remove any accidentally included secrets from response (Req 2.3).
    Defense-in-depth: even if internal code leaks a token, strip it here.
    """
    sensitive_keys = {
        "access_token", "refresh_token", "token",
        "password", "tl_password", "tl_email",
        "jwt", "secret", "credential",
    }

    def _strip(obj: Any) -> None:
        if isinstance(obj, dict):
            keys_to_remove = [
                k for k in obj
                if any(s in k.lower() for s in sensitive_keys)
            ]
            for k in keys_to_remove:
                del obj[k]
            for v in obj.values():
                _strip(v)
        elif isinstance(obj, list):
            for item in obj:
                _strip(item)

    _strip(state)
