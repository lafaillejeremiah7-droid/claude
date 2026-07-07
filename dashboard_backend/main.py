"""
Dashboard Backend – Main FastAPI application entry point.

Launches the read-only FastAPI server that:
  - Reads bot files and TradeLocker API
  - Computes metrics (P&L, streaks, confidence, equity curve, countdown)
  - Streams real-time updates via SSE to the frontend

Run with:
    uvicorn dashboard_backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dashboard_backend.config import get_settings
from dashboard_backend.guards.safety import apply_safety_guards
from dashboard_backend.api.routes import router as api_router
from dashboard_backend.stream.manager import StreamManager

logger = logging.getLogger("dashboard_backend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

settings = get_settings()
stream_manager: StreamManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global stream_manager

    logger.info("Dashboard Backend starting up...")

    # Validate credentials (Req 2.8)
    if not settings.credentials_valid:
        missing = ", ".join(settings.missing_credentials)
        logger.error(f"Missing credentials: {missing}. API reads disabled.")

    # Apply read-only safety guards (Req 1)
    apply_safety_guards(settings)

    # Start the streaming/polling manager
    stream_manager = StreamManager(settings)
    await stream_manager.start()

    logger.info("Dashboard Backend ready.")
    yield

    # Shutdown
    logger.info("Dashboard Backend shutting down...")
    if stream_manager:
        await stream_manager.stop()


app = FastAPI(
    title="TradeLocker Dashboard Backend",
    description="Read-only real-time dashboard for the TradeLocker trading bot.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow the local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local deployment; tighten in production
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api")
