"""
Configuration module for the Dashboard Backend.

Loads TradeLocker credentials from the bot's .env file and provides
all path/interval constants. Credentials are NEVER exposed to the frontend.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Resolve the tradelocker_bot directory (sibling to dashboard_backend)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_BOT_DIR = _THIS_DIR.parent / "tradelocker_bot"


class Settings(BaseSettings):
    """
    All settings are loaded from the bot's .env file on the server.
    No credential is ever accepted from a frontend request.
    """

    # --- TradeLocker credentials (Req 2) ---
    tl_email: str = Field(default="", alias="TL_EMAIL")
    tl_password: str = Field(default="", alias="TL_PASSWORD")
    tl_server: str = Field(default="", alias="TL_SERVER")
    tl_environment: str = Field(default="demo", alias="TL_ENVIRONMENT")
    tl_account_id: str = Field(default="", alias="TL_ACCOUNT_ID")

    # --- Bot configuration ---
    instruments: str = Field(default="BTCUSD,XAUUSD", alias="INSTRUMENTS")
    scan_interval_seconds: int = Field(default=60, alias="SCAN_INTERVAL_SECONDS")

    # --- Paths (derived, not from .env) ---
    bot_dir: Path = _DEFAULT_BOT_DIR

    # --- Polling intervals (seconds) ---
    file_poll_interval: float = 2.0
    api_poll_interval: float = 5.0

    # --- Freshness thresholds ---
    stale_data_threshold_seconds: float = 15.0
    bot_offline_threshold_seconds: float = 90.0

    # --- Token refresh ---
    token_refresh_margin_seconds: float = 30.0
    token_refresh_timeout_seconds: float = 10.0

    # --- SSE reconnect ---
    sse_reconnect_interval_seconds: float = 5.0
    sse_max_reconnect_attempts: int = 12

    # --- Equity curve baseline ---
    starting_equity_baseline: float = Field(default=0.0, alias="STARTING_EQUITY")

    class Config:
        env_file = str(_DEFAULT_BOT_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"
        populate_by_name = True

    # ----- Derived paths -----

    @property
    def logs_dir(self) -> Path:
        return self.bot_dir / "logs"

    @property
    def journal_dir(self) -> Path:
        return self.bot_dir / "journal"

    @property
    def daily_stats_file(self) -> Path:
        return self.logs_dir / "daily_stats.json"

    @property
    def positions_file(self) -> Path:
        return self.logs_dir / "active_positions.json"

    @property
    def adaptive_config_file(self) -> Path:
        return self.logs_dir / "adaptive_config.json"

    @property
    def trade_features_file(self) -> Path:
        return self.logs_dir / "trade_features.jsonl"

    @property
    def instrument_list(self) -> List[str]:
        return [i.strip() for i in self.instruments.split(",") if i.strip()]

    # ----- Credential validation (Req 2.8) -----

    @property
    def missing_credentials(self) -> List[str]:
        """Return list of credential names that are missing/empty."""
        missing: List[str] = []
        if not self.tl_email:
            missing.append("TL_EMAIL")
        if not self.tl_password:
            missing.append("TL_PASSWORD")
        if not self.tl_server:
            missing.append("TL_SERVER")
        if not self.tl_environment:
            missing.append("TL_ENVIRONMENT")
        return missing

    @property
    def credentials_valid(self) -> bool:
        return len(self.missing_credentials) == 0


def get_settings() -> Settings:
    """
    Factory that returns a Settings instance.
    Attempts to load the bot .env; if not found, uses defaults/env vars.
    """
    env_path = _DEFAULT_BOT_DIR / ".env"
    if env_path.exists():
        return Settings(_env_file=str(env_path))
    return Settings()
