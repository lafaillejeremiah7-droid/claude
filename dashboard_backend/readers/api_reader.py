"""
API_Reader – Read-only TradeLocker API client.

Responsibilities (Requirements 1, 2, 3, 9, 15):
  - Authenticate using credentials from server-side .env ONLY.
  - Refresh JWT tokens proactively (30s before expiry, 10s timeout).
  - Fetch account state (balance, equity, free margin) via GET.
  - Fetch live quotes per instrument via GET.
  - NEVER issue order-create, order-modify, position-close, or position-modify requests.
  - NEVER expose tokens or credentials to any response/stream.

All outbound requests are limited to GET (data reads) and POST (auth/refresh only).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from dashboard_backend.config import Settings

logger = logging.getLogger("dashboard_backend.api_reader")

# ---------------------------------------------------------------------------
# Blocked HTTP methods / paths (Req 1.2, 1.7)
# ---------------------------------------------------------------------------
_BLOCKED_PATTERNS = [
    "order",       # order-create, order-modify
    "position",    # position-close, position-modify (POST/PUT/DELETE)
]


@dataclass
class TokenState:
    """Holds current auth tokens and expiry."""
    access_token: str = ""
    refresh_token: str = ""
    access_expiry: float = 0.0  # epoch seconds
    account_id: str = ""
    auth_failed: bool = False
    last_error: str = ""


@dataclass
class AccountState:
    """Live account state from API."""
    balance: Optional[float] = None
    equity: Optional[float] = None
    free_margin: Optional[float] = None
    timestamp: float = 0.0
    source: str = "api"  # "api" or "fallback"
    error: str = ""


@dataclass
class Quote:
    """Live quote for an instrument."""
    symbol: str = ""
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    timestamp: float = 0.0
    error: str = ""


class APIReader:
    """
    Read-only TradeLocker API client.
    
    SAFETY: Only GET requests for data. POST only for /auth/jwt/token and
    /auth/jwt/refresh. All other mutation verbs are blocked.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._token = TokenState()
        self._client: Optional[httpx.AsyncClient] = None
        self._base_url = self._resolve_base_url()
        self._lock = asyncio.Lock()

    def _resolve_base_url(self) -> str:
        """Resolve TradeLocker API base URL from environment setting."""
        env = self._settings.tl_environment.lower()
        server = self._settings.tl_server.rstrip("/")
        if server:
            return server
        # Default TradeLocker endpoints
        if env == "demo":
            return "https://demo.tradelocker.com"
        return "https://live.tradelocker.com"

    async def initialize(self) -> None:
        """Create HTTP client and attempt initial authentication."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            follow_redirects=True,
        )
        if self._settings.credentials_valid:
            await self._authenticate()
        else:
            self._token.auth_failed = True
            self._token.last_error = (
                f"Missing credentials: {', '.join(self._settings.missing_credentials)}"
            )
            logger.error(self._token.last_error)

    async def close(self) -> None:
        """Shutdown HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Authentication (Req 2.1, 2.5, 2.6)
    # ------------------------------------------------------------------

    async def _authenticate(self) -> bool:
        """Perform initial authentication with email/password."""
        try:
            resp = await self._client.post(
                f"{self._base_url}/backend-api/auth/jwt/token",
                json={
                    "email": self._settings.tl_email,
                    "password": self._settings.tl_password,
                    "server": self._settings.tl_server,
                },
                timeout=self._settings.token_refresh_timeout_seconds,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._token.access_token = data.get("accessToken", "")
                self._token.refresh_token = data.get("refreshToken", "")
                # Assume token lifetime of 300s if not provided
                expires_in = data.get("expiresIn", 300)
                self._token.access_expiry = time.time() + expires_in
                self._token.auth_failed = False
                self._token.last_error = ""

                # Resolve account ID
                if not self._settings.tl_account_id:
                    await self._resolve_account_id()
                else:
                    self._token.account_id = self._settings.tl_account_id

                logger.info("TradeLocker authentication successful.")
                return True
            else:
                self._token.auth_failed = True
                self._token.last_error = f"Auth failed: HTTP {resp.status_code}"
                logger.error(self._token.last_error)
                return False
        except Exception as e:
            self._token.auth_failed = True
            self._token.last_error = f"Auth exception: {e}"
            logger.error(self._token.last_error)
            return False

    async def _refresh_token(self) -> bool:
        """Refresh the JWT access token (Req 2.5, 2.6)."""
        if not self._token.refresh_token:
            return await self._authenticate()

        try:
            resp = await self._client.post(
                f"{self._base_url}/backend-api/auth/jwt/refresh",
                json={"refreshToken": self._token.refresh_token},
                timeout=self._settings.token_refresh_timeout_seconds,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._token.access_token = data.get("accessToken", "")
                self._token.refresh_token = data.get("refreshToken", self._token.refresh_token)
                expires_in = data.get("expiresIn", 300)
                self._token.access_expiry = time.time() + expires_in
                self._token.auth_failed = False
                self._token.last_error = ""
                logger.debug("Token refreshed successfully.")
                return True
            else:
                self._token.auth_failed = True
                self._token.last_error = f"Token refresh failed: HTTP {resp.status_code}"
                logger.warning(self._token.last_error)
                return False
        except Exception as e:
            self._token.auth_failed = True
            self._token.last_error = f"Token refresh exception: {e}"
            logger.warning(self._token.last_error)
            return False

    async def _ensure_token(self) -> bool:
        """Ensure we have a valid token, refreshing if needed (Req 2.5)."""
        if self._token.auth_failed and not self._token.refresh_token:
            # Try full re-auth
            return await self._authenticate()

        time_to_expiry = self._token.access_expiry - time.time()
        if time_to_expiry <= self._settings.token_refresh_margin_seconds:
            return await self._refresh_token()
        return True

    async def _resolve_account_id(self) -> None:
        """Fetch the first trading account ID."""
        try:
            resp = await self._safe_get("/backend-api/trade/accounts")
            if resp and resp.status_code == 200:
                accounts = resp.json().get("accounts", [])
                if accounts:
                    self._token.account_id = str(accounts[0].get("id", ""))
                    logger.info(f"Resolved account ID: {self._token.account_id}")
        except Exception as e:
            logger.warning(f"Could not resolve account ID: {e}")

    # ------------------------------------------------------------------
    # Safe HTTP methods (Req 1.1, 1.2, 1.7)
    # ------------------------------------------------------------------

    async def _safe_get(self, path: str, params: Optional[Dict] = None) -> Optional[httpx.Response]:
        """
        Execute a GET request. This is the ONLY data-read verb allowed.
        All mutation requests are blocked by design.
        """
        if not self._client:
            return None

        async with self._lock:
            if not await self._ensure_token():
                return None

        headers = {"Authorization": f"Bearer {self._token.access_token}"}
        try:
            resp = await self._client.get(
                f"{self._base_url}{path}",
                headers=headers,
                params=params,
            )
            return resp
        except Exception as e:
            logger.warning(f"GET {path} failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Public data methods
    # ------------------------------------------------------------------

    @property
    def is_authenticated(self) -> bool:
        return not self._token.auth_failed and bool(self._token.access_token)

    @property
    def auth_error(self) -> str:
        return self._token.last_error

    async def get_account_state(self) -> AccountState:
        """
        Fetch account balance, equity, free margin (Req 3.1).
        Returns AccountState with error info if unavailable.
        """
        if not self.is_authenticated:
            return AccountState(error=self._token.last_error or "Not authenticated")

        account_id = self._token.account_id
        if not account_id:
            return AccountState(error="No account ID available")

        resp = await self._safe_get(
            f"/backend-api/trade/accounts/{account_id}/state"
        )

        if resp is None:
            return AccountState(error="API request failed")

        if resp.status_code != 200:
            return AccountState(error=f"HTTP {resp.status_code}")

        try:
            data = resp.json()
            # TradeLocker returns account state in various formats
            balance = self._parse_numeric(data.get("balance"))
            equity = self._parse_numeric(data.get("equity"))
            free_margin = self._parse_numeric(data.get("freeMargin") or data.get("availableBalance"))

            # Req 3.4: equity must be present, numeric, non-negative
            if equity is None or equity < 0:
                return AccountState(
                    error="Equity missing, null, non-numeric, or negative",
                    timestamp=time.time(),
                )

            return AccountState(
                balance=balance,
                equity=equity,
                free_margin=free_margin,
                timestamp=time.time(),
                source="api",
            )
        except Exception as e:
            return AccountState(error=f"Parse error: {e}")

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Fetch live quotes for each instrument (Req 15.3, 15.4).
        Issues a separate request per instrument.
        """
        results: Dict[str, Quote] = {}

        if not self.is_authenticated:
            for sym in symbols:
                results[sym] = Quote(symbol=sym, error="Not authenticated")
            return results

        account_id = self._token.account_id
        if not account_id:
            for sym in symbols:
                results[sym] = Quote(symbol=sym, error="No account ID")
            return results

        for symbol in symbols:
            quote = await self._fetch_single_quote(account_id, symbol)
            results[symbol] = quote

        return results

    async def _fetch_single_quote(self, account_id: str, symbol: str) -> Quote:
        """Fetch quote for a single instrument."""
        try:
            # First, try to get instrument info to find the route/tradable ID
            resp = await self._safe_get(
                f"/backend-api/trade/accounts/{account_id}/instruments",
                params={"name": symbol},
            )

            if resp is None or resp.status_code != 200:
                return Quote(symbol=symbol, error="Instrument lookup failed")

            data = resp.json()
            instruments = data.get("instruments", data.get("d", []))

            if not instruments:
                return Quote(symbol=symbol, error="Instrument not found")

            # Get the first matching instrument
            inst = instruments[0] if isinstance(instruments, list) else instruments
            route_id = inst.get("routeId") or inst.get("route_id", "")
            tradable_id = inst.get("tradableInstrumentId") or inst.get("id", "")

            # Fetch the actual quote
            quote_resp = await self._safe_get(
                f"/backend-api/trade/accounts/{account_id}/instruments/{tradable_id}/quote",
                params={"routeId": route_id} if route_id else None,
            )

            if quote_resp is None or quote_resp.status_code != 200:
                return Quote(symbol=symbol, error="Quote request failed")

            qdata = quote_resp.json()
            bid = self._parse_numeric(qdata.get("bid") or qdata.get("b"))
            ask = self._parse_numeric(qdata.get("ask") or qdata.get("a"))
            last_price = bid  # Use bid as last price if no explicit last

            if ask is not None and bid is not None:
                last_price = (bid + ask) / 2.0

            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last_price=last_price,
                timestamp=time.time(),
            )
        except Exception as e:
            return Quote(symbol=symbol, error=f"Exception: {e}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_numeric(value: Any) -> Optional[float]:
        """Parse a value as float, returning None if invalid."""
        if value is None:
            return None
        try:
            result = float(value)
            return result
        except (TypeError, ValueError):
            return None
