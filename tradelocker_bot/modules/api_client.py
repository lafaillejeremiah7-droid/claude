"""
TradeLocker API Client Module

Handles authentication, token management, order execution,
position management, and market data retrieval.
Uses the official tradelocker Python library as base with
additional custom methods for advanced functionality.
"""
import time
import random
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from config import (
    BASE_URL,
    TL_EMAIL,
    TL_PASSWORD,
    TL_SERVER,
    API_MIN_REQUEST_INTERVAL,
    API_MAX_RETRIES,
    API_BACKOFF_BASE,
    API_BACKOFF_MAX,
    HISTORY_CACHE_TTL,
    HISTORY_CACHE_TTL_DEFAULT,
)

logger = logging.getLogger(__name__)


# Map integer-seconds resolutions to the TradeLocker /trade/history API's
# required STRING resolution codes. The history endpoint returns an HTTP 200
# error envelope ({"s":"error","errmsg":...}) if it receives integer seconds
# instead of one of these codes. The three the bot actually uses are
# 300->"5m", 1800->"30m", 14400->"4H".
RESOLUTION_CODES = {
    60: "1m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1H",
    14400: "4H",
    86400: "1D",
    604800: "1W",
}


def resolution_to_code(resolution: int) -> Optional[str]:
    """
    Convert an integer-seconds resolution to the TradeLocker API's string code.

    Returns the exact mapped code when known (e.g. 300 -> "5m"). For unknown
    values a clear warning is logged and a best-effort conversion is attempted:
    exact hour multiples become "<n>H" and sub-hour values become "<n>m".
    Returns None if no reasonable conversion is possible.
    """
    try:
        seconds = int(resolution)
    except (ValueError, TypeError):
        logger.warning(f"Invalid resolution {resolution!r}; cannot convert to API code")
        return None

    code = RESOLUTION_CODES.get(seconds)
    if code is not None:
        return code

    logger.warning(
        f"Resolution {seconds}s not in RESOLUTION_CODES; attempting best-effort "
        f"conversion to an API resolution code"
    )
    if seconds <= 0:
        return None
    if seconds % 3600 == 0:
        return f"{seconds // 3600}H"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    logger.warning(
        f"Could not convert resolution {seconds}s to a valid API code"
    )
    return None


class TradeLockerClient:
    """
    TradeLocker REST API Client.
    Handles JWT authentication, token refresh, and all trading operations.
    """

    def __init__(self):
        self.base_url = BASE_URL
        self.email = TL_EMAIL
        self.password = TL_PASSWORD
        self.server = TL_SERVER

        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: float = 0
        self.refresh_expiry: float = 0

        self.account_id: Optional[int] = None
        self.acc_num: Optional[int] = None

        # Instrument cache: symbol -> {instrument_id, route_id_trade, route_id_info, ...}
        self.instruments_cache: dict = {}

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # --- Rate-limiting / resilience state ---
        # Timestamp (monotonic) of the last outbound TradeLocker HTTP request,
        # used by _throttle() to enforce a minimum spacing between calls.
        self._last_request_ts: float = 0.0
        self._min_request_interval: float = API_MIN_REQUEST_INTERVAL
        self._max_retries: int = API_MAX_RETRIES
        self._backoff_base: float = API_BACKOFF_BASE
        self._backoff_max: float = API_BACKOFF_MAX
        # Short-lived price-history bar cache:
        #   key   -> (tradableInstrumentId, resolution)
        #   value -> {"df": <DataFrame copy>, "expires": <monotonic ts>}
        self._history_cache: dict = {}

    # ========================================
    # RATE LIMITING / RESILIENCE HELPERS
    # ========================================

    def _throttle(self) -> None:
        """
        Sleep just enough to keep at least ``_min_request_interval`` seconds
        between consecutive outbound TradeLocker HTTP requests. This protects
        the DEMO server's aggressive per-second rate limit (HTTP 429).
        """
        interval = self._min_request_interval
        if interval <= 0:
            self._last_request_ts = time.monotonic()
            return

        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if 0 <= elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_ts = time.monotonic()

    def _history_ttl(self, resolution: int) -> float:
        """Return the cache TTL (seconds) for a given resolution."""
        return HISTORY_CACHE_TTL.get(int(resolution), HISTORY_CACHE_TTL_DEFAULT)

    def _compute_backoff(self, attempt: int, retry_after: Optional[str]) -> float:
        """
        Compute how long to sleep before the next retry.

        Honors the ``Retry-After`` response header when present (seconds form),
        otherwise uses exponential backoff: base * 2**attempt, capped at
        ``_backoff_max``, plus a small random jitter to avoid thundering herds.
        ``attempt`` is 0-based (0 for the first retry).
        """
        if retry_after:
            try:
                wait = float(retry_after)
                if wait >= 0:
                    return min(wait, self._backoff_max)
            except (ValueError, TypeError):
                pass  # Non-numeric (HTTP-date) Retry-After: fall back to backoff.

        wait = self._backoff_base * (2 ** attempt)
        wait = min(wait, self._backoff_max)
        jitter = random.uniform(0, self._backoff_base * 0.25)
        return wait + jitter

    def _request_with_retry(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Perform a throttled HTTP request with exponential-backoff retries on
        transient failures (HTTP 429 and 5xx).

        Retries up to ``_max_retries`` times, honoring ``Retry-After`` on 429.
        On persistent failure (or a connection error) returns the last response
        if one exists, otherwise ``None``. Non-transient HTTP responses (e.g.
        200, 4xx other than 429) are returned immediately for the caller to
        handle.
        """
        last_resp: Optional[requests.Response] = None
        total_attempts = self._max_retries + 1

        for attempt in range(total_attempts):
            self._throttle()
            try:
                resp = self.session.request(method, url, **kwargs)
            except requests.exceptions.RequestException as e:
                # Network-level error: back off and retry if attempts remain.
                if attempt < total_attempts - 1:
                    wait = self._compute_backoff(attempt, None)
                    logger.warning(
                        f"Request error ({e.__class__.__name__}) on {url}; "
                        f"retry {attempt + 1}/{self._max_retries} in {wait:.2f}s"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Request failed after retries: {e}")
                return None

            last_resp = resp

            # Retry on 429 (rate limited) and 5xx (server errors).
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < total_attempts - 1:
                    retry_after = resp.headers.get("Retry-After")
                    wait = self._compute_backoff(attempt, retry_after)
                    logger.warning(
                        f"HTTP {resp.status_code} on {url}; backing off {wait:.2f}s "
                        f"(retry {attempt + 1}/{self._max_retries}"
                        + (f", Retry-After={retry_after}" if retry_after else "")
                        + ")"
                    )
                    time.sleep(wait)
                    continue
                logger.warning(
                    f"HTTP {resp.status_code} on {url}; exhausted "
                    f"{self._max_retries} retries"
                )
                return resp

            # Success or a non-transient status: hand back to caller.
            return resp

        return last_resp

    # ========================================
    # AUTHENTICATION
    # ========================================

    def authenticate(self) -> bool:
        """Authenticate with TradeLocker and obtain JWT tokens."""
        url = f"{self.base_url}/auth/jwt/token"
        payload = {
            "email": self.email,
            "password": self.password,
            "server": self.server,
        }

        try:
            self._throttle()
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            self.access_token = data["accessToken"]
            self.refresh_token = data["refreshToken"]
            self.token_expiry = time.time() + data.get("expireInMs", 300000) / 1000
            self.refresh_expiry = time.time() + data.get("refreshExpireInMs", 3600000) / 1000

            self.session.headers.update(
                {"Authorization": f"Bearer {self.access_token}"}
            )

            logger.info("Successfully authenticated with TradeLocker")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def refresh_access_token(self) -> bool:
        """Refresh the access token using the refresh token."""
        if not self.refresh_token:
            return self.authenticate()

        url = f"{self.base_url}/auth/jwt/refresh"
        payload = {"refreshToken": self.refresh_token}

        try:
            self._throttle()
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            self.access_token = data["accessToken"]
            self.refresh_token = data["refreshToken"]
            self.token_expiry = time.time() + data.get("expireInMs", 300000) / 1000
            self.refresh_expiry = time.time() + data.get("refreshExpireInMs", 3600000) / 1000

            self.session.headers.update(
                {"Authorization": f"Bearer {self.access_token}"}
            )

            logger.info("Token refreshed successfully")
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Token refresh failed, re-authenticating: {e}")
            return self.authenticate()

    def ensure_authenticated(self) -> bool:
        """Ensure we have a valid token, refreshing if needed.

        NOTE: Multiple modules call ensure_authenticated() defensively before
        each API call (e.g. get_instrument_id -> get_instruments ->
        ensure_authenticated). This is intentional for safety — the fast path
        (token still valid) returns immediately with negligible overhead, and
        it guarantees no method ever makes an unauthenticated request even if
        called in isolation or a different order.
        """
        if not self.access_token:
            return self.authenticate()

        if time.time() >= self.token_expiry - 30:  # 30 sec buffer
            return self.refresh_access_token()

        return True

    # ========================================
    # ACCOUNT SETUP
    # ========================================

    def get_accounts(self) -> list:
        """Retrieve all accounts and set the first one as active."""
        if not self.ensure_authenticated():
            return []

        url = f"{self.base_url}/auth/jwt/all-accounts"
        try:
            resp = self.session.get(url)
            resp.raise_for_status()
            data = resp.json()
            accounts = data.get("accounts", [])

            if accounts:
                # Use first account by default
                self.account_id = accounts[0]["id"]
                self.acc_num = accounts[0]["accNum"]
                logger.info(
                    f"Active account: ID={self.account_id}, accNum={self.acc_num}"
                )

            return accounts

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get accounts: {e}")
            return []

    def setup(self) -> bool:
        """Full setup: authenticate, get accounts, load instruments."""
        if not self.authenticate():
            return False
        if not self.get_accounts():
            return False
        return True

    # ========================================
    # INSTRUMENTS
    # ========================================

    def get_instruments(self) -> dict:
        """Fetch all tradable instruments for the account."""
        if not self.ensure_authenticated():
            return {}

        url = f"{self.base_url}/trade/accounts/{self.account_id}/instruments"
        headers = {"accNum": str(self.acc_num)}

        try:
            resp = self.session.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            instruments = data.get("d", {}).get("instruments", [])
            for inst in instruments:
                symbol = inst.get("name", "")
                self.instruments_cache[symbol] = {
                    "tradableInstrumentId": inst.get("tradableInstrumentId"),
                    "instrumentId": inst.get("instrumentId"),
                    "routes": inst.get("routes", {}),
                    "lotSize": inst.get("lotSize", 1),
                    "lotStep": inst.get("lotStep", 0.01),
                    "minLot": inst.get("minLotSize", 0.01),
                    "maxLot": inst.get("maxLotSize", 100),
                    "pipSize": inst.get("pipSize", 0.01),
                }

            logger.info(f"Loaded {len(self.instruments_cache)} instruments")
            return self.instruments_cache

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get instruments: {e}")
            return {}

    def get_instrument_id(self, symbol: str) -> Optional[int]:
        """Get the tradable instrument ID for a symbol."""
        if not self.instruments_cache:
            self.get_instruments()

        inst = self.instruments_cache.get(symbol)
        if inst:
            return inst["tradableInstrumentId"]

        # Try partial match
        for name, data in self.instruments_cache.items():
            if symbol.upper() in name.upper():
                return data["tradableInstrumentId"]

        logger.warning(f"Instrument not found: {symbol}")
        return None

    def get_route_id(self, symbol: str, route_type: str = "TRADE") -> Optional[int]:
        """Get route ID for a symbol. route_type: 'TRADE' or 'INFO'"""
        if not self.instruments_cache:
            self.get_instruments()

        inst = self.instruments_cache.get(symbol)
        if not inst:
            # Try partial match
            for name, data in self.instruments_cache.items():
                if symbol.upper() in name.upper():
                    inst = data
                    break

        if inst and inst.get("routes"):
            routes = inst["routes"]
            if isinstance(routes, dict):
                return routes.get(route_type)
            elif isinstance(routes, list):
                for route in routes:
                    if route.get("type") == route_type:
                        return route.get("id")

        return None

    # ========================================
    # MARKET DATA
    # ========================================

    def get_price_history(
        self,
        symbol: str,
        resolution: int,
        lookback_bars: int = 250,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data (OHLCV).

        Args:
            symbol: Instrument symbol (e.g., 'BTCUSD')
            resolution: Timeframe in seconds (300=5m, 1800=30m, 14400=4h)
            lookback_bars: Number of bars to fetch

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not self.ensure_authenticated():
            return None

        instrument_id = self.get_instrument_id(symbol)
        route_id = self.get_route_id(symbol, "INFO")

        if not instrument_id or not route_id:
            logger.error(f"Cannot get price history: instrument/route not found for {symbol}")
            return None

        # --- Short-lived bar cache -------------------------------------------
        # Keyed by (tradableInstrumentId, resolution). Return a copy of cached
        # bars while fresh so the bot doesn't refetch identical bars every 60s.
        cache_key = (instrument_id, int(resolution))
        cached = self._history_cache.get(cache_key)
        if cached is not None and time.monotonic() < cached["expires"]:
            logger.debug(
                f"Price history cache HIT for {symbol} @ {resolution}s "
                f"(age within {self._history_ttl(resolution):.0f}s TTL)"
            )
            return cached["df"].copy()

        # Calculate timestamps using the integer SECONDS resolution (unchanged).
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (lookback_bars * resolution * 1000)

        # The API requires a STRING resolution code (e.g. "5m"), NOT integer
        # seconds. Convert here; bail out if we can't produce a valid code.
        resolution_code = resolution_to_code(resolution)
        if resolution_code is None:
            logger.error(
                f"Cannot get price history for {symbol}: unsupported resolution "
                f"{resolution!r} (no valid API resolution code)"
            )
            return None

        url = f"{self.base_url}/trade/history"
        headers = {"accNum": str(self.acc_num)}
        params = {
            "routeId": route_id,
            "tradableInstrumentId": instrument_id,
            "resolution": resolution_code,
            "from": start_ms,
            "to": now_ms,
        }

        resp = self._request_with_retry("GET", url, headers=headers, params=params)
        if resp is None:
            logger.warning(
                f"Failed to get price history for {symbol} @ {resolution}s "
                f"(no response after retries)"
            )
            return None

        if resp.status_code != 200:
            logger.warning(
                f"Failed to get price history for {symbol} @ {resolution}s: "
                f"HTTP {resp.status_code}"
            )
            return None

        try:
            data = resp.json()
        except ValueError as e:
            logger.warning(
                f"Price history for {symbol} @ {resolution}s: non-JSON response "
                f"(HTTP {resp.status_code}): {e}"
            )
            return None

        # Detect the TradeLocker JSON error envelope. The history endpoint can
        # return HTTP 200 with {"s":"error","errmsg":...} (e.g. a bad resolution
        # param). Surface errmsg and return None so API errors are obvious and
        # are NOT cached.
        if isinstance(data, dict) and data.get("s") == "error":
            errmsg = data.get("errmsg", "<no errmsg>")
            logger.warning(
                f"Price history API error for {symbol} @ {resolution}s "
                f"(resolution={resolution_code!r}): {errmsg}"
            )
            return None

        bars = (data.get("d", {}).get("bars") or data.get("d", {}).get("barDetails") or []) if isinstance(data, dict) else []
        if not bars:
            # Better diagnostics: distinguish a genuinely empty payload from a
            # masked throttle. Log status + top-level JSON keys + a short,
            # secret-free snippet of the body.
            top_keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            inner_keys = (
                list(data.get("d", {}).keys())
                if isinstance(data, dict) and isinstance(data.get("d"), dict)
                else None
            )
            snippet = resp.text[:200].replace("\n", " ") if resp.text else ""
            logger.warning(
                f"No price data returned for {symbol} @ {resolution}s | "
                f"HTTP {resp.status_code} | top_keys={top_keys} | "
                f"d_keys={inner_keys} | body[:200]={snippet!r}"
            )
            return None

        df = pd.DataFrame(bars)
        # TradeLocker returns: t(timestamp), o(open), h(high), l(low), c(close), v(volume)
        df.rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"},
            inplace=True,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Store a copy in the cache with a resolution-appropriate TTL.
        self._history_cache[cache_key] = {
            "df": df.copy(),
            "expires": time.monotonic() + self._history_ttl(resolution),
        }

        # Evict oldest entries if cache exceeds 50 items (simple LRU-lite).
        if len(self._history_cache) > 50:
            oldest_key = min(
                self._history_cache, key=lambda k: self._history_cache[k]["expires"]
            )
            del self._history_cache[oldest_key]

        return df

    def get_latest_price(self, symbol: str) -> Optional[dict]:
        """Get latest bid/ask price for a symbol.

        Routed through _request_with_retry for resilience against transient
        network errors and 429/5xx responses (read-only, safe to retry).
        """
        if not self.ensure_authenticated():
            return None

        instrument_id = self.get_instrument_id(symbol)
        route_id = self.get_route_id(symbol, "INFO")

        if not instrument_id or not route_id:
            return None

        url = f"{self.base_url}/trade/quotes"
        headers = {"accNum": str(self.acc_num)}
        params = {
            "routeId": route_id,
            "tradableInstrumentId": instrument_id,
        }

        resp = self._request_with_retry("GET", url, headers=headers, params=params)
        if resp is None or resp.status_code != 200:
            logger.error(f"Failed to get latest price for {symbol}")
            return None

        try:
            data = resp.json()
            quotes = data.get("d", {})

            return {
                "bid": float(quotes.get("bp", 0)),
                "ask": float(quotes.get("ap", 0)),
                "mid": (float(quotes.get("bp", 0)) + float(quotes.get("ap", 0))) / 2,
            }
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to parse latest price for {symbol}: {e}")
            return None

    # ========================================
    # ACCOUNT INFO
    # ========================================

    def get_account_balance(self) -> Optional[dict]:
        """Get account balance, equity, and margin info."""
        if not self.ensure_authenticated():
            return None

        headers = {"accNum": str(self.acc_num)}

        # Try multiple endpoint patterns (varies by API version)
        endpoints = [
            f"{self.base_url}/trade/accounts/{self.account_id}/accountDetails",
            f"{self.base_url}/trade/accounts/{self.account_id}/details",
            f"{self.base_url}/trade/accountDetails",
            f"{self.base_url}/trade/accounts/{self.account_id}/state",
        ]

        # PRIMARY: Use /state endpoint with accountDetailsData array
        # Confirmed working for this account
        try:
            url = f"{self.base_url}/trade/accounts/{self.account_id}/state"
            resp = self.session.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                raw_d = data.get("d", {})
                arr = raw_d.get("accountDetailsData", [])
                if arr and len(arr) > 4:
                    result = {
                        "balance": float(arr[0]),
                        "equity": float(arr[1]),
                        "freeMargin": float(arr[4]),
                        "marginLevel": 0,
                        "unrealizedPnL": 0,
                    }
                    logger.info(f"Account balance: ${result['equity']:.2f} equity")
                    return result
        except Exception as e:
            logger.debug(f"State endpoint error: {e}")

        # SECONDARY: Try other endpoints as fallback
        for url in endpoints:
            try:
                resp = self.session.get(url, headers=headers)
                if resp.status_code in (404, 405):
                    continue
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("d", data)

                if isinstance(raw, dict):
                    balance = float(raw.get("balance", 0) or raw.get("accountBalance", 0) or 0)
                    equity = float(raw.get("equity", 0) or raw.get("accountEquity", 0) or balance)
                    if balance > 0 or equity > 0:
                        return {
                            "balance": balance,
                            "equity": equity if equity > 0 else balance,
                            "freeMargin": float(raw.get("freeMargin", 0) or 0),
                            "marginLevel": 0,
                            "unrealizedPnL": 0,
                        }

                elif isinstance(raw, list) and len(raw) > 0:
                    try:
                        balance = float(raw[0])
                        equity = float(raw[1]) if len(raw) > 1 else balance
                        if balance > 0 or equity > 0:
                            return {
                                "balance": balance,
                                "equity": equity,
                                "freeMargin": float(raw[2]) if len(raw) > 2 else 0,
                                "marginLevel": 0,
                                "unrealizedPnL": 0,
                            }
                    except (ValueError, TypeError, IndexError):
                        pass

            except Exception:
                continue

        # LAST RESORT: Use ACCOUNT_BALANCE from .env
        import os
        fallback_equity = float(os.getenv("ACCOUNT_BALANCE", "10109.58"))
        logger.warning(f"Using fallback balance: ${fallback_equity:.2f}")
        return {
            "balance": fallback_equity,
            "equity": fallback_equity,
            "freeMargin": fallback_equity,
            "marginLevel": 0,
            "unrealizedPnL": 0,
            "_source": "fallback",  # Flag for circuit breaker in main.py
        }

    # ========================================
    # ORDER MANAGEMENT
    # ========================================

    def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss_type: str = "absolute",
        take_profit_type: str = "absolute",
    ) -> Optional[str]:
        """
        Place a trading order.

        NOTE: This method intentionally does NOT retry on failure. Retrying
        order placement on network errors risks duplicate fills (the original
        order may have been accepted by the exchange despite the client
        receiving a timeout/disconnect). Only network-level connection errors
        are caught; we do NOT retry on 429 to avoid double orders. If the
        request fails, we log clearly and return None so the caller knows no
        order was confirmed.

        Args:
            symbol: Instrument symbol
            side: 'buy' or 'sell'
            quantity: Lot size
            order_type: 'market', 'limit', or 'stop'
            price: Limit/stop price (required for limit/stop orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            stop_loss_type: 'absolute' or 'pips'
            take_profit_type: 'absolute' or 'pips'

        Returns:
            Order ID if successful, None otherwise
        """
        if not self.ensure_authenticated():
            return None

        instrument_id = self.get_instrument_id(symbol)
        route_id = self.get_route_id(symbol, "TRADE")

        if not instrument_id or not route_id:
            logger.error(f"Cannot create order: instrument/route not found for {symbol}")
            return None

        url = f"{self.base_url}/trade/accounts/{self.account_id}/orders"
        headers = {"accNum": str(self.acc_num)}

        payload = {
            "tradableInstrumentId": instrument_id,
            "routeId": route_id,
            "side": side.lower(),
            "type": order_type.lower(),
            "qty": quantity,
        }

        if price and order_type.lower() != "market":
            payload["price"] = price

        if stop_loss:
            payload["stopLoss"] = stop_loss
            payload["stopLossType"] = stop_loss_type

        if take_profit:
            payload["takeProfit"] = take_profit
            payload["takeProfitType"] = take_profit_type

        try:
            self._throttle()
            resp = self.session.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            order_id = data.get("d", {}).get("orderId")
            if order_id:
                logger.info(
                    f"Order placed: {side.upper()} {quantity} {symbol} | "
                    f"SL={stop_loss} TP={take_profit} | ID={order_id}"
                )
                return str(order_id)
            else:
                logger.error(f"Order response missing orderId: {data}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create order for {symbol}: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None

    def modify_order(
        self,
        order_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Modify an existing order's SL/TP."""
        if not self.ensure_authenticated():
            return False

        url = f"{self.base_url}/trade/accounts/{self.account_id}/orders/{order_id}"
        headers = {"accNum": str(self.acc_num)}
        payload = {}

        if stop_loss is not None:
            payload["stopLoss"] = stop_loss
        if take_profit is not None:
            payload["takeProfit"] = take_profit

        try:
            resp = self.session.patch(url, headers=headers, json=payload)
            resp.raise_for_status()
            logger.info(f"Order {order_id} modified: SL={stop_loss}, TP={take_profit}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return False

    def close_position(self, position_id: str, quantity: Optional[float] = None) -> bool:
        """
        Close a position (fully or partially).

        Args:
            position_id: The position ID to close
            quantity: If provided, close partial quantity. None = close all.
        """
        if not self.ensure_authenticated():
            return False

        url = f"{self.base_url}/trade/accounts/{self.account_id}/positions/{position_id}"
        headers = {"accNum": str(self.acc_num)}
        payload = {}

        if quantity:
            payload["qty"] = quantity

        try:
            resp = self.session.delete(url, headers=headers, json=payload)
            resp.raise_for_status()
            logger.info(f"Position {position_id} closed (qty={quantity})")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to close position {position_id}: {e}")
            return False

    def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Modify a position's SL/TP (e.g., move to breakeven)."""
        if not self.ensure_authenticated():
            return False

        url = f"{self.base_url}/trade/accounts/{self.account_id}/positions/{position_id}"
        headers = {"accNum": str(self.acc_num)}
        payload = {}

        if stop_loss is not None:
            payload["stopLoss"] = stop_loss
        if take_profit is not None:
            payload["takeProfit"] = take_profit

        try:
            resp = self.session.patch(url, headers=headers, json=payload)
            resp.raise_for_status()
            logger.info(f"Position {position_id} modified: SL={stop_loss}, TP={take_profit}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to modify position {position_id}: {e}")
            return False

    # ========================================
    # POSITIONS & ORDERS QUERIES
    # ========================================

    def get_open_positions(self) -> list:
        """Get all open positions."""
        if not self.ensure_authenticated():
            return []

        url = f"{self.base_url}/trade/accounts/{self.account_id}/positions"
        headers = {"accNum": str(self.acc_num)}

        try:
            resp = self.session.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            positions = data.get("d", {}).get("positions", [])
            return positions

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get open positions: {e}")
            return []

    def get_open_orders(self) -> list:
        """Get all pending orders."""
        if not self.ensure_authenticated():
            return []

        url = f"{self.base_url}/trade/accounts/{self.account_id}/orders"
        headers = {"accNum": str(self.acc_num)}

        try:
            resp = self.session.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            orders = data.get("d", {}).get("orders", [])
            return orders

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_orders_history(self) -> list:
        """Get order history for today."""
        if not self.ensure_authenticated():
            return []

        url = f"{self.base_url}/trade/accounts/{self.account_id}/ordersHistory"
        headers = {"accNum": str(self.acc_num)}

        try:
            resp = self.session.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data.get("d", {}).get("ordersHistory", [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get order history: {e}")
            return []
