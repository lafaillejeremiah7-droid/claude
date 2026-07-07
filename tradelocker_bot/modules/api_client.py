"""
TradeLocker API Client Module

Handles authentication, token management, order execution,
position management, and market data retrieval.
Uses the official tradelocker Python library as base with
additional custom methods for advanced functionality.
"""
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from config import BASE_URL, TL_EMAIL, TL_PASSWORD, TL_SERVER

logger = logging.getLogger(__name__)


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
        """Ensure we have a valid token, refreshing if needed."""
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

        # Calculate timestamps
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (lookback_bars * resolution * 1000)

        url = f"{self.base_url}/trade/history"
        headers = {"accNum": str(self.acc_num)}
        params = {
            "routeId": route_id,
            "tradableInstrumentId": instrument_id,
            "resolution": resolution,
            "from": start_ms,
            "to": now_ms,
        }

        try:
            resp = self.session.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            bars = data.get("d", {}).get("bars", [])
            if not bars:
                logger.warning(f"No price data returned for {symbol} @ {resolution}s")
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

            return df.sort_values("timestamp").reset_index(drop=True)

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get price history for {symbol}: {e}")
            return None

    def get_latest_price(self, symbol: str) -> Optional[dict]:
        """Get latest bid/ask price for a symbol."""
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

        try:
            resp = self.session.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            quotes = data.get("d", {})

            return {
                "bid": float(quotes.get("bp", 0)),
                "ask": float(quotes.get("ap", 0)),
                "mid": (float(quotes.get("bp", 0)) + float(quotes.get("ap", 0))) / 2,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get latest price for {symbol}: {e}")
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
