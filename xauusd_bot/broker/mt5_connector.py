"""
MetaTrader 5 Connector for XAUUSD Signal Bot.

Handles:
    - Connection to MT5 terminal
    - Sending trade orders (market execution)
    - Monitoring open positions
    - Fetching live price data
    - Account info

Requirements:
    pip install MetaTrader5

Note: MT5 Python API only works on Windows.
For Linux/Mac, use the REST API bridge or docker approach.
"""

import os
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False


@dataclass
class MT5Config:
    """MT5 connection config."""
    login: int = 0
    password: str = ""
    server: str = ""
    path: str = ""  # Path to MT5 terminal exe
    symbol: str = "XAUUSD"
    magic_number: int = 202607  # Unique ID for our bot's trades
    deviation: int = 20  # Max slippage in points


class MT5Connector:
    """
    Interface to MetaTrader 5 for live trade execution.

    Usage:
        mt5 = MT5Connector(config)
        mt5.connect()
        mt5.send_order("BUY", entry=4100.0, sl=4098.50, tp=4101.50, lot=0.1)
        mt5.disconnect()
    """

    def __init__(self, config: Optional[MT5Config] = None):
        if config is None:
            config = MT5Config(
                login=int(os.getenv("MT5_LOGIN", "0")),
                password=os.getenv("MT5_PASSWORD", ""),
                server=os.getenv("MT5_SERVER", ""),
            )
        self.config = config
        self.connected = False

    def connect(self) -> bool:
        """Initialize connection to MT5 terminal."""
        if not HAS_MT5:
            print("[MT5] MetaTrader5 package not installed.")
            print("[MT5] Install with: pip install MetaTrader5")
            print("[MT5] Note: Only works on Windows")
            return False

        if self.config.path:
            init = mt5.initialize(self.config.path)
        else:
            init = mt5.initialize()

        if not init:
            print(f"[MT5] Init failed: {mt5.last_error()}")
            return False

        # Login
        if self.config.login:
            auth = mt5.login(
                self.config.login,
                password=self.config.password,
                server=self.config.server,
            )
            if not auth:
                print(f"[MT5] Login failed: {mt5.last_error()}")
                return False

        self.connected = True
        info = mt5.account_info()
        print(f"[MT5] Connected: {info.server} | "
              f"Account: {info.login} | "
              f"Balance: ${info.balance:.2f}")
        return True

    def disconnect(self):
        """Shutdown MT5 connection."""
        if HAS_MT5:
            mt5.shutdown()
        self.connected = False

    def get_price(self) -> Optional[dict]:
        """Get current bid/ask for XAUUSD."""
        if not HAS_MT5 or not self.connected:
            return None

        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            return None

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round((tick.ask - tick.bid) / 0.01, 1),
            "time": datetime.fromtimestamp(tick.time),
        }

    def get_account_info(self) -> Optional[dict]:
        """Get account balance and equity."""
        if not HAS_MT5 or not self.connected:
            return None

        info = mt5.account_info()
        if info is None:
            return None

        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin_free": info.margin_free,
            "profit": info.profit,
            "leverage": info.leverage,
        }

    def send_order(self, direction: str, sl: float, tp: float,
                   lot: float, comment: str = "XAUUSD_BOT_V2") -> dict:
        """
        Send a market order. Set and forget — SL and TP are set immediately.

        Args:
            direction: "BUY" or "SELL"
            sl: Stop loss price
            tp: Take profit price
            lot: Lot size
            comment: Order comment

        Returns:
            dict with order result
        """
        if not HAS_MT5 or not self.connected:
            return {"success": False, "error": "Not connected to MT5"}

        symbol = self.config.symbol
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"success": False, "error": "Cannot get price"}

        # Determine order type and price
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif direction == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            return {"success": False, "error": f"Invalid direction: {direction}"}

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt5.order_send(request)

        if result is None:
            return {"success": False, "error": str(mt5.last_error())}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "success": False,
                "error": f"Order failed: {result.comment}",
                "retcode": result.retcode,
            }

        return {
            "success": True,
            "order_id": result.order,
            "price": result.price,
            "volume": result.volume,
            "comment": result.comment,
        }

    def close_position(self, ticket: int) -> dict:
        """Close a specific position by ticket number."""
        if not HAS_MT5 or not self.connected:
            return {"success": False, "error": "Not connected"}

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": f"Position {ticket} not found"}

        pos = position[0]
        symbol = pos.symbol
        tick = mt5.symbol_info_tick(symbol)

        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.config.deviation,
            "magic": self.config.magic_number,
            "comment": "CLOSE_BOT_V2",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"success": True, "price": result.price}
        return {"success": False, "error": str(mt5.last_error())}

    def get_open_positions(self) -> list[dict]:
        """Get all open positions for our magic number."""
        if not HAS_MT5 or not self.connected:
            return []

        positions = mt5.positions_get(
            symbol=self.config.symbol,
        )

        if positions is None:
            return []

        return [
            {
                "ticket": p.ticket,
                "direction": "BUY" if p.type == 0 else "SELL",
                "volume": p.volume,
                "price_open": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "time": datetime.fromtimestamp(p.time),
                "magic": p.magic,
            }
            for p in positions
            if p.magic == self.config.magic_number
        ]

    def get_bars(self, timeframe: str = "M15",
                 count: int = 200) -> Optional[list]:
        """Fetch recent OHLC bars."""
        if not HAS_MT5 or not self.connected:
            return None

        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        tf = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
        rates = mt5.copy_rates_from_pos(self.config.symbol, tf, 0, count)

        if rates is None:
            return None

        return [
            {
                "time": datetime.fromtimestamp(r[0]),
                "open": r[1],
                "high": r[2],
                "low": r[3],
                "close": r[4],
                "volume": r[5],
            }
            for r in rates
        ]
