"""
Technical Indicators Calculator for XAUUSD Bot.
All indicators computed from OHLCV data (numpy arrays).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class IndicatorSnapshot:
    """All indicator values at a single point in time."""
    # EMAs
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_bias: float = 0.0

    # ADX system
    adx: float = 0.0
    di_plus: float = 0.0
    di_minus: float = 0.0

    # ATR
    atr: float = 0.0
    atr_avg: float = 0.0  # 20-period average of ATR itself

    # RSI
    rsi: float = 50.0

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_bandwidth: float = 0.0
    bb_bandwidth_avg: float = 0.0

    # Stochastic
    stoch_k: float = 50.0
    stoch_d: float = 50.0

    # Price context
    current_price: float = 0.0
    session_high: float = 0.0
    session_low: float = 0.0


class TechnicalIndicators:
    """Computes all technical indicators needed by the multi-factor system."""

    def __init__(self, config: dict):
        self.cfg = config

    # ------------------------------------------------------------------
    # EXPONENTIAL MOVING AVERAGE
    # ------------------------------------------------------------------

    def ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA for a price series."""
        if len(data) < period:
            return np.full_like(data, np.nan)

        multiplier = 2.0 / (period + 1)
        ema_values = np.zeros_like(data, dtype=float)
        ema_values[period - 1] = np.mean(data[:period])

        for i in range(period, len(data)):
            ema_values[i] = (data[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]

        ema_values[:period - 1] = np.nan
        return ema_values

    # ------------------------------------------------------------------
    # AVERAGE TRUE RANGE
    # ------------------------------------------------------------------

    def atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int) -> np.ndarray:
        """Calculate ATR (Average True Range)."""
        if len(high) < 2:
            return np.zeros_like(high)

        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        return self._wilder_smooth(tr, period)

    # ------------------------------------------------------------------
    # ADX (Average Directional Index)
    # ------------------------------------------------------------------

    def adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ADX, DI+, and DI-."""
        n = len(high)
        if n < period + 1:
            return np.zeros(n), np.zeros(n), np.zeros(n)

        # Directional movement
        dm_plus = np.zeros(n)
        dm_minus = np.zeros(n)

        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            dm_plus[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
            dm_minus[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        # True range
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )

        # Smoothed values (Wilder's smoothing)
        atr_smooth = self._wilder_smooth(tr, period)
        dm_plus_smooth = self._wilder_smooth(dm_plus, period)
        dm_minus_smooth = self._wilder_smooth(dm_minus, period)

        # DI+ and DI-
        di_plus = np.zeros(n)
        di_minus = np.zeros(n)
        dx = np.zeros(n)

        for i in range(period, n):
            if atr_smooth[i] > 0:
                di_plus[i] = 100.0 * dm_plus_smooth[i] / atr_smooth[i]
                di_minus[i] = 100.0 * dm_minus_smooth[i] / atr_smooth[i]

            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum

        # ADX = smoothed DX
        adx_values = self._wilder_smooth(dx, period)

        return adx_values, di_plus, di_minus

    # ------------------------------------------------------------------
    # RSI (Relative Strength Index)
    # ------------------------------------------------------------------

    def rsi(self, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI."""
        n = len(close)
        if n < period + 1:
            return np.full(n, 50.0)

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)

        # Initial averages
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        # Wilder smoothing
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

        rsi_values = np.full(n, 50.0)
        for i in range(period, n):
            if avg_loss[i] == 0:
                rsi_values[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

        return rsi_values

    # ------------------------------------------------------------------
    # BOLLINGER BANDS
    # ------------------------------------------------------------------

    def bollinger_bands(self, close: np.ndarray, period: int,
                        std_mult: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands. Returns (upper, middle, lower, bandwidth)."""
        n = len(close)
        upper = np.zeros(n)
        middle = np.zeros(n)
        lower = np.zeros(n)
        bandwidth = np.zeros(n)

        for i in range(period - 1, n):
            window = close[i - period + 1:i + 1]
            sma = np.mean(window)
            std = np.std(window, ddof=0)

            middle[i] = sma
            upper[i] = sma + std_mult * std
            lower[i] = sma - std_mult * std
            bandwidth[i] = (upper[i] - lower[i]) / sma if sma > 0 else 0.0

        return upper, middle, lower, bandwidth

    # ------------------------------------------------------------------
    # STOCHASTIC OSCILLATOR
    # ------------------------------------------------------------------

    def stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   k_period: int, d_period: int, smooth: int) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic %K and %D."""
        n = len(close)
        raw_k = np.zeros(n)

        for i in range(k_period - 1, n):
            highest = np.max(high[i - k_period + 1:i + 1])
            lowest = np.min(low[i - k_period + 1:i + 1])
            denom = highest - lowest
            raw_k[i] = ((close[i] - lowest) / denom * 100.0) if denom > 0 else 50.0

        # Smooth %K
        stoch_k = self._sma(raw_k, smooth)
        # %D = SMA of %K
        stoch_d = self._sma(stoch_k, d_period)

        return stoch_k, stoch_d

    # ------------------------------------------------------------------
    # COMPUTE ALL — returns snapshot
    # ------------------------------------------------------------------

    def compute_all(self, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    session_high: Optional[float] = None,
                    session_low: Optional[float] = None) -> IndicatorSnapshot:
        """Compute all indicators and return a snapshot of the latest values."""
        cfg = self.cfg

        # EMAs
        ema_fast = self.ema(close, cfg["ema_fast"])
        ema_slow = self.ema(close, cfg["ema_slow"])
        ema_bias = self.ema(close, cfg["ema_bias"])

        # ADX
        adx_vals, di_plus, di_minus = self.adx(high, low, close, cfg["adx_period"])

        # ATR
        atr_vals = self.atr(high, low, close, cfg["atr_period"])
        atr_avg = self._sma(atr_vals, 20)

        # RSI
        rsi_vals = self.rsi(close, cfg["rsi_period"])

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_bw = self.bollinger_bands(
            close, cfg["bb_period"], cfg["bb_std"]
        )
        bb_bw_avg = self._sma(bb_bw, 20)

        # Stochastic
        stoch_k, stoch_d = self.stochastic(
            high, low, close, cfg["stoch_k"], cfg["stoch_d"], cfg["stoch_smooth"]
        )

        # Build snapshot from latest values
        idx = -1  # Last bar
        return IndicatorSnapshot(
            ema_fast=ema_fast[idx] if not np.isnan(ema_fast[idx]) else close[idx],
            ema_slow=ema_slow[idx] if not np.isnan(ema_slow[idx]) else close[idx],
            ema_bias=ema_bias[idx] if not np.isnan(ema_bias[idx]) else close[idx],
            adx=adx_vals[idx],
            di_plus=di_plus[idx],
            di_minus=di_minus[idx],
            atr=atr_vals[idx],
            atr_avg=atr_avg[idx] if atr_avg[idx] > 0 else atr_vals[idx],
            rsi=rsi_vals[idx],
            bb_upper=bb_upper[idx],
            bb_middle=bb_middle[idx],
            bb_lower=bb_lower[idx],
            bb_bandwidth=bb_bw[idx],
            bb_bandwidth_avg=bb_bw_avg[idx] if bb_bw_avg[idx] > 0 else bb_bw[idx],
            stoch_k=stoch_k[idx],
            stoch_d=stoch_d[idx],
            current_price=close[idx],
            session_high=session_high if session_high else np.max(high[-50:]),
            session_low=session_low if session_low else np.min(low[-50:]),
        )

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing method (used by ATR, ADX)."""
        n = len(data)
        result = np.zeros(n)
        if n < period:
            return result

        result[period - 1] = np.mean(data[:period])
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period

        return result

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        n = len(data)
        result = np.zeros(n)
        if n < period:
            return result

        for i in range(period - 1, n):
            result[i] = np.mean(data[i - period + 1:i + 1])

        return result
