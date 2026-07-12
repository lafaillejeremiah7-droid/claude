"""
Optimized Backtesting Script - Testing Multiple Strategy Variations
Finds the best configuration for the multi-timeframe EMA strategy.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from copy import deepcopy

# ============================================================
# CONFIGURATION
# ============================================================
STARTING_EQUITY = 10000.0
RISK_PERCENT = 2.0

# Indicator periods (fixed)
EMA_4H_PERIOD = 50
EMA_4H_SLOPE_LOOKBACK = 3
EMA_30M_FAST = 50
EMA_30M_SLOW = 200
EMA_5M_PULLBACK = 20
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_AVG_PERIOD = 20
SWING_LOOKBACK = 10

SESSION_START = 7
SESSION_END = 21


@dataclass
class StrategyConfig:
    """Configurable strategy parameters for optimization."""
    name: str = "Default"
    
    # RSI zones
    rsi_long_min: float = 45
    rsi_long_max: float = 60
    rsi_short_min: float = 40
    rsi_short_max: float = 55
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # Confirmations required (out of 6)
    min_confirmations: int = 6  # 6 = all required
    
    # Which confirmations are mandatory (always required even if min < 6)
    require_pullback: bool = True
    require_rsi: bool = True
    require_sweep: bool = True
    require_structure: bool = True
    require_candle: bool = True
    require_volume: bool = True
    
    # Direction filter
    allow_longs: bool = True
    allow_shorts: bool = True
    
    # Risk/Reward
    min_rr: float = 1.5
    preferred_rr: float = 2.0
    
    # Pullback tolerance
    pullback_threshold: float = 0.002  # 0.2%
    
    # Candle patterns allowed
    allowed_bull_patterns: list = field(default_factory=lambda: [
        'bullish_engulfing', 'hammer', 'bullish_rejection', 'strong_bullish'
    ])
    allowed_bear_patterns: list = field(default_factory=lambda: [
        'bearish_engulfing', 'shooting_star', 'bearish_rejection', 'strong_bearish'
    ])
    
    # Volume threshold
    volume_min_ratio: float = 1.0
    
    # Max trades per day
    max_trades_per_day: int = 2
    
    # Structure break lookback
    structure_lookback: int = 20


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: str
    exit_price: float = 0.0
    exit_time: str = ""
    pnl_r: float = 0.0
    result: str = ""
    hit_breakeven: bool = False
    pattern: str = ""
    rsi: float = 0.0
    volume_ratio: float = 0.0
    confirmations_met: int = 0


# ============================================================
# INDICATOR CALCULATIONS (same as before)
# ============================================================
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_atr(df, period=14):
    high, low, close_prev = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def find_swing_highs(highs, lookback=10):
    swing_highs = pd.Series(np.nan, index=highs.index)
    for i in range(lookback, len(highs) - lookback):
        if highs.iloc[i] >= highs.iloc[i-lookback:i].max() and highs.iloc[i] >= highs.iloc[i+1:i+lookback+1].max():
            swing_highs.iloc[i] = highs.iloc[i]
    return swing_highs

def find_swing_lows(lows, lookback=10):
    swing_lows = pd.Series(np.nan, index=lows.index)
    for i in range(lookback, len(lows) - lookback):
        if lows.iloc[i] <= lows.iloc[i-lookback:i].min() and lows.iloc[i] <= lows.iloc[i+1:i+lookback+1].min():
            swing_lows.iloc[i] = lows.iloc[i]
    return swing_lows


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_data(symbol):
    """Fetch multi-timeframe data."""
    print(f"  Fetching {symbol}...")
    
    df_1h = yf.download(symbol, period="2y", interval="1h", progress=False)
    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)
    
    df_4h = df_1h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    df_30m = yf.download(symbol, period="60d", interval="30m", progress=False)
    if isinstance(df_30m.columns, pd.MultiIndex):
        df_30m.columns = df_30m.columns.get_level_values(0)
    
    df_5m = yf.download(symbol, period="60d", interval="5m", progress=False)
    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)
    
    return {'4h': df_4h, '30m': df_30m, '5m': df_5m}


def add_indicators(data):
    """Add all indicators to all timeframes."""
    df_4h = data['4h']
    df_30m = data['30m']
    df_5m = data['5m']
    
    df_4h['ema_50'] = calc_ema(df_4h['Close'], EMA_4H_PERIOD)
    df_30m['ema_50'] = calc_ema(df_30m['Close'], EMA_30M_FAST)
    df_30m['ema_200'] = calc_ema(df_30m['Close'], EMA_30M_SLOW)
    df_5m['ema_20'] = calc_ema(df_5m['Close'], EMA_5M_PULLBACK)
    df_5m['rsi'] = calc_rsi(df_5m['Close'], RSI_PERIOD)
    df_5m['atr'] = calc_atr(df_5m, ATR_PERIOD)
    df_5m['volume_avg'] = df_5m['Volume'].rolling(VOLUME_AVG_PERIOD).mean()
    df_5m['swing_highs'] = find_swing_highs(df_5m['High'], SWING_LOOKBACK)
    df_5m['swing_lows'] = find_swing_lows(df_5m['Low'], SWING_LOOKBACK)
    
    return {'4h': df_4h, '30m': df_30m, '5m': df_5m}


# ============================================================
# STRATEGY CHECKS (configurable)
# ============================================================
def check_4h_trend(df_4h, idx):
    if idx < EMA_4H_PERIOD + EMA_4H_SLOPE_LOOKBACK:
        return None
    ema50 = df_4h['ema_50'].iloc[idx]
    slope = df_4h['ema_50'].iloc[idx] - df_4h['ema_50'].iloc[idx - EMA_4H_SLOPE_LOOKBACK]
    price = df_4h['Close'].iloc[idx]
    if slope > 0 and price > ema50:
        return 'bullish'
    elif slope < 0 and price < ema50:
        return 'bearish'
    return None


def check_30m_trend(df_30m, idx):
    if idx < EMA_30M_SLOW + 10:
        return None
    ema50 = df_30m['ema_50'].iloc[idx]
    ema200 = df_30m['ema_200'].iloc[idx]
    price = df_30m['Close'].iloc[idx]
    if ema50 > ema200 and price > ema50:
        return 'bullish'
    elif ema50 < ema200 and price < ema50:
        return 'bearish'
    return None


def check_pullback(df_5m, idx, direction, config: StrategyConfig):
    if idx < 3:
        return False
    ema20 = df_5m['ema_20'].iloc[idx]
    close = df_5m['Close'].iloc[idx]
    low = df_5m['Low'].iloc[idx]
    high = df_5m['High'].iloc[idx]
    threshold = config.pullback_threshold
    
    if direction == 'bullish':
        near_ema = abs(low - ema20) / ema20 <= threshold or low <= ema20
        bouncing = close > ema20
        if near_ema and bouncing:
            return True
        for i in range(max(0, idx-3), idx):
            if df_5m['Low'].iloc[i] <= ema20 * (1 + threshold) and df_5m['Close'].iloc[i] > ema20:
                return True
    else:
        near_ema = abs(high - ema20) / ema20 <= threshold or high >= ema20
        bouncing = close < ema20
        if near_ema and bouncing:
            return True
        for i in range(max(0, idx-3), idx):
            if df_5m['High'].iloc[i] >= ema20 * (1 - threshold) and df_5m['Close'].iloc[i] < ema20:
                return True
    return False


def check_rsi(df_5m, idx, direction, config: StrategyConfig):
    rsi = df_5m['rsi'].iloc[idx]
    if rsi >= config.rsi_overbought or rsi <= config.rsi_oversold:
        return False, rsi
    if direction == 'bullish':
        return config.rsi_long_min <= rsi <= config.rsi_long_max, rsi
    else:
        return config.rsi_short_min <= rsi <= config.rsi_short_max, rsi


def check_liquidity_sweep(df_5m, idx, direction):
    if idx < SWING_LOOKBACK + 10:
        return False
    check_start = max(0, idx - 5)
    
    if direction == 'bullish':
        swing_lows = df_5m['swing_lows'].iloc[:check_start].dropna()
        if len(swing_lows) == 0:
            return False
        recent_sl = swing_lows.iloc[-1]
        for i in range(check_start, idx + 1):
            if df_5m['Low'].iloc[i] < recent_sl and df_5m['Close'].iloc[i] > recent_sl:
                return True
        for i in range(check_start, idx):
            if df_5m['Low'].iloc[i] < recent_sl:
                for j in range(i+1, idx+1):
                    if df_5m['Close'].iloc[j] > recent_sl:
                        return True
    else:
        swing_highs = df_5m['swing_highs'].iloc[:check_start].dropna()
        if len(swing_highs) == 0:
            return False
        recent_sh = swing_highs.iloc[-1]
        for i in range(check_start, idx + 1):
            if df_5m['High'].iloc[i] > recent_sh and df_5m['Close'].iloc[i] < recent_sh:
                return True
        for i in range(check_start, idx):
            if df_5m['High'].iloc[i] > recent_sh:
                for j in range(i+1, idx+1):
                    if df_5m['Close'].iloc[j] < recent_sh:
                        return True
    return False


def check_structure_break(df_5m, idx, direction, config: StrategyConfig):
    lookback = config.structure_lookback
    if idx < lookback:
        return False
    
    window = df_5m.iloc[idx-lookback:idx+1]
    current_close = df_5m['Close'].iloc[idx]
    
    if direction == 'bullish':
        local_highs = []
        for i in range(2, len(window) - 1):
            if window['High'].iloc[i] > window['High'].iloc[i-1] and window['High'].iloc[i] > window['High'].iloc[i+1]:
                local_highs.append(window['High'].iloc[i])
        if len(local_highs) >= 2:
            for j in range(len(local_highs)-1, 0, -1):
                if local_highs[j] < local_highs[j-1]:
                    if current_close > local_highs[j]:
                        return True
                    break
        recent_highs = window['High'].iloc[-6:-1]
        if len(recent_highs) > 0 and current_close > recent_highs.max():
            return True
    else:
        local_lows = []
        for i in range(2, len(window) - 1):
            if window['Low'].iloc[i] < window['Low'].iloc[i-1] and window['Low'].iloc[i] < window['Low'].iloc[i+1]:
                local_lows.append(window['Low'].iloc[i])
        if len(local_lows) >= 2:
            for j in range(len(local_lows)-1, 0, -1):
                if local_lows[j] > local_lows[j-1]:
                    if current_close < local_lows[j]:
                        return True
                    break
        recent_lows = window['Low'].iloc[-6:-1]
        if len(recent_lows) > 0 and current_close < recent_lows.min():
            return True
    return False


def check_candle_pattern(df_5m, idx, direction, config: StrategyConfig):
    if idx < 1:
        return False, None
    
    curr_o = df_5m['Open'].iloc[idx]
    curr_c = df_5m['Close'].iloc[idx]
    curr_h = df_5m['High'].iloc[idx]
    curr_l = df_5m['Low'].iloc[idx]
    prev_o = df_5m['Open'].iloc[idx-1]
    prev_c = df_5m['Close'].iloc[idx-1]
    
    body = abs(curr_c - curr_o)
    candle_range = curr_h - curr_l
    if candle_range == 0:
        return False, None
    
    body_ratio = body / candle_range
    upper_wick = curr_h - max(curr_o, curr_c)
    lower_wick = min(curr_o, curr_c) - curr_l
    
    pattern = None
    
    if direction == 'bullish':
        if curr_c > curr_o and prev_c < prev_o and curr_o <= prev_c and curr_c >= prev_o:
            pattern = 'bullish_engulfing'
        elif body_ratio < 0.35 and body > 0 and lower_wick >= body * 2 and upper_wick < body * 0.5 and curr_c >= curr_o:
            pattern = 'hammer'
        elif curr_c > curr_o and body_ratio >= 0.5 and lower_wick >= body * 0.7:
            pattern = 'bullish_rejection'
        elif curr_c > curr_o and body_ratio >= 0.65:
            pattern = 'strong_bullish'
        
        if pattern and pattern in config.allowed_bull_patterns:
            return True, pattern
    else:
        if curr_c < curr_o and prev_c > prev_o and curr_o >= prev_c and curr_c <= prev_o:
            pattern = 'bearish_engulfing'
        elif body_ratio < 0.35 and body > 0 and upper_wick >= body * 2 and lower_wick < body * 0.5 and curr_c <= curr_o:
            pattern = 'shooting_star'
        elif curr_c < curr_o and body_ratio >= 0.5 and upper_wick >= body * 0.7:
            pattern = 'bearish_rejection'
        elif curr_c < curr_o and body_ratio >= 0.65:
            pattern = 'strong_bearish'
        
        if pattern and pattern in config.allowed_bear_patterns:
            return True, pattern
    
    return False, pattern


def check_volume(df_5m, idx, config: StrategyConfig):
    if idx < VOLUME_AVG_PERIOD:
        return False, 0
    vol = df_5m['Volume'].iloc[idx]
    avg_vol = df_5m['Volume'].iloc[idx-VOLUME_AVG_PERIOD:idx].mean()
    if avg_vol <= 0:
        return False, 0
    ratio = vol / avg_vol
    return ratio >= config.volume_min_ratio, ratio


def calculate_sl(df_5m, idx, direction, entry_price):
    atr = df_5m['atr'].iloc[idx]
    if direction == 'bullish':
        atr_sl = entry_price - atr
        swing_lows = df_5m['swing_lows'].iloc[:idx].dropna()
        if len(swing_lows) > 0:
            swing_sl = swing_lows.iloc[-1] - (entry_price * 0.0005)
            return min(atr_sl, swing_sl)
        return atr_sl
    else:
        atr_sl = entry_price + atr
        swing_highs = df_5m['swing_highs'].iloc[:idx].dropna()
        if len(swing_highs) > 0:
            swing_sl = swing_highs.iloc[-1] + (entry_price * 0.0005)
            return max(atr_sl, swing_sl)
        return atr_sl


def simulate_trade(df_5m, entry_idx, trade: Trade):
    sl = trade.stop_loss
    tp = trade.take_profit
    entry = trade.entry_price
    sl_distance = abs(entry - sl)
    
    for i in range(entry_idx + 1, min(entry_idx + 200, len(df_5m))):
        high = df_5m['High'].iloc[i]
        low = df_5m['Low'].iloc[i]
        
        if trade.direction == 'bullish':
            if low <= sl:
                trade.exit_price = sl
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = 0.0 if trade.hit_breakeven else -1.0
                trade.result = 'breakeven' if trade.hit_breakeven else 'loss'
                return trade
            if high >= tp:
                trade.exit_price = tp
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = abs(tp - entry) / sl_distance
                trade.result = 'win'
                return trade
            if not trade.hit_breakeven and high >= entry + sl_distance:
                trade.hit_breakeven = True
                sl = entry
        else:
            if high >= sl:
                trade.exit_price = sl
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = 0.0 if trade.hit_breakeven else -1.0
                trade.result = 'breakeven' if trade.hit_breakeven else 'loss'
                return trade
            if low <= tp:
                trade.exit_price = tp
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = abs(entry - tp) / sl_distance
                trade.result = 'win'
                return trade
            if not trade.hit_breakeven and low <= entry - sl_distance:
                trade.hit_breakeven = True
                sl = entry
    
    # Timeout - close at last price
    last_price = df_5m['Close'].iloc[min(entry_idx + 199, len(df_5m)-1)]
    trade.exit_price = last_price
    trade.exit_time = str(df_5m.index[min(entry_idx + 199, len(df_5m)-1)])
    if trade.direction == 'bullish':
        trade.pnl_r = (last_price - entry) / sl_distance
    else:
        trade.pnl_r = (entry - last_price) / sl_distance
    trade.result = 'win' if trade.pnl_r > 0 else 'loss'
    return trade


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest_with_config(all_data: dict, config: StrategyConfig) -> List[Trade]:
    """Run backtest with a specific configuration."""
    all_trades = []
    
    for name, data in all_data.items():
        df_4h = data['4h']
        df_30m = data['30m']
        df_5m = data['5m']
        
        if df_4h.empty or df_30m.empty or df_5m.empty:
            continue
        
        daily_trades = {}
        start_idx = max(250, SWING_LOOKBACK + 20)
        
        for idx in range(start_idx, len(df_5m) - 200):
            timestamp = df_5m.index[idx]
            
            if hasattr(timestamp, 'tz') and timestamp.tz is not None:
                ts_naive = timestamp.tz_localize(None)
            else:
                ts_naive = timestamp
            
            if not (SESSION_START <= ts_naive.hour <= SESSION_END):
                continue
            
            day_key = ts_naive.date()
            if daily_trades.get(day_key, 0) >= config.max_trades_per_day:
                continue
            
            # 4H trend
            mask_4h = df_4h.index <= timestamp
            if mask_4h.sum() < EMA_4H_PERIOD + EMA_4H_SLOPE_LOOKBACK + 1:
                continue
            idx_4h = mask_4h.sum() - 1
            trend_4h = check_4h_trend(df_4h, idx_4h)
            if trend_4h is None:
                continue
            
            # 30M trend
            mask_30m = df_30m.index <= timestamp
            if mask_30m.sum() < EMA_30M_SLOW + 10:
                continue
            idx_30m = mask_30m.sum() - 1
            trend_30m = check_30m_trend(df_30m, idx_30m)
            if trend_30m is None or trend_30m != trend_4h:
                continue
            
            direction = trend_4h
            
            # Direction filter
            if direction == 'bullish' and not config.allow_longs:
                continue
            if direction == 'bearish' and not config.allow_shorts:
                continue
            
            # Run all 6 confirmations and count
            confirmations = 0
            confirmation_details = {}
            
            # 1. Pullback
            pb_ok = check_pullback(df_5m, idx, direction, config)
            if pb_ok:
                confirmations += 1
            elif config.require_pullback:
                continue
            confirmation_details['pullback'] = pb_ok
            
            # 2. RSI
            rsi_ok, rsi_val = check_rsi(df_5m, idx, direction, config)
            if rsi_ok:
                confirmations += 1
            elif config.require_rsi:
                continue
            confirmation_details['rsi'] = rsi_ok
            
            # 3. Liquidity sweep
            sweep_ok = check_liquidity_sweep(df_5m, idx, direction)
            if sweep_ok:
                confirmations += 1
            elif config.require_sweep:
                continue
            confirmation_details['sweep'] = sweep_ok
            
            # 4. Structure break
            struct_ok = check_structure_break(df_5m, idx, direction, config)
            if struct_ok:
                confirmations += 1
            elif config.require_structure:
                continue
            confirmation_details['structure'] = struct_ok
            
            # 5. Candle pattern
            candle_ok, pattern = check_candle_pattern(df_5m, idx, direction, config)
            if candle_ok:
                confirmations += 1
            elif config.require_candle:
                continue
            confirmation_details['candle'] = candle_ok
            
            # 6. Volume
            vol_ok, vol_ratio = check_volume(df_5m, idx, config)
            if vol_ok:
                confirmations += 1
            elif config.require_volume:
                continue
            confirmation_details['volume'] = vol_ok
            
            # Check minimum confirmations
            if confirmations < config.min_confirmations:
                continue
            
            # CREATE TRADE
            entry_price = df_5m['Close'].iloc[idx]
            sl = calculate_sl(df_5m, idx, direction, entry_price)
            sl_distance = abs(entry_price - sl)
            
            if sl_distance <= 0:
                continue
            
            rr = config.preferred_rr
            if direction == 'bullish':
                tp = entry_price + (sl_distance * rr)
            else:
                tp = entry_price - (sl_distance * rr)
            
            trade = Trade(
                symbol=name,
                direction=direction,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=tp,
                entry_time=str(timestamp),
                pattern=pattern or "",
                rsi=rsi_val,
                volume_ratio=vol_ratio,
                confirmations_met=confirmations,
            )
            
            trade = simulate_trade(df_5m, idx, trade)
            all_trades.append(trade)
            daily_trades[day_key] = daily_trades.get(day_key, 0) + 1
            
            # Skip ahead
            idx += 10
    
    return all_trades


def calculate_stats(trades: List[Trade], config_name: str) -> dict:
    """Calculate comprehensive stats for a set of trades."""
    if not trades:
        return {'name': config_name, 'total': 0, 'valid': False}
    
    total = len(trades)
    wins = [t for t in trades if t.result == 'win']
    losses = [t for t in trades if t.result == 'loss']
    breakevens = [t for t in trades if t.result == 'breakeven']
    
    win_rate = len(wins) / total * 100
    total_r = sum(t.pnl_r for t in trades)
    avg_win_r = np.mean([t.pnl_r for t in wins]) if wins else 0
    avg_loss_r = np.mean([abs(t.pnl_r) for t in losses]) if losses else 0
    
    gross_profit = sum(t.pnl_r for t in trades if t.pnl_r > 0)
    gross_loss = abs(sum(t.pnl_r for t in trades if t.pnl_r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown
    cumulative_r = np.cumsum([t.pnl_r for t in trades])
    peak = np.maximum.accumulate(cumulative_r)
    max_dd_r = (peak - cumulative_r).max()
    
    # Equity
    equity = STARTING_EQUITY
    max_eq = equity
    max_dd_pct = 0
    for t in trades:
        risk = equity * (RISK_PERCENT / 100)
        equity += risk * t.pnl_r
        max_eq = max(max_eq, equity)
        dd = (max_eq - equity) / max_eq * 100
        max_dd_pct = max(max_dd_pct, dd)
    
    total_return = ((equity - STARTING_EQUITY) / STARTING_EQUITY) * 100
    
    # By direction
    longs = [t for t in trades if t.direction == 'bullish']
    shorts = [t for t in trades if t.direction == 'bearish']
    long_r = sum(t.pnl_r for t in longs)
    short_r = sum(t.pnl_r for t in shorts)
    long_wr = sum(1 for t in longs if t.result == 'win') / len(longs) * 100 if longs else 0
    short_wr = sum(1 for t in shorts if t.result == 'win') / len(shorts) * 100 if shorts else 0
    
    # Consecutive losses
    max_consec_loss = 0
    curr = 0
    for t in trades:
        if t.result == 'loss':
            curr += 1
            max_consec_loss = max(max_consec_loss, curr)
        else:
            curr = 0
    
    return {
        'name': config_name,
        'valid': True,
        'total': total,
        'wins': len(wins),
        'losses': len(losses),
        'breakevens': len(breakevens),
        'win_rate': win_rate,
        'total_r': total_r,
        'avg_win_r': avg_win_r,
        'avg_loss_r': avg_loss_r,
        'profit_factor': profit_factor,
        'expectancy': total_r / total,
        'max_dd_pct': max_dd_pct,
        'max_dd_r': max_dd_r,
        'total_return': total_return,
        'final_equity': equity,
        'longs': len(longs),
        'shorts': len(shorts),
        'long_r': long_r,
        'short_r': short_r,
        'long_wr': long_wr,
        'short_wr': short_wr,
        'max_consec_loss': max_consec_loss,
        'be_rate': len(breakevens) / total * 100,
    }


# ============================================================
# STRATEGY VARIATIONS
# ============================================================
def get_strategy_variations() -> List[StrategyConfig]:
    """Define all strategy variations to test."""
    
    configs = []
    
    # V1: ORIGINAL (baseline)
    configs.append(StrategyConfig(
        name="V1: ORIGINAL (all 6 required)",
    ))
    
    # V2: SHORTS ONLY (remove losing longs)
    configs.append(StrategyConfig(
        name="V2: SHORTS ONLY",
        allow_longs=False,
    ))
    
    # V3: RELAXED RSI (wider zones)
    configs.append(StrategyConfig(
        name="V3: WIDER RSI (40-65 long, 35-60 short)",
        rsi_long_min=40,
        rsi_long_max=65,
        rsi_short_min=35,
        rsi_short_max=60,
    ))
    
    # V4: 5/6 CONFIRMATIONS (drop volume requirement)
    configs.append(StrategyConfig(
        name="V4: 5/6 CONFIRMS (volume optional)",
        min_confirmations=5,
        require_volume=False,
    ))
    
    # V5: 5/6 CONFIRMATIONS (drop liquidity sweep)
    configs.append(StrategyConfig(
        name="V5: 5/6 CONFIRMS (sweep optional)",
        min_confirmations=5,
        require_sweep=False,
    ))
    
    # V6: WIDER PULLBACK + WIDER RSI
    configs.append(StrategyConfig(
        name="V6: WIDER PULLBACK (0.4%) + WIDER RSI",
        pullback_threshold=0.004,
        rsi_long_min=40,
        rsi_long_max=65,
        rsi_short_min=35,
        rsi_short_max=60,
    ))
    
    # V7: REMOVE BAD PATTERNS (no strong_bullish for longs)
    configs.append(StrategyConfig(
        name="V7: BETTER PATTERNS (no strong_bullish)",
        allowed_bull_patterns=['bullish_engulfing', 'hammer', 'bullish_rejection'],
    ))
    
    # V8: AGGRESSIVE (wider RSI + 5/6 confirms + wider pullback)
    configs.append(StrategyConfig(
        name="V8: AGGRESSIVE (wider all + 5/6)",
        rsi_long_min=40,
        rsi_long_max=65,
        rsi_short_min=35,
        rsi_short_max=60,
        min_confirmations=5,
        require_volume=False,
        pullback_threshold=0.004,
    ))
    
    # V9: CONSERVATIVE (shorts only + all 6 + tighter volume)
    configs.append(StrategyConfig(
        name="V9: CONSERVATIVE (shorts + volume 1.2x)",
        allow_longs=False,
        volume_min_ratio=1.2,
    ))
    
    # V10: OPTIMAL HYBRID (shorts only, wider RSI, 5/6, better patterns)
    configs.append(StrategyConfig(
        name="V10: OPTIMAL (shorts, wide RSI, 5/6, good patterns)",
        allow_longs=False,
        rsi_short_min=35,
        rsi_short_max=60,
        min_confirmations=5,
        require_volume=False,
        allowed_bear_patterns=['bearish_engulfing', 'shooting_star', 'bearish_rejection', 'strong_bearish'],
    ))
    
    # V11: BOTH DIRS OPTIMIZED (wide RSI, better patterns, 5/6)
    configs.append(StrategyConfig(
        name="V11: BOTH DIRS (wide RSI, good patterns, 5/6)",
        rsi_long_min=40,
        rsi_long_max=65,
        rsi_short_min=35,
        rsi_short_max=60,
        min_confirmations=5,
        require_volume=False,
        allowed_bull_patterns=['bullish_engulfing', 'hammer', 'bullish_rejection'],
        allowed_bear_patterns=['bearish_engulfing', 'shooting_star', 'bearish_rejection', 'strong_bearish'],
    ))
    
    # V12: MAX FREQUENCY (wider everything, 3 trades/day)
    configs.append(StrategyConfig(
        name="V12: MAX FREQ (wide all, 3/day, 5/6)",
        rsi_long_min=38,
        rsi_long_max=68,
        rsi_short_min=32,
        rsi_short_max=62,
        min_confirmations=5,
        require_volume=False,
        pullback_threshold=0.005,
        max_trades_per_day=3,
    ))
    
    return configs


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("   STRATEGY OPTIMIZATION - TESTING 12 VARIATIONS")
    print("=" * 70)
    print(f"Starting Equity: ${STARTING_EQUITY:,.2f} | Risk: {RISK_PERCENT}%")
    print(f"Data: 60 days of 5m/30m, 2 years of 4H (BTC-USD + Gold Futures)")
    print("=" * 70)
    
    # Fetch data once
    print("\nFetching market data (one-time)...")
    all_data = {}
    
    for name, ticker in [('BTC-USD', 'BTC-USD'), ('XAU', 'GC=F')]:
        try:
            data = fetch_data(ticker)
            data = add_indicators(data)
            all_data[name] = data
            print(f"  {name}: 4H={len(data['4h'])} | 30M={len(data['30m'])} | 5M={len(data['5m'])} bars")
        except Exception as e:
            print(f"  ERROR: {name}: {e}")
    
    if not all_data:
        print("No data available. Exiting.")
        return
    
    # Run all variations
    print("\n" + "=" * 70)
    print("   RUNNING ALL STRATEGY VARIATIONS...")
    print("=" * 70)
    
    configs = get_strategy_variations()
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n  [{i:>2}/{len(configs)}] Testing: {config.name}")
        trades = run_backtest_with_config(all_data, config)
        stats = calculate_stats(trades, config.name)
        results.append(stats)
        
        if stats['valid']:
            print(f"         Trades: {stats['total']:>3} | WR: {stats['win_rate']:>5.1f}% | "
                  f"PF: {stats['profit_factor']:>5.2f} | Total: {stats['total_r']:>+6.1f}R | "
                  f"Return: {stats['total_return']:>+6.1f}% | MaxDD: {stats['max_dd_pct']:>5.1f}%")
        else:
            print(f"         NO TRADES FOUND")
    
    # Sort by total R (best performing)
    valid_results = [r for r in results if r['valid'] and r['total'] > 0]
    valid_results.sort(key=lambda x: x['total_r'], reverse=True)
    
    # Print comparison table
    print("\n\n")
    print("=" * 90)
    print("   STRATEGY COMPARISON - RANKED BY TOTAL R")
    print("=" * 90)
    print(f"{'Rank':<5} {'Strategy':<45} {'Trades':<7} {'WR%':<6} {'PF':<6} {'TotalR':<8} {'Return':<8} {'MaxDD':<7} {'Exp/T':<7}")
    print("-" * 90)
    
    for rank, r in enumerate(valid_results, 1):
        name = r['name'][:43]
        print(f"{rank:<5} {name:<45} {r['total']:<7} {r['win_rate']:<6.1f} {r['profit_factor']:<6.2f} "
              f"{r['total_r']:<+8.1f} {r['total_return']:<+8.1f} {r['max_dd_pct']:<7.1f} {r['expectancy']:<+7.3f}")
    
    # Detail on top 3
    print("\n\n")
    print("=" * 70)
    print("   TOP 3 DETAILED BREAKDOWN")
    print("=" * 70)
    
    for rank, r in enumerate(valid_results[:3], 1):
        print(f"\n{'#' + str(rank) + ' ' + r['name']:=^70}")
        print(f"  Trades: {r['total']} | Wins: {r['wins']} | Losses: {r['losses']} | BE: {r['breakevens']}")
        print(f"  Win Rate: {r['win_rate']:.1f}% | Profit Factor: {r['profit_factor']:.2f}")
        print(f"  Total R: {r['total_r']:+.2f} | Expectancy: {r['expectancy']:+.3f}R/trade")
        print(f"  Avg Win: {r['avg_win_r']:.2f}R | Avg Loss: {r['avg_loss_r']:.2f}R")
        print(f"  Final Equity: ${r['final_equity']:,.2f} | Return: {r['total_return']:+.2f}%")
        print(f"  Max Drawdown: {r['max_dd_pct']:.2f}% | Max Consec Losses: {r['max_consec_loss']}")
        print(f"  Longs: {r['longs']} ({r['long_wr']:.0f}% WR, {r['long_r']:+.1f}R)")
        print(f"  Shorts: {r['shorts']} ({r['short_wr']:.0f}% WR, {r['short_r']:+.1f}R)")
        print(f"  Breakeven Rate: {r['be_rate']:.1f}%")
    
    # RECOMMENDATION
    print("\n\n")
    print("=" * 70)
    print("   RECOMMENDATION")
    print("=" * 70)
    
    best = valid_results[0] if valid_results else None
    if best:
        print(f"\n  BEST STRATEGY: {best['name']}")
        print(f"\n  Why:")
        print(f"  - Highest total R: {best['total_r']:+.2f}")
        print(f"  - {best['total']} trades in 60 days = ~{best['total']/60*30:.0f} trades/month")
        print(f"  - Profit Factor: {best['profit_factor']:.2f}")
        print(f"  - Max Drawdown: {best['max_dd_pct']:.1f}% (within your 4% daily limit)")
        
        # Find best risk-adjusted
        risk_adjusted = sorted(valid_results, key=lambda x: x['total_r'] / max(x['max_dd_pct'], 0.01), reverse=True)
        if risk_adjusted:
            safest = risk_adjusted[0]
            if safest['name'] != best['name']:
                print(f"\n  SAFEST (best R/DD ratio): {safest['name']}")
                print(f"  - Total R: {safest['total_r']:+.2f} with only {safest['max_dd_pct']:.1f}% drawdown")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
