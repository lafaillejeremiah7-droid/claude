"""
Win Rate Optimization Backtest

Goal: Increase win rate WITHOUT significantly hurting total profit.

Levers to increase win rate:
1. TIGHTER TAKE PROFIT (1.5R instead of 2R) - easier to hit
2. TRAILING STOP instead of fixed TP - lock in partial gains
3. STRONGER TREND FILTER - only trade when trend is very strong
4. TIGHTER STRUCTURE BREAK - more decisive breaks only
5. HIGHER VOLUME THRESHOLD - only enter on strong volume spikes
6. ADD EMA20 SLOPE FILTER on 5M - momentum must align
7. REQUIRE 2+ CANDLE CONFIRMATION - wait for follow-through
8. TIME-BASED EXIT - close after X bars if no TP/SL hit
9. COMBINATION APPROACHES

Each tweak targets "more trades hitting TP" without removing too many trades.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional

# ============================================================
# CONFIGURATION
# ============================================================
STARTING_EQUITY = 10000.0
RISK_PERCENT = 2.0
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
class WRConfig:
    """Configuration for win-rate optimized variants."""
    name: str = "Default"
    
    # RSI zones
    rsi_long_min: float = 40
    rsi_long_max: float = 65
    rsi_short_min: float = 35
    rsi_short_max: float = 60
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # Confirmations
    require_sweep: bool = False  # V5 baseline: sweep optional
    require_volume: bool = True
    min_confirmations: int = 5
    
    # Direction
    allow_longs: bool = True
    allow_shorts: bool = True
    
    # R:R targets
    rr_ratio: float = 2.0  # Default 2R TP
    
    # Win rate boosters
    use_trailing_stop: bool = False
    trailing_trigger_r: float = 1.0  # Start trailing after 1R
    trailing_distance_r: float = 0.5  # Trail by 0.5R behind
    
    use_time_exit: bool = False
    max_bars_in_trade: int = 100  # Close after N bars if no TP/SL
    
    require_ema20_slope: bool = False  # 5M EMA20 must slope in direction
    ema20_slope_bars: int = 3
    
    require_momentum_candle: bool = False  # Entry candle body > 60% of range
    
    volume_min_ratio: float = 1.0
    
    # Trend strength filter
    require_strong_trend: bool = False  # 4H EMA must be steeply sloped
    min_4h_slope_pct: float = 0.001  # Minimum slope as % of price
    
    # Pattern filter
    allowed_bull_patterns: list = field(default_factory=lambda: [
        'bullish_engulfing', 'hammer', 'bullish_rejection', 'strong_bullish'
    ])
    allowed_bear_patterns: list = field(default_factory=lambda: [
        'bearish_engulfing', 'shooting_star', 'bearish_rejection', 'strong_bearish'
    ])
    
    max_trades_per_day: int = 2
    pullback_threshold: float = 0.002


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


# ============================================================
# INDICATORS
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
    return (100 - (100 / (1 + rs))).fillna(50)

def calc_atr(df, period=14):
    high, low, close_prev = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def find_swing_highs(highs, lookback=10):
    sh = pd.Series(np.nan, index=highs.index)
    for i in range(lookback, len(highs) - lookback):
        if highs.iloc[i] >= highs.iloc[i-lookback:i].max() and highs.iloc[i] >= highs.iloc[i+1:i+lookback+1].max():
            sh.iloc[i] = highs.iloc[i]
    return sh

def find_swing_lows(lows, lookback=10):
    sl = pd.Series(np.nan, index=lows.index)
    for i in range(lookback, len(lows) - lookback):
        if lows.iloc[i] <= lows.iloc[i-lookback:i].min() and lows.iloc[i] <= lows.iloc[i+1:i+lookback+1].min():
            sl.iloc[i] = lows.iloc[i]
    return sl


# ============================================================
# DATA
# ============================================================
def fetch_and_prepare():
    """Fetch and prepare all data."""
    print("Fetching market data...")
    all_data = {}
    
    for name, ticker in [('BTC-USD', 'BTC-USD'), ('XAU', 'GC=F')]:
        print(f"  {name}...")
        df_1h = yf.download(ticker, period="2y", interval="1h", progress=False)
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.get_level_values(0)
        
        df_4h = df_1h.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        df_30m = yf.download(ticker, period="60d", interval="30m", progress=False)
        if isinstance(df_30m.columns, pd.MultiIndex):
            df_30m.columns = df_30m.columns.get_level_values(0)
        
        df_5m = yf.download(ticker, period="60d", interval="5m", progress=False)
        if isinstance(df_5m.columns, pd.MultiIndex):
            df_5m.columns = df_5m.columns.get_level_values(0)
        
        # Add indicators
        df_4h['ema_50'] = calc_ema(df_4h['Close'], EMA_4H_PERIOD)
        df_30m['ema_50'] = calc_ema(df_30m['Close'], EMA_30M_FAST)
        df_30m['ema_200'] = calc_ema(df_30m['Close'], EMA_30M_SLOW)
        df_5m['ema_20'] = calc_ema(df_5m['Close'], EMA_5M_PULLBACK)
        df_5m['ema_20_slope'] = df_5m['ema_20'].diff(3)
        df_5m['rsi'] = calc_rsi(df_5m['Close'], RSI_PERIOD)
        df_5m['atr'] = calc_atr(df_5m, ATR_PERIOD)
        df_5m['volume_avg'] = df_5m['Volume'].rolling(VOLUME_AVG_PERIOD).mean()
        df_5m['swing_highs'] = find_swing_highs(df_5m['High'], SWING_LOOKBACK)
        df_5m['swing_lows'] = find_swing_lows(df_5m['Low'], SWING_LOOKBACK)
        
        all_data[name] = {'4h': df_4h, '30m': df_30m, '5m': df_5m}
        print(f"    4H={len(df_4h)} | 30M={len(df_30m)} | 5M={len(df_5m)}")
    
    return all_data


# ============================================================
# STRATEGY CHECKS
# ============================================================
def check_4h_trend(df_4h, idx, config: WRConfig):
    if idx < EMA_4H_PERIOD + EMA_4H_SLOPE_LOOKBACK:
        return None
    ema50 = df_4h['ema_50'].iloc[idx]
    slope = df_4h['ema_50'].iloc[idx] - df_4h['ema_50'].iloc[idx - EMA_4H_SLOPE_LOOKBACK]
    price = df_4h['Close'].iloc[idx]
    
    # Strong trend filter
    if config.require_strong_trend:
        slope_pct = abs(slope) / ema50 if ema50 > 0 else 0
        if slope_pct < config.min_4h_slope_pct:
            return None
    
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

def check_pullback(df_5m, idx, direction, config: WRConfig):
    if idx < 3:
        return False
    ema20 = df_5m['ema_20'].iloc[idx]
    close = df_5m['Close'].iloc[idx]
    low = df_5m['Low'].iloc[idx]
    high = df_5m['High'].iloc[idx]
    t = config.pullback_threshold
    
    if direction == 'bullish':
        near = abs(low - ema20) / ema20 <= t or low <= ema20
        bounce = close > ema20
        if near and bounce:
            return True
        for i in range(max(0, idx-3), idx):
            if df_5m['Low'].iloc[i] <= ema20 * (1 + t) and df_5m['Close'].iloc[i] > ema20:
                return True
    else:
        near = abs(high - ema20) / ema20 <= t or high >= ema20
        bounce = close < ema20
        if near and bounce:
            return True
        for i in range(max(0, idx-3), idx):
            if df_5m['High'].iloc[i] >= ema20 * (1 - t) and df_5m['Close'].iloc[i] < ema20:
                return True
    return False

def check_rsi(df_5m, idx, direction, config: WRConfig):
    rsi = df_5m['rsi'].iloc[idx]
    if rsi >= config.rsi_overbought or rsi <= config.rsi_oversold:
        return False, rsi
    if direction == 'bullish':
        return config.rsi_long_min <= rsi <= config.rsi_long_max, rsi
    else:
        return config.rsi_short_min <= rsi <= config.rsi_short_max, rsi

def check_sweep(df_5m, idx, direction):
    if idx < SWING_LOOKBACK + 10:
        return False
    check_start = max(0, idx - 5)
    if direction == 'bullish':
        sl = df_5m['swing_lows'].iloc[:check_start].dropna()
        if len(sl) == 0:
            return False
        level = sl.iloc[-1]
        for i in range(check_start, idx + 1):
            if df_5m['Low'].iloc[i] < level and df_5m['Close'].iloc[i] > level:
                return True
        for i in range(check_start, idx):
            if df_5m['Low'].iloc[i] < level:
                for j in range(i+1, idx+1):
                    if df_5m['Close'].iloc[j] > level:
                        return True
    else:
        sh = df_5m['swing_highs'].iloc[:check_start].dropna()
        if len(sh) == 0:
            return False
        level = sh.iloc[-1]
        for i in range(check_start, idx + 1):
            if df_5m['High'].iloc[i] > level and df_5m['Close'].iloc[i] < level:
                return True
        for i in range(check_start, idx):
            if df_5m['High'].iloc[i] > level:
                for j in range(i+1, idx+1):
                    if df_5m['Close'].iloc[j] < level:
                        return True
    return False

def check_structure_break(df_5m, idx, direction):
    if idx < 20:
        return False
    window = df_5m.iloc[idx-20:idx+1]
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
        recent = window['High'].iloc[-6:-1]
        if len(recent) > 0 and current_close > recent.max():
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
        recent = window['Low'].iloc[-6:-1]
        if len(recent) > 0 and current_close < recent.min():
            return True
    return False

def check_candle(df_5m, idx, direction, config: WRConfig):
    if idx < 1:
        return False, None
    curr_o, curr_c = df_5m['Open'].iloc[idx], df_5m['Close'].iloc[idx]
    curr_h, curr_l = df_5m['High'].iloc[idx], df_5m['Low'].iloc[idx]
    prev_o, prev_c = df_5m['Open'].iloc[idx-1], df_5m['Close'].iloc[idx-1]
    
    body = abs(curr_c - curr_o)
    rng = curr_h - curr_l
    if rng == 0:
        return False, None
    body_ratio = body / rng
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

def check_volume(df_5m, idx, config: WRConfig):
    if idx < VOLUME_AVG_PERIOD:
        return False, 0
    vol = df_5m['Volume'].iloc[idx]
    avg = df_5m['Volume'].iloc[idx-VOLUME_AVG_PERIOD:idx].mean()
    if avg <= 0:
        return False, 0
    ratio = vol / avg
    return ratio >= config.volume_min_ratio, ratio

def check_ema20_slope(df_5m, idx, direction, config: WRConfig):
    """Check if EMA20 slope aligns with trade direction on 5M."""
    if not config.require_ema20_slope:
        return True
    slope = df_5m['ema_20_slope'].iloc[idx]
    if direction == 'bullish':
        return slope > 0
    else:
        return slope < 0

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


# ============================================================
# TRADE SIMULATION (with trailing stop & time exit options)
# ============================================================
def simulate_trade(df_5m, entry_idx, trade: Trade, config: WRConfig):
    """Simulate trade with optional trailing stop and time-based exit."""
    sl = trade.stop_loss
    tp = trade.take_profit
    entry = trade.entry_price
    sl_distance = abs(entry - sl)
    
    if sl_distance <= 0:
        trade.result = 'loss'
        trade.pnl_r = -1.0
        return trade
    
    highest_profit_r = 0.0
    trailing_active = False
    
    max_bars = config.max_bars_in_trade if config.use_time_exit else 200
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df_5m))):
        high = df_5m['High'].iloc[i]
        low = df_5m['Low'].iloc[i]
        
        if trade.direction == 'bullish':
            current_profit_r = (high - entry) / sl_distance
            highest_profit_r = max(highest_profit_r, current_profit_r)
            
            # Trailing stop logic
            if config.use_trailing_stop and highest_profit_r >= config.trailing_trigger_r:
                trailing_active = True
                trail_level = entry + (highest_profit_r - config.trailing_distance_r) * sl_distance
                sl = max(sl, trail_level)  # Only move up
            
            # Check SL hit
            if low <= sl:
                trade.exit_price = sl
                trade.exit_time = str(df_5m.index[i])
                if trailing_active:
                    trade.pnl_r = (sl - entry) / sl_distance
                    trade.result = 'win' if trade.pnl_r > 0 else ('breakeven' if trade.pnl_r == 0 else 'loss')
                elif trade.hit_breakeven:
                    trade.pnl_r = 0.0
                    trade.result = 'breakeven'
                else:
                    trade.pnl_r = -1.0
                    trade.result = 'loss'
                return trade
            
            # Check TP hit
            if high >= tp:
                trade.exit_price = tp
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = (tp - entry) / sl_distance
                trade.result = 'win'
                return trade
            
            # Standard breakeven (if not using trailing)
            if not config.use_trailing_stop and not trade.hit_breakeven and high >= entry + sl_distance:
                trade.hit_breakeven = True
                sl = entry
                
        else:  # bearish
            current_profit_r = (entry - low) / sl_distance
            highest_profit_r = max(highest_profit_r, current_profit_r)
            
            if config.use_trailing_stop and highest_profit_r >= config.trailing_trigger_r:
                trailing_active = True
                trail_level = entry - (highest_profit_r - config.trailing_distance_r) * sl_distance
                sl = min(sl, trail_level)  # Only move down
            
            if high >= sl:
                trade.exit_price = sl
                trade.exit_time = str(df_5m.index[i])
                if trailing_active:
                    trade.pnl_r = (entry - sl) / sl_distance
                    trade.result = 'win' if trade.pnl_r > 0 else ('breakeven' if trade.pnl_r == 0 else 'loss')
                elif trade.hit_breakeven:
                    trade.pnl_r = 0.0
                    trade.result = 'breakeven'
                else:
                    trade.pnl_r = -1.0
                    trade.result = 'loss'
                return trade
            
            if low <= tp:
                trade.exit_price = tp
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = (entry - tp) / sl_distance
                trade.result = 'win'
                return trade
            
            if not config.use_trailing_stop and not trade.hit_breakeven and low <= entry - sl_distance:
                trade.hit_breakeven = True
                sl = entry
    
    # Time exit or end of data
    last_idx = min(entry_idx + max_bars - 1, len(df_5m) - 1)
    last_price = df_5m['Close'].iloc[last_idx]
    trade.exit_price = last_price
    trade.exit_time = str(df_5m.index[last_idx])
    if trade.direction == 'bullish':
        trade.pnl_r = (last_price - entry) / sl_distance
    else:
        trade.pnl_r = (entry - last_price) / sl_distance
    trade.result = 'win' if trade.pnl_r > 0.1 else ('loss' if trade.pnl_r < -0.1 else 'breakeven')
    return trade


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(all_data: dict, config: WRConfig) -> List[Trade]:
    trades = []
    
    for name, data in all_data.items():
        df_4h, df_30m, df_5m = data['4h'], data['30m'], data['5m']
        if df_5m.empty:
            continue
        
        daily_trades = {}
        start_idx = max(250, SWING_LOOKBACK + 20)
        
        for idx in range(start_idx, len(df_5m) - 200):
            timestamp = df_5m.index[idx]
            ts_naive = timestamp.tz_localize(None) if hasattr(timestamp, 'tz') and timestamp.tz else timestamp
            
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
            trend_4h = check_4h_trend(df_4h, idx_4h, config)
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
            if direction == 'bullish' and not config.allow_longs:
                continue
            if direction == 'bearish' and not config.allow_shorts:
                continue
            
            # Check confirmations
            confirmations = 0
            
            if check_pullback(df_5m, idx, direction, config):
                confirmations += 1
            else:
                continue  # Pullback always required
            
            rsi_ok, rsi_val = check_rsi(df_5m, idx, direction, config)
            if rsi_ok:
                confirmations += 1
            else:
                continue  # RSI always required
            
            sweep_ok = check_sweep(df_5m, idx, direction)
            if sweep_ok:
                confirmations += 1
            elif config.require_sweep:
                continue
            
            if check_structure_break(df_5m, idx, direction):
                confirmations += 1
            else:
                continue  # Structure break always required
            
            candle_ok, pattern = check_candle(df_5m, idx, direction, config)
            if candle_ok:
                confirmations += 1
            else:
                continue  # Candle always required
            
            vol_ok, vol_ratio = check_volume(df_5m, idx, config)
            if vol_ok:
                confirmations += 1
            elif config.require_volume:
                continue
            
            if confirmations < config.min_confirmations:
                continue
            
            # Additional filters
            if not check_ema20_slope(df_5m, idx, direction, config):
                continue
            
            # ENTRY
            entry_price = df_5m['Close'].iloc[idx]
            sl = calculate_sl(df_5m, idx, direction, entry_price)
            sl_distance = abs(entry_price - sl)
            if sl_distance <= 0:
                continue
            
            if direction == 'bullish':
                tp = entry_price + (sl_distance * config.rr_ratio)
            else:
                tp = entry_price - (sl_distance * config.rr_ratio)
            
            trade = Trade(
                symbol=name, direction=direction,
                entry_price=entry_price, stop_loss=sl, take_profit=tp,
                entry_time=str(timestamp), pattern=pattern or "",
                rsi=rsi_val, volume_ratio=vol_ratio,
            )
            
            trade = simulate_trade(df_5m, idx, trade, config)
            trades.append(trade)
            daily_trades[day_key] = daily_trades.get(day_key, 0) + 1
            idx += 10
    
    return trades


def calc_stats(trades: List[Trade], config_name: str) -> dict:
    if not trades:
        return {'name': config_name, 'total': 0, 'valid': False}
    
    total = len(trades)
    wins = [t for t in trades if t.result == 'win']
    losses = [t for t in trades if t.result == 'loss']
    breakevens = [t for t in trades if t.result == 'breakeven']
    
    win_rate = len(wins) / total * 100
    total_r = sum(t.pnl_r for t in trades)
    avg_win = np.mean([t.pnl_r for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t.pnl_r) for t in losses]) if losses else 0
    
    gross_profit = sum(t.pnl_r for t in trades if t.pnl_r > 0)
    gross_loss = abs(sum(t.pnl_r for t in trades if t.pnl_r < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    equity = STARTING_EQUITY
    max_eq = equity
    max_dd = 0
    for t in trades:
        equity += (equity * RISK_PERCENT / 100) * t.pnl_r
        max_eq = max(max_eq, equity)
        dd = (max_eq - equity) / max_eq * 100
        max_dd = max(max_dd, dd)
    
    ret = ((equity - STARTING_EQUITY) / STARTING_EQUITY) * 100
    
    return {
        'name': config_name, 'valid': True,
        'total': total, 'wins': len(wins), 'losses': len(losses), 'be': len(breakevens),
        'win_rate': win_rate, 'total_r': total_r,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'pf': pf, 'expectancy': total_r / total,
        'max_dd': max_dd, 'return': ret, 'equity': equity,
    }


# ============================================================
# STRATEGY VARIATIONS - WIN RATE FOCUS
# ============================================================
def get_wr_configs():
    configs = []
    
    # BASELINE: V5 from previous test
    configs.append(WRConfig(
        name="BASELINE (V5: sweep optional, 2R TP)",
        require_sweep=False,
    ))
    
    # WR1: Lower TP to 1.5R (easier to hit)
    configs.append(WRConfig(
        name="WR1: 1.5R Take Profit (easier target)",
        require_sweep=False,
        rr_ratio=1.5,
    ))
    
    # WR2: Trailing stop (lock in profits, let winners run)
    configs.append(WRConfig(
        name="WR2: Trailing Stop (trigger 0.8R, trail 0.4R)",
        require_sweep=False,
        use_trailing_stop=True,
        trailing_trigger_r=0.8,
        trailing_distance_r=0.4,
        rr_ratio=3.0,  # High TP, but trailing will catch most
    ))
    
    # WR3: Trailing stop tighter
    configs.append(WRConfig(
        name="WR3: Trailing Stop (trigger 1R, trail 0.3R)",
        require_sweep=False,
        use_trailing_stop=True,
        trailing_trigger_r=1.0,
        trailing_distance_r=0.3,
        rr_ratio=3.0,
    ))
    
    # WR4: Strong trend only (steep 4H EMA slope)
    configs.append(WRConfig(
        name="WR4: Strong Trend Filter (steep 4H EMA)",
        require_sweep=False,
        require_strong_trend=True,
        min_4h_slope_pct=0.001,
    ))
    
    # WR5: EMA20 slope must confirm on 5M
    configs.append(WRConfig(
        name="WR5: EMA20 Slope Confirmation on 5M",
        require_sweep=False,
        require_ema20_slope=True,
    ))
    
    # WR6: Higher volume threshold (1.3x)
    configs.append(WRConfig(
        name="WR6: Volume 1.3x Average Required",
        require_sweep=False,
        volume_min_ratio=1.3,
    ))
    
    # WR7: Time exit at 60 bars (5 hours) + 1.5R TP
    configs.append(WRConfig(
        name="WR7: 1.5R TP + Time Exit (5hr max hold)",
        require_sweep=False,
        rr_ratio=1.5,
        use_time_exit=True,
        max_bars_in_trade=60,
    ))
    
    # WR8: Trailing + EMA20 slope
    configs.append(WRConfig(
        name="WR8: Trailing (1R/0.4R) + EMA20 Slope",
        require_sweep=False,
        use_trailing_stop=True,
        trailing_trigger_r=1.0,
        trailing_distance_r=0.4,
        rr_ratio=3.0,
        require_ema20_slope=True,
    ))
    
    # WR9: 1.5R + Strong trend + EMA20 slope
    configs.append(WRConfig(
        name="WR9: 1.5R + Strong Trend + EMA20 Slope",
        require_sweep=False,
        rr_ratio=1.5,
        require_strong_trend=True,
        min_4h_slope_pct=0.0008,
        require_ema20_slope=True,
    ))
    
    # WR10: Best of all - trailing + strong trend + high volume
    configs.append(WRConfig(
        name="WR10: Trailing + Strong Trend + Vol 1.2x",
        require_sweep=False,
        use_trailing_stop=True,
        trailing_trigger_r=0.8,
        trailing_distance_r=0.4,
        rr_ratio=3.0,
        require_strong_trend=True,
        min_4h_slope_pct=0.0008,
        volume_min_ratio=1.2,
    ))
    
    # WR11: Conservative high WR - 1.5R + Vol 1.3x + EMA slope + strong trend
    configs.append(WRConfig(
        name="WR11: MAX WR (1.5R+trend+slope+vol1.3x)",
        require_sweep=False,
        rr_ratio=1.5,
        require_strong_trend=True,
        min_4h_slope_pct=0.0008,
        require_ema20_slope=True,
        volume_min_ratio=1.3,
    ))
    
    # WR12: Trailing + remove bad long patterns
    configs.append(WRConfig(
        name="WR12: Trailing + No strong_bullish",
        require_sweep=False,
        use_trailing_stop=True,
        trailing_trigger_r=0.8,
        trailing_distance_r=0.4,
        rr_ratio=3.0,
        allowed_bull_patterns=['bullish_engulfing', 'hammer', 'bullish_rejection'],
    ))
    
    return configs


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("   WIN RATE OPTIMIZATION - TESTING 12 VARIATIONS")
    print("=" * 70)
    print("   Goal: Increase WR% without killing profitability")
    print("=" * 70)
    
    all_data = fetch_and_prepare()
    
    configs = get_wr_configs()
    results = []
    
    print("\n" + "-" * 70)
    for i, config in enumerate(configs, 1):
        print(f"  [{i:>2}/{len(configs)}] {config.name}")
        trades = run_backtest(all_data, config)
        stats = calc_stats(trades, config.name)
        results.append(stats)
        if stats['valid']:
            print(f"         Trades:{stats['total']:>3} | WR:{stats['win_rate']:>5.1f}% | "
                  f"PF:{stats['pf']:>5.2f} | R:{stats['total_r']:>+6.1f} | "
                  f"${stats['equity']:>9,.0f} | DD:{stats['max_dd']:>5.1f}%")
        else:
            print(f"         NO TRADES")
    
    # Sort by win rate (primary) then total R (secondary)
    valid = [r for r in results if r['valid'] and r['total'] >= 10]
    
    # Create two rankings
    by_winrate = sorted(valid, key=lambda x: (x['win_rate'], x['total_r']), reverse=True)
    by_profit = sorted(valid, key=lambda x: x['total_r'], reverse=True)
    
    # Best "balanced" score: WR * Total_R (maximizes both)
    for r in valid:
        r['score'] = r['win_rate'] * r['total_r'] / 100  # Balanced metric
    by_score = sorted(valid, key=lambda x: x['score'], reverse=True)
    
    print("\n\n")
    print("=" * 95)
    print("   RESULTS RANKED BY WIN RATE")
    print("=" * 95)
    print(f"{'Rk':<3} {'Strategy':<47} {'Trades':<7} {'WR%':<7} {'AvgWin':<7} {'AvgLoss':<8} {'PF':<6} {'TotalR':<8} {'Return':<8} {'DD%':<6}")
    print("-" * 95)
    for rank, r in enumerate(by_winrate, 1):
        n = r['name'][:45]
        print(f"{rank:<3} {n:<47} {r['total']:<7} {r['win_rate']:<7.1f} {r['avg_win']:<7.2f} {r['avg_loss']:<8.2f} "
              f"{r['pf']:<6.2f} {r['total_r']:<+8.1f} {r['return']:<+8.1f} {r['max_dd']:<6.1f}")
    
    print("\n\n")
    print("=" * 95)
    print("   RESULTS RANKED BY BALANCED SCORE (WR% x Total R)")
    print("=" * 95)
    print(f"{'Rk':<3} {'Strategy':<47} {'Trades':<7} {'WR%':<7} {'PF':<6} {'TotalR':<8} {'Return':<8} {'DD%':<6} {'Score':<7}")
    print("-" * 95)
    for rank, r in enumerate(by_score, 1):
        n = r['name'][:45]
        print(f"{rank:<3} {n:<47} {r['total']:<7} {r['win_rate']:<7.1f} {r['pf']:<6.2f} "
              f"{r['total_r']:<+8.1f} {r['return']:<+8.1f} {r['max_dd']:<6.1f} {r['score']:<7.1f}")
    
    # DETAILED TOP PICKS
    print("\n\n")
    print("=" * 70)
    print("   TOP RECOMMENDATIONS")
    print("=" * 70)
    
    baseline = next((r for r in results if 'BASELINE' in r['name']), None)
    
    if baseline and baseline['valid']:
        print(f"\n  BASELINE (your V5):")
        print(f"    WR: {baseline['win_rate']:.1f}% | Trades: {baseline['total']} | "
              f"Total: {baseline['total_r']:+.1f}R | Return: {baseline['return']:+.1f}%")
    
    if by_winrate:
        best_wr = by_winrate[0]
        print(f"\n  HIGHEST WIN RATE: {best_wr['name']}")
        print(f"    WR: {best_wr['win_rate']:.1f}% | Trades: {best_wr['total']} | "
              f"Total: {best_wr['total_r']:+.1f}R | Return: {best_wr['return']:+.1f}%")
        if baseline and baseline['valid']:
            wr_diff = best_wr['win_rate'] - baseline['win_rate']
            r_diff = best_wr['total_r'] - baseline['total_r']
            print(f"    vs Baseline: WR {wr_diff:+.1f}% | Profit {r_diff:+.1f}R")
    
    if by_score:
        best_bal = by_score[0]
        print(f"\n  BEST BALANCED (high WR + high profit): {best_bal['name']}")
        print(f"    WR: {best_bal['win_rate']:.1f}% | Trades: {best_bal['total']} | "
              f"Total: {best_bal['total_r']:+.1f}R | Return: {best_bal['return']:+.1f}%")
        if baseline and baseline['valid']:
            wr_diff = best_bal['win_rate'] - baseline['win_rate']
            r_diff = best_bal['total_r'] - baseline['total_r']
            print(f"    vs Baseline: WR {wr_diff:+.1f}% | Profit {r_diff:+.1f}R")
    
    # Dollar comparison
    print("\n\n")
    print("=" * 70)
    print("   DOLLAR COMPARISON ($100 RISK PER TRADE)")
    print("=" * 70)
    
    if baseline and baseline['valid']:
        print(f"\n  {'Strategy':<40} {'WR%':<7} {'Trades':<7} {'Profit':<10} {'$/Trade':<8}")
        print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*10} {'-'*8}")
        
        for r in by_score[:5]:
            profit = r['total_r'] * 100  # $100 risk per trade
            per_trade = profit / r['total'] if r['total'] > 0 else 0
            n = r['name'][:38]
            print(f"  {n:<40} {r['win_rate']:<7.1f} {r['total']:<7} ${profit:<9,.0f} ${per_trade:<7.1f}")
        
        print(f"\n  {'BASELINE (V5)':<40} {baseline['win_rate']:<7.1f} {baseline['total']:<7} "
              f"${baseline['total_r']*100:<9,.0f} ${baseline['total_r']*100/baseline['total']:<7.1f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
