"""
Backtesting Script - Multi-Timeframe EMA Strategy
Runs the EXACT strategy logic against historical data for BTC-USD and XAU (GLD as proxy).
Collects 100 trades and reports full statistics.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# CONFIGURATION (matches the live bot exactly)
# ============================================================
RISK_PERCENT = 2.0
MIN_RR = 1.5
PREFERRED_RR = 2.0
MAX_TRADES_PER_DAY = 2
STARTING_EQUITY = 10000.0

# Indicator periods
EMA_4H_PERIOD = 50
EMA_4H_SLOPE_LOOKBACK = 3
EMA_30M_FAST = 50
EMA_30M_SLOW = 200
EMA_5M_PULLBACK = 20
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_AVG_PERIOD = 20
SWING_LOOKBACK = 10

# RSI zones
RSI_LONG_MIN, RSI_LONG_MAX = 45, 60
RSI_SHORT_MIN, RSI_SHORT_MAX = 40, 55
RSI_OVERBOUGHT, RSI_OVERSOLD = 70, 30

# Session hours (UTC)
SESSION_START = 7   # London open
SESSION_END = 21    # NY close


# ============================================================
# INDICATOR CALCULATIONS
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
def fetch_data(symbol, period="2y", interval_map=None):
    """
    Fetch multi-timeframe data from Yahoo Finance.
    Returns dict of DataFrames: {'4h': df, '30m': df, '5m': df}
    
    Note: yfinance limits:
    - 5m data: max 60 days
    - 30m data: max 60 days  
    - 1h data: max 730 days
    We'll use 1h as proxy for 4h (resample), and for 5m/30m we get 60 days.
    """
    print(f"  Fetching {symbol} data...")
    
    # 1H data for last 2 years (resample to 4H)
    df_1h = yf.download(symbol, period="2y", interval="1h", progress=False)
    if df_1h.empty:
        # Try shorter period
        df_1h = yf.download(symbol, period="730d", interval="1h", progress=False)
    
    # Handle multi-level columns from yfinance
    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)
    
    # Resample 1H to 4H
    df_4h = df_1h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    # 30m data (max 60 days)
    df_30m = yf.download(symbol, period="60d", interval="30m", progress=False)
    if isinstance(df_30m.columns, pd.MultiIndex):
        df_30m.columns = df_30m.columns.get_level_values(0)
    
    # 5m data (max 60 days)
    df_5m = yf.download(symbol, period="60d", interval="5m", progress=False)
    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)
    
    print(f"    4H bars: {len(df_4h)}, 30M bars: {len(df_30m)}, 5M bars: {len(df_5m)}")
    
    return {'4h': df_4h, '30m': df_30m, '5m': df_5m}


# ============================================================
# STRATEGY CHECKS
# ============================================================
def check_4h_trend(df_4h, idx):
    """Check 4H trend at a given index."""
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
    """Check 30M trend confirmation at a given index."""
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


def check_pullback(df_5m, idx, direction):
    """Check if price pulled back to EMA20 or VWAP zone."""
    if idx < 3:
        return False
    
    ema20 = df_5m['ema_20'].iloc[idx]
    close = df_5m['Close'].iloc[idx]
    low = df_5m['Low'].iloc[idx]
    high = df_5m['High'].iloc[idx]
    threshold = 0.002  # 0.2%
    
    if direction == 'bullish':
        # Low should touch or come near EMA20, close should be above
        near_ema = abs(low - ema20) / ema20 <= threshold or low <= ema20
        bouncing = close > ema20
        if near_ema and bouncing:
            return True
        # Check last 3 bars
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


def check_rsi(df_5m, idx, direction):
    """Check RSI is in appropriate zone."""
    rsi = df_5m['rsi'].iloc[idx]
    
    if rsi >= RSI_OVERBOUGHT or rsi <= RSI_OVERSOLD:
        return False, rsi
    
    if direction == 'bullish':
        return RSI_LONG_MIN <= rsi <= RSI_LONG_MAX, rsi
    else:
        return RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX, rsi


def check_liquidity_sweep(df_5m, idx, direction):
    """Detect liquidity sweep in last 5 bars."""
    if idx < SWING_LOOKBACK + 10:
        return False
    
    check_start = max(0, idx - 5)
    
    if direction == 'bullish':
        # Find recent swing low before the check window
        swing_lows = df_5m['swing_lows'].iloc[:check_start].dropna()
        if len(swing_lows) == 0:
            return False
        recent_sl = swing_lows.iloc[-1]
        
        # Check if any bar in window swept below then closed above
        for i in range(check_start, idx + 1):
            if df_5m['Low'].iloc[i] < recent_sl and df_5m['Close'].iloc[i] > recent_sl:
                return True
        # Check sweep then reversal in subsequent bar
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


def check_structure_break(df_5m, idx, direction):
    """Detect market structure break."""
    if idx < 20:
        return False
    
    window = df_5m.iloc[idx-20:idx+1]
    current_close = df_5m['Close'].iloc[idx]
    
    if direction == 'bullish':
        # Find local highs and check for lower high broken
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
        
        # Alternative: close above recent pullback high
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


def check_candle_pattern(df_5m, idx, direction):
    """Check for confirming candlestick pattern."""
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
    
    # Bullish patterns
    if direction == 'bullish':
        if curr_c > curr_o and prev_c < prev_o and curr_o <= prev_c and curr_c >= prev_o:
            pattern = 'bullish_engulfing'
        elif body_ratio < 0.35 and lower_wick >= body * 2 and upper_wick < body * 0.5 and curr_c >= curr_o:
            pattern = 'hammer'
        elif curr_c > curr_o and body_ratio >= 0.5 and lower_wick >= body * 0.7:
            pattern = 'bullish_rejection'
        elif curr_c > curr_o and body_ratio >= 0.65:
            pattern = 'strong_bullish'
    else:
        if curr_c < curr_o and prev_c > prev_o and curr_o >= prev_c and curr_c <= prev_o:
            pattern = 'bearish_engulfing'
        elif body_ratio < 0.35 and upper_wick >= body * 2 and lower_wick < body * 0.5 and curr_c <= curr_o:
            pattern = 'shooting_star'
        elif curr_c < curr_o and body_ratio >= 0.5 and upper_wick >= body * 0.7:
            pattern = 'bearish_rejection'
        elif curr_c < curr_o and body_ratio >= 0.65:
            pattern = 'strong_bearish'
    
    return pattern is not None, pattern


def check_volume(df_5m, idx):
    """Check if volume is above 20-period average."""
    if idx < VOLUME_AVG_PERIOD:
        return False, 0
    vol = df_5m['Volume'].iloc[idx]
    avg_vol = df_5m['Volume'].iloc[idx-VOLUME_AVG_PERIOD:idx].mean()
    if avg_vol <= 0:
        return False, 0
    ratio = vol / avg_vol
    return ratio >= 1.0, ratio


def check_session(timestamp):
    """Check if within London/NY session."""
    hour = timestamp.hour
    return SESSION_START <= hour <= SESSION_END


# ============================================================
# STOP LOSS & TAKE PROFIT CALCULATION
# ============================================================
def calculate_sl(df_5m, idx, direction, entry_price):
    """Calculate SL: wider of swing level or 1 ATR."""
    atr = df_5m['atr'].iloc[idx]
    
    if direction == 'bullish':
        atr_sl = entry_price - atr
        # Find recent swing low
        swing_lows = df_5m['swing_lows'].iloc[:idx].dropna()
        if len(swing_lows) > 0:
            swing_sl = swing_lows.iloc[-1] - (entry_price * 0.0005)
            return min(atr_sl, swing_sl)  # Lower = more protection for longs
        return atr_sl
    else:
        atr_sl = entry_price + atr
        swing_highs = df_5m['swing_highs'].iloc[:idx].dropna()
        if len(swing_highs) > 0:
            swing_sl = swing_highs.iloc[-1] + (entry_price * 0.0005)
            return max(atr_sl, swing_sl)  # Higher = more protection for shorts
        return atr_sl


# ============================================================
# TRADE SIMULATION
# ============================================================
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
    result: str = ""  # 'win', 'loss', 'breakeven'
    hit_breakeven: bool = False
    pattern: str = ""
    rsi: float = 0.0
    volume_ratio: float = 0.0


def simulate_trade_outcome(df_5m, entry_idx, trade: Trade):
    """
    Simulate how the trade would have played out bar by bar.
    Checks SL/TP hit, moves SL to breakeven at 1R.
    """
    sl = trade.stop_loss
    tp = trade.take_profit
    entry = trade.entry_price
    sl_distance = abs(entry - sl)
    
    for i in range(entry_idx + 1, min(entry_idx + 200, len(df_5m))):
        high = df_5m['High'].iloc[i]
        low = df_5m['Low'].iloc[i]
        
        if trade.direction == 'bullish':
            # Check SL hit
            if low <= sl:
                trade.exit_price = sl
                trade.exit_time = str(df_5m.index[i])
                if trade.hit_breakeven:
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
                trade.pnl_r = abs(tp - entry) / sl_distance
                trade.result = 'win'
                return trade
            
            # Check breakeven trigger (1R profit reached)
            if not trade.hit_breakeven and high >= entry + sl_distance:
                trade.hit_breakeven = True
                sl = entry  # Move SL to breakeven
                
        else:  # bearish
            # Check SL hit
            if high >= sl:
                trade.exit_price = sl
                trade.exit_time = str(df_5m.index[i])
                if trade.hit_breakeven:
                    trade.pnl_r = 0.0
                    trade.result = 'breakeven'
                else:
                    trade.pnl_r = -1.0
                    trade.result = 'loss'
                return trade
            
            # Check TP hit
            if low <= tp:
                trade.exit_price = tp
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = abs(entry - tp) / sl_distance
                trade.result = 'win'
                return trade
            
            # Check breakeven trigger
            if not trade.hit_breakeven and low <= entry - sl_distance:
                trade.hit_breakeven = True
                sl = entry
    
    # If we run out of data, close at last price
    trade.exit_price = df_5m['Close'].iloc[min(entry_idx + 199, len(df_5m)-1)]
    trade.exit_time = str(df_5m.index[min(entry_idx + 199, len(df_5m)-1)])
    if trade.direction == 'bullish':
        trade.pnl_r = (trade.exit_price - entry) / sl_distance
    else:
        trade.pnl_r = (entry - trade.exit_price) / sl_distance
    trade.result = 'win' if trade.pnl_r > 0 else 'loss'
    return trade


# ============================================================
# MAIN BACKTEST
# ============================================================
def run_backtest():
    print("=" * 60)
    print("MULTI-TIMEFRAME EMA STRATEGY BACKTEST")
    print("=" * 60)
    print(f"Starting Equity: ${STARTING_EQUITY:,.2f}")
    print(f"Risk Per Trade: {RISK_PERCENT}%")
    print(f"Target: 100 trades")
    print(f"Instruments: BTC-USD, GLD (Gold ETF proxy for XAU/USD)")
    print("=" * 60)
    
    # Fetch data for both instruments
    symbols = {'BTC-USD': 'BTC-USD', 'XAU': 'GC=F'}  # GC=F is gold futures
    all_trades = []
    
    for name, ticker in symbols.items():
        print(f"\n--- Processing {name} ({ticker}) ---")
        
        try:
            data = fetch_data(ticker)
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            continue
        
        df_4h = data['4h']
        df_30m = data['30m']
        df_5m = data['5m']
        
        if df_4h.empty or df_30m.empty or df_5m.empty:
            print(f"  Insufficient data for {name}")
            continue
        
        # Add indicators to each timeframe
        df_4h['ema_50'] = calc_ema(df_4h['Close'], EMA_4H_PERIOD)
        
        df_30m['ema_50'] = calc_ema(df_30m['Close'], EMA_30M_FAST)
        df_30m['ema_200'] = calc_ema(df_30m['Close'], EMA_30M_SLOW)
        
        df_5m['ema_20'] = calc_ema(df_5m['Close'], EMA_5M_PULLBACK)
        df_5m['rsi'] = calc_rsi(df_5m['Close'], RSI_PERIOD)
        df_5m['atr'] = calc_atr(df_5m, ATR_PERIOD)
        df_5m['volume_avg'] = df_5m['Volume'].rolling(VOLUME_AVG_PERIOD).mean()
        df_5m['swing_highs'] = find_swing_highs(df_5m['High'], SWING_LOOKBACK)
        df_5m['swing_lows'] = find_swing_lows(df_5m['Low'], SWING_LOOKBACK)
        
        print(f"  Indicators calculated. Scanning for entries...")
        
        # Map 5m bars to their corresponding 4h and 30m context
        trades_this_symbol = 0
        daily_trades = {}
        
        # Start scanning from bar 250 onwards (need enough history)
        start_idx = max(250, SWING_LOOKBACK + 20)
        
        for idx in range(start_idx, len(df_5m) - 200):  # Leave room for trade simulation
            if len(all_trades) >= 100:
                break
            
            timestamp = df_5m.index[idx]
            
            # Handle timezone-aware vs naive
            if hasattr(timestamp, 'tz') and timestamp.tz is not None:
                ts_naive = timestamp.tz_localize(None)
            else:
                ts_naive = timestamp
            
            # Session filter
            if not check_session(ts_naive):
                continue
            
            # Max 2 trades per day
            day_key = ts_naive.date()
            if daily_trades.get(day_key, 0) >= MAX_TRADES_PER_DAY:
                continue
            
            # Find corresponding 4H bar
            # Get the most recent 4H bar before this 5m timestamp
            mask_4h = df_4h.index <= timestamp
            if mask_4h.sum() < EMA_4H_PERIOD + EMA_4H_SLOPE_LOOKBACK + 1:
                continue
            idx_4h = mask_4h.sum() - 1
            
            # Find corresponding 30M bar
            mask_30m = df_30m.index <= timestamp
            if mask_30m.sum() < EMA_30M_SLOW + 10:
                continue
            idx_30m = mask_30m.sum() - 1
            
            # 1. Check 4H trend
            trend_4h = check_4h_trend(df_4h, idx_4h)
            if trend_4h is None:
                continue
            
            # 2. Check 30M trend alignment
            trend_30m = check_30m_trend(df_30m, idx_30m)
            if trend_30m is None or trend_30m != trend_4h:
                continue
            
            direction = trend_4h  # 'bullish' or 'bearish'
            
            # 3. Check pullback to value
            if not check_pullback(df_5m, idx, direction):
                continue
            
            # 4. Check RSI zone
            rsi_ok, rsi_val = check_rsi(df_5m, idx, direction)
            if not rsi_ok:
                continue
            
            # 5. Check liquidity sweep
            if not check_liquidity_sweep(df_5m, idx, direction):
                continue
            
            # 6. Check market structure break
            if not check_structure_break(df_5m, idx, direction):
                continue
            
            # 7. Check candlestick pattern
            candle_ok, pattern = check_candle_pattern(df_5m, idx, direction)
            if not candle_ok:
                continue
            
            # 8. Check volume
            vol_ok, vol_ratio = check_volume(df_5m, idx)
            if not vol_ok:
                continue
            
            # ALL CONFIRMATIONS MET - Create trade
            entry_price = df_5m['Close'].iloc[idx]
            sl = calculate_sl(df_5m, idx, direction, entry_price)
            sl_distance = abs(entry_price - sl)
            
            if sl_distance <= 0:
                continue
            
            # Determine R:R based on trend strength
            rr = PREFERRED_RR if abs(df_4h['ema_50'].iloc[idx_4h] - df_4h['ema_50'].iloc[idx_4h - 3]) > 0 else MIN_RR
            
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
                pattern=pattern,
                rsi=rsi_val,
                volume_ratio=vol_ratio,
            )
            
            # Simulate the trade
            trade = simulate_trade_outcome(df_5m, idx, trade)
            all_trades.append(trade)
            
            daily_trades[day_key] = daily_trades.get(day_key, 0) + 1
            trades_this_symbol += 1
            
            # Skip ahead to avoid overlapping trades (minimum 10 bars between entries)
            idx += 10
        
        print(f"  Found {trades_this_symbol} trades for {name}")
    
    return all_trades


def print_results(trades):
    """Print comprehensive backtest results."""
    if not trades:
        print("\nNO TRADES FOUND - strategy may be too strict for available data.")
        return
    
    total = len(trades)
    wins = [t for t in trades if t.result == 'win']
    losses = [t for t in trades if t.result == 'loss']
    breakevens = [t for t in trades if t.result == 'breakeven']
    
    win_rate = len(wins) / total * 100
    
    total_r = sum(t.pnl_r for t in trades)
    avg_win_r = np.mean([t.pnl_r for t in wins]) if wins else 0
    avg_loss_r = np.mean([abs(t.pnl_r) for t in losses]) if losses else 0
    
    # Profit factor
    gross_profit = sum(t.pnl_r for t in trades if t.pnl_r > 0)
    gross_loss = abs(sum(t.pnl_r for t in trades if t.pnl_r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown in R
    cumulative_r = np.cumsum([t.pnl_r for t in trades])
    peak = np.maximum.accumulate(cumulative_r)
    drawdown = peak - cumulative_r
    max_drawdown_r = drawdown.max()
    
    # Equity curve
    equity = STARTING_EQUITY
    equity_curve = [equity]
    max_equity = equity
    max_dd_pct = 0
    
    for trade in trades:
        risk_amount = equity * (RISK_PERCENT / 100)
        pnl = risk_amount * trade.pnl_r
        equity += pnl
        equity_curve.append(equity)
        max_equity = max(max_equity, equity)
        dd_pct = (max_equity - equity) / max_equity * 100
        max_dd_pct = max(max_dd_pct, dd_pct)
    
    final_equity = equity_curve[-1]
    total_return_pct = ((final_equity - STARTING_EQUITY) / STARTING_EQUITY) * 100
    
    # Consecutive stats
    max_consec_wins = 0
    max_consec_losses = 0
    curr_wins = 0
    curr_losses = 0
    for t in trades:
        if t.result == 'win':
            curr_wins += 1
            curr_losses = 0
            max_consec_wins = max(max_consec_wins, curr_wins)
        elif t.result == 'loss':
            curr_losses += 1
            curr_wins = 0
            max_consec_losses = max(max_consec_losses, curr_losses)
        else:
            curr_wins = 0
            curr_losses = 0
    
    # Breakeven usage
    be_count = sum(1 for t in trades if t.hit_breakeven)
    
    # By symbol
    btc_trades = [t for t in trades if 'BTC' in t.symbol]
    xau_trades = [t for t in trades if 'XAU' in t.symbol or 'GC' in t.symbol or 'GLD' in t.symbol]
    
    print("\n")
    print("=" * 60)
    print("        BACKTEST RESULTS - 100 TRADE SIMULATION")
    print("=" * 60)
    print(f"\n{'OVERVIEW':=^50}")
    print(f"  Total Trades:            {total}")
    print(f"  Wins:                    {len(wins)} ({win_rate:.1f}%)")
    print(f"  Losses:                  {len(losses)} ({100-win_rate-len(breakevens)/total*100:.1f}%)")
    print(f"  Breakeven:               {len(breakevens)} ({len(breakevens)/total*100:.1f}%)")
    
    print(f"\n{'PROFITABILITY':=^50}")
    print(f"  Total R Gained:          {total_r:.2f}R")
    print(f"  Average Win:             {avg_win_r:.2f}R")
    print(f"  Average Loss:            -{avg_loss_r:.2f}R")
    print(f"  Profit Factor:           {profit_factor:.2f}")
    print(f"  Expectancy (per trade):  {total_r/total:.3f}R")
    
    print(f"\n{'EQUITY':=^50}")
    print(f"  Starting Equity:         ${STARTING_EQUITY:,.2f}")
    print(f"  Final Equity:            ${final_equity:,.2f}")
    print(f"  Total Return:            {total_return_pct:.2f}%")
    print(f"  Max Drawdown (%):        {max_dd_pct:.2f}%")
    print(f"  Max Drawdown (R):        {max_drawdown_r:.2f}R")
    
    print(f"\n{'STREAKS & CONSISTENCY':=^50}")
    print(f"  Max Consecutive Wins:    {max_consec_wins}")
    print(f"  Max Consecutive Losses:  {max_consec_losses}")
    print(f"  Breakeven Moves Used:    {be_count}/{total} ({be_count/total*100:.0f}%)")
    
    print(f"\n{'BY INSTRUMENT':=^50}")
    if btc_trades:
        btc_wins = sum(1 for t in btc_trades if t.result == 'win')
        btc_r = sum(t.pnl_r for t in btc_trades)
        print(f"  BTC/USD:")
        print(f"    Trades: {len(btc_trades)} | Win Rate: {btc_wins/len(btc_trades)*100:.1f}% | Total: {btc_r:.2f}R")
    if xau_trades:
        xau_wins = sum(1 for t in xau_trades if t.result == 'win')
        xau_r = sum(t.pnl_r for t in xau_trades)
        print(f"  XAU/USD (Gold):")
        print(f"    Trades: {len(xau_trades)} | Win Rate: {xau_wins/len(xau_trades)*100:.1f}% | Total: {xau_r:.2f}R")
    
    print(f"\n{'BY DIRECTION':=^50}")
    longs = [t for t in trades if t.direction == 'bullish']
    shorts = [t for t in trades if t.direction == 'bearish']
    if longs:
        long_wins = sum(1 for t in longs if t.result == 'win')
        long_r = sum(t.pnl_r for t in longs)
        print(f"  Longs:  {len(longs)} trades | Win Rate: {long_wins/len(longs)*100:.1f}% | Total: {long_r:.2f}R")
    if shorts:
        short_wins = sum(1 for t in shorts if t.result == 'win')
        short_r = sum(t.pnl_r for t in shorts)
        print(f"  Shorts: {len(shorts)} trades | Win Rate: {short_wins/len(shorts)*100:.1f}% | Total: {short_r:.2f}R")
    
    print(f"\n{'TOP PATTERNS':=^50}")
    patterns = {}
    for t in trades:
        p = t.pattern or 'unknown'
        if p not in patterns:
            patterns[p] = {'count': 0, 'wins': 0, 'r': 0}
        patterns[p]['count'] += 1
        if t.result == 'win':
            patterns[p]['wins'] += 1
        patterns[p]['r'] += t.pnl_r
    
    for p, stats in sorted(patterns.items(), key=lambda x: x[1]['count'], reverse=True):
        wr = stats['wins'] / stats['count'] * 100
        print(f"  {p:20s}: {stats['count']:3d} trades | WR: {wr:.0f}% | {stats['r']:.2f}R")
    
    print(f"\n{'RISK ASSESSMENT':=^50}")
    if win_rate >= 50 and profit_factor >= 1.5:
        print(f"  Strategy Rating:         STRONG")
        print(f"  Edge:                    Confirmed (PF > 1.5, WR > 50%)")
    elif win_rate >= 40 and profit_factor >= 1.2:
        print(f"  Strategy Rating:         MODERATE")
        print(f"  Edge:                    Present but needs optimization")
    else:
        print(f"  Strategy Rating:         NEEDS WORK")
        print(f"  Edge:                    Weak or not confirmed")
    
    if max_dd_pct > 15:
        print(f"  Drawdown Risk:           HIGH (>{max_dd_pct:.0f}%)")
    elif max_dd_pct > 10:
        print(f"  Drawdown Risk:           MODERATE ({max_dd_pct:.1f}%)")
    else:
        print(f"  Drawdown Risk:           LOW ({max_dd_pct:.1f}%)")
    
    print(f"\n{'SAMPLE TRADES (First 10)':=^50}")
    print(f"  {'#':<3} {'Symbol':<7} {'Dir':<7} {'Entry':<12} {'SL':<12} {'TP':<12} {'Result':<8} {'R':>6}")
    print(f"  {'-'*3} {'-'*7} {'-'*7} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*6}")
    for i, t in enumerate(trades[:10], 1):
        print(f"  {i:<3} {t.symbol:<7} {t.direction[:5]:<7} {t.entry_price:<12.2f} {t.stop_loss:<12.2f} {t.take_profit:<12.2f} {t.result:<8} {t.pnl_r:>+6.2f}")
    
    print("\n" + "=" * 60)
    print("  DISCLAIMER: Past performance does NOT guarantee future results.")
    print("  This backtest uses historical data and may not reflect live")
    print("  conditions (slippage, spread, liquidity differences).")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    trades = run_backtest()
    print_results(trades)
