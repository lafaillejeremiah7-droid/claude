"""
FINAL OPTIMIZATION - Two Levers

Lever 2 (Free Money): Early breakeven stop
- Move SL to breakeven (or +0.1R) once trade hits +0.4R to +0.5R
- This happens BEFORE the trailing stop triggers at 1R
- Converts "almost-winners-that-reversed" from -1R losses into scratches

Lever 1 (Entry Quality): Analyze losing trades for filterable clusters
- Tag all 45 losers by: time of day, ATR percentile, distance from EMA20,
  slope strength, higher TF trend condition
- Find if losses cluster on any dimension
- Only add a filter if it makes structural market sense

This script:
1. Runs WR8 with various early-breakeven thresholds
2. Finds the best one
3. Then analyzes losing trades from that best version to find clusters
4. Tests whether any cluster-based filter improves results without curve-fitting
"""
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import List, Optional
from collections import Counter

# ============================================================
# CONFIG
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
class Config:
    name: str = ""
    # Base WR8 settings
    rsi_long_min: float = 40
    rsi_long_max: float = 65
    rsi_short_min: float = 35
    rsi_short_max: float = 60
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    require_sweep: bool = False
    require_volume: bool = True
    volume_min_ratio: float = 1.0
    require_ema20_slope: bool = True  # WR8 addition
    pullback_threshold: float = 0.002
    max_trades_per_day: int = 2
    allowed_bull_patterns: list = field(default_factory=lambda: [
        'bullish_engulfing', 'hammer', 'bullish_rejection', 'strong_bullish'
    ])
    allowed_bear_patterns: list = field(default_factory=lambda: [
        'bearish_engulfing', 'shooting_star', 'bearish_rejection', 'strong_bearish'
    ])
    
    # Trailing stop (WR8)
    use_trailing: bool = True
    trail_trigger_r: float = 1.0
    trail_distance_r: float = 0.4
    rr_target: float = 3.0  # High TP, trailing catches most
    
    # LEVER 2: Early breakeven
    use_early_be: bool = False
    early_be_trigger_r: float = 0.4  # Move to BE when +0.4R
    early_be_level_r: float = 0.0    # 0 = exact entry, 0.1 = +0.1R (covers fees)
    
    # LEVER 1: Entry filters
    min_atr_percentile: float = 0.0     # 0 = no filter
    max_ema20_distance_pct: float = 1.0  # 1.0 = no filter (100%)
    min_4h_slope_strength: float = 0.0   # 0 = no filter


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
    pattern: str = ""
    rsi: float = 0.0
    volume_ratio: float = 0.0
    
    # Analysis tags
    hour_of_day: int = 0
    atr_value: float = 0.0
    atr_percentile: float = 0.0
    ema20_distance_pct: float = 0.0
    slope_4h_strength: float = 0.0
    slope_5m_strength: float = 0.0
    max_favorable_r: float = 0.0  # How far trade went in your favor before failing
    
    # Exit tracking
    hit_early_be: bool = False
    hit_trailing: bool = False


# ============================================================
# INDICATORS
# ============================================================
def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    ag = g.ewm(alpha=1.0/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1.0/p, min_periods=p, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def calc_atr(df, p=14):
    h, l, cp = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([h-l, (h-cp).abs(), (l-cp).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def find_swings(series, lookback, is_high=True):
    result = pd.Series(np.nan, index=series.index)
    for i in range(lookback, len(series) - lookback):
        if is_high:
            if series.iloc[i] >= series.iloc[i-lookback:i].max() and series.iloc[i] >= series.iloc[i+1:i+lookback+1].max():
                result.iloc[i] = series.iloc[i]
        else:
            if series.iloc[i] <= series.iloc[i-lookback:i].min() and series.iloc[i] <= series.iloc[i+1:i+lookback+1].min():
                result.iloc[i] = series.iloc[i]
    return result


# ============================================================
# DATA
# ============================================================
def fetch_and_prepare():
    print("Fetching market data...")
    all_data = {}
    for name, ticker in [('BTC-USD', 'BTC-USD'), ('XAU', 'GC=F')]:
        print(f"  {name}...")
        df_1h = yf.download(ticker, period="2y", interval="1h", progress=False)
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.get_level_values(0)
        df_4h = df_1h.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        
        df_30m = yf.download(ticker, period="60d", interval="30m", progress=False)
        if isinstance(df_30m.columns, pd.MultiIndex):
            df_30m.columns = df_30m.columns.get_level_values(0)
        
        df_5m = yf.download(ticker, period="60d", interval="5m", progress=False)
        if isinstance(df_5m.columns, pd.MultiIndex):
            df_5m.columns = df_5m.columns.get_level_values(0)
        
        df_4h['ema_50'] = calc_ema(df_4h['Close'], EMA_4H_PERIOD)
        df_4h['ema_50_slope'] = df_4h['ema_50'].diff(EMA_4H_SLOPE_LOOKBACK)
        df_4h['ema_50_slope_pct'] = df_4h['ema_50_slope'] / df_4h['ema_50']
        
        df_30m['ema_50'] = calc_ema(df_30m['Close'], EMA_30M_FAST)
        df_30m['ema_200'] = calc_ema(df_30m['Close'], EMA_30M_SLOW)
        
        df_5m['ema_20'] = calc_ema(df_5m['Close'], EMA_5M_PULLBACK)
        df_5m['ema_20_slope'] = df_5m['ema_20'].diff(3)
        df_5m['rsi'] = calc_rsi(df_5m['Close'], RSI_PERIOD)
        df_5m['atr'] = calc_atr(df_5m, ATR_PERIOD)
        df_5m['atr_percentile'] = df_5m['atr'].rolling(200).rank(pct=True)
        df_5m['volume_avg'] = df_5m['Volume'].rolling(VOLUME_AVG_PERIOD).mean()
        df_5m['swing_highs'] = find_swings(df_5m['High'], SWING_LOOKBACK, True)
        df_5m['swing_lows'] = find_swings(df_5m['Low'], SWING_LOOKBACK, False)
        df_5m['ema20_dist_pct'] = ((df_5m['Close'] - df_5m['ema_20']) / df_5m['ema_20']).abs()
        
        all_data[name] = {'4h': df_4h, '30m': df_30m, '5m': df_5m}
        print(f"    4H={len(df_4h)} | 30M={len(df_30m)} | 5M={len(df_5m)}")
    return all_data


# ============================================================
# ENTRY CHECKS (same as WR8)
# ============================================================
def check_4h(df_4h, idx, config):
    if idx < EMA_4H_PERIOD + EMA_4H_SLOPE_LOOKBACK:
        return None, 0
    ema50 = df_4h['ema_50'].iloc[idx]
    slope = df_4h['ema_50_slope'].iloc[idx]
    slope_pct = abs(df_4h['ema_50_slope_pct'].iloc[idx]) if not pd.isna(df_4h['ema_50_slope_pct'].iloc[idx]) else 0
    price = df_4h['Close'].iloc[idx]
    
    if config.min_4h_slope_strength > 0 and slope_pct < config.min_4h_slope_strength:
        return None, slope_pct
    
    if slope > 0 and price > ema50:
        return 'bullish', slope_pct
    elif slope < 0 and price < ema50:
        return 'bearish', slope_pct
    return None, slope_pct

def check_30m(df_30m, idx):
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

def check_pullback(df, idx, direction, config):
    if idx < 3:
        return False
    ema20 = df['ema_20'].iloc[idx]
    close, low, high = df['Close'].iloc[idx], df['Low'].iloc[idx], df['High'].iloc[idx]
    t = config.pullback_threshold
    if direction == 'bullish':
        if (abs(low - ema20)/ema20 <= t or low <= ema20) and close > ema20:
            return True
        for i in range(max(0,idx-3), idx):
            if df['Low'].iloc[i] <= ema20*(1+t) and df['Close'].iloc[i] > ema20:
                return True
    else:
        if (abs(high - ema20)/ema20 <= t or high >= ema20) and close < ema20:
            return True
        for i in range(max(0,idx-3), idx):
            if df['High'].iloc[i] >= ema20*(1-t) and df['Close'].iloc[i] < ema20:
                return True
    return False

def check_rsi(df, idx, direction, config):
    rsi = df['rsi'].iloc[idx]
    if rsi >= config.rsi_overbought or rsi <= config.rsi_oversold:
        return False, rsi
    if direction == 'bullish':
        return config.rsi_long_min <= rsi <= config.rsi_long_max, rsi
    else:
        return config.rsi_short_min <= rsi <= config.rsi_short_max, rsi

def check_structure(df, idx, direction):
    if idx < 20:
        return False
    w = df.iloc[idx-20:idx+1]
    cc = df['Close'].iloc[idx]
    if direction == 'bullish':
        lh = []
        for i in range(2, len(w)-1):
            if w['High'].iloc[i] > w['High'].iloc[i-1] and w['High'].iloc[i] > w['High'].iloc[i+1]:
                lh.append(w['High'].iloc[i])
        if len(lh) >= 2:
            for j in range(len(lh)-1, 0, -1):
                if lh[j] < lh[j-1]:
                    if cc > lh[j]:
                        return True
                    break
        rh = w['High'].iloc[-6:-1]
        if len(rh) > 0 and cc > rh.max():
            return True
    else:
        ll = []
        for i in range(2, len(w)-1):
            if w['Low'].iloc[i] < w['Low'].iloc[i-1] and w['Low'].iloc[i] < w['Low'].iloc[i+1]:
                ll.append(w['Low'].iloc[i])
        if len(ll) >= 2:
            for j in range(len(ll)-1, 0, -1):
                if ll[j] > ll[j-1]:
                    if cc < ll[j]:
                        return True
                    break
        rl = w['Low'].iloc[-6:-1]
        if len(rl) > 0 and cc < rl.min():
            return True
    return False

def check_candle(df, idx, direction, config):
    if idx < 1:
        return False, None
    co, cc = df['Open'].iloc[idx], df['Close'].iloc[idx]
    ch, cl = df['High'].iloc[idx], df['Low'].iloc[idx]
    po, pc = df['Open'].iloc[idx-1], df['Close'].iloc[idx-1]
    body = abs(cc - co)
    rng = ch - cl
    if rng == 0:
        return False, None
    br = body / rng
    uw = ch - max(co, cc)
    lw = min(co, cc) - cl
    
    pattern = None
    if direction == 'bullish':
        if cc > co and pc < po and co <= pc and cc >= po:
            pattern = 'bullish_engulfing'
        elif br < 0.35 and body > 0 and lw >= body*2 and uw < body*0.5 and cc >= co:
            pattern = 'hammer'
        elif cc > co and br >= 0.5 and lw >= body*0.7:
            pattern = 'bullish_rejection'
        elif cc > co and br >= 0.65:
            pattern = 'strong_bullish'
        if pattern and pattern in config.allowed_bull_patterns:
            return True, pattern
    else:
        if cc < co and pc > po and co >= pc and cc <= po:
            pattern = 'bearish_engulfing'
        elif br < 0.35 and body > 0 and uw >= body*2 and lw < body*0.5 and cc <= co:
            pattern = 'shooting_star'
        elif cc < co and br >= 0.5 and uw >= body*0.7:
            pattern = 'bearish_rejection'
        elif cc < co and br >= 0.65:
            pattern = 'strong_bearish'
        if pattern and pattern in config.allowed_bear_patterns:
            return True, pattern
    return False, pattern

def check_volume(df, idx, config):
    if idx < VOLUME_AVG_PERIOD:
        return False, 0
    v = df['Volume'].iloc[idx]
    a = df['volume_avg'].iloc[idx]
    if a <= 0:
        return False, 0
    r = v / a
    return r >= config.volume_min_ratio, r

def calc_sl(df, idx, direction, entry):
    atr = df['atr'].iloc[idx]
    if direction == 'bullish':
        atr_sl = entry - atr
        sw = df['swing_lows'].iloc[:idx].dropna()
        if len(sw) > 0:
            return min(atr_sl, sw.iloc[-1] - entry*0.0005)
        return atr_sl
    else:
        atr_sl = entry + atr
        sw = df['swing_highs'].iloc[:idx].dropna()
        if len(sw) > 0:
            return max(atr_sl, sw.iloc[-1] + entry*0.0005)
        return atr_sl


# ============================================================
# TRADE SIMULATION — WITH EARLY BREAKEVEN + TRAILING
# ============================================================
def simulate(df_5m, entry_idx, trade: Trade, config: Config):
    """
    Exit priority chain:
    1. SL hit (initial or moved) → loss or scratch
    2. Early BE trigger → move SL to entry+fees
    3. Trailing trigger → start trailing
    4. TP hit → full win (rarely reached due to trailing)
    """
    sl = trade.stop_loss
    tp = trade.take_profit
    entry = trade.entry_price
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        trade.result = 'loss'
        trade.pnl_r = -1.0
        return trade
    
    early_be_active = False
    trailing_active = False
    max_favorable = 0.0
    
    for i in range(entry_idx + 1, min(entry_idx + 200, len(df_5m))):
        high = df_5m['High'].iloc[i]
        low = df_5m['Low'].iloc[i]
        
        if trade.direction == 'bullish':
            current_r = (high - entry) / sl_dist
            max_favorable = max(max_favorable, current_r)
            
            # 1. Check SL
            if low <= sl:
                trade.exit_price = sl
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = (sl - entry) / sl_dist
                trade.max_favorable_r = max_favorable
                if trailing_active:
                    trade.hit_trailing = True
                    trade.result = 'win' if trade.pnl_r > 0.05 else 'breakeven'
                elif early_be_active:
                    trade.hit_early_be = True
                    trade.result = 'breakeven' if trade.pnl_r >= -0.05 else 'loss'
                else:
                    trade.result = 'loss'
                return trade
            
            # 2. TP hit
            if high >= tp:
                trade.exit_price = tp
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = (tp - entry) / sl_dist
                trade.result = 'win'
                trade.max_favorable_r = max_favorable
                return trade
            
            # 3. Trailing stop management
            if config.use_trailing and max_favorable >= config.trail_trigger_r:
                trailing_active = True
                trail_sl = entry + (max_favorable - config.trail_distance_r) * sl_dist
                sl = max(sl, trail_sl)
            
            # 4. Early breakeven (only if trailing not yet active)
            elif config.use_early_be and not early_be_active and max_favorable >= config.early_be_trigger_r:
                early_be_active = True
                be_level = entry + config.early_be_level_r * sl_dist
                sl = max(sl, be_level)
        
        else:  # bearish
            current_r = (entry - low) / sl_dist
            max_favorable = max(max_favorable, current_r)
            
            if high >= sl:
                trade.exit_price = sl
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = (entry - sl) / sl_dist
                trade.max_favorable_r = max_favorable
                if trailing_active:
                    trade.hit_trailing = True
                    trade.result = 'win' if trade.pnl_r > 0.05 else 'breakeven'
                elif early_be_active:
                    trade.hit_early_be = True
                    trade.result = 'breakeven' if trade.pnl_r >= -0.05 else 'loss'
                else:
                    trade.result = 'loss'
                return trade
            
            if low <= tp:
                trade.exit_price = tp
                trade.exit_time = str(df_5m.index[i])
                trade.pnl_r = (entry - tp) / sl_dist
                trade.result = 'win'
                trade.max_favorable_r = max_favorable
                return trade
            
            if config.use_trailing and max_favorable >= config.trail_trigger_r:
                trailing_active = True
                trail_sl = entry - (max_favorable - config.trail_distance_r) * sl_dist
                sl = min(sl, trail_sl)
            elif config.use_early_be and not early_be_active and max_favorable >= config.early_be_trigger_r:
                early_be_active = True
                be_level = entry - config.early_be_level_r * sl_dist
                sl = min(sl, be_level)
    
    # Timeout
    li = min(entry_idx + 199, len(df_5m)-1)
    trade.exit_price = df_5m['Close'].iloc[li]
    trade.exit_time = str(df_5m.index[li])
    if trade.direction == 'bullish':
        trade.pnl_r = (trade.exit_price - entry) / sl_dist
    else:
        trade.pnl_r = (entry - trade.exit_price) / sl_dist
    trade.max_favorable_r = max_favorable
    trade.result = 'win' if trade.pnl_r > 0.1 else ('loss' if trade.pnl_r < -0.1 else 'breakeven')
    return trade


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run(all_data, config: Config) -> List[Trade]:
    trades = []
    for name, data in all_data.items():
        df_4h, df_30m, df_5m = data['4h'], data['30m'], data['5m']
        if df_5m.empty:
            continue
        daily = {}
        start = max(250, SWING_LOOKBACK + 20)
        
        for idx in range(start, len(df_5m) - 200):
            ts = df_5m.index[idx]
            ts_n = ts.tz_localize(None) if hasattr(ts, 'tz') and ts.tz else ts
            if not (SESSION_START <= ts_n.hour <= SESSION_END):
                continue
            dk = ts_n.date()
            if daily.get(dk, 0) >= config.max_trades_per_day:
                continue
            
            m4 = df_4h.index <= ts
            if m4.sum() < EMA_4H_PERIOD + EMA_4H_SLOPE_LOOKBACK + 1:
                continue
            i4 = m4.sum() - 1
            trend_4h, slope_4h_str = check_4h(df_4h, i4, config)
            if not trend_4h:
                continue
            
            m30 = df_30m.index <= ts
            if m30.sum() < EMA_30M_SLOW + 10:
                continue
            i30 = m30.sum() - 1
            trend_30m = check_30m(df_30m, i30)
            if trend_30m != trend_4h:
                continue
            
            direction = trend_4h
            
            # Entry checks
            if not check_pullback(df_5m, idx, direction, config):
                continue
            rsi_ok, rsi_v = check_rsi(df_5m, idx, direction, config)
            if not rsi_ok:
                continue
            if not check_structure(df_5m, idx, direction):
                continue
            candle_ok, pattern = check_candle(df_5m, idx, direction, config)
            if not candle_ok:
                continue
            vol_ok, vol_r = check_volume(df_5m, idx, config)
            if config.require_volume and not vol_ok:
                continue
            
            # EMA20 slope check (WR8)
            if config.require_ema20_slope:
                slope_5m = df_5m['ema_20_slope'].iloc[idx]
                if direction == 'bullish' and slope_5m <= 0:
                    continue
                if direction == 'bearish' and slope_5m >= 0:
                    continue
            
            # LEVER 1 FILTERS
            atr_pctile = df_5m['atr_percentile'].iloc[idx]
            if not pd.isna(atr_pctile) and config.min_atr_percentile > 0:
                if atr_pctile < config.min_atr_percentile:
                    continue
            
            ema20_dist = df_5m['ema20_dist_pct'].iloc[idx]
            if config.max_ema20_distance_pct < 1.0:
                if ema20_dist > config.max_ema20_distance_pct:
                    continue
            
            # ENTRY
            entry = df_5m['Close'].iloc[idx]
            sl = calc_sl(df_5m, idx, direction, entry)
            sl_dist = abs(entry - sl)
            if sl_dist <= 0:
                continue
            
            if direction == 'bullish':
                tp = entry + sl_dist * config.rr_target
            else:
                tp = entry - sl_dist * config.rr_target
            
            trade = Trade(
                symbol=name, direction=direction,
                entry_price=entry, stop_loss=sl, take_profit=tp,
                entry_time=str(ts), pattern=pattern or "", rsi=rsi_v,
                volume_ratio=vol_r, hour_of_day=ts_n.hour,
                atr_value=df_5m['atr'].iloc[idx],
                atr_percentile=atr_pctile if not pd.isna(atr_pctile) else 0.5,
                ema20_distance_pct=ema20_dist,
                slope_4h_strength=slope_4h_str,
                slope_5m_strength=abs(df_5m['ema_20_slope'].iloc[idx]) / entry if entry > 0 else 0,
            )
            
            trade = simulate(df_5m, idx, trade, config)
            trades.append(trade)
            daily[dk] = daily.get(dk, 0) + 1
            idx += 10
    
    return trades


def stats(trades, name=""):
    if not trades:
        return {'name': name, 'valid': False, 'total': 0}
    total = len(trades)
    wins = [t for t in trades if t.result == 'win']
    losses = [t for t in trades if t.result == 'loss']
    bes = [t for t in trades if t.result == 'breakeven']
    
    wr = len(wins)/total*100
    total_r = sum(t.pnl_r for t in trades)
    avg_win = np.mean([t.pnl_r for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_r for t in losses]) if losses else 0
    
    gp = sum(t.pnl_r for t in trades if t.pnl_r > 0)
    gl = abs(sum(t.pnl_r for t in trades if t.pnl_r < 0))
    pf = gp / gl if gl > 0 else float('inf')
    
    eq = STARTING_EQUITY
    mx = eq
    dd = 0
    for t in trades:
        eq += (eq * RISK_PERCENT/100) * t.pnl_r
        mx = max(mx, eq)
        dd = max(dd, (mx-eq)/mx*100)
    
    # Count early BE saves
    be_saves = sum(1 for t in trades if t.hit_early_be)
    
    return {
        'name': name, 'valid': True, 'total': total,
        'wins': len(wins), 'losses': len(losses), 'bes': len(bes),
        'wr': wr, 'total_r': total_r,
        'avg_win': avg_win, 'avg_loss': avg_loss,
        'pf': pf, 'exp': total_r/total,
        'dd': dd, 'ret': ((eq-STARTING_EQUITY)/STARTING_EQUITY)*100,
        'equity': eq, 'be_saves': be_saves,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("   FINAL OPTIMIZATION: LEVER 2 (EARLY BE) + LEVER 1 (CLUSTERS)")
    print("=" * 70)
    
    all_data = fetch_and_prepare()
    
    # ========================================
    # PART 1: Test early breakeven thresholds
    # ========================================
    print("\n" + "=" * 70)
    print("   PART 1: LEVER 2 — EARLY BREAKEVEN STOP TESTS")
    print("   Base: WR8 (trailing 1R/0.4R + EMA20 slope)")
    print("=" * 70)
    
    be_configs = [
        Config(name="WR8 BASELINE (no early BE)", use_early_be=False),
        Config(name="Early BE @ +0.3R → entry", use_early_be=True, early_be_trigger_r=0.3, early_be_level_r=0.0),
        Config(name="Early BE @ +0.4R → entry", use_early_be=True, early_be_trigger_r=0.4, early_be_level_r=0.0),
        Config(name="Early BE @ +0.4R → +0.1R", use_early_be=True, early_be_trigger_r=0.4, early_be_level_r=0.1),
        Config(name="Early BE @ +0.5R → entry", use_early_be=True, early_be_trigger_r=0.5, early_be_level_r=0.0),
        Config(name="Early BE @ +0.5R → +0.1R", use_early_be=True, early_be_trigger_r=0.5, early_be_level_r=0.1),
        Config(name="Early BE @ +0.6R → +0.1R", use_early_be=True, early_be_trigger_r=0.6, early_be_level_r=0.1),
    ]
    
    be_results = []
    baseline_trades = None
    
    for i, cfg in enumerate(be_configs, 1):
        trades = run(all_data, cfg)
        s = stats(trades, cfg.name)
        be_results.append(s)
        if i == 1:
            baseline_trades = trades
        
        if s['valid']:
            saves = s.get('be_saves', 0)
            print(f"  [{i}] {cfg.name:<35} | WR:{s['wr']:>5.1f}% | Losses:{s['losses']:>3} | "
                  f"AvgLoss:{s['avg_loss']:>+5.2f}R | TotalR:{s['total_r']:>+6.1f} | "
                  f"Ret:{s['ret']:>+6.1f}% | DD:{s['dd']:>5.1f}% | BE_Saves:{saves}")
    
    # ========================================
    # PART 2: Analyze losing trades from baseline
    # ========================================
    print("\n\n" + "=" * 70)
    print("   PART 2: LEVER 1 — LOSING TRADE CLUSTER ANALYSIS")
    print("=" * 70)
    
    if baseline_trades:
        losers = [t for t in baseline_trades if t.result == 'loss']
        winners = [t for t in baseline_trades if t.result == 'win']
        
        print(f"\n  Analyzing {len(losers)} losing trades vs {len(winners)} winning trades...\n")
        
        # How far did losers go in your favor before failing?
        print(f"  {'MAX FAVORABLE EXCURSION (how far losers went before dying)':=^60}")
        fav_ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7), (0.7, 1.0)]
        for lo, hi in fav_ranges:
            count = sum(1 for t in losers if lo <= t.max_favorable_r < hi)
            pct = count/len(losers)*100 if losers else 0
            bar = "█" * int(pct/2)
            print(f"    +{lo:.1f}R to +{hi:.1f}R: {count:>3} losers ({pct:>5.1f}%) {bar}")
        
        salvageable = sum(1 for t in losers if t.max_favorable_r >= 0.4)
        print(f"\n    → {salvageable}/{len(losers)} losers ({salvageable/len(losers)*100:.0f}%) reached +0.4R before reversing")
        print(f"    → These could be converted to breakeven/scratch with early BE at +0.4R")
        
        # ATR percentile distribution
        print(f"\n  {'ATR PERCENTILE (volatility) DISTRIBUTION':=^60}")
        atr_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        print(f"    {'Range':<15} {'Losers':<12} {'Winners':<12} {'Loss Rate':<12}")
        print(f"    {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
        for lo, hi in atr_ranges:
            l_count = sum(1 for t in losers if lo <= t.atr_percentile < hi)
            w_count = sum(1 for t in winners if lo <= t.atr_percentile < hi)
            total_in_range = l_count + w_count
            loss_rate = l_count / total_in_range * 100 if total_in_range > 0 else 0
            print(f"    {lo:.1f}-{hi:.1f}         {l_count:<12} {w_count:<12} {loss_rate:<.1f}%")
        
        # Hour of day
        print(f"\n  {'HOUR OF DAY DISTRIBUTION':=^60}")
        hours_loss = Counter(t.hour_of_day for t in losers)
        hours_win = Counter(t.hour_of_day for t in winners)
        all_hours = sorted(set(list(hours_loss.keys()) + list(hours_win.keys())))
        print(f"    {'Hour':<6} {'Losers':<8} {'Winners':<8} {'Loss Rate':<10}")
        print(f"    {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
        for h in all_hours:
            lc = hours_loss.get(h, 0)
            wc = hours_win.get(h, 0)
            total_h = lc + wc
            lr = lc / total_h * 100 if total_h > 0 else 0
            flag = " ⚠️" if lr > 60 and total_h >= 5 else ""
            print(f"    {h:>2}:00  {lc:<8} {wc:<8} {lr:<.0f}%{flag}")
        
        # EMA20 distance at entry
        print(f"\n  {'EMA20 DISTANCE AT ENTRY':=^60}")
        dist_ranges = [(0, 0.001), (0.001, 0.002), (0.002, 0.004), (0.004, 0.008), (0.008, 0.02)]
        print(f"    {'Distance':<15} {'Losers':<8} {'Winners':<8} {'Loss Rate':<10}")
        print(f"    {'-'*15} {'-'*8} {'-'*8} {'-'*10}")
        for lo, hi in dist_ranges:
            lc = sum(1 for t in losers if lo <= t.ema20_distance_pct < hi)
            wc = sum(1 for t in winners if lo <= t.ema20_distance_pct < hi)
            total_d = lc + wc
            lr = lc / total_d * 100 if total_d > 0 else 0
            flag = " ⚠️" if lr > 60 and total_d >= 5 else ""
            print(f"    {lo:.3f}-{hi:.3f}   {lc:<8} {wc:<8} {lr:.0f}%{flag}")
        
        # 4H slope strength
        print(f"\n  {'4H EMA SLOPE STRENGTH':=^60}")
        slope_ranges = [(0, 0.0005), (0.0005, 0.001), (0.001, 0.002), (0.002, 0.005), (0.005, 0.1)]
        print(f"    {'Slope %':<15} {'Losers':<8} {'Winners':<8} {'Loss Rate':<10}")
        print(f"    {'-'*15} {'-'*8} {'-'*8} {'-'*10}")
        for lo, hi in slope_ranges:
            lc = sum(1 for t in losers if lo <= t.slope_4h_strength < hi)
            wc = sum(1 for t in winners if lo <= t.slope_4h_strength < hi)
            total_s = lc + wc
            lr = lc / total_s * 100 if total_s > 0 else 0
            flag = " ⚠️" if lr > 60 and total_s >= 5 else ""
            print(f"    {lo:.4f}-{hi:.4f} {lc:<8} {wc:<8} {lr:.0f}%{flag}")
        
        # Direction breakdown
        print(f"\n  {'DIRECTION BREAKDOWN':=^60}")
        l_longs = sum(1 for t in losers if t.direction == 'bullish')
        l_shorts = sum(1 for t in losers if t.direction == 'bearish')
        w_longs = sum(1 for t in winners if t.direction == 'bullish')
        w_shorts = sum(1 for t in winners if t.direction == 'bearish')
        print(f"    Longs:  {l_longs} losses / {w_longs} wins = {l_longs/(l_longs+w_longs)*100:.0f}% loss rate" if (l_longs+w_longs) > 0 else "    Longs: N/A")
        print(f"    Shorts: {l_shorts} losses / {w_shorts} wins = {l_shorts/(l_shorts+w_shorts)*100:.0f}% loss rate" if (l_shorts+w_shorts) > 0 else "    Shorts: N/A")
    
    # ========================================
    # PART 3: Test Lever 1 filters based on clusters
    # ========================================
    print("\n\n" + "=" * 70)
    print("   PART 3: LEVER 1 — TESTING CLUSTER-BASED FILTERS")
    print("=" * 70)
    
    filter_configs = [
        Config(name="BASELINE WR8 + Early BE @0.4R", use_early_be=True, early_be_trigger_r=0.4, early_be_level_r=0.1),
        Config(name="+ ATR > 20th percentile", use_early_be=True, early_be_trigger_r=0.4, early_be_level_r=0.1, min_atr_percentile=0.20),
        Config(name="+ ATR > 30th percentile", use_early_be=True, early_be_trigger_r=0.4, early_be_level_r=0.1, min_atr_percentile=0.30),
        Config(name="+ 4H slope > 0.0008", use_early_be=True, early_be_trigger_r=0.4, early_be_level_r=0.1, min_4h_slope_strength=0.0008),
        Config(name="+ ATR>20% + slope>0.0008", use_early_be=True, early_be_trigger_r=0.4, early_be_level_r=0.1, min_atr_percentile=0.20, min_4h_slope_strength=0.0008),
        Config(name="+ EMA dist < 0.4% + ATR>20%", use_early_be=True, early_be_trigger_r=0.4, early_be_level_r=0.1, min_atr_percentile=0.20, max_ema20_distance_pct=0.004),
    ]
    
    print(f"\n  {'Strategy':<40} {'Trades':<7} {'WR%':<7} {'Losses':<8} {'AvgLoss':<8} {'TotalR':<8} {'Ret%':<8} {'DD%':<6} {'Saves':<6}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    
    for cfg in filter_configs:
        trades = run(all_data, cfg)
        s = stats(trades, cfg.name)
        if s['valid']:
            print(f"  {cfg.name:<40} {s['total']:<7} {s['wr']:<7.1f} {s['losses']:<8} "
                  f"{s['avg_loss']:<+8.2f} {s['total_r']:<+8.1f} {s['ret']:<+8.1f} {s['dd']:<6.1f} {s['be_saves']:<6}")
    
    # ========================================
    # FINAL RECOMMENDATION
    # ========================================
    print("\n\n" + "=" * 70)
    print("   FINAL RECOMMENDATION")
    print("=" * 70)
    
    # Run the recommended config
    final_config = Config(
        name="FINAL: WR8 + Early BE@0.4R(+0.1R) + ATR>20% + Slope>0.0008",
        use_early_be=True,
        early_be_trigger_r=0.4,
        early_be_level_r=0.1,
        min_atr_percentile=0.20,
        min_4h_slope_strength=0.0008,
    )
    final_trades = run(all_data, final_config)
    final_s = stats(final_trades, final_config.name)
    baseline_s = be_results[0] if be_results else None
    
    if final_s['valid'] and baseline_s and baseline_s['valid']:
        print(f"\n  {'Metric':<25} {'WR8 Baseline':<18} {'FINAL Optimized':<18} {'Change':<15}")
        print(f"  {'-'*25} {'-'*18} {'-'*18} {'-'*15}")
        print(f"  {'Win Rate':<25} {baseline_s['wr']:<18.1f} {final_s['wr']:<18.1f} {final_s['wr']-baseline_s['wr']:+.1f}%")
        print(f"  {'Total Trades':<25} {baseline_s['total']:<18} {final_s['total']:<18} {final_s['total']-baseline_s['total']:+d}")
        print(f"  {'Wins':<25} {baseline_s['wins']:<18} {final_s['wins']:<18} {final_s['wins']-baseline_s['wins']:+d}")
        print(f"  {'Losses':<25} {baseline_s['losses']:<18} {final_s['losses']:<18} {final_s['losses']-baseline_s['losses']:+d}")
        print(f"  {'Avg Loss':<25} {baseline_s['avg_loss']:<+18.3f} {final_s['avg_loss']:<+18.3f} {final_s['avg_loss']-baseline_s['avg_loss']:+.3f}R")
        print(f"  {'Avg Win':<25} {baseline_s['avg_win']:<+18.3f} {final_s['avg_win']:<+18.3f} {final_s['avg_win']-baseline_s['avg_win']:+.3f}R")
        print(f"  {'Total R':<25} {baseline_s['total_r']:<+18.1f} {final_s['total_r']:<+18.1f} {final_s['total_r']-baseline_s['total_r']:+.1f}R")
        print(f"  {'Profit Factor':<25} {baseline_s['pf']:<18.2f} {final_s['pf']:<18.2f} {final_s['pf']-baseline_s['pf']:+.2f}")
        print(f"  {'Expectancy/Trade':<25} {baseline_s['exp']:<+18.3f} {final_s['exp']:<+18.3f} {final_s['exp']-baseline_s['exp']:+.3f}R")
        print(f"  {'Max Drawdown':<25} {baseline_s['dd']:<18.1f} {final_s['dd']:<18.1f} {final_s['dd']-baseline_s['dd']:+.1f}%")
        print(f"  {'Return ($10k)':<25} {baseline_s['ret']:<+18.1f} {final_s['ret']:<+18.1f} {final_s['ret']-baseline_s['ret']:+.1f}%")
        print(f"  {'Final Equity':<25} ${baseline_s['equity']:<17,.0f} ${final_s['equity']:<17,.0f} ${final_s['equity']-baseline_s['equity']:+,.0f}")
        print(f"  {'BE Saves (from losses)':<25} {'N/A':<18} {final_s['be_saves']:<18}")
        
        # Dollar comparison
        print(f"\n  DOLLAR PROFIT @ $100 RISK/TRADE:")
        print(f"    WR8 Baseline: ${baseline_s['total_r']*100:+,.0f}")
        print(f"    FINAL:        ${final_s['total_r']*100:+,.0f}")
        print(f"    Difference:   ${(final_s['total_r']-baseline_s['total_r'])*100:+,.0f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
