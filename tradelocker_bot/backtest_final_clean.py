"""
FINAL CLEAN RUN: WR8 + Session Filter (no 15-17 UTC) + ATR < 80th percentile
Just gives the numbers.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import List

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

# NEW FILTERS
AVOID_HOURS = [15, 16, 17]  # UTC hours to skip
MAX_ATR_PERCENTILE = 0.80   # Skip when ATR > 80th percentile

@dataclass
class Trade:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: str
    exit_price: float = 0.0
    pnl_r: float = 0.0
    result: str = ""
    pattern: str = ""
    max_favorable_r: float = 0.0

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

def fetch_and_prepare():
    print("Fetching data...")
    all_data = {}
    for name, ticker in [('BTC-USD', 'BTC-USD'), ('XAU', 'GC=F')]:
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
        
        all_data[name] = {'4h': df_4h, '30m': df_30m, '5m': df_5m}
    return all_data

def check_4h(df_4h, idx):
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

def check_pullback(df, idx, direction):
    if idx < 3:
        return False
    ema20 = df['ema_20'].iloc[idx]
    close, low, high = df['Close'].iloc[idx], df['Low'].iloc[idx], df['High'].iloc[idx]
    t = 0.002
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

def check_rsi(df, idx, direction):
    rsi = df['rsi'].iloc[idx]
    if rsi >= 70 or rsi <= 30:
        return False
    if direction == 'bullish':
        return 40 <= rsi <= 65
    else:
        return 35 <= rsi <= 60

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

def check_candle(df, idx, direction):
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
    
    bull = ['bullish_engulfing', 'hammer', 'bullish_rejection', 'strong_bullish']
    bear = ['bearish_engulfing', 'shooting_star', 'bearish_rejection', 'strong_bearish']
    
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
        if pattern and pattern in bull:
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
        if pattern and pattern in bear:
            return True, pattern
    return False, pattern

def check_volume(df, idx):
    if idx < VOLUME_AVG_PERIOD:
        return False
    v = df['Volume'].iloc[idx]
    a = df['volume_avg'].iloc[idx]
    if a <= 0:
        return False
    return v / a >= 1.0

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

def simulate(df_5m, entry_idx, trade: Trade):
    """WR8 trailing stop: trigger at 1R, trail by 0.4R. No early BE."""
    sl = trade.stop_loss
    tp = trade.take_profit
    entry = trade.entry_price
    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        trade.result = 'loss'
        trade.pnl_r = -1.0
        return trade
    
    trailing_active = False
    max_fav = 0.0
    
    for i in range(entry_idx + 1, min(entry_idx + 200, len(df_5m))):
        high = df_5m['High'].iloc[i]
        low = df_5m['Low'].iloc[i]
        
        if trade.direction == 'bullish':
            current_r = (high - entry) / sl_dist
            max_fav = max(max_fav, current_r)
            
            # Trailing stop
            if max_fav >= 1.0:
                trailing_active = True
                trail_sl = entry + (max_fav - 0.4) * sl_dist
                sl = max(sl, trail_sl)
            
            # SL hit
            if low <= sl:
                trade.exit_price = sl
                trade.pnl_r = (sl - entry) / sl_dist
                trade.max_favorable_r = max_fav
                if trailing_active and trade.pnl_r > 0.05:
                    trade.result = 'win'
                elif trailing_active:
                    trade.result = 'breakeven'
                else:
                    trade.result = 'loss'
                return trade
            
            # TP hit
            if high >= tp:
                trade.exit_price = tp
                trade.pnl_r = (tp - entry) / sl_dist
                trade.result = 'win'
                trade.max_favorable_r = max_fav
                return trade
        else:
            current_r = (entry - low) / sl_dist
            max_fav = max(max_fav, current_r)
            
            if max_fav >= 1.0:
                trailing_active = True
                trail_sl = entry - (max_fav - 0.4) * sl_dist
                sl = min(sl, trail_sl)
            
            if high >= sl:
                trade.exit_price = sl
                trade.pnl_r = (entry - sl) / sl_dist
                trade.max_favorable_r = max_fav
                if trailing_active and trade.pnl_r > 0.05:
                    trade.result = 'win'
                elif trailing_active:
                    trade.result = 'breakeven'
                else:
                    trade.result = 'loss'
                return trade
            
            if low <= tp:
                trade.exit_price = tp
                trade.pnl_r = (entry - tp) / sl_dist
                trade.result = 'win'
                trade.max_favorable_r = max_fav
                return trade
    
    # Timeout
    li = min(entry_idx + 199, len(df_5m)-1)
    trade.exit_price = df_5m['Close'].iloc[li]
    if trade.direction == 'bullish':
        trade.pnl_r = (trade.exit_price - entry) / sl_dist
    else:
        trade.pnl_r = (entry - trade.exit_price) / sl_dist
    trade.max_favorable_r = max_fav
    trade.result = 'win' if trade.pnl_r > 0.1 else ('loss' if trade.pnl_r < -0.1 else 'breakeven')
    return trade


def run_backtest(all_data, use_filters=True):
    """Run with or without the new filters for comparison."""
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
            
            # Session filter
            if not (SESSION_START <= ts_n.hour <= SESSION_END):
                continue
            
            # NEW FILTER 1: Avoid 15:00-17:00 UTC
            if use_filters and ts_n.hour in AVOID_HOURS:
                continue
            
            dk = ts_n.date()
            if daily.get(dk, 0) >= 2:
                continue
            
            # NEW FILTER 2: Skip high ATR
            if use_filters:
                atr_pctile = df_5m['atr_percentile'].iloc[idx]
                if not pd.isna(atr_pctile) and atr_pctile > MAX_ATR_PERCENTILE:
                    continue
            
            # 4H trend
            m4 = df_4h.index <= ts
            if m4.sum() < EMA_4H_PERIOD + EMA_4H_SLOPE_LOOKBACK + 1:
                continue
            i4 = m4.sum() - 1
            trend_4h = check_4h(df_4h, i4)
            if not trend_4h:
                continue
            
            # 30M trend
            m30 = df_30m.index <= ts
            if m30.sum() < EMA_30M_SLOW + 10:
                continue
            i30 = m30.sum() - 1
            if check_30m(df_30m, i30) != trend_4h:
                continue
            
            direction = trend_4h
            
            # Entry checks (5/6: sweep optional)
            if not check_pullback(df_5m, idx, direction):
                continue
            if not check_rsi(df_5m, idx, direction):
                continue
            if not check_structure(df_5m, idx, direction):
                continue
            candle_ok, pattern = check_candle(df_5m, idx, direction)
            if not candle_ok:
                continue
            if not check_volume(df_5m, idx):
                continue
            
            # EMA20 slope (WR8)
            slope_5m = df_5m['ema_20_slope'].iloc[idx]
            if direction == 'bullish' and slope_5m <= 0:
                continue
            if direction == 'bearish' and slope_5m >= 0:
                continue
            
            # ENTRY
            entry = df_5m['Close'].iloc[idx]
            sl = calc_sl(df_5m, idx, direction, entry)
            sl_dist = abs(entry - sl)
            if sl_dist <= 0:
                continue
            
            tp = entry + sl_dist * 3.0 if direction == 'bullish' else entry - sl_dist * 3.0
            
            trade = Trade(symbol=name, direction=direction, entry_price=entry,
                         stop_loss=sl, take_profit=tp, entry_time=str(ts), pattern=pattern or "")
            trade = simulate(df_5m, idx, trade)
            trades.append(trade)
            daily[dk] = daily.get(dk, 0) + 1
            idx += 10
    
    return trades


def main():
    all_data = fetch_and_prepare()
    
    # Run both versions
    print("\nRunning WR8 WITHOUT new filters (baseline)...")
    baseline_trades = run_backtest(all_data, use_filters=False)
    
    print("Running WR8 WITH new filters (session + ATR)...")
    filtered_trades = run_backtest(all_data, use_filters=True)
    
    # Calculate stats
    def get_stats(trades):
        total = len(trades)
        wins = [t for t in trades if t.result == 'win']
        losses = [t for t in trades if t.result == 'loss']
        bes = [t for t in trades if t.result == 'breakeven']
        total_r = sum(t.pnl_r for t in trades)
        avg_win = np.mean([t.pnl_r for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_r for t in losses]) if losses else 0
        
        eq = STARTING_EQUITY
        mx = eq
        dd = 0
        for t in trades:
            eq += (eq * RISK_PERCENT/100) * t.pnl_r
            mx = max(mx, eq)
            dd = max(dd, (mx-eq)/mx*100)
        
        # Consecutive losses
        max_cl = 0
        cl = 0
        for t in trades:
            if t.result == 'loss':
                cl += 1
                max_cl = max(max_cl, cl)
            else:
                cl = 0
        
        return {
            'total': total, 'wins': len(wins), 'losses': len(losses), 'bes': len(bes),
            'wr': len(wins)/total*100 if total > 0 else 0,
            'total_r': total_r,
            'avg_win': avg_win, 'avg_loss': avg_loss,
            'equity': eq, 'dd': dd, 'max_cl': max_cl,
            'ret': ((eq - STARTING_EQUITY)/STARTING_EQUITY)*100,
        }
    
    b = get_stats(baseline_trades)
    f = get_stats(filtered_trades)
    
    print("\n")
    print("=" * 65)
    print("         FINAL RESULTS: WR8 vs WR8 + FILTERS")
    print("=" * 65)
    print(f"\n  {'Metric':<30} {'WR8 (Before)':<20} {'WR8+Filters (After)':<20}")
    print(f"  {'='*30} {'='*20} {'='*20}")
    print(f"  {'Total Trades':<30} {b['total']:<20} {f['total']:<20}")
    print(f"  {'Wins':<30} {b['wins']:<20} {f['wins']:<20}")
    print(f"  {'Losses':<30} {b['losses']:<20} {f['losses']:<20}")
    print(f"  {'Breakevens':<30} {b['bes']:<20} {f['bes']:<20}")
    print(f"  {'WIN RATE':<30} {b['wr']:<20.1f} {f['wr']:<20.1f}")
    print(f"  {'Avg Win (R)':<30} {b['avg_win']:<+20.3f} {f['avg_win']:<+20.3f}")
    print(f"  {'Avg Loss (R)':<30} {b['avg_loss']:<+20.3f} {f['avg_loss']:<+20.3f}")
    print(f"  {'Total R Gained':<30} {b['total_r']:<+20.1f} {f['total_r']:<+20.1f}")
    print(f"  {'Max Drawdown':<30} {b['dd']:<20.1f}% {f['dd']:<19.1f}%")
    print(f"  {'Max Consecutive Losses':<30} {b['max_cl']:<20} {f['max_cl']:<20}")
    print(f"  {'Return on $10,000':<30} {b['ret']:<+20.1f}% {f['ret']:<+19.1f}%")
    print(f"  {'Final Equity':<30} ${b['equity']:<19,.0f} ${f['equity']:<18,.0f}")
    
    print(f"\n  {'CHANGE SUMMARY':=^65}")
    print(f"  Win Rate:    {b['wr']:.1f}%  →  {f['wr']:.1f}%  ({f['wr']-b['wr']:+.1f}%)")
    print(f"  Losses:      {b['losses']}  →  {f['losses']}  ({f['losses']-b['losses']:+d} fewer losses)")
    print(f"  Wins:        {b['wins']}  →  {f['wins']}  ({f['wins']-b['wins']:+d})")
    print(f"  Total R:     {b['total_r']:+.1f}  →  {f['total_r']:+.1f}  ({f['total_r']-b['total_r']:+.1f}R)")
    print(f"  Drawdown:    {b['dd']:.1f}%  →  {f['dd']:.1f}%  ({f['dd']-b['dd']:+.1f}%)")
    
    print(f"\n  {'DOLLAR PROFIT @ $100 RISK/TRADE':=^65}")
    print(f"  Before: ${b['total_r']*100:+,.0f}")
    print(f"  After:  ${f['total_r']*100:+,.0f}")
    print(f"  Diff:   ${(f['total_r']-b['total_r'])*100:+,.0f}")
    
    # Removed trades analysis
    removed = b['total'] - f['total']
    removed_losses = b['losses'] - f['losses']
    removed_wins = b['wins'] - f['wins']
    print(f"\n  {'WHAT THE FILTERS REMOVED':=^65}")
    print(f"  Total trades removed:  {removed}")
    print(f"  Of which were LOSSES:  {removed_losses}")
    print(f"  Of which were WINS:    {removed_wins}")
    if removed > 0:
        print(f"  Filter accuracy:       {removed_losses/removed*100:.0f}% of removed trades were losers")
    
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
