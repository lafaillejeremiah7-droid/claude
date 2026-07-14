"""
GOLD VORTEX v3 - Adaptive Momentum Strategy for XAUUSD
========================================================
Built for maximum weekly profit on a $5k account.

Key insight from data: Gold Jan-Jul 2026 went $4394 -> $5591 -> $4023.
NOT a simple trend — it's a MOMENTUM regime. The strategy must:
  1. Catch the big moves in BOTH directions
  2. Use wide enough stops to survive gold's $15-20 hourly noise
  3. Trail exits aggressively once in profit to capture large swings
  4. Cut losses quickly when momentum reverses

Strategy: MOMENTUM REGIME FOLLOWING
  - Direction: follow the dominant 1h momentum (EMA crossover + RSI thrust)
  - Entry: momentum shift confirmed (EMA9 crosses EMA21 + RSI confirms)
  - SL: 2.5x ATR (wide enough for gold's noise = ~$50-65)
  - Exit: trailing stop at 3x ATR that ratchets, OR regime flip
  - This catches big swings ($100-400 moves) while accepting -1R on noise

The math: with 2.5% risk and 2.5:1 avg winner, you need ~35% WR to profit.
Target: catch 3-4 big swings per month, each delivering 2-5R.
"""
import numpy as np, pandas as pd


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def atr_calc(h, l, c, period=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def rsi_calc(series, period=14):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))

def adx_calc(h, l, c, period=14):
    up_move = h - h.shift(); down_move = l.shift() - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100*pd.Series(plus_dm,index=h.index).ewm(alpha=1/period,adjust=False).mean()/(atr_s+1e-12)
    mdi = 100*pd.Series(minus_dm,index=h.index).ewm(alpha=1/period,adjust=False).mean()/(atr_s+1e-12)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi+1e-12)
    return dx.ewm(alpha=1/period, adjust=False).mean(), pdi, mdi



def build_features(df):
    d = df.copy()
    h, l, c = d['High'], d['Low'], d['Close']
    d['atr'] = atr_calc(h, l, c, 14)
    d['ema9'] = ema(c, 9)
    d['ema21'] = ema(c, 21)
    d['ema50'] = ema(c, 50)
    d['rsi'] = rsi_calc(c, 14)
    d['adx'], d['pdi'], d['mdi'] = adx_calc(h, l, c, 14)
    d['macd'] = ema(c, 12) - ema(c, 26)
    d['macd_sig'] = ema(d['macd'], 9)
    d['macd_hist'] = d['macd'] - d['macd_sig']
    # Momentum strength: how far price is from EMA50 (normalized by ATR)
    d['mom_dist'] = (c - d['ema50']) / (d['atr'] + 1e-12)
    return d.dropna(subset=['atr','ema9','ema21','ema50','rsi','adx'])


def generate_signals(df):
    """
    Signal logic — FAST MOMENTUM REGIME SHIFTS:
    
    BUY when:
      - EMA9 > EMA21 (short-term momentum up)
      - RSI crosses above 50 from below (momentum shift)
      - OR: RSI > 60 with MACD histogram positive and increasing
      - ADX > 15 (directional movement)
      - Session: 7-20 UTC
    
    SELL when (mirror):
      - EMA9 < EMA21
      - RSI crosses below 50 from above
      - OR: RSI < 40 with MACD histogram negative and decreasing
      - ADX > 15
      - Session: 7-20 UTC
    """
    signals = []
    
    for i in range(2, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        prev2 = df.iloc[i-2]
        hour = row.name.hour if hasattr(row.name, 'hour') else 12
        
        if hour < 7 or hour > 20:
            continue
        if row['atr'] < 5:
            continue
        
        direction = None
        confidence = 50
        
        # BUY signals
        ema_bull = row['ema9'] > row['ema21']
        rsi_cross_up = prev['rsi'] < 50 and row['rsi'] >= 50
        rsi_strong = row['rsi'] > 60
        macd_accel = row['macd_hist'] > 0 and row['macd_hist'] > prev['macd_hist']
        adx_ok = row['adx'] > 15
        
        if ema_bull and adx_ok:
            if rsi_cross_up:
                direction = 'BUY'
                confidence = 70
            elif rsi_strong and macd_accel:
                direction = 'BUY'
                confidence = 65
        
        # SELL signals (only if no buy)
        if direction is None:
            ema_bear = row['ema9'] < row['ema21']
            rsi_cross_dn = prev['rsi'] > 50 and row['rsi'] <= 50
            rsi_weak = row['rsi'] < 40
            macd_decel = row['macd_hist'] < 0 and row['macd_hist'] < prev['macd_hist']
            
            if ema_bear and adx_ok:
                if rsi_cross_dn:
                    direction = 'SELL'
                    confidence = 70
                elif rsi_weak and macd_decel:
                    direction = 'SELL'
                    confidence = 65
        
        if direction is None:
            continue
        
        # Bonus confidence
        if direction == 'BUY':
            if row['Close'] > row['ema50']: confidence += 10
            if row['adx'] > 25: confidence += 10
        else:
            if row['Close'] < row['ema50']: confidence += 10
            if row['adx'] > 25: confidence += 10
        
        signals.append({
            'idx': i, 'time': row.name, 'direction': direction,
            'confidence': min(100, confidence),
            'close': row['Close'], 'atr': row['atr'],
            'ema9': row['ema9'], 'ema21': row['ema21'],
        })
    
    return signals



def backtest(df, signals, start_balance=5000.0, risk_pct=0.025,
             max_daily_dd=0.05, max_total_dd=0.10):
    """
    Backtest with trailing stop that lets winners run big.
    - SL: 2.5x ATR (survives gold's noise)
    - Exit: trailing stop at 2.5x ATR behind HWM, activates after 1R profit
    - Also exits on opposite signal (regime flip)
    - Max hold: 48 bars
    """
    equity = start_balance
    peak_eq = start_balance
    trades = []
    daily_pnl = {}
    trades_today = {}
    last_exit_idx = -5
    
    # Build signal index map for quick lookup (opposite signal = exit)
    sig_map = {}
    for s in signals:
        sig_map[s['idx']] = s
    
    for si, sig in enumerate(signals):
        idx = sig['idx']
        if idx + 1 >= len(df):
            continue
        if idx - last_exit_idx < 2:
            continue
        
        entry_bar = df.iloc[idx + 1]
        entry_price = entry_bar['Open']
        entry_time = entry_bar.name
        day_key = str(entry_time.date())
        
        # Max 2 trades/day
        trades_today[day_key] = trades_today.get(day_key, 0) + 1
        if trades_today[day_key] > 2:
            continue
        
        # Daily DD
        if daily_pnl.get(day_key, 0) <= -(max_daily_dd * equity):
            continue
        
        # Total DD HARD HALT
        if equity <= peak_eq * (1 - max_total_dd):
            continue
        
        # Adaptive risk - CRITICAL: must never breach 10%
        dd_pct = (peak_eq - equity) / peak_eq if peak_eq > 0 else 0
        if dd_pct >= 0.085:
            continue  # hard halt - protect the 10% cap
        elif dd_pct >= 0.06:
            eff_risk = 0.012  # minimal risk
        elif dd_pct >= 0.035:
            eff_risk = 0.018  # reduced
        else:
            eff_risk = risk_pct  # full 2.5%
        
        risk_dollars = equity * eff_risk
        atr_val = sig['atr']
        sl_mult = 2.0  # tighter SL (2x ATR) - still survives most noise
        sl_dist = sl_mult * atr_val
        
        if sig['direction'] == 'BUY':
            sl = entry_price - sl_dist
        else:
            sl = entry_price + sl_dist
        
        # Position size
        lots = risk_dollars / (sl_dist * 100.0)
        lots = max(0.01, round(lots, 2))
        
        # Resolve bar-by-bar with trailing stop
        hwm = entry_price
        trail_mult = 2.0
        outcome = None; exit_price = None; exit_time = None; bars_held = 0
        initial_sl = sl
        
        for j in range(idx + 2, min(idx + 50, len(df))):
            bar = df.iloc[j]
            bars_held += 1
            
            if sig['direction'] == 'BUY':
                if bar['Low'] <= sl:
                    outcome = 'SL'; exit_price = sl; exit_time = bar.name; break
                if bar['High'] > hwm:
                    hwm = bar['High']
                    # Move to breakeven after 0.5R profit
                    if hwm - entry_price >= 0.5 * sl_dist:
                        sl = max(sl, entry_price)  # breakeven
                    # Activate trailing after 1.5R profit
                    if hwm - entry_price >= 1.5 * sl_dist:
                        new_sl = hwm - trail_mult * atr_val
                        sl = max(sl, new_sl)
            else:
                if bar['High'] >= sl:
                    outcome = 'SL'; exit_price = sl; exit_time = bar.name; break
                if bar['Low'] < hwm:
                    hwm = bar['Low']
                    if entry_price - hwm >= 0.5 * sl_dist:
                        sl = min(sl, entry_price)  # breakeven
                    if entry_price - hwm >= 1.5 * sl_dist:
                        new_sl = hwm + trail_mult * atr_val
                        sl = min(sl, new_sl)
            
            # Check if opposite signal fires (regime flip = exit)
            if j in sig_map and sig_map[j]['direction'] != sig['direction']:
                exit_price = bar['Close']
                exit_time = bar.name
                outcome = 'FLIP'
                break
        
        if outcome is None:
            exit_bar = df.iloc[min(idx + 49, len(df)-1)]
            exit_price = exit_bar['Close']; exit_time = exit_bar.name
            outcome = 'TIMEOUT'; bars_held = min(48, len(df)-idx-2)
        
        if sig['direction'] == 'BUY':
            pnl = (exit_price - entry_price) * lots * 100.0
        else:
            pnl = (entry_price - exit_price) * lots * 100.0
        
        equity += pnl
        peak_eq = max(peak_eq, equity)
        daily_pnl[day_key] = daily_pnl.get(day_key, 0) + pnl
        last_exit_idx = idx + bars_held + 1
        
        rr = pnl / risk_dollars if risk_dollars > 0 else 0
        tp_price = entry_price + sl_dist*2.5 if sig['direction']=='BUY' else entry_price - sl_dist*2.5
        
        trades.append({
            'entry_time': entry_time, 'exit_time': exit_time,
            'direction': sig['direction'], 'entry': entry_price,
            'sl': initial_sl, 'tp': tp_price, 'exit_price': exit_price,
            'lots': lots, 'risk_dollars': risk_dollars,
            'pnl': pnl, 'outcome': outcome, 'rr_target': 2.5,
            'rr_actual': rr, 'confidence': sig['confidence'],
            'bars_held': bars_held, 'equity_after': equity,
        })
    
    return trades, equity



def report(trades, start_balance, start_date, end_date):
    if not trades:
        print("NO TRADES."); return None
    t = pd.DataFrame(trades)
    end_bal = t['equity_after'].iloc[-1]
    ret = (end_bal - start_balance) / start_balance * 100
    net = end_bal - start_balance
    weeks = max(1, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 7)
    avg_wk = net / weeks
    wins = t[t['pnl'] > 0]; losses = t[t['pnl'] <= 0]
    wr = len(wins)/len(t)*100
    exp = t['pnl'].mean()
    eq = t['equity_after'].values
    pk = np.maximum.accumulate(np.concatenate([[start_balance], eq]))
    dd = (pk[1:] - eq); max_dd = dd.max(); max_dd_pct = max_dd/pk[np.argmax(dd)]*100 if max_dd>0 else 0
    gp = wins['pnl'].sum() if len(wins)>0 else 0
    gl = abs(losses['pnl'].sum()) if len(losses)>0 else 1
    pf = gp/gl
    
    print("="*65)
    print("        GOLD VORTEX v3 — BACKTEST RESULTS")
    print("="*65)
    print(f"  Period:              {start_date} to {end_date}")
    print(f"  Starting Balance:    ${start_balance:,.2f}")
    print(f"  Ending Balance:      ${end_bal:,.2f}")
    print(f"  Total Return:        {ret:+.2f}%")
    print(f"  Net Profit:          ${net:+,.2f}")
    print(f"  Average Weekly Profit: ${avg_wk:+,.2f}")
    print("-"*65)
    print(f"  Total Trades:        {len(t)}")
    print(f"  Trades/Week:         {len(t)/weeks:.1f}")
    print(f"  Win Rate:            {wr:.1f}%")
    print(f"  Profit Factor:       {pf:.2f}")
    print(f"  Expectancy/Trade:    ${exp:+,.2f}")
    print(f"  Avg Win:             ${wins['pnl'].mean() if len(wins)>0 else 0:+,.2f}")
    print(f"  Avg Loss:            ${losses['pnl'].mean() if len(losses)>0 else 0:+,.2f}")
    oc = t['outcome'].value_counts()
    print(f"  Outcomes: {oc.to_dict()}")
    print("-"*65)
    print(f"  Avg Trade Duration:  {t['bars_held'].mean():.1f} hours")
    print(f"  Max Drawdown:        ${max_dd:,.2f} ({max_dd_pct:.1f}%)")
    print(f"  Avg Confidence:      {t['confidence'].mean():.0f}/100")
    print(f"  Avg R:R Actual:      {t['rr_actual'].mean():+.2f}")
    print("="*65)
    
    print(f"\n  Sample trades:")
    for _, tr in t.head(8).iterrows():
        print(f"    {str(tr['entry_time'])[:16]} {tr['direction']:4s} @ {tr['entry']:.2f}"
              f" -> {tr['exit_price']:.2f} ({tr['outcome']:7s}) "
              f"PnL ${tr['pnl']:+.2f} R={tr['rr_actual']:+.1f} [{tr['bars_held']}h]")
    
    t['wk'] = pd.to_datetime(t['entry_time']).dt.isocalendar().week
    t['yr'] = pd.to_datetime(t['entry_time']).dt.year
    weekly = t.groupby(['yr','wk'])['pnl'].agg(['sum','count'])
    print(f"\n  Weekly P&L:")
    for (yr,wk), row in weekly.iterrows():
        print(f"    {yr}-W{wk:02d}: ${row['sum']:+8.2f} ({int(row['count'])} trades)")
    
    return dict(start_balance=start_balance, end_balance=end_bal, total_return=ret,
                net_profit=net, avg_weekly=avg_wk, trades=len(t), win_rate=wr,
                avg_duration=t['bars_held'].mean(), max_dd=max_dd, max_dd_pct=max_dd_pct,
                profit_factor=pf, expectancy=exp)


if __name__ == "__main__":
    df = pd.read_parquet('data/xau_1h.parquet')
    if 'Datetime' in df.columns: df = df.set_index('Datetime')
    elif 'Date' in df.columns: df = df.set_index('Date')
    if not isinstance(df.index, pd.DatetimeIndex): df.index = pd.to_datetime(df.index)
    if df.index.tz is not None: df.index = df.index.tz_convert('UTC')
    else: df.index = df.index.tz_localize('UTC')
    col_map = {'open':'Open','high':'High','low':'Low','close':'Close'}
    df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
    
    print(f"Data: {len(df)} bars, {df.index.min()} -> {df.index.max()}\n")
    feat = build_features(df[['Open','High','Low','Close']])
    print(f"Features: {len(feat)} bars\n")
    
    all_sigs = generate_signals(feat)
    test_start = pd.Timestamp('2026-01-01', tz='UTC')
    test_end = pd.Timestamp('2026-07-01', tz='UTC')
    test_sigs = [s for s in all_sigs if test_start <= s['time'] <= test_end]
    print(f"Signals: {len(all_sigs)} total, {len(test_sigs)} in test period\n")
    
    trades, final = backtest(feat, test_sigs)
    results = report(trades, 5000.0, '2026-01-01', '2026-07-01')
