"""
Full-year XAUUSD backtest using real 1-minute data (Hugging Face Ashraf-CK/XAUUSD).

Replicates the bot's exact ASWP logic:
  - GATE:     EMA20 vs EMA50 on gate timeframe  -> direction
  - PULLBACK: price within 1.5x ATR of EMA20 on pullback timeframe
  - ENTRY:    prev candle touches EMA20 + closes in direction + RSI filter

Multi-TP exit: 10% at 1R, 20% at 2R, 70% at 3R, SL at 1x ATR (entry TF).
Uses merge_asof to align higher-timeframe signals onto the entry timeline
(no look-ahead: only completed higher-TF bars are used).

Tests the top combos on a full year (most recent 12 months available).
"""
import pandas as pd
import numpy as np

TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}


def load_data():
    df1 = pd.read_parquet("data/xau_train.parquet")
    df2 = pd.read_parquet("data/xau_test.parquet")
    df = pd.concat([df1, df2], ignore_index=True)
    # Build datetime from Date (YYYYMMDD) + Time (HHMMSS)
    dt = pd.to_datetime(
        df["Date"].astype(str).str.zfill(8) + df["Time"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S", errors="coerce"
    )
    df = df.assign(dt=dt).dropna(subset=["dt"])
    df = df[["dt", "Open", "High", "Low", "Close"]].rename(
        columns={"Open": "o", "High": "h", "Low": "l", "Close": "c"})
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    # Most recent full 12 months
    end = df.index.max()
    start = end - pd.Timedelta(days=365)
    df = df.loc[start:end]
    return df


def resample_tf(df1m, minutes):
    if minutes == 1:
        return df1m.copy()
    rule = f"{minutes}min"
    out = df1m.resample(rule, label="right", closed="right").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    return out


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def atr(df, period=14):
    h, l, c = df["h"], df["l"], df["c"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def backtest(df1m, gate_tf, pb_tf, entry_tf, cooldown_min=180, max_hold_bars=200,
             exit_atr_tf=None, return_signals=False):
    # Build timeframes
    g = resample_tf(df1m, TF_MIN[gate_tf])
    p = resample_tf(df1m, TF_MIN[pb_tf])
    e = resample_tf(df1m, TF_MIN[entry_tf])

    if len(g) < 60 or len(p) < 30 or len(e) < 50:
        return None

    # Optional: source the exit (SL/TP) ATR from a higher timeframe for
    # practical stop sizing, aligned onto the entry timeline.
    exit_atr_map = None
    if exit_atr_tf is not None and exit_atr_tf != entry_tf:
        xa_df = resample_tf(df1m, TF_MIN[exit_atr_tf])
        xa = atr(xa_df).dropna()
        exit_atr_map = pd.DataFrame({"dt": xa.index, "x_atr": xa.values}).sort_values("dt")

    # Gate indicators
    g_e20 = ema(g["c"], 20)
    g_e50 = ema(g["c"], 50)
    gate = pd.DataFrame({"g_e20": g_e20, "g_e50": g_e50}, index=g.index).dropna()

    # Pullback indicators
    p_e20 = ema(p["c"], 20)
    p_atr = atr(p)
    pb = pd.DataFrame({"p_c": p["c"], "p_e20": p_e20, "p_atr": p_atr}, index=p.index).dropna()

    # Entry indicators
    e_e20 = ema(e["c"], 20)
    e_rsi = rsi(e["c"])
    e_atr = atr(e)
    ent = e.copy()
    ent["e20"] = e_e20
    ent["rsi"] = e_rsi
    ent["atr"] = e_atr
    ent["prev_o"] = ent["o"].shift(1)
    ent["prev_h"] = ent["h"].shift(1)
    ent["prev_l"] = ent["l"].shift(1)
    ent["prev_c"] = ent["c"].shift(1)
    ent["prev_e20"] = ent["e20"].shift(1)
    ent = ent.dropna()

    # Align higher TFs onto entry timeline (only completed bars -> merge_asof backward)
    ent = ent.reset_index().rename(columns={"index": "dt", "dt": "dt"})
    ent = ent.sort_values("dt")
    gate_r = gate.reset_index().rename(columns={"index": "dt"}).sort_values("dt")
    pb_r = pb.reset_index().rename(columns={"index": "dt"}).sort_values("dt")
    if "dt" not in gate_r.columns:
        gate_r = gate_r.rename(columns={gate_r.columns[0]: "dt"})
    if "dt" not in pb_r.columns:
        pb_r = pb_r.rename(columns={pb_r.columns[0]: "dt"})

    m = pd.merge_asof(ent, gate_r, on="dt", direction="backward")
    m = pd.merge_asof(m, pb_r, on="dt", direction="backward")
    if exit_atr_map is not None:
        m = pd.merge_asof(m, exit_atr_map, on="dt", direction="backward")
    else:
        m["x_atr"] = m["atr"]
    m = m.dropna(subset=["g_e20", "g_e50", "p_e20", "p_atr", "x_atr"])

    # Direction from gate
    m["dir"] = np.where(m["g_e20"] > m["g_e50"], 1, np.where(m["g_e20"] < m["g_e50"], -1, 0))

    # Pullback zone
    m["pb_dist"] = (m["p_c"] - m["p_e20"]).abs() / m["p_atr"].replace(0, np.nan)
    pb_ok = m["pb_dist"] <= 1.5

    # Entry trigger
    buy_trig = (m["prev_l"] <= m["prev_e20"] * 1.002) & (m["prev_c"] > m["prev_o"]) & (m["rsi"] < 65)
    sell_trig = (m["prev_h"] >= m["prev_e20"] * 0.998) & (m["prev_c"] < m["prev_o"]) & (m["rsi"] > 35)
    trig = ((m["dir"] == 1) & buy_trig) | ((m["dir"] == -1) & sell_trig)

    m["signal"] = (m["dir"] != 0) & pb_ok & trig & (m["atr"] > 0)

    cand = m[m["signal"]].copy()
    if cand.empty:
        return {"combo": f"{gate_tf}/{pb_tf}/{entry_tf}", "signals": 0, "wins": 0,
                "losses": 0, "win_rate": 0, "total_r": 0, "avg_r": 0, "tp_final_pct": 0}

    # Prepare forward arrays for outcome simulation
    e_h = e["h"].values
    e_l = e["l"].values
    e_c = e["c"].values
    e_idx = {ts: i for i, ts in enumerate(e.index)}

    cooldown_bars = max(1, cooldown_min // TF_MIN[entry_tf])
    results = []
    sig_records = []  # (dt, r, exit_atr, entry_price)
    last_i = -10 ** 9

    for _, row in cand.iterrows():
        i = e_idx.get(row["dt"])
        if i is None or i < 1:
            continue
        if i - last_i < cooldown_bars:
            continue
        direction = int(row["dir"])
        entry_price = e_c[i]
        a = row["x_atr"]
        if a <= 0:
            continue
        spread = 0.30
        sl_dist = a + spread / 2
        tp1 = a + spread
        tp2 = 2 * a + spread
        tp3 = 3 * a + spread

        tp1_hit = tp2_hit = False
        outcome = "timeout"
        for j in range(i + 1, min(i + max_hold_bars, len(e_c))):
            hi, lo = e_h[j], e_l[j]
            if direction == 1:
                # SL check (moves to BE after tp1, to tp1 after tp2)
                sl_level = entry_price - sl_dist
                if tp2_hit:
                    sl_level = entry_price + tp1
                elif tp1_hit:
                    sl_level = entry_price
                if lo <= sl_level:
                    outcome = "final_be" if tp2_hit else ("tp1_be" if tp1_hit else "sl")
                    break
                if not tp1_hit and hi >= entry_price + tp1:
                    tp1_hit = True
                if not tp2_hit and hi >= entry_price + tp2:
                    tp2_hit = True
                if hi >= entry_price + tp3:
                    outcome = "tp_final"
                    break
            else:
                sl_level = entry_price + sl_dist
                if tp2_hit:
                    sl_level = entry_price - tp1
                elif tp1_hit:
                    sl_level = entry_price
                if hi >= sl_level:
                    outcome = "final_be" if tp2_hit else ("tp1_be" if tp1_hit else "sl")
                    break
                if not tp1_hit and lo <= entry_price - tp1:
                    tp1_hit = True
                if not tp2_hit and lo <= entry_price - tp2:
                    tp2_hit = True
                if lo <= entry_price - tp3:
                    outcome = "tp_final"
                    break

        # P&L in R (risk = sl_dist). Multi-TP allocation 10/20/70.
        if outcome == "tp_final":
            r = (0.10 * tp1 + 0.20 * tp2 + 0.70 * tp3) / sl_dist
        elif outcome == "final_be":
            # tp1 + tp2 banked, remaining 70% exits at tp1 level
            r = (0.10 * tp1 + 0.20 * tp2 + 0.70 * tp1) / sl_dist
        elif outcome == "tp1_be":
            # tp1 banked (10%), remaining 90% exits at breakeven
            r = (0.10 * tp1) / sl_dist
        elif outcome == "sl":
            r = -1.0
        else:  # timeout - close at last price
            j2 = min(i + max_hold_bars - 1, len(e_c) - 1)
            move = (e_c[j2] - entry_price) * direction
            r = move / sl_dist
        results.append(r)
        sig_records.append((row["dt"], r, a, entry_price))
        last_i = i

    if not results:
        return {"combo": f"{gate_tf}/{pb_tf}/{entry_tf}", "signals": 0, "wins": 0,
                "losses": 0, "win_rate": 0, "total_r": 0, "avg_r": 0, "tp_final_pct": 0}

    results = np.array(results)
    wins = int((results > 0).sum())
    losses = int((results < 0).sum())
    total = len(results)
    out = {
        "combo": f"{gate_tf}/{pb_tf}/{entry_tf}",
        "signals": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1),
        "total_r": round(float(results.sum()), 2),
        "avg_r": round(float(results.mean()), 3),
        "tp_final_pct": round(float((results == results.max()).mean() * 100), 1),
    }
    if return_signals:
        out["records"] = sig_records
    return out


if __name__ == "__main__":
    print("Loading full-year 1m XAUUSD data...")
    df1m = load_data()
    print(f"Rows: {len(df1m)} | {df1m.index.min()} -> {df1m.index.max()}")
    print()

    # Test the key combos: current, optimal candidate, and neighbors
    combos = [
        ("30m", "15m", "1m"),   # optimizer winner
        ("30m", "5m", "1m"),
        ("1h", "15m", "1m"),
        ("1h", "30m", "1m"),
        ("15m", "5m", "1m"),
        ("1h", "15m", "5m"),    # CURRENT bot setting
        ("1h", "30m", "5m"),
        ("30m", "15m", "5m"),
        ("4h", "1h", "15m"),
        ("4h", "30m", "5m"),
    ]

    # PRACTICAL variants: 1m entry precision but stops sized from a higher TF
    # so the ~$0.30 spread is a small fraction of the stop.
    practical = [
        ("30m", "5m", "1m", "5m"),    # 1m entry, 5m ATR stops
        ("30m", "5m", "1m", "30m"),   # 1m entry, 30m ATR stops
        ("30m", "15m", "1m", "15m"),  # 1m entry, 15m ATR stops
        ("1h", "15m", "1m", "15m"),
        ("30m", "5m", "5m", "5m"),    # 5m entry, 5m ATR (like current but 30m gate)
        ("1h", "15m", "5m", "5m"),    # CURRENT bot
    ]

    print(f"{'COMBO (G/PB/E, exitATR)':<28} {'SIG':>5} {'WR%':>6} {'TOT R':>9} {'AVG R':>7}")
    print("-" * 62)
    prows = []
    for g, p, e, xa in practical:
        res = backtest(df1m, g, p, e, exit_atr_tf=xa)
        if res:
            res["label"] = f"{g}/{p}/{e} atr:{xa}"
            prows.append(res)
            tag = "  <- CURRENT" if (g, p, e, xa) == ("1h", "15m", "5m", "5m") else ""
            print(f"{res['label']:<28} {res['signals']:>5} {res['win_rate']:>5.1f}% "
                  f"{res['total_r']:>+9.2f} {res['avg_r']:>+7.3f}{tag}")

    prows.sort(key=lambda x: x["total_r"], reverse=True)
    print()
    print("=" * 62)
    best = prows[0]
    print(f"OPTIMAL PRACTICAL: {best['label']}")
    print(f"  {best['signals']} signals/yr (~{best['signals']/252:.1f}/day)  |  WR {best['win_rate']}%  |  "
          f"Total {best['total_r']:+.2f}R  |  Avg {best['avg_r']:+.3f}R")
    print("=" * 62)
