"""
ADAPTIVE SL/TP backtest.

The static-1.0x-ATR sweep proved the current SL width is near-optimal ON
AVERAGE. But sl_then_tp_check.py proved 103/367 stop-outs (28%) later ran to
where the final TP would have been anyway - a real, measurable stop-hunt
pattern. A single fixed multiplier can't fix that: some signals need a wider
stop (choppy/expanding volatility), others can use a tighter one (calm,
strongly-trending). This module makes SL/TP width ADAPTIVE per-signal.

Two features, both computed ONLY from data available AT signal time (no
look-ahead):

  1. VOL_RATIO = current 15m ATR / rolling 50-bar average 15m ATR
     >1 => volatility is expanding right now (more likely to get stop-hunted
     by noise before the real move) => widen the stop.
     <1 => calm/contracting volatility => a tighter stop is efficient.

  2. TREND_STRENGTH = |EMA20_1h - EMA50_1h| / ATR_1h
     Higher => the 1H trend is strongly separated (more conviction the move
     continues) => extend how far TP3 (the runner) is allowed to go.
     Lower => weak/borderline trend => keep TPs closer, don't overstay.

Adaptive formula (both multiplicatively clipped so it never runs away):
  sl_mult_i = base_sl * clip(vol_ratio_i, vol_lo, vol_hi)
  tp_ext_i  = clip(1 + trend_gain * (trend_strength_i - 1), tp_lo, tp_hi)
  tp1 = sl_dist + spread
  tp2 = 2*sl_dist + spread
  tp3 = 3*sl_dist*tp_ext_i + spread      <- only the runner leg is extended
"""
import pandas as pd
import numpy as np
import fullyear_backtest as fb


def compute_features(df1m, gate_tf="1h", pb_tf="15m", entry_tf="1m", exit_atr_tf="15m"):
    """Build the merged signal dataframe (same as fullyear_backtest) plus the
    two adaptive features, all backward-looking (no look-ahead)."""
    g = fb.resample_tf(df1m, fb.TF_MIN[gate_tf])
    p = fb.resample_tf(df1m, fb.TF_MIN[pb_tf])
    e = fb.resample_tf(df1m, fb.TF_MIN[entry_tf])
    xa_df = fb.resample_tf(df1m, fb.TF_MIN[exit_atr_tf])

    # Gate: EMA20/50 + trend strength (normalized separation)
    g_e20 = fb.ema(g["c"], 20)
    g_e50 = fb.ema(g["c"], 50)
    g_atr = fb.atr(g, 14)
    trend_strength = ((g_e20 - g_e50).abs() / g_atr.replace(0, np.nan))
    gate = pd.DataFrame({"g_e20": g_e20, "g_e50": g_e50, "trend_strength": trend_strength},
                        index=g.index).dropna()

    # Pullback zone
    p_e20 = fb.ema(p["c"], 20)
    p_atr = fb.atr(p)
    pb = pd.DataFrame({"p_c": p["c"], "p_e20": p_e20, "p_atr": p_atr}, index=p.index).dropna()

    # Exit ATR (15m) + its own 50-bar rolling average -> volatility-expansion ratio
    xa = fb.atr(xa_df, 14)
    xa_avg = xa.rolling(50).mean()
    vol_ratio = (xa / xa_avg.replace(0, np.nan))
    exit_map = pd.DataFrame({"x_atr": xa, "vol_ratio": vol_ratio}, index=xa.index).dropna()

    # Entry
    e_e20 = fb.ema(e["c"], 20)
    e_rsi = fb.rsi(e["c"])
    e_atr = fb.atr(e)
    ent = e.copy()
    ent["e20"] = e_e20
    ent["rsi"] = e_rsi
    ent["atr"] = e_atr
    ent["prev_o"] = ent["o"].shift(1)
    ent["prev_h"] = ent["h"].shift(1)
    ent["prev_l"] = ent["l"].shift(1)
    ent["prev_c"] = ent["c"].shift(1)
    ent["prev_e20"] = ent["e20"].shift(1)
    ent = ent.dropna().reset_index().rename(columns={"index": "dt"}).sort_values("dt")

    gate_r = gate.reset_index().rename(columns={"index": "dt"}).sort_values("dt")
    pb_r = pb.reset_index().rename(columns={"index": "dt"}).sort_values("dt")
    exit_r = exit_map.reset_index().rename(columns={"index": "dt"}).sort_values("dt")

    m = pd.merge_asof(ent, gate_r, on="dt", direction="backward")
    m = pd.merge_asof(m, pb_r, on="dt", direction="backward")
    m = pd.merge_asof(m, exit_r, on="dt", direction="backward")
    m = m.dropna(subset=["g_e20", "g_e50", "p_e20", "p_atr", "x_atr", "vol_ratio", "trend_strength"])

    m["dir"] = np.where(m["g_e20"] > m["g_e50"], 1, np.where(m["g_e20"] < m["g_e50"], -1, 0))
    m["pb_dist"] = (m["p_c"] - m["p_e20"]).abs() / m["p_atr"].replace(0, np.nan)
    pb_ok = m["pb_dist"] <= 1.5
    buy_trig = (m["prev_l"] <= m["prev_e20"] * 1.002) & (m["prev_c"] > m["prev_o"]) & (m["rsi"] < 65)
    sell_trig = (m["prev_h"] >= m["prev_e20"] * 0.998) & (m["prev_c"] < m["prev_o"]) & (m["rsi"] > 35)
    trig = ((m["dir"] == 1) & buy_trig) | ((m["dir"] == -1) & sell_trig)
    m["signal"] = (m["dir"] != 0) & pb_ok & trig & (m["atr"] > 0)

    return m, e


def backtest_adaptive(df1m, base_sl=1.0, vol_lo=0.7, vol_hi=1.6, trend_gain=0.5,
                      tp_lo=0.7, tp_hi=1.8, cooldown_min=180, max_hold_bars=200,
                      return_signals=False):
    m, e = compute_features(df1m)
    cand = m[m["signal"]].copy()
    if cand.empty:
        return None

    e_h = e["h"].values
    e_l = e["l"].values
    e_c = e["c"].values
    e_idx = {ts: i for i, ts in enumerate(e.index)}

    cooldown_bars = max(1, cooldown_min // 1)  # entry_tf = 1m
    results = []
    sig_records = []
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

        vol_ratio = np.clip(row["vol_ratio"], vol_lo, vol_hi)
        trend_strength = row["trend_strength"]
        tp_ext = np.clip(1 + trend_gain * (trend_strength - 1), tp_lo, tp_hi)

        spread = 0.30
        sl_mult_i = base_sl * vol_ratio
        sl_dist = sl_mult_i * a + spread / 2
        tp1 = sl_dist + spread
        tp2 = 2 * sl_dist + spread
        tp3 = 3 * sl_dist * tp_ext + spread   # only the runner leg adapts to trend strength

        tp1_hit = tp2_hit = False
        outcome = "timeout"
        for j in range(i + 1, min(i + max_hold_bars, len(e_c))):
            hi, lo = e_h[j], e_l[j]
            if direction == 1:
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

        if outcome == "tp_final":
            r = (0.10 * tp1 + 0.20 * tp2 + 0.70 * tp3) / sl_dist
        elif outcome == "final_be":
            r = (0.10 * tp1 + 0.20 * tp2 + 0.70 * tp1) / sl_dist
        elif outcome == "tp1_be":
            r = (0.10 * tp1) / sl_dist
        elif outcome == "sl":
            r = -1.0
        else:
            j2 = min(i + max_hold_bars - 1, len(e_c) - 1)
            move = (e_c[j2] - entry_price) * direction
            r = move / sl_dist

        results.append(r)
        sig_records.append((row["dt"], r, sl_dist, entry_price))
        last_i = i

    if not results:
        return None

    results = np.array(results)
    wins = int((results > 0).sum())
    losses = int((results < 0).sum())
    total = len(results)
    out = {
        "signals": total, "wins": wins, "losses": losses,
        "win_rate": round(wins / total * 100, 1),
        "total_r": round(float(results.sum()), 2),
        "avg_r": round(float(results.mean()), 3),
    }
    if return_signals:
        out["records"] = sig_records
    return out


def dollar_pnl(r, sl_dist, entry, equity=5000.0):
    contract = 100.0
    leverage = 10.0
    loss_per_lot = contract * sl_dist
    lots = min(0.12, 60.0 / loss_per_lot)
    lots = max(0.01, round(int(lots * 100) / 100, 2))
    margin = lots * contract * entry / leverage
    if margin > equity * 0.95:
        lots = max(0.01, round(int((equity * 0.95 * leverage / (contract * entry)) * 100) / 100, 2))
    risk = lots * loss_per_lot
    return r * risk, risk


def dollar_stats(records):
    rows = []
    for dt, r, sd, e in records:
        pnl, risk = dollar_pnl(r, sd, e)
        rows.append({"dt": pd.Timestamp(dt), "pnl": pnl, "risk": risk})
    d = pd.DataFrame(rows).set_index("dt").sort_index()
    eq = 5000 + d["pnl"].cumsum()
    dd = (eq.cummax() - eq).max()
    return {
        "total_d": round(float(d["pnl"].sum()), 2),
        "avg_d": round(float(d["pnl"].mean()), 2),
        "max_dd": round(float(dd), 2),
        "avg_risk": round(float(d["risk"].mean()), 2),
    }


if __name__ == "__main__":
    df1m = fb.load_data()
    print(f"IN-SAMPLE window: {df1m.index.min()} -> {df1m.index.max()} | {len(df1m)} bars")
    print()

    # Baseline: static 1.0x (from previous sweep)
    base = fb.backtest(df1m, "1h", "15m", "1m", exit_atr_tf="15m", return_signals=True, sl_mult=1.0)
    base_d = dollar_stats(base["records"])
    print(f"BASELINE (static 1.0x ATR): signals={base['signals']} WR={base['win_rate']}% "
          f"total_R={base['total_r']:+.2f} | ${base_d['total_d']:+.2f} avg=${base_d['avg_d']:.2f} "
          f"maxDD=${base_d['max_dd']:.2f}")
    print()

    # Adaptive parameter grid
    print(f"{'vol_lo':>7} {'vol_hi':>7} {'tgain':>6} {'tp_lo':>6} {'tp_hi':>6} "
          f"{'SIG':>5} {'WR%':>6} {'TOT_R':>8} {'TOT_$':>10} {'AVG_$':>7} {'MAXDD_$':>8}")
    print("-" * 88)
    grid = []
    for vol_lo, vol_hi in [(0.6, 1.4), (0.7, 1.6), (0.8, 1.8), (0.5, 2.0)]:
        for trend_gain in [0.0, 0.3, 0.5, 0.8]:
            for tp_lo, tp_hi in [(0.8, 1.5), (0.7, 1.8), (1.0, 1.0)]:
                res = backtest_adaptive(df1m, base_sl=1.0, vol_lo=vol_lo, vol_hi=vol_hi,
                                        trend_gain=trend_gain, tp_lo=tp_lo, tp_hi=tp_hi,
                                        return_signals=True)
                if not res:
                    continue
                d = dollar_stats(res["records"])
                grid.append((vol_lo, vol_hi, trend_gain, tp_lo, tp_hi, res, d))
                print(f"{vol_lo:>7.1f} {vol_hi:>7.1f} {trend_gain:>6.1f} {tp_lo:>6.1f} {tp_hi:>6.1f} "
                      f"{res['signals']:>5} {res['win_rate']:>5.1f}% {res['total_r']:>+8.2f} "
                      f"${d['total_d']:>9.2f} ${d['avg_d']:>6.2f} ${d['max_dd']:>7.2f}")

    grid.sort(key=lambda x: x[6]["total_d"], reverse=True)
    print()
    print("=" * 88)
    best = grid[0]
    print(f"BEST ADAPTIVE: vol_lo={best[0]} vol_hi={best[1]} trend_gain={best[2]} "
          f"tp_lo={best[3]} tp_hi={best[4]}")
    print(f"  signals={best[5]['signals']} WR={best[5]['win_rate']}% total_R={best[5]['total_r']:+.2f} "
          f"| ${best[6]['total_d']:+.2f} avg=${best[6]['avg_d']:.2f} maxDD=${best[6]['max_dd']:.2f}")
    print(f"  vs BASELINE: ${base_d['total_d']:+.2f} avg=${base_d['avg_d']:.2f} maxDD=${base_d['max_dd']:.2f}")
    improvement = best[6]["total_d"] - base_d["total_d"]
    print(f"  IMPROVEMENT: ${improvement:+.2f} ({improvement/abs(base_d['total_d'])*100:+.1f}%)")
    print("=" * 88)
