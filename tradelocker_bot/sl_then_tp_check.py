"""
For every signal that got fully stopped out (SL hit before TP1), check what
price did AFTERWARD: did it go on to reach the original TP1 / TP2 / Final
level anyway (just after the stop had already closed the signal)?

This directly answers: "how many times did price hit SL and then go right
to where the TP would have been?"

Uses the exact same signal detection as fullyear_backtest.py (1H gate / 15m
pullback / 1m entry, 15m-ATR stops, sl_mult=1.0 - the live config).
"""
import pandas as pd
import numpy as np
import fullyear_backtest as fb

FOLLOWUP_BARS = 500  # how many 1m bars to watch AFTER the SL hit (~8h of entry-TF bars)


def run(sl_mult=1.0, followup_bars=FOLLOWUP_BARS):
    df1m = fb.load_data()
    gate_tf, pb_tf, entry_tf = "1h", "15m", "1m"

    g = fb.resample_tf(df1m, fb.TF_MIN[gate_tf])
    p = fb.resample_tf(df1m, fb.TF_MIN[pb_tf])
    e = fb.resample_tf(df1m, fb.TF_MIN[entry_tf])

    xa_df = fb.resample_tf(df1m, fb.TF_MIN["15m"])
    xa = fb.atr(xa_df).dropna()
    exit_atr_map = pd.DataFrame({"dt": xa.index, "x_atr": xa.values}).sort_values("dt")

    g_e20 = fb.ema(g["c"], 20)
    g_e50 = fb.ema(g["c"], 50)
    gate = pd.DataFrame({"g_e20": g_e20, "g_e50": g_e50}, index=g.index).dropna()

    p_e20 = fb.ema(p["c"], 20)
    p_atr = fb.atr(p)
    pb = pd.DataFrame({"p_c": p["c"], "p_e20": p_e20, "p_atr": p_atr}, index=p.index).dropna()

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

    m = pd.merge_asof(ent, gate_r, on="dt", direction="backward")
    m = pd.merge_asof(m, pb_r, on="dt", direction="backward")
    m = pd.merge_asof(m, exit_atr_map, on="dt", direction="backward")
    m = m.dropna(subset=["g_e20", "g_e50", "p_e20", "p_atr", "x_atr"])

    m["dir"] = np.where(m["g_e20"] > m["g_e50"], 1, np.where(m["g_e20"] < m["g_e50"], -1, 0))
    m["pb_dist"] = (m["p_c"] - m["p_e20"]).abs() / m["p_atr"].replace(0, np.nan)
    pb_ok = m["pb_dist"] <= 1.5
    buy_trig = (m["prev_l"] <= m["prev_e20"] * 1.002) & (m["prev_c"] > m["prev_o"]) & (m["rsi"] < 65)
    sell_trig = (m["prev_h"] >= m["prev_e20"] * 0.998) & (m["prev_c"] < m["prev_o"]) & (m["rsi"] > 35)
    trig = ((m["dir"] == 1) & buy_trig) | ((m["dir"] == -1) & sell_trig)
    m["signal"] = (m["dir"] != 0) & pb_ok & trig & (m["atr"] > 0)
    cand = m[m["signal"]].copy()

    e_h = e["h"].values
    e_l = e["l"].values
    e_c = e["c"].values
    e_idx = {ts: i for i, ts in enumerate(e.index)}

    cooldown_bars = max(1, 180 // fb.TF_MIN[entry_tf])
    last_i = -10 ** 9
    max_hold_bars = 200

    total_signals = 0
    pure_sl_count = 0        # signals fully stopped (SL hit before TP1)
    reached_tp1_after_sl = 0
    reached_tp2_after_sl = 0
    reached_tp3_after_sl = 0
    examples = []

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
        sl_dist = sl_mult * a + spread / 2
        tp1 = sl_dist + spread
        tp2 = 2 * sl_dist + spread
        tp3 = 3 * sl_dist + spread

        outcome = "timeout"
        sl_bar = None
        for j in range(i + 1, min(i + max_hold_bars, len(e_c))):
            hi, lo = e_h[j], e_l[j]
            if direction == 1:
                if lo <= entry_price - sl_dist:
                    outcome = "sl"
                    sl_bar = j
                    break
                if hi >= entry_price + tp1:
                    outcome = "tp1_or_better"
                    break
            else:
                if hi >= entry_price + sl_dist:
                    outcome = "sl"
                    sl_bar = j
                    break
                if lo <= entry_price - tp1:
                    outcome = "tp1_or_better"
                    break

        total_signals += 1
        last_i = i

        if outcome != "sl":
            continue
        pure_sl_count += 1

        # Now watch what price does AFTER the SL bar, relative to the
        # ORIGINAL entry price's TP1/TP2/TP3 (same direction).
        end_j = min(sl_bar + followup_bars, len(e_h) - 1, len(e_l) - 1)
        window_h = e_h[sl_bar:end_j + 1]
        window_l = e_l[sl_bar:end_j + 1]

        if direction == 1:
            target1, target2, target3 = entry_price + tp1, entry_price + tp2, entry_price + tp3
            hit1 = (window_h >= target1).any()
            hit2 = (window_h >= target2).any()
            hit3 = (window_h >= target3).any()
        else:
            target1, target2, target3 = entry_price - tp1, entry_price - tp2, entry_price - tp3
            hit1 = (window_l <= target1).any()
            hit2 = (window_l <= target2).any()
            hit3 = (window_l <= target3).any()

        if hit1:
            reached_tp1_after_sl += 1
        if hit2:
            reached_tp2_after_sl += 1
        if hit3:
            reached_tp3_after_sl += 1
            if len(examples) < 8:
                examples.append({
                    "dt": row["dt"], "dir": "buy" if direction == 1 else "sell",
                    "entry": round(entry_price, 2), "sl_dist": round(sl_dist, 2),
                    "would_be_final_tp": round(target3, 2),
                })

    print(f"Config: 1H/15m/1m, sl_mult={sl_mult}, followup window={followup_bars} entry-TF bars")
    print(f"Total signals: {total_signals}")
    print(f"Fully stopped out (SL before TP1): {pure_sl_count}  "
          f"({pure_sl_count/total_signals*100:.1f}% of all signals)")
    print()
    print(f"Of those {pure_sl_count} stopped-out signals, price LATER reached "
          f"(within {followup_bars} bars after the stop):")
    print(f"  -> original TP1 level  : {reached_tp1_after_sl}  "
          f"({reached_tp1_after_sl/pure_sl_count*100:.1f}% of stop-outs)")
    print(f"  -> original TP2 level  : {reached_tp2_after_sl}  "
          f"({reached_tp2_after_sl/pure_sl_count*100:.1f}% of stop-outs)")
    print(f"  -> original FINAL(3R)  : {reached_tp3_after_sl}  "
          f"({reached_tp3_after_sl/pure_sl_count*100:.1f}% of stop-outs)")
    print()
    if examples:
        print("Examples where price hit SL then later reached the would-be FINAL TP:")
        for ex in examples:
            print(f"  {ex['dt']} | {ex['dir'].upper()} entry {ex['entry']} | "
                  f"SL dist {ex['sl_dist']} | would-be final TP {ex['would_be_final_tp']}")
    return {
        "total_signals": total_signals,
        "pure_sl_count": pure_sl_count,
        "reached_tp1_after_sl": reached_tp1_after_sl,
        "reached_tp2_after_sl": reached_tp2_after_sl,
        "reached_tp3_after_sl": reached_tp3_after_sl,
    }


if __name__ == "__main__":
    run()
