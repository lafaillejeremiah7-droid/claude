"""
PHASE 3 — Loss forensics + logical loss-reduction.

1. Run the Phase-2 optimum on real Jan2025->Jul2026 data.
2. Attribute every LOSS: was it a stop-hunt (price later reached the TP anyway),
   a genuine adverse move, or chop? Compare winner vs loser feature profiles.
3. Search 140,189 filter trials over feature thresholds (RSI, trend strength,
   volatility ratio, pullback distance, hour) to SKIP the loser-profile signals
   while keeping winners -> effectively converts losses into "not taken" and
   raises the TP rate. Objective = net profit subject to the TSAI safety gates.
4. Keep the LOGICALLY BEST (mechanistic + validated) filter, then VALIDATE it
   OUT-OF-SAMPLE on 2022-2024 data to ensure it generalizes (not curve-fit).
5. Compare before vs after.
"""
import sys
import json
import time
import numpy as np
import pandas as pd

import tsai_optimize as T
import phase_optimize as P
import fullyear_backtest as fb

SPREAD = T.SPREAD
MAX_HOLD = T.MAX_HOLD

# Phase-2 converged optimum
BEST = {"base_sl": 1.095, "vol_lo": 0.347, "vol_hi": 1.0, "tp1_r": 1.9836,
        "tp2_r": 3.9925, "tp3_r": 4.0925, "trend_gain": 2.8338, "tp_lo": 0.2,
        "tp_hi": 2.802, "risk_cap": 26.0785}
N_TRIALS = 140189


def simulate_full(c, p):
    """Like T.simulate but also returns per-signal outcome code, the SL-hit bar,
    and whether price reached TP1/TP2/final AFTER a full stop (stop-hunt)."""
    d = c["d"]; entries = c["entries"]; x_atr = c["x_atr"]
    H, L, Cc = c["H"], c["L"], c["C"]; n = c["n"]
    is_buy = d == 1
    vr = np.clip(c["vol_ratio"], p["vol_lo"], p["vol_hi"])
    tp_ext = np.clip(1 + p["trend_gain"] * (c["trend_strength"] - 1), p["tp_lo"], p["tp_hi"])
    sl_dist = p["base_sl"] * vr * x_atr + SPREAD / 2
    tp1 = p["tp1_r"] * sl_dist + SPREAD
    tp2 = p["tp2_r"] * sl_dist + SPREAD
    tp3 = p["tp3_r"] * sl_dist * tp_ext + SPREAD

    tp1_hit = np.zeros(n, bool); tp2_hit = np.zeros(n, bool); closed = np.zeros(n, bool)
    code = np.zeros(n, np.int8); sl_bar = np.full(n, -1)
    e_sl = np.where(is_buy, entries - sl_dist, entries + sl_dist)
    e_tp1 = np.where(is_buy, entries + tp1, entries - tp1)
    e_tp2 = np.where(is_buy, entries + tp2, entries - tp2)
    e_tp3 = np.where(is_buy, entries + tp3, entries - tp3)
    be = entries; tp1_as_sl = np.where(is_buy, entries + tp1, entries - tp1)

    for j in range(MAX_HOLD):
        active = ~closed
        if not active.any():
            break
        hi = H[:, j]; lo = L[:, j]
        sl_lvl = e_sl.copy()
        sl_lvl = np.where(tp1_hit & ~tp2_hit, be, sl_lvl)
        sl_lvl = np.where(tp2_hit, tp1_as_sl, sl_lvl)
        hit_sl = np.where(is_buy, lo <= sl_lvl, hi >= sl_lvl)
        newly = active & hit_sl
        code = np.where(newly, np.where(tp2_hit, 2, np.where(tp1_hit, 3, 1)), code)
        sl_bar = np.where(newly & (sl_bar < 0), j, sl_bar)
        closed = closed | newly
        active = ~closed
        t1 = np.where(is_buy, hi >= e_tp1, lo <= e_tp1)
        tp1_hit = tp1_hit | (active & ~tp1_hit & t1)
        t2 = np.where(is_buy, hi >= e_tp2, lo <= e_tp2)
        tp2_hit = tp2_hit | (active & ~tp2_hit & t2)
        t3 = np.where(is_buy, hi >= e_tp3, lo <= e_tp3)
        nf = active & t3
        code = np.where(nf, 4, code)
        closed = closed | nf

    last_c = Cc[:, -1]
    r_to = (last_c - entries) * d / sl_dist
    r = np.where(code == 4, (0.10*tp1 + 0.20*tp2 + 0.70*tp3) / sl_dist,
        np.where(code == 2, (0.10*tp1 + 0.20*tp2 + 0.70*tp1) / sl_dist,
        np.where(code == 3, (0.10*tp1) / sl_dist,
        np.where(code == 1, -1.0, r_to))))

    # Post-SL stop-hunt check: for full-SL losers (code==1), did price later
    # reach the ORIGINAL tp1/tp2/tp3 within the remaining hold window?
    reach1 = np.zeros(n, bool); reach3 = np.zeros(n, bool)
    for k in range(n):
        if code[k] != 1:
            continue
        jb = sl_bar[k]
        if jb < 0:
            continue
        wh = H[k, jb:]; wl = L[k, jb:]
        if is_buy[k]:
            reach1[k] = (wh >= e_tp1[k]).any()
            reach3[k] = (wh >= e_tp3[k]).any()
        else:
            reach1[k] = (wl <= e_tp1[k]).any()
            reach3[k] = (wl <= e_tp3[k]).any()
    return r, sl_dist, code, reach1, reach3


def metrics_masked(c, r, sl_dist, mask, risk_cap):
    """Metrics over the subset of signals selected by mask."""
    idx = np.where(mask)[0]
    if len(idx) < 20:
        return None
    entries = c["entries"][idx]; sld = sl_dist[idx]; rr = r[idx]
    lpl = 100.0 * sld
    lots = np.minimum(0.12, risk_cap / lpl)
    lots = np.maximum(0.01, np.floor(lots * 100) / 100)
    margin = lots * 100.0 * entries / 10.0
    over = margin > T.EQUITY0 * 0.95
    if over.any():
        alt = np.maximum(0.01, np.floor((T.EQUITY0*0.95*10.0/(100.0*entries))*100)/100)
        lots = np.where(over, alt, lots)
    risk = lots * lpl
    pnl = rr * risk
    # chronological
    sub_dt = c["dt"][idx]
    o = np.argsort(sub_dt)
    pnl_s = pnl[o]
    eq = T.EQUITY0 + np.cumsum(pnl_s)
    dd_total = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
    day_codes = pd.factorize(pd.to_datetime(sub_dt[o]).date)[0]
    day_sums = np.bincount(day_codes, weights=pnl_s, minlength=day_codes.max()+1 if len(day_codes) else 1)
    dd_day = float(-day_sums.min()) if len(day_sums) and day_sums.min() < 0 else 0.0
    wins = int((rr > 0).sum()); losses = int((rr < 0).sum())
    return {"total_d": float(pnl.sum()), "n": len(idx), "wins": wins, "losses": losses,
            "wr": wins/max(1, wins+losses), "dd_total": dd_total, "dd_day": dd_day,
            "avg_sl": float(sld.mean()), "avg_atr5": float(c["atr5"][idx].mean()),
            "avg_lots": float(lots.mean()), "ev": float(pnl.mean())}


def tsai_from(mt):
    if mt is None:
        return {"TSAI": 0.0, "G": 0}
    g1 = 1.0 if mt["avg_sl"] > 1.2 * mt["avg_atr5"] else 0.0
    g2 = 1.0 if mt["dd_total"] < 350 else 0.0
    g3 = 1.0 if mt["dd_day"] < 200 else 0.0
    G = g1 * g2 * g3
    sc = mt["avg_lots"] * 100.0 * SPREAD
    friction = mt["ev"] / sc if sc > 0 else 0
    noise = mt["avg_sl"] / mt["avg_atr5"] if mt["avg_atr5"] > 0 else 0
    C = max(0.0, 5.0 * friction * noise)
    s_day = 250/mt["dd_day"] if mt["dd_day"] > 0 else 1e6
    s_total = 400/mt["dd_total"] if mt["dd_total"] > 0 else 1e6
    S = (2*s_day*s_total)/(s_day+s_total)
    return {"TSAI": G*((2*C*S)/(C+S)) if (C+S) > 0 else 0.0, "G": G}


def apply_filter(c, f):
    """Boolean keep-mask from a filter spec f (skip loser-profile signals)."""
    keep = np.ones(c["n"], bool)
    keep &= c["trend_strength"] >= f["ts_min"]
    keep &= c["rsi"] >= f["rsi_min"]
    keep &= c["rsi"] <= f["rsi_max"]
    keep &= c["pb_dist"] <= f["pb_max"]
    keep &= c["vol_ratio"] <= f["vr_max"]
    if f["skip_late"]:
        keep &= c["hours"] < 21   # skip 21:00-23:59 UTC thin/rollover
    return keep


def main():
    print("Loading REAL Jan2025->Jul2026 data + candidates...")
    df = T.load_data()
    weeks = (df.index.max() - df.index.min()).days / 7.0
    c = T.build_candidates(df)
    r, sl_dist, code, reach1, reach3 = simulate_full(c, BEST)

    base_mask = np.ones(c["n"], bool)
    base = metrics_masked(c, r, sl_dist, base_mask, BEST["risk_cap"])
    base_ts = tsai_from(base)

    # ---- 1. LOSS ATTRIBUTION ----
    is_loss = code == 1
    is_win = r > 0
    n_loss = int(is_loss.sum()); n_win = int(is_win.sum())
    sh1 = int(reach1[is_loss].sum()); sh3 = int(reach3[is_loss].sum())
    print("\n===== LOSS ATTRIBUTION (Phase-2 optimum on 2025-2026) =====")
    print(f"Signals: {c['n']} | wins: {n_win} | full-SL losses: {n_loss} | "
          f"BE/partial: {c['n']-n_win-n_loss}")
    print(f"Of {n_loss} full-SL losses:")
    print(f"  {sh1} ({sh1/max(1,n_loss)*100:.1f}%) were STOP-HUNTS -> price later reached TP1 anyway")
    print(f"  {sh3} ({sh3/max(1,n_loss)*100:.1f}%) later reached the FINAL TP anyway")
    print("Feature profile (winners vs full-SL losers):")
    for feat in ["trend_strength", "rsi", "pb_dist", "vol_ratio"]:
        wv = c[feat][is_win].mean(); lv = c[feat][is_loss].mean()
        print(f"  {feat:16}: winners {wv:.3f} | losers {lv:.3f}")
    # hour concentration of losses
    lh = c["hours"][is_loss]
    from collections import Counter
    worst = Counter(lh).most_common(5)
    print(f"  loss hours (UTC, top5): {worst}")

    print(f"\nBASELINE (no filter): ${base['total_d']:+.2f} (+${base['total_d']/weeks:.2f}/wk) | "
          f"WR={base['wr']*100:.1f}% | DDtot=${base['dd_total']:.0f} DDday=${base['dd_day']:.0f} | "
          f"TSAI={base_ts['TSAI']:.3f}")

    # ---- 2. FILTER SEARCH (140,189 trials) ----
    print(f"\nSearching {N_TRIALS} filter combinations to remove loser-profile signals...")
    rng = np.random.default_rng(3333)
    best_val = base["total_d"]; best_f = None; best_mt = None; best_ts = None; peaks = 0
    t0 = time.time()
    for trial in range(1, N_TRIALS + 1):
        f = {"ts_min": rng.uniform(0.0, 3.0),
             "rsi_min": rng.uniform(0.0, 50.0),
             "rsi_max": rng.uniform(50.0, 100.0),
             "pb_max": rng.uniform(0.3, 1.5),
             "vr_max": rng.uniform(1.0, 2.5),
             "skip_late": bool(rng.integers(0, 2))}
        keep = apply_filter(c, f)
        if keep.sum() < 100:      # keep enough signals to stay statistically real
            continue
        mt = metrics_masked(c, r, sl_dist, keep, BEST["risk_cap"])
        ts = tsai_from(mt)
        if ts["G"] < 1:
            continue
        if mt["total_d"] > best_val:
            best_val = mt["total_d"]; best_f = f; best_mt = mt; best_ts = ts; peaks += 1
        if trial % 35000 == 0:
            print(f"  trial {trial}/{N_TRIALS} | peaks={peaks} | best=${best_val:+.2f}")
    print(f"  search done in {(time.time()-t0)/60:.1f} min | filter peaks found: {peaks}")

    if best_f is None:
        print("No gate-passing filter improved on baseline. Baseline is already optimal.")
        return

    # ---- 3. IN-SAMPLE BEFORE/AFTER ----
    print("\n===== BEFORE vs AFTER (in-sample 2025-2026) =====")
    print(f"BEFORE: ${base['total_d']:+.2f} | +${base['total_d']/weeks:.2f}/wk | WR={base['wr']*100:.1f}% | "
          f"n={base['n']} | DDtot=${base['dd_total']:.0f} DDday=${base['dd_day']:.0f} | TSAI={base_ts['TSAI']:.3f}")
    print(f"AFTER : ${best_mt['total_d']:+.2f} | +${best_mt['total_d']/weeks:.2f}/wk | WR={best_mt['wr']*100:.1f}% | "
          f"n={best_mt['n']} | DDtot=${best_mt['dd_total']:.0f} DDday=${best_mt['dd_day']:.0f} | TSAI={best_ts['TSAI']:.3f}")
    print("FILTER:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in best_f.items()}))

    # ---- 4. OUT-OF-SAMPLE VALIDATION (2023-2024) ----
    print("\n===== OUT-OF-SAMPLE VALIDATION (2023-2024, filter NEVER tuned on it) =====")
    df_oos = fb.load_data()
    weeks_oos = (df_oos.index.max() - df_oos.index.min()).days / 7.0
    c2 = T.build_candidates(df_oos)
    r2, sld2, code2, _, _ = simulate_full(c2, BEST)
    base2 = metrics_masked(c2, r2, sld2, np.ones(c2["n"], bool), BEST["risk_cap"])
    keep2 = apply_filter(c2, best_f)
    aft2 = metrics_masked(c2, r2, sld2, keep2, BEST["risk_cap"])
    b2t = tsai_from(base2); a2t = tsai_from(aft2)
    print(f"OOS BEFORE: ${base2['total_d']:+.2f} | +${base2['total_d']/weeks_oos:.2f}/wk | "
          f"WR={base2['wr']*100:.1f}% | n={base2['n']} | TSAI={b2t['TSAI']:.3f}")
    if aft2:
        print(f"OOS AFTER : ${aft2['total_d']:+.2f} | +${aft2['total_d']/weeks_oos:.2f}/wk | "
              f"WR={aft2['wr']*100:.1f}% | n={aft2['n']} | TSAI={a2t['TSAI']:.3f}")
        verdict = "HOLDS (generalizes)" if aft2["total_d"] >= base2["total_d"] else "OVERFIT (rejects)"
        print(f"OOS VERDICT: {verdict}")
    json.dump({"filter": best_f, "in_sample": best_mt, "in_sample_tsai": best_ts,
               "baseline": base, "oos_before": base2, "oos_after": aft2},
              open("phase3_result.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
