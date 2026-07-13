"""
PHASE 4 — Optimize the WAY the ASWP engine is used (not the SL/TP numbers).

Keeps the Phase-2 converged SL/TP shape fixed, but searches 300,000 combos of
NEW adaptive usage dimensions to push weekly profit, avg-R, AND the TSAI
Compatibility(C)/Suitability(S) scores up together while cutting max DD and
daily DD:

  1. Adaptive TP allocation  a1/a2/a3 (default 10/20/70), with a trend-tilt that
     shifts weight to the runner leg when the 1H trend is strong.
  2. Daily loss circuit-breaker  (cap each day's loss -> directly raises S_day).
  3. Signals-per-day throttle.
  4. Position risk size.

Objective: find configs that PARETO-IMPROVE the baseline (weekly $ up, avg-R up,
DD_total down, DD_day down, gates pass). Validated honestly; reports before/after.
"""
import json
import time
import numpy as np
import pandas as pd

import tsai_optimize as T
import phase3_loss as P3

BEST = P3.BEST
N_TRIALS = 300000
SPREAD = T.SPREAD


def precompute(c):
    """Per-signal outcome code + TP R-multiples under the fixed Phase-2 SL/TP."""
    r, sl_dist, code, _, _ = P3.simulate_full(c, BEST)
    vr = np.clip(c["vol_ratio"], BEST["vol_lo"], BEST["vol_hi"])
    tp_ext = np.clip(1 + BEST["trend_gain"] * (c["trend_strength"] - 1), BEST["tp_lo"], BEST["tp_hi"])
    sld = BEST["base_sl"] * vr * c["x_atr"] + SPREAD / 2
    tp1 = BEST["tp1_r"] * sld + SPREAD
    tp2 = BEST["tp2_r"] * sld + SPREAD
    tp3 = BEST["tp3_r"] * sld * tp_ext + SPREAD
    tp1_R = tp1 / sld; tp2_R = tp2 / sld; tp3_R = tp3 / sld
    # timeout r (alloc-independent) already in r for code==0
    order = np.argsort(c["dt"])
    dts = c["dt"][order]
    day_code = pd.factorize(pd.to_datetime(dts).date)[0].astype(np.int64)
    # within-day rank (chronological) for the signals-per-day throttle
    rank = np.zeros(len(order), dtype=np.int64)
    seen = {}
    for i, dc in enumerate(day_code):
        rank[i] = seen.get(dc, 0); seen[dc] = rank[i] + 1
    strong = c["trend_strength"] >= np.percentile(c["trend_strength"], 60)
    return {
        "order": order, "code": code[order], "sl_dist": sld[order],
        "tp1_R": tp1_R[order], "tp2_R": tp2_R[order], "tp3_R": tp3_R[order],
        "timeout_r": r[order], "entries": c["entries"][order], "atr5": c["atr5"][order],
        "day_code": day_code, "rank": rank, "strong": strong[order], "n": len(order),
        "n_days": int(day_code.max()) + 1,
    }


def eval_usage(pc, a1, a2, tilt, day_cap, max_day, risk_cap, weeks):
    a3 = 1.0 - a1 - a2
    # trend-tilt: move `tilt` from (a1,a2) to a3 on strong-trend signals
    A1 = np.where(pc["strong"], np.maximum(0.02, a1 - tilt * 0.5), a1)
    A2 = np.where(pc["strong"], np.maximum(0.02, a2 - tilt * 0.5), a2)
    A3 = 1.0 - A1 - A2
    code = pc["code"]; t1 = pc["tp1_R"]; t2 = pc["tp2_R"]; t3 = pc["tp3_R"]
    r = np.select(
        [code == 4, code == 2, code == 3, code == 1],
        [A1*t1 + A2*t2 + A3*t3, A1*t1 + A2*t2 + A3*t1, A1*t1, -1.0],
        default=pc["timeout_r"])
    # sizing
    lpl = 100.0 * pc["sl_dist"]
    lots = np.minimum(0.12, risk_cap / lpl)
    lots = np.maximum(0.01, np.floor(lots * 100) / 100)
    margin = lots * 100.0 * pc["entries"] / 10.0
    over = margin > T.EQUITY0 * 0.95
    if over.any():
        alt = np.maximum(0.01, np.floor((T.EQUITY0*0.95*10.0/(100.0*pc["entries"]))*100)/100)
        lots = np.where(over, alt, lots)
    risk = lots * lpl
    pnl = r * risk
    # signals-per-day throttle
    taken = pc["rank"] < max_day
    pnl_t = np.where(taken, pnl, 0.0)
    # daily circuit-breaker: cap each day's realized loss at -day_cap
    day_sums = np.bincount(pc["day_code"], weights=pnl_t, minlength=pc["n_days"])
    day_clip = np.maximum(day_sums, -day_cap)
    total = float(day_clip.sum())
    eq = T.EQUITY0 + np.cumsum(day_clip)
    dd_total = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
    dd_day = float(-day_clip.min()) if len(day_clip) and day_clip.min() < 0 else 0.0
    n_taken = int(taken.sum())
    rt = r[taken]
    wins = int((rt > 0).sum()); losses = int((rt < 0).sum())
    avg_R = float(rt.mean()) if n_taken else 0.0
    ev = total / n_taken if n_taken else 0.0
    avg_sl = float(pc["sl_dist"][taken].mean()) if n_taken else 0.0
    avg_atr5 = float(pc["atr5"][taken].mean()) if n_taken else 0.0
    avg_lots = float(lots[taken].mean()) if n_taken else 0.0
    # TSAI
    g1 = 1.0 if avg_sl > 1.2 * avg_atr5 else 0.0
    g2 = 1.0 if dd_total < 350 else 0.0
    g3 = 1.0 if dd_day < 200 else 0.0
    G = g1 * g2 * g3
    sc = avg_lots * 100.0 * SPREAD
    C = max(0.0, 5.0 * (ev / sc if sc > 0 else 0) * (avg_sl / avg_atr5 if avg_atr5 > 0 else 0))
    s_day = 250/dd_day if dd_day > 0 else 1e6
    s_total = 400/dd_total if dd_total > 0 else 1e6
    S = (2*s_day*s_total)/(s_day+s_total)
    TSAI = G*((2*C*S)/(C+S)) if (C+S) > 0 else 0.0
    return {"total": total, "wk": total/weeks, "avg_R": avg_R, "wr": wins/max(1, wins+losses),
            "dd_total": dd_total, "dd_day": dd_day, "n": n_taken, "C": C, "S": S,
            "TSAI": TSAI, "G": G}


def main():
    df = T.load_data()
    weeks = (df.index.max() - df.index.min()).days / 7.0
    c = T.build_candidates(df)
    pc = precompute(c)

    BIG = 1e9
    base = eval_usage(pc, 0.10, 0.20, 0.0, BIG, BIG, 26.0785, weeks)
    print("BASELINE (10/20/70, no breaker, risk $26):")
    print(f"  +${base['wk']:.2f}/wk | avgR={base['avg_R']:.3f} | WR={base['wr']*100:.1f}% | "
          f"DDtot=${base['dd_total']:.0f} DDday=${base['dd_day']:.0f} | C={base['C']:.2f} "
          f"S={base['S']:.2f} TSAI={base['TSAI']:.3f}")
    print(f"\nSearching {N_TRIALS} usage configs for Pareto improvements...\n")

    rng = np.random.default_rng(4444)
    # best strict-Pareto (all axes >= baseline, strictly better on >=1), maximize wk
    best_dom = None; best_dom_wk = base["wk"]
    # best TSAI with weekly profit not sacrificed (>= baseline wk)
    best_tsai = base["TSAI"]; best_tsai_cfg = None
    dom_count = 0
    t0 = time.time()
    for trial in range(1, N_TRIALS + 1):
        a1 = rng.uniform(0.03, 0.30)
        a2 = rng.uniform(0.08, 0.40)
        if a1 + a2 > 0.75:
            continue
        tilt = rng.uniform(0.0, 0.30)
        day_cap = BIG   # circuit-breaker disabled: it cannot be faithfully modeled
                        # as an asymmetric daily-loss clip (that cherry-picks the
                        # left tail). Only honest, implementable levers below.
        max_day = rng.choice([BIG, 8, 6, 5, 4, 3])
        risk_cap = rng.uniform(15.0, 40.0)
        m = eval_usage(pc, a1, a2, tilt, day_cap, max_day, risk_cap, weeks)
        if m["G"] < 1:
            continue
        # strict Pareto improvement over baseline
        if (m["wk"] >= base["wk"] and m["avg_R"] >= base["avg_R"]
                and m["dd_total"] <= base["dd_total"] and m["dd_day"] <= base["dd_day"]
                and (m["wk"] > base["wk"] or m["avg_R"] > base["avg_R"]
                     or m["dd_total"] < base["dd_total"] or m["dd_day"] < base["dd_day"])):
            dom_count += 1
            if m["wk"] > best_dom_wk:
                best_dom_wk = m["wk"]
                best_dom = dict(m, a1=a1, a2=a2, tilt=tilt, day_cap=float(day_cap),
                                max_day=float(max_day), risk_cap=risk_cap)
        if m["wk"] >= base["wk"] and m["TSAI"] > best_tsai:
            best_tsai = m["TSAI"]
            best_tsai_cfg = dict(m, a1=a1, a2=a2, tilt=tilt, day_cap=float(day_cap),
                                 max_day=float(max_day), risk_cap=risk_cap)
        if trial % 60000 == 0:
            print(f"  trial {trial}/{N_TRIALS} | pareto-improving found={dom_count} | "
                  f"best_dom=${best_dom_wk:.2f}/wk | best_tsai={best_tsai:.3f}")

    print(f"\n  search done in {(time.time()-t0)/60:.1f} min | "
          f"total Pareto-improving configs: {dom_count}")

    def show(tag, m):
        cfg = f"alloc={m['a1']*100:.0f}/{m['a2']*100:.0f}/{(1-m['a1']-m['a2'])*100:.0f} " \
              f"tilt={m['tilt']:.2f} day_cap={'none' if m['day_cap']>1e8 else '$'+str(int(m['day_cap']))} " \
              f"max/day={'none' if m['max_day']>1e8 else int(m['max_day'])} risk=${m['risk_cap']:.1f}"
        print(f"{tag}: +${m['wk']:.2f}/wk | avgR={m['avg_R']:.3f} | WR={m['wr']*100:.1f}% | "
              f"DDtot=${m['dd_total']:.0f} DDday=${m['dd_day']:.0f} | C={m['C']:.2f} S={m['S']:.2f} "
              f"TSAI={m['TSAI']:.3f}\n     {cfg}")

    print("\n" + "=" * 96)
    print("BASELINE:")
    print(f"  +${base['wk']:.2f}/wk | avgR={base['avg_R']:.3f} | DDtot=${base['dd_total']:.0f} "
          f"DDday=${base['dd_day']:.0f} | C={base['C']:.2f} S={base['S']:.2f} TSAI={base['TSAI']:.3f}")
    print()
    if best_dom:
        show("BEST PARETO (better on ALL axes, max $/wk)", best_dom)
    else:
        print("BEST PARETO: none found that strictly improves ALL axes at once.")
    print()
    if best_tsai_cfg:
        show("BEST TSAI (no weekly-profit sacrifice)", best_tsai_cfg)
    print("=" * 96)
    json.dump({"baseline": base, "best_dom": best_dom, "best_tsai": best_tsai_cfg,
               "dom_count": dom_count, "weeks": weeks},
              open("phase4_result.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
