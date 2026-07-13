"""
PHASE 11 — Trend-exhaustion filter.

Hypothesis: skip signals when the 1H trend has already run > N × ATR(1H)
from its recent swing. An exhausted trend has low remaining fuel — entering
at the tail-end produces the stop-outs like today's 3× loss at the floor.

Protocol:
  TRAIN on 2023-2024 (find the best exhaustion threshold).
  TEST on 2025-2026 (true out-of-sample).
  Baseline = current engine (30/40/30, circuit-breaker, Flow, range filter).
  Push ONLY if it improves weekly profit OOS and stays gate-safe.
"""
import numpy as np
import pandas as pd

import tsai_optimize as T
import phase4_usage as P4
import phase7_candleEV as P7
import walk_forward as W
import fullyear_backtest as fb


def compute_exhaustion(df1m, c):
    """For each signal, compute how far the 1H price has moved from its recent
    swing in the trend direction, normalized by 1H ATR. Higher = more exhausted."""
    b1h = fb.resample_tf(df1m, 60)
    # 1H ATR series
    atr1h = fb.atr(b1h, 14)
    # Recent swing: highest high (for sell) or lowest low (for buy) over last 20 1H bars
    hi20 = b1h["h"].rolling(20).max()
    lo20 = b1h["l"].rolling(20).min()
    close1h = b1h["c"]

    # Align to signal times
    dts = pd.to_datetime(c["dt"])
    # merge_asof to get latest completed 1H bar for each signal
    sig_df = pd.DataFrame({"dt": dts}).sort_values("dt")
    h1_df = pd.DataFrame({"dt": b1h.index, "c1h": close1h.values, "hi20": hi20.values,
                           "lo20": lo20.values, "atr1h": atr1h.values}).dropna().sort_values("dt")
    m = pd.merge_asof(sig_df, h1_df, on="dt", direction="backward")

    d = c["d"]  # direction: 1=buy, -1=sell
    exhaustion = np.zeros(len(d))
    for i in range(len(d)):
        a = m.iloc[i]["atr1h"]
        if a <= 0 or pd.isna(a):
            continue
        if d[i] == -1:  # SELL: how far has price fallen from the recent 20-bar high?
            drop = m.iloc[i]["hi20"] - m.iloc[i]["c1h"]
            exhaustion[i] = drop / a
        else:  # BUY: how far has price risen from the recent 20-bar low?
            rise = m.iloc[i]["c1h"] - m.iloc[i]["lo20"]
            exhaustion[i] = rise / a
    return exhaustion


def eval_with_filter(pc, exhaustion, threshold, weeks):
    """Keep only signals where exhaustion < threshold (trend still has fuel)."""
    keep = exhaustion[pc["order"]] < threshold
    idx = np.where(keep)[0]
    if len(idx) < 50:
        return None
    r = P7.per_signal_r(pc, 0.30, 0.40, 0.19)
    sld = pc["sl_dist"]; ent = pc["entries"]
    # Apply circuit-breaker sizing (simplified: flat $45 for this comparison)
    risk = 45.0
    lpl = 100.0 * sld[idx]; lots = np.minimum(0.12, risk / lpl)
    lots = np.maximum(0.01, np.floor(lots * 100) / 100)
    pnl = r[idx] * lots * lpl
    ddt, ddd = W.stitched_dd(pnl, pc["dt_sorted"][idx])
    w = int((pnl > 0).sum()); l = int((pnl < 0).sum()); n = len(idx)
    return {"wk": float(pnl.sum()) / weeks, "n": n, "per_day": n / (weeks * 5),
            "wr": w / max(1, w + l) * 100, "dd_total": ddt, "dd_day": ddd,
            "gate": ddt < 350 and ddd < 200}


def run_window(df, label):
    c = T.build_candidates(df)
    pc = P4.precompute(c); pc["dt_sorted"] = c["dt"][pc["order"]]
    weeks = (df.index.max() - df.index.min()).days / 7.0
    exhaustion = compute_exhaustion(df, c)

    # Baseline (no exhaustion filter)
    base = eval_with_filter(pc, exhaustion, 9999.0, weeks)

    print(f"\n{label} ({weeks:.0f}w):")
    print(f"  {'threshold':>10} {'signals':>8} {'sig/day':>8} {'WR%':>6} {'$/wk':>9} {'DDtot':>7} {'DDday':>7} {'gate':>5}")
    print(f"  {'-'*65}")
    print(f"  {'BASELINE':>10} {base['n']:>8} {base['per_day']:>8.1f} {base['wr']:>5.1f}% ${base['wk']:>8.2f} "
          f"${base['dd_total']:>6.0f} ${base['dd_day']:>6.0f} {'PASS' if base['gate'] else 'FAIL':>5}")

    results = []
    for thr in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]:
        m = eval_with_filter(pc, exhaustion, thr, weeks)
        if m is None:
            continue
        results.append((thr, m))
        print(f"  {f'< {thr:.1f}x ATR':>10} {m['n']:>8} {m['per_day']:>8.1f} {m['wr']:>5.1f}% ${m['wk']:>8.2f} "
              f"${m['dd_total']:>6.0f} ${m['dd_day']:>6.0f} {'PASS' if m['gate'] else 'FAIL':>5}")

    return base, results, exhaustion


def main():
    print("=" * 70)
    print("PHASE 11: Trend-exhaustion filter (skip signals when trend is spent)")
    print("=" * 70)

    # TRAIN on 2023-2024: find best threshold
    df_tr = fb.load_data()
    base_tr, res_tr, _ = run_window(df_tr, "TRAIN 2023-2024")

    # Find best threshold on TRAIN (maximizes weekly $ while gate-safe)
    best_thr = None; best_wk = base_tr["wk"]
    for thr, m in res_tr:
        if m["gate"] and m["wk"] > best_wk:
            best_wk = m["wk"]; best_thr = thr

    if best_thr is None:
        print("\nNo exhaustion threshold improves TRAIN profit. Checking if it helps OOS anyway...")
        # Still test the most common sense thresholds on OOS
        best_thr = 4.0  # reasonable default to test

    print(f"\nBest TRAIN threshold: < {best_thr}x ATR(1H) (train ${best_wk:.2f}/wk)")

    # TEST on 2025-2026 (true out-of-sample)
    df_te = T.load_data()
    base_te, res_te, _ = run_window(df_te, "TEST 2025-2026 (out-of-sample)")

    # Find the test result at the train-selected threshold
    test_result = None
    for thr, m in res_te:
        if abs(thr - best_thr) < 0.01:
            test_result = m
            break

    print("\n" + "=" * 70)
    if test_result and test_result["gate"]:
        imp = test_result["wk"] - base_te["wk"]
        print(f"OOS result at threshold < {best_thr}x:")
        print(f"  Baseline: ${base_te['wk']:.2f}/wk | With filter: ${test_result['wk']:.2f}/wk | Delta: ${imp:+.2f}/wk")
        if imp > 0:
            print(f"  VERDICT: IMPROVES weekly profit OOS by ${imp:+.2f}/wk -> AUTO-PUSH.")
        else:
            print(f"  VERDICT: does NOT improve OOS (${imp:+.2f}/wk) -> DO NOT PUSH.")
    else:
        print("  No gate-safe improvement found. DO NOT PUSH.")
    print("=" * 70)

    return best_thr, base_te, test_result


if __name__ == "__main__":
    best_thr, base, result = main()
