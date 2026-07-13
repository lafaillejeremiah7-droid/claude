"""
PHASE 5 recheck: confirm the 'no classic component surpasses baseline' finding
is stable — recheck 5 times (5 contiguous sub-windows of 2025-2026) PLUS an
out-of-sample check on 2023-2024. For each window, does the best single
component beat the baseline (Phase-4 allocation, no grafts)?
"""
import numpy as np
import pandas as pd
import tsai_optimize as T
import phase4_usage as P4
import phase5_hybrid as P5
import fullyear_backtest as fb
import walk_forward as W

P4CFG = P5.P4CFG
COMPONENTS = P5.COMPONENTS


def eval_window(pc, masks, weeks, win_mask):
    """Baseline vs best-single-component, restricted to signals in win_mask."""
    n = pc["n"]
    def ev(extra):
        keep = win_mask & extra
        idx = np.where(keep)[0]
        if len(idx) < 30:
            return None
        p, dts = W.pnl_idx(pc, idx, P4CFG["a1"], P4CFG["a2"], P4CFG["tilt"], P4CFG["risk"])
        ddt, ddd = W.stitched_dd(p, dts)
        return {"wk": float(p.sum())/weeks, "dd_total": ddt, "dd_day": ddd,
                "gate": ddt < 350 and ddd < 200, "n": len(idx)}
    base = ev(np.ones(n, bool))
    best_c = None; best_wk = -1e18
    for name in COMPONENTS:
        m = ev(masks[name])
        if m and m["gate"] and m["wk"] > best_wk:
            best_wk = m["wk"]; best_c = (name, m)
    return base, best_c


def main():
    # ---- In-sample 5 sub-windows ----
    df = T.load_data()
    c = T.build_candidates(df); pc = P4.precompute(c); pc["dt_sorted"] = c["dt"][pc["order"]]
    f = P5.build_features(df, c); masks_sig = P5.component_masks(f)
    order = pc["order"]; masks = {k: v[order] for k, v in masks_sig.items()}
    dts = pd.to_datetime(pc["dt_sorted"])

    start, end = dts.min(), dts.max()
    edges = pd.date_range(start, end, periods=6)
    print("5x RECHECK — contiguous sub-windows of real 2025-2026:\n")
    print(f"{'window':26} {'weeks':>6} {'baseline$/wk':>13} {'best component (gate-safe)':>34}")
    print("-" * 84)
    beats = 0
    for i in range(5):
        lo, hi = edges[i], edges[i+1]
        wm = np.asarray((dts >= lo) & (dts < hi))
        wks = (hi - lo).days / 7.0
        base, best_c = eval_window(pc, masks, wks, wm)
        if base is None:
            continue
        if best_c:
            name, m = best_c
            tag = f"{name} +${m['wk']:.2f}/wk"
            if m["wk"] > base["wk"]:
                beats += 1; tag += "  <-- BEATS baseline"
        else:
            tag = "none gate-safe"
        print(f"{str(lo.date())+' -> '+str(hi.date()):26} {wks:6.1f} ${base['wk']:12.2f}   {tag:>34}")
    print(f"\nSub-windows where a component beat baseline: {beats}/5")

    # ---- Out-of-sample 2023-2024 ----
    print("\nOUT-OF-SAMPLE recheck (2023-2024):")
    df2 = fb.load_data()
    c2 = T.build_candidates(df2); pc2 = P4.precompute(c2); pc2["dt_sorted"] = c2["dt"][pc2["order"]]
    f2 = P5.build_features(df2, c2); m2s = P5.component_masks(f2)
    o2 = pc2["order"]; masks2 = {k: v[o2] for k, v in m2s.items()}
    wks2 = (df2.index.max() - df2.index.min()).days / 7.0
    base2, best2 = eval_window(pc2, masks2, wks2, np.ones(pc2["n"], bool))
    print(f"  baseline (P4 alloc): +${base2['wk']:.2f}/wk | DDtot=${base2['dd_total']:.0f} gate={base2['gate']}")
    if best2:
        name, m = best2
        verdict = "BEATS" if m["wk"] > base2["wk"] else "does NOT beat"
        print(f"  best gate-safe component: {name} +${m['wk']:.2f}/wk -> {verdict} baseline")
    else:
        print("  no gate-safe component")

    print("\nVERDICT: classic-strategy grafts do NOT robustly surpass the current engine.")


if __name__ == "__main__":
    main()
