"""
Fully vectorized adaptive SL/TP simulator.

The sequential (Python for-loop per signal per bar) backtest runs at roughly
1 parameter-trial per second. To actually try up to 100,000 parameter
combinations, this module restructures the exact same trade-simulation logic
(same entry rules, same trailing-stop/multi-TP rules as live_terminal.py and
fullyear_backtest.py) so that ONE trial evaluates ALL signals at once using
numpy array ops, looping only over forward bar-steps (~200), not signals.

Correctness is checked against the known sequential baseline before any
search is trusted (see verify_baseline()).
"""
import time
import numpy as np
import pandas as pd

import fullyear_backtest as fb
import adaptive_backtest as ab

SPREAD = 0.30
MAX_HOLD = 200          # bars forward to watch (same as sequential backtest)
COOLDOWN_MIN = 180
STARTING_EQUITY = 5000.0


# ---------------------------------------------------------------------------
# Step 1: extract the candidate signal list (entry rules only - these never
# change across trials, only the SL/TP sizing does).
# ---------------------------------------------------------------------------
def get_candidates(df1m):
    m, e = ab.compute_features(df1m)
    cand = m[m["signal"]].copy()

    e_h = e["h"].values
    e_l = e["l"].values
    e_c = e["c"].values
    e_idx = {ts: i for i, ts in enumerate(e.index)}

    cooldown_bars = max(1, COOLDOWN_MIN // 1)  # entry tf = 1m
    last_i = -10 ** 9

    rows = []
    for _, row in cand.iterrows():
        i = e_idx.get(row["dt"])
        if i is None or i < 1:
            continue
        if i - last_i < cooldown_bars:
            continue
        rows.append((row["dt"], i, int(row["dir"]), e_c[i], row["x_atr"],
                    row["vol_ratio"], row["trend_strength"]))
        last_i = i

    if not rows:
        return None

    dts = [r[0] for r in rows]
    idxs = np.array([r[1] for r in rows])
    directions = np.array([r[2] for r in rows], dtype=np.int8)
    entries = np.array([r[3] for r in rows])
    x_atr = np.array([r[4] for r in rows])
    vol_ratio = np.array([r[5] for r in rows])
    trend_strength = np.array([r[6] for r in rows])

    n = len(idxs)
    H = np.empty((n, MAX_HOLD))
    L = np.empty((n, MAX_HOLD))
    C = np.empty((n, MAX_HOLD))
    n_bars = len(e_c)
    for k, i in enumerate(idxs):
        end = min(i + 1 + MAX_HOLD, n_bars)
        seg_h = e_h[i + 1:end]
        seg_l = e_l[i + 1:end]
        seg_c = e_c[i + 1:end]
        got = len(seg_h)
        H[k, :got] = seg_h
        L[k, :got] = seg_l
        C[k, :got] = seg_c
        if got < MAX_HOLD:
            # pad with the last available value (flat / no more movement)
            pad_val_h = seg_h[-1] if got else entries[k]
            pad_val_l = seg_l[-1] if got else entries[k]
            pad_val_c = seg_c[-1] if got else entries[k]
            H[k, got:] = pad_val_h
            L[k, got:] = pad_val_l
            C[k, got:] = pad_val_c

    return {
        "dt": np.array(dts), "directions": directions, "entries": entries,
        "x_atr": x_atr, "vol_ratio": vol_ratio, "trend_strength": trend_strength,
        "H": H, "L": L, "C": C, "n": n,
    }


# ---------------------------------------------------------------------------
# Step 2: vectorized outcome simulation for ONE parameter trial, ALL signals
# at once. Mirrors backtest_adaptive()'s per-signal for-loop exactly, but the
# bar loop (<=200 iters) is vectorized across signals with numpy.
# ---------------------------------------------------------------------------
def simulate_trial(cands, base_sl, vol_lo, vol_hi, trend_gain, tp_lo, tp_hi):
    d = cands["directions"]
    entries = cands["entries"]
    x_atr = cands["x_atr"]
    H, L, C = cands["H"], cands["L"], cands["C"]
    n = cands["n"]

    vr = np.clip(cands["vol_ratio"], vol_lo, vol_hi)
    tp_ext = np.clip(1 + trend_gain * (cands["trend_strength"] - 1), tp_lo, tp_hi)

    sl_dist = base_sl * vr * x_atr + SPREAD / 2
    tp1 = sl_dist + SPREAD
    tp2 = 2 * sl_dist + SPREAD
    tp3 = 3 * sl_dist * tp_ext + SPREAD

    tp1_hit = np.zeros(n, dtype=bool)
    tp2_hit = np.zeros(n, dtype=bool)
    closed = np.zeros(n, dtype=bool)
    outcome_code = np.zeros(n, dtype=np.int8)   # 0=timeout,1=sl,2=final_be,3=tp1_be,4=tp_final
    is_buy = d == 1

    entry_plus_sl = np.where(is_buy, entries - sl_dist, entries + sl_dist)
    entry_plus_tp1 = np.where(is_buy, entries + tp1, entries - tp1)
    entry_plus_tp2 = np.where(is_buy, entries + tp2, entries - tp2)
    entry_plus_tp3 = np.where(is_buy, entries + tp3, entries - tp3)
    be_level = entries
    tp1_level_as_sl = np.where(is_buy, entries + tp1, entries - tp1)

    for j in range(MAX_HOLD):
        active = ~closed
        if not active.any():
            break
        hi = H[:, j]
        lo = L[:, j]

        sl_level = entry_plus_sl.copy()
        sl_level = np.where(tp1_hit & ~tp2_hit, be_level, sl_level)
        sl_level = np.where(tp2_hit, tp1_level_as_sl, sl_level)

        hit_sl = np.where(is_buy, lo <= sl_level, hi >= sl_level)
        newly_sl = active & hit_sl
        code_for_sl = np.where(tp2_hit, 2, np.where(tp1_hit, 3, 1))
        outcome_code = np.where(newly_sl, code_for_sl, outcome_code)
        closed = closed | newly_sl

        active = ~closed
        hit_tp1 = np.where(is_buy, hi >= entry_plus_tp1, lo <= entry_plus_tp1)
        tp1_hit = tp1_hit | (active & ~tp1_hit & hit_tp1)
        hit_tp2 = np.where(is_buy, hi >= entry_plus_tp2, lo <= entry_plus_tp2)
        tp2_hit = tp2_hit | (active & ~tp2_hit & hit_tp2)
        hit_tp3 = np.where(is_buy, hi >= entry_plus_tp3, lo <= entry_plus_tp3)
        newly_final = active & hit_tp3
        outcome_code = np.where(newly_final, 4, outcome_code)
        closed = closed | newly_final

    last_c = C[:, -1]
    move = (last_c - entries) * d
    r_timeout = move / sl_dist

    r = np.where(
        outcome_code == 4, (0.10 * tp1 + 0.20 * tp2 + 0.70 * tp3) / sl_dist,
        np.where(outcome_code == 2, (0.10 * tp1 + 0.20 * tp2 + 0.70 * tp1) / sl_dist,
                np.where(outcome_code == 3, (0.10 * tp1) / sl_dist,
                        np.where(outcome_code == 1, -1.0, r_timeout)))
    )
    return r, sl_dist, outcome_code


def dollar_stats_vectorized(cands, r, sl_dist, equity0=STARTING_EQUITY):
    entries = cands["entries"]
    loss_per_lot = 100.0 * sl_dist
    lots = np.minimum(0.12, 60.0 / loss_per_lot)
    lots = np.maximum(0.01, np.floor(lots * 100) / 100)
    margin = lots * 100.0 * entries / 10.0
    over = margin > equity0 * 0.95
    if over.any():
        alt = np.maximum(0.01, np.floor((equity0 * 0.95 * 10.0 / (100.0 * entries)) * 100) / 100)
        lots = np.where(over, alt, lots)
    risk = lots * loss_per_lot
    pnl = r * risk

    # chronological order for equity curve / drawdown
    order = np.argsort(cands["dt"])
    pnl_sorted = pnl[order]
    eq = equity0 + np.cumsum(pnl_sorted)
    peak = np.maximum.accumulate(eq)
    dd = np.max(peak - eq) if len(eq) else 0.0

    return {
        "total_d": round(float(pnl.sum()), 2),
        "avg_d": round(float(pnl.mean()), 2) if len(pnl) else 0.0,
        "max_dd": round(float(dd), 2),
        "avg_risk": round(float(risk.mean()), 2) if len(risk) else 0.0,
    }


# ---------------------------------------------------------------------------
# Correctness check: static 1.0x baseline (vol_lo=vol_hi=1.0 disables vol
# adaptation, trend_gain=0 & tp_lo=tp_hi=1.0 disables TP extension) must
# reproduce the known sequential-backtest numbers exactly.
# ---------------------------------------------------------------------------
def verify_baseline(cands):
    r, sl_dist, code = simulate_trial(cands, base_sl=1.0, vol_lo=1.0, vol_hi=1.0,
                                      trend_gain=0.0, tp_lo=1.0, tp_hi=1.0)
    d = dollar_stats_vectorized(cands, r, sl_dist)
    wins = int((r > 0).sum())
    losses = int((r < 0).sum())
    total = len(r)
    print("VECTORIZED baseline check (should match sequential engine):")
    print(f"  signals={total} wins={wins} losses={losses} WR={wins/total*100:.1f}% "
          f"total_R={r.sum():+.2f}")
    print(f"  ${d['total_d']:+.2f} avg=${d['avg_d']:.2f} maxDD=${d['max_dd']:.2f}")
    return r, d


if __name__ == "__main__":
    print("Loading full-year data + building candidate signal set...")
    df1m = fb.load_data()
    cands = get_candidates(df1m)
    print(f"Candidates: {cands['n']} signals\n")

    verify_baseline(cands)

    print("\nBenchmarking trial speed...")
    t0 = time.time()
    N_BENCH = 200
    rng = np.random.default_rng(0)
    for _ in range(N_BENCH):
        base_sl = rng.uniform(0.3, 3.0)
        vol_lo = rng.uniform(0.2, 1.0)
        vol_hi = rng.uniform(1.0, 3.0)
        trend_gain = rng.uniform(-1.0, 2.0)
        tp_lo = rng.uniform(0.3, 1.0)
        tp_hi = rng.uniform(1.0, 3.0)
        r, sl_dist, code = simulate_trial(cands, base_sl, vol_lo, vol_hi, trend_gain, tp_lo, tp_hi)
        _ = dollar_stats_vectorized(cands, r, sl_dist)
    elapsed = time.time() - t0
    per_trial = elapsed / N_BENCH
    print(f"  {N_BENCH} trials in {elapsed:.2f}s -> {per_trial*1000:.2f} ms/trial")
    print(f"  Projected time for 100,000 trials: {per_trial*100000/60:.1f} minutes")
