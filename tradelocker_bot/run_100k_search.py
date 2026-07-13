"""
Run a 100,000-trial random search over adaptive SL/TP parameters using the
vectorized engine, on the in-sample year. Tracks the best-by-total-$ config
and periodically reports progress + how many trials beat the static baseline.
"""
import time
import numpy as np

import fullyear_backtest as fb
import vectorized_adaptive as va

N_TRIALS = 100_000
SEED = 42

if __name__ == "__main__":
    print("Loading full-year data + building candidate signal set...")
    df1m = fb.load_data()
    cands = va.get_candidates(df1m)
    print(f"Candidates: {cands['n']} signals\n")

    base_r, base_d = va.verify_baseline(cands)
    baseline_total = base_d["total_d"]
    print()

    rng = np.random.default_rng(SEED)

    best_total = -1e18
    best_params = None
    best_stats = None
    beat_baseline = 0

    t0 = time.time()
    log_every = 5000

    for trial in range(1, N_TRIALS + 1):
        base_sl = rng.uniform(0.3, 3.0)
        vol_lo = rng.uniform(0.2, 1.0)
        vol_hi = rng.uniform(1.0, 3.0)
        trend_gain = rng.uniform(-1.0, 2.0)
        tp_lo = rng.uniform(0.3, 1.0)
        tp_hi = rng.uniform(1.0, 3.0)

        r, sl_dist, code = va.simulate_trial(cands, base_sl, vol_lo, vol_hi, trend_gain, tp_lo, tp_hi)
        d = va.dollar_stats_vectorized(cands, r, sl_dist)

        if d["total_d"] > baseline_total:
            beat_baseline += 1

        if d["total_d"] > best_total:
            best_total = d["total_d"]
            best_params = (base_sl, vol_lo, vol_hi, trend_gain, tp_lo, tp_hi)
            best_stats = dict(d)
            wins = int((r > 0).sum())
            total = len(r)
            best_stats["win_rate"] = round(wins / total * 100, 1)
            best_stats["total_r"] = round(float(r.sum()), 2)
            best_stats["signals"] = total

        if trial % log_every == 0:
            elapsed = time.time() - t0
            rate = trial / elapsed
            eta = (N_TRIALS - trial) / rate
            print(f"  trial {trial:>7}/{N_TRIALS} | {rate:.0f} trials/s | ETA {eta/60:.1f} min | "
                  f"beat_baseline={beat_baseline} | best_so_far=${best_total:+.2f}")

    elapsed = time.time() - t0
    print()
    print("=" * 90)
    print(f"SEARCH COMPLETE: {N_TRIALS} trials in {elapsed/60:.1f} minutes "
          f"({N_TRIALS/elapsed:.0f} trials/sec)")
    print(f"Trials that beat the static baseline (\${baseline_total:+.2f}): "
          f"{beat_baseline} / {N_TRIALS} ({beat_baseline/N_TRIALS*100:.3f}%)")
    print()
    print("BASELINE (static 1.0x ATR, no adaptation):")
    print(f"  \${baseline_total:+.2f}")
    print()
    print("BEST FOUND (adaptive):")
    bs = best_stats
    print(f"  base_sl={best_params[0]:.4f}  vol_lo={best_params[1]:.4f}  vol_hi={best_params[2]:.4f}")
    print(f"  trend_gain={best_params[3]:.4f}  tp_lo={best_params[4]:.4f}  tp_hi={best_params[5]:.4f}")
    print(f"  signals={bs['signals']}  WR={bs['win_rate']}%  total_R={bs['total_r']:+.2f}")
    print(f"  \${bs['total_d']:+.2f}  avg=\${bs['avg_d']:.2f}  maxDD=\${bs['max_dd']:.2f}  "
          f"avg_risk=\${bs['avg_risk']:.2f}")
    improvement = bs["total_d"] - baseline_total
    print(f"  IMPROVEMENT: \${improvement:+.2f} ({improvement/abs(baseline_total)*100:+.1f}%)")
    print("=" * 90)

    # Save best params for the out-of-sample validation step
    import json
    with open("best_adaptive_params.json", "w") as f:
        json.dump({
            "base_sl": best_params[0], "vol_lo": best_params[1], "vol_hi": best_params[2],
            "trend_gain": best_params[3], "tp_lo": best_params[4], "tp_hi": best_params[5],
            "in_sample_total_d": bs["total_d"], "in_sample_baseline_d": baseline_total,
            "beat_baseline_count": beat_baseline, "n_trials": N_TRIALS,
        }, f, indent=2)
    print("\nSaved best params to best_adaptive_params.json")
