"""
PHASE 9 — Can we run $60 risk safely? (drawdown circuit-breaker search)

Flat $60 risk blew the drawdown gate ($1,293 DD). The one legitimate mechanism
to run bigger risk without shrinking every trade: a DRAWDOWN CIRCUIT-BREAKER —
trade full $60 at/near equity peak, but cut size (throttle) and/or stop the day
when in drawdown, so a losing streak can't run the account past the gate.

This is a REAL sequential path-dependent simulation (no cherry-picking): each
signal in chronological order, tracking equity/peak/daily-loss, applying:
  - daily stop: skip remaining same-day signals after daily loss <= -daily_stop
  - dd throttle: risk *= factor while (peak - equity) >= dd_thr, until new peak

Objective: beat the flat-$25 baseline (+$307/wk) while passing gates
(DD_total<350, DD_day<200). Validate OOS on 2023-2024 + 4.5yr.
"""
import numpy as np
import pandas as pd

import tsai_optimize as T
import phase4_usage as P4
import phase7_candleEV as P7
import phase8_candleEV_pretrain as P8

ALLOC = dict(a1=0.30, a2=0.40, tilt=0.19)
EQ0 = 5000.0


def simulate(pc, r, base_risk, daily_stop, dd_thr, factor):
    """Sequential risk-managed equity simulation on chronological pc arrays."""
    n = pc["n"]
    sld = pc["sl_dist"]; ent = pc["entries"]; day = pc["day_code"]
    equity = EQ0; peak = EQ0
    pnls = np.zeros(n)
    cur_day = -1; day_pnl = 0.0; blocked = False
    for k in range(n):
        if day[k] != cur_day:
            cur_day = day[k]; day_pnl = 0.0; blocked = False
        if blocked:
            continue
        dd = peak - equity
        risk = base_risk * (factor if (dd_thr is not None and dd >= dd_thr) else 1.0)
        lpl = 100.0 * sld[k]
        lots = min(0.12, risk / lpl)
        lots = max(0.01, np.floor(lots * 100) / 100)
        if lots * 100.0 * ent[k] / 10.0 > equity * 0.95:
            lots = max(0.01, np.floor((equity * 0.95 * 10.0 / (100.0 * ent[k])) * 100) / 100)
        pnl = r[k] * lots * lpl
        pnls[k] = pnl
        equity += pnl; day_pnl += pnl
        if equity > peak:
            peak = equity
        if daily_stop is not None and day_pnl <= -daily_stop:
            blocked = True
    eq = EQ0 + np.cumsum(pnls)
    dd_total = float(np.max(np.maximum.accumulate(eq) - eq))
    dsum = np.bincount(day, weights=pnls, minlength=day.max()+1)
    dd_day = float(-dsum.min()) if dsum.min() < 0 else 0.0
    return {"total": float(pnls.sum()), "dd_total": dd_total, "dd_day": dd_day,
            "gate": dd_total < 350 and dd_day < 200}


def prep(df):
    c = T.build_candidates(df)
    pc = P4.precompute(c); pc["dt_sorted"] = c["dt"][pc["order"]]
    r = P7.per_signal_r(pc, ALLOC["a1"], ALLOC["a2"], ALLOC["tilt"])
    weeks = (df.index.max() - df.index.min()).days / 7.0
    return pc, r, weeks


def search(pc, r, weeks):
    best = None; best_wk = -1e18
    for base in [60.0, 55.0, 50.0, 45.0, 40.0]:
        for daily_stop in [None, 250, 200, 150, 120, 100]:
            for dd_thr in [None, 250, 200, 150, 120, 90]:
                for factor in [0.25, 0.35, 0.5, 0.65]:
                    m = simulate(pc, r, base, daily_stop, dd_thr, factor)
                    if not m["gate"]:
                        continue
                    wk = m["total"] / weeks
                    if wk > best_wk:
                        best_wk = wk
                        best = dict(m, base=base, daily_stop=daily_stop, dd_thr=dd_thr,
                                    factor=factor, wk=wk)
    return best


def main():
    print("=== IN-SAMPLE 2025-2026 ===")
    pc, r, weeks = prep(T.load_data())
    flat25 = simulate(pc, r, 25.0, None, None, 1.0)
    flat60 = simulate(pc, r, 60.0, None, None, 1.0)
    print(f"flat $25 (baseline): +${flat25['total']/weeks:.2f}/wk | DDtot=${flat25['dd_total']:.0f} "
          f"DDday=${flat25['dd_day']:.0f} | gate={flat25['gate']}")
    print(f"flat $60 (naive):    +${flat60['total']/weeks:.2f}/wk | DDtot=${flat60['dd_total']:.0f} "
          f"DDday=${flat60['dd_day']:.0f} | gate={flat60['gate']}")

    best = search(pc, r, weeks)
    if best is None:
        print("\nNo gate-safe circuit-breaker config found at any base risk >= $40.")
        return
    print(f"\nBEST circuit-breaker config (in-sample, gate-safe):")
    print(f"  base=${best['base']:.0f} | daily_stop={best['daily_stop']} | "
          f"dd_throttle={best['dd_thr']} | factor={best['factor']}")
    print(f"  +${best['wk']:.2f}/wk | DDtot=${best['dd_total']:.0f} DDday=${best['dd_day']:.0f}")
    print(f"  vs flat $25 (+${flat25['total']/weeks:.2f}/wk): "
          f"{'BEATS' if best['wk'] > flat25['total']/weeks else 'does not beat'} "
          f"(${best['wk']-flat25['total']/weeks:+.2f}/wk)")

    # OOS validation on 2023-2024 and 4.5yr
    print("\n=== OUT-OF-SAMPLE validation of the best config ===")
    for label, df in [("2023-2024", P8.load_hf_window("2023-01-01", "2024-11-29")),
                      ("4.5yr 2020-2024", P8.load_hf_window("2020-05-29", "2024-11-29"))]:
        pc2, r2, wk2 = prep(df)
        m = simulate(pc2, r2, best["base"], best["daily_stop"], best["dd_thr"], best["factor"])
        b25 = simulate(pc2, r2, 25.0, None, None, 1.0)
        print(f"{label}: CB +${m['total']/wk2:.2f}/wk (DDtot=${m['dd_total']:.0f} "
              f"DDday=${m['dd_day']:.0f} gate={m['gate']}) | flat$25 +${b25['total']/wk2:.2f}/wk "
              f"(gate={b25['gate']})")


if __name__ == "__main__":
    main()
