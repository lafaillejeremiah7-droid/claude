"""
PHASE 17 — where are the dollars actually being left on the table?
Sweep trades/day cap x session width x risk for the shipped phase-15 geometry.
Find the max gate-safe $/wk at each volume level, on BOTH datasets. Honest answer
to: "how do we make this not $54/wk?"
"""
import numpy as np
import tsai_optimize as T
import fullyear_backtest as fb
import phase12_ideal as P12

CUR = {"base_sl": 1.1342, "vol_lo": 0.3460, "vol_hi": 1.2653, "tp1_r": 0.45,
       "tp2_r": 3.16, "tp3_r": 6.26, "trend_gain": 1.1363, "tp_lo": 0.849, "tp_hi": 2.509}
A1, A2, TILT = 0.45, 0.16, 0.08

SESSIONS = {"12-20 UTC (current)": range(12, 21),
            "10-21 UTC (wider)":   range(10, 22),
            "07-21 UTC (Ldn+NY)":  range(7, 22),
            "all 24h":             range(0, 24)}


def best_risk(S_tr, S_te, hrs, max_day):
    """Highest gate-safe risk on BOTH datasets; return its metrics."""
    best = None
    for risk in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
        a = P12.run(S_tr, CUR, A1, A2, TILT, base=risk, hrs=hrs, max_day=max_day)
        b = P12.run(S_te, CUR, A1, A2, TILT, base=risk, hrs=hrs, max_day=max_day)
        if a["gate"] and b["gate"]:
            best = (risk, a, b)   # keep climbing to the max gate-safe risk
    return best


def main():
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    print("Max gate-safe $/wk (worst-case of both datasets) by VOLUME:\n")
    print(f"{'session':>22} {'max/day':>7} {'risk':>5} | {'23-24 $/wk':>10} {'WR':>4} {'sig/d':>5} "
          f"{'DDt':>5} | {'25-26 $/wk':>10} {'WR':>4} {'sig/d':>5} {'DDt':>5}")
    print("-" * 104)
    for sname, hrs in SESSIONS.items():
        for md in [2, 3, 4, 6]:
            r = best_risk(tr, te, hrs, md)
            if r is None:
                print(f"{sname:>22} {md:>7} {'--':>5} | {'no gate-safe risk':>30}")
                continue
            risk, a, b = r
            print(f"{sname:>22} {md:>7} ${risk:>4} | ${a['wk']:>+9.1f} {a['wr']:>3.0f}% {a['sig_d']:>4.1f} "
                  f"${a['ddt']:>4.0f} | ${b['wk']:>+9.1f} {b['wr']:>3.0f}% {b['sig_d']:>4.1f} ${b['ddt']:>4.0f}")
        print()


if __name__ == "__main__":
    main()
