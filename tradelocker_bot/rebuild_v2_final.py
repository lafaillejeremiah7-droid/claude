"""
REBUILD v2 FINAL — apples-to-apples vs the ACTUAL deployed live config.

Live bot today: base $45 + circuit-breaker (daily_stop $120, dd_throttle $200,
factor 0.35), MAX 2/day, 60-min cooldown, all hours except 21-22 UTC & Fri eve.

Unified sim reproduces ALL of that. Then compares:
  (A) CURRENT   : all hours (minus 21,22), $45 base + CB, 2/day, cooldown
  (B) SESSION   : only 12-20 UTC (London/NY overlap + NY), base risk swept, same CB
Validated on 2023-24 AND 2025-26. Push only if B beats A and passes gates on both.
"""
import numpy as np
import pandas as pd
import tsai_optimize as T
import phase4_usage as P4
import phase7_candleEV as P7
import fullyear_backtest as fb

EQ0 = 5000.0


def prep(df):
    c = T.build_candidates(df)
    pc = P4.precompute(c)
    dt = pd.to_datetime(c["dt"][pc["order"]])
    r = P7.per_signal_r(pc, 0.30, 0.40, 0.19)
    return dict(dt=dt, hour=dt.hour.values, wd=dt.weekday.values, r=r,
                sld=pc["sl_dist"], ent=pc["entries"], day=pc["day_code"],
                n=pc["n"], weeks=(df.index.max() - df.index.min()).days / 7.0)


def sim(S, allowed_hours, base, daily_stop=120.0, dd_thr=200.0, factor=0.35,
        max_day=2, cooldown=3600):
    hour = S["hour"]; wd = S["wd"]; r = S["r"]; sld = S["sld"]; ent = S["ent"]
    day = S["day"]; dts = S["dt"]; n = S["n"]
    allow = np.isin(hour, list(allowed_hours))
    equity = EQ0; peak = EQ0
    pnls = np.zeros(n)
    cur = -1; dpnl = 0.0; blk = False; dc = 0; taken = 0
    lt = pd.Timestamp("2000-01-01")
    for k in range(n):
        if day[k] != cur:
            cur = day[k]; dpnl = 0.0; blk = False; dc = 0
        # live session guards: skip thin liquidity 21,22 and Fri-eve
        if hour[k] in (21, 22):
            continue
        if wd[k] == 4 and hour[k] >= 19:
            continue
        if not allow[k] or blk or dc >= max_day:
            continue
        if (dts[k] - lt).total_seconds() < cooldown:
            continue
        dd = peak - equity
        risk = base * (factor if dd >= dd_thr else 1.0)
        lpl = 100.0 * sld[k]
        lots = min(0.12, risk / lpl)
        lots = max(0.01, np.floor(lots * 100) / 100)
        if lots * 100.0 * ent[k] / 10.0 > equity * 0.95:
            lots = max(0.01, np.floor((equity * 0.95 * 10.0 / (100.0 * ent[k])) * 100) / 100)
        pnl = r[k] * lots * lpl
        pnls[k] = pnl
        equity += pnl; dpnl += pnl; taken += 1; dc += 1; lt = dts[k]
        if equity > peak:
            peak = equity
        if dpnl <= -daily_stop:
            blk = True
    eq = EQ0 + np.cumsum(pnls)
    ddt = float(np.max(np.maximum.accumulate(eq) - eq))
    dsum = np.bincount(day, weights=pnls, minlength=day.max() + 1)
    ddd = float(-dsum.min()) if dsum.min() < 0 else 0.0
    w = int((pnls[pnls != 0] > 0).sum()); l = int((pnls[pnls != 0] < 0).sum())
    return dict(wk=pnls.sum() / S["weeks"], ddt=ddt, ddd=ddd,
                gate=ddt < 350 and ddd < 200, wr=w / max(1, w + l) * 100,
                sig_d=taken / (S["weeks"] * 5), n=taken)


def row(tag, a, b):
    print(f"{tag:>26} | ${a['wk']:>+7.1f} {a['wr']:>3.0f}% ${a['ddt']:>4.0f}/${a['ddd']:>3.0f} "
          f"{'PASS' if a['gate'] else 'FAIL':>4} | ${b['wk']:>+7.1f} {b['wr']:>3.0f}% "
          f"${b['ddt']:>4.0f}/${b['ddd']:>3.0f} {'PASS' if b['gate'] else 'FAIL':>4}")


def main():
    tr = prep(fb.load_data())   # 2023-24
    te = prep(T.load_data())    # 2025-26
    ALL = set(range(24))
    SESS = set(range(12, 21))   # 12-20 UTC

    print("=" * 92)
    print("APPLES-TO-APPLES: current live vs session-filtered (same circuit-breaker, 2/day)")
    print(f"{'config':>26} | {'2023-24 $/wk WR  DDt/DDd  gate':>38} | {'2025-26 $/wk WR  DDt/DDd  gate':>38}")
    print("-" * 92)

    # (A) CURRENT deployed: all hours, $45 base + CB
    a_cur = sim(tr, ALL, 45); b_cur = sim(te, ALL, 45)
    row("A) CURRENT all-hrs $45", a_cur, b_cur)

    print("  --- (B) SESSION 12-20 UTC + same CB, base-risk sweep ---")
    best = None
    for base in [30, 35, 40, 45, 50, 55, 60]:
        a = sim(tr, SESS, base); b = sim(te, SESS, base)
        row(f"B) sess 12-20 ${base}", a, b)
        if a["gate"] and b["gate"]:
            # track the highest-base gate-safe (max profit) on both sets
            if best is None or base > best[0]:
                best = (base, a, b)

    print("=" * 92)
    if best:
        base, a, b = best
        print(f"MAX gate-safe SESSION config: 12-20 UTC, base ${base} + CB")
        print(f"  2023-24: ${a['wk']:+.1f}/wk WR{a['wr']:.0f}% DDt${a['ddt']:.0f} sig/d {a['sig_d']:.1f}")
        print(f"  2025-26: ${b['wk']:+.1f}/wk WR{b['wr']:.0f}% DDt${b['ddt']:.0f} sig/d {b['sig_d']:.1f}")
        print(f"  vs CURRENT: 2023-24 ${a['wk']-a_cur['wk']:+.1f}/wk, "
              f"2025-26 ${b['wk']-b_cur['wk']:+.1f}/wk  "
              f"(current gate: TR {'PASS' if a_cur['gate'] else 'FAIL'}, "
              f"TE {'PASS' if b_cur['gate'] else 'FAIL'})")
    else:
        print("No session config passes gates on both sets.")


if __name__ == "__main__":
    main()
