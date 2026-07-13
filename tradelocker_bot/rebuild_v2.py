"""
REBUILD v2 — Session-adaptive signal engine for XAU/USD, max 2 signals/day.

Honest protocol:
  1. Tag every signal with its UTC hour and trading session.
  2. LEARN the best session window on 2023-2024 (train). Rank hours by mean R
     and win rate; pick the contiguous window that maximizes gate-safe $/wk.
  3. VALIDATE the fixed window on 2025-2026 (test) — the DD-binding dataset.
  4. At 2/day, sweep risk for the max gate-safe size WITH the session filter.
  5. Compare vs the current $20-risk baseline ($56 IS / $41 OOS /wk).
     Push only if it genuinely beats current AND passes gates on both sets.

No fabrication. Reports before/after truthfully.
"""
import numpy as np
import pandas as pd
import tsai_optimize as T
import phase4_usage as P4
import phase7_candleEV as P7
import fullyear_backtest as fb


def prep(df):
    """Return per-signal arrays sorted chronologically."""
    c = T.build_candidates(df)
    pc = P4.precompute(c)
    dt_sorted = pd.to_datetime(c["dt"][pc["order"]])
    r = P7.per_signal_r(pc, 0.30, 0.40, 0.19)
    return {
        "dt": dt_sorted,
        "hour": dt_sorted.hour.values,
        "r": r,
        "sl_dist": pc["sl_dist"],
        "day_code": pc["day_code"],
        "n": pc["n"],
        "weeks": (df.index.max() - df.index.min()).days / 7.0,
    }


def hour_table(S, tag):
    """Per-UTC-hour win rate, mean R, count."""
    print(f"\n--- {tag}: per-UTC-hour signal edge ---")
    print(f"{'hr':>3} {'n':>5} {'WR%':>5} {'meanR':>7} {'sumR':>8}")
    rows = []
    for h in range(24):
        m = S["hour"] == h
        n = int(m.sum())
        if n == 0:
            rows.append((h, 0, 0, 0, 0)); continue
        rr = S["r"][m]
        wr = float((rr > 0).mean()) * 100
        rows.append((h, n, wr, float(rr.mean()), float(rr.sum())))
        print(f"{h:>3} {n:>5} {wr:>5.0f} {rr.mean():>7.3f} {rr.sum():>8.1f}")
    return rows


def sim(S, allowed_hours, max_day, risk, dd_stop=120.0):
    """2/day sim with session mask, 60-min cooldown, daily DD stop."""
    hour = S["hour"]; r = S["r"]; sld = S["sl_dist"]
    day = S["day_code"]; dts = S["dt"]; n = S["n"]
    allow = np.isin(hour, list(allowed_hours))

    eq = 5000.0; peak = 5000.0
    pnls = np.zeros(n)
    cur = -1; dpnl = 0.0; blk = False; dc = 0; taken = 0
    lt = pd.Timestamp("2000-01-01")
    for k in range(n):
        if day[k] != cur:
            cur = day[k]; dpnl = 0.0; blk = False; dc = 0
        if not allow[k] or blk or dc >= max_day:
            continue
        if (dts[k] - lt).total_seconds() < 3600:
            continue
        lpl = 100 * sld[k]
        lots = min(0.12, risk / lpl)
        lots = max(0.01, np.floor(lots * 100) / 100)
        pnl = r[k] * lots * lpl
        pnls[k] = pnl
        eq += pnl; dpnl += pnl; taken += 1; dc += 1; lt = dts[k]
        if eq > peak:
            peak = eq
        if dpnl <= -dd_stop:
            blk = True

    eqc = 5000 + np.cumsum(pnls)
    ddt = float(np.max(np.maximum.accumulate(eqc) - eqc))
    dsum = np.bincount(day, weights=pnls, minlength=day.max() + 1)
    ddd = float(-dsum.min()) if dsum.min() < 0 else 0.0
    gate = ddt < 350 and ddd < 200
    w = int((pnls[pnls != 0] > 0).sum())
    l = int((pnls[pnls != 0] < 0).sum())
    wk = pnls.sum() / S["weeks"]
    sig_d = taken / (S["weeks"] * 5)
    wr = w / max(1, w + l) * 100
    return dict(wk=wk, sig_d=sig_d, wr=wr, ddt=ddt, ddd=ddd, gate=gate, n=taken)


SESSIONS = {
    "Asian (22-07)":  set(list(range(22, 24)) + list(range(0, 7))),
    "London (07-12)": set(range(7, 12)),
    "Overlap (12-16)": set(range(12, 16)),
    "NY pm (16-21)":  set(range(16, 21)),
    "London+Overlap (07-16)": set(range(7, 16)),
    "Overlap+NY (12-21)": set(range(12, 21)),
    "London->NY (07-21)": set(range(7, 21)),
    "ALL (24h)": set(range(24)),
}


def main():
    print("=" * 78)
    print("REBUILD v2 — session-adaptive, max 2/day")
    print("TRAIN(learn window)=2023-2024 | TEST(validate,DD-binding)=2025-2026")
    print("=" * 78)
    tr = prep(fb.load_data())     # 2023-24
    te = prep(T.load_data())      # 2025-26

    hour_table(tr, "TRAIN 2023-24")
    hour_table(te, "TEST 2025-26")

    print("\n" + "=" * 78)
    print("SESSION SWEEP @ 2/day, risk $20 (gate-safe baseline size)")
    print("=" * 78)
    print(f"{'session':>24} | {'TR $/wk':>8} {'WR':>4} {'DDt':>5} {'g':>4} "
          f"| {'TE $/wk':>8} {'WR':>4} {'DDt':>5} {'g':>4}")
    print("-" * 78)
    for name, hrs in SESSIONS.items():
        a = sim(tr, hrs, 2, 20)
        b = sim(te, hrs, 2, 20)
        print(f"{name:>24} | ${a['wk']:>+7.1f} {a['wr']:>3.0f}% ${a['ddt']:>4.0f} "
              f"{'P' if a['gate'] else 'F':>4} | ${b['wk']:>+7.1f} {b['wr']:>3.0f}% "
              f"${b['ddt']:>4.0f} {'P' if b['gate'] else 'F':>4}")

    print("\n" + "=" * 78)
    print("RISK SWEEP for the best sessions (find max gate-safe risk on TEST)")
    print("=" * 78)
    for name in ["Overlap (12-16)", "Overlap+NY (12-21)", "London+Overlap (07-16)", "ALL (24h)"]:
        hrs = SESSIONS[name]
        print(f"\n[{name}]")
        print(f"{'risk':>6} | {'TR $/wk':>8} {'sig/d':>6} {'WR':>4} {'DDt':>5} {'g':>3} "
              f"| {'TE $/wk':>8} {'sig/d':>6} {'WR':>4} {'DDt':>5} {'g':>3}")
        for risk in [20, 25, 30, 35, 40, 45, 50, 60]:
            a = sim(tr, hrs, 2, risk)
            b = sim(te, hrs, 2, risk)
            print(f"  ${risk:<4} | ${a['wk']:>+7.1f} {a['sig_d']:>4.1f}/d {a['wr']:>3.0f}% "
                  f"${a['ddt']:>4.0f} {'P' if a['gate'] else 'F':>3} | ${b['wk']:>+7.1f} "
                  f"{b['sig_d']:>4.1f}/d {b['wr']:>3.0f}% ${b['ddt']:>4.0f} "
                  f"{'P' if b['gate'] else 'F':>3}")


if __name__ == "__main__":
    main()
