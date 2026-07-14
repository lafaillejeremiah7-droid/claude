"""
PHASE 16 — realistic FIXED reward:risk structures (1.5:1 .. 3:1).

User: 6:1 is absurd. Test clean structures with a fixed final R:R swept 1.5->3.0,
and keep SL/TP distances sane (no micro-stops, no moon-shot targets).

Structures tested (12-20 UTC, max 2/day, $45+circuit-breaker):
  A) SINGLE      : 100% at the final target, hard SL. Pure k:1 trade.
  B) TP1+FINAL   : half at a near partial (0.5R), SL->breakeven, half to k:1.
  C) TP2+FINAL   : half at a mid partial (1.0R),  SL->breakeven, half to k:1.

SL = base_sl * clip(vol_ratio, 0.80, 1.30) * ATR(15m)  -> always a sane multiple
of ATR. We report the median SL and TP distance in DOLLARS so you can eyeball it.
Validated on BOTH 2023-24 and 2025-26.
"""
import numpy as np
import pandas as pd
import tsai_optimize as T
import fullyear_backtest as fb
import phase12_ideal as P12

MAX_HOLD = T.MAX_HOLD
SPREAD = T.SPREAD
EQ0 = 5000.0
VLO, VHI = 0.80, 1.30


def ladder_R(c, base_sl, tgts_R, alloc):
    """Per-signal realized R for a target ladder. After the FIRST partial the
    stop moves to breakeven (single-target => no BE, hard stop). Returns
    (R array, sl_dist array)."""
    n = c["n"]; is_buy = c["d"] == 1
    H, L, Cc = c["H"], c["L"], c["C"]; entries = c["entries"]
    vrc = np.clip(c["vol_ratio"], VLO, VHI)
    sl_dist = base_sl * vrc * c["x_atr"] + SPREAD / 2
    e_sl0 = np.where(is_buy, entries - sl_dist, entries + sl_dist)
    tgt_px = [np.where(is_buy, entries + t * sl_dist + SPREAD, entries - t * sl_dist - SPREAD)
              for t in tgts_R]
    m = len(tgts_R)
    multi = m > 1
    res = np.zeros(n); level = np.zeros(n, int); done = np.zeros(n, bool)
    be = np.zeros(n, bool)
    cum_alloc = np.concatenate([[0.0], np.cumsum(alloc)])  # banked fraction by level
    for j in range(MAX_HOLD):
        if done.all():
            break
        hi = H[:, j]; lo = L[:, j]
        sl_lvl = np.where(be, entries, e_sl0)
        hitsl = (~done) & np.where(is_buy, lo <= sl_lvl, hi >= sl_lvl)
        # full stop (no partial yet) => -1R whole position; BE stop => keep banked, rest at 0
        res = np.where(hitsl & ~be, -1.0, res)
        done = done | hitsl
        for i in range(m):
            tpx = tgt_px[i]
            reach = (~done) & (level == i) & np.where(is_buy, hi >= tpx, lo <= tpx)
            res = np.where(reach, res + alloc[i] * tgts_R[i], res)
            level = np.where(reach, i + 1, level)
            if multi:
                be = np.where(reach, True, be)
            if i == m - 1:
                done = done | reach
    # timeout: mark remaining open portion to close price
    if not done.all():
        lastc = Cc[:, -1]
        rC = (lastc - entries) * np.where(is_buy, 1.0, -1.0) / sl_dist
        rem = 1.0 - cum_alloc[level]
        res = np.where(~done, res + rem * rC, res)
    return res, sl_dist


def cb_sim(S, r, sld, base=45.0, ds=120.0, ddt_thr=200.0, fac=0.35,
           hrs=range(12, 21), max_day=2, cd=3600):
    o = S["order"]; r = r[o]; sld = sld[o]
    hour = S["hour"]; wd = S["wd"]; day = S["day"]; dts = S["dt"]
    allow = np.isin(hour, list(hrs)); n = len(r)
    eq = EQ0; peak = EQ0; pnls = np.zeros(n)
    cur = -1; dp = 0.0; blk = False; dc = 0; taken = 0
    lt = pd.Timestamp("2000-01-01")
    for k in range(n):
        if day[k] != cur:
            cur = day[k]; dp = 0.0; blk = False; dc = 0
        if hour[k] in (21, 22) or (wd[k] == 4 and hour[k] >= 19):
            continue
        if not allow[k] or blk or dc >= max_day:
            continue
        if (dts[k] - lt).total_seconds() < cd:
            continue
        risk = base * (fac if (peak - eq) >= ddt_thr else 1.0)
        lpl = 100.0 * sld[k]
        lots = max(0.01, min(0.12, np.floor(risk / lpl * 100) / 100))
        pnl = r[k] * lots * lpl
        pnls[k] = pnl; eq += pnl; dp += pnl; taken += 1; dc += 1; lt = dts[k]
        if eq > peak:
            peak = eq
        if dp <= -ds:
            blk = True
    eqc = EQ0 + np.cumsum(pnls)
    ddt = float(np.max(np.maximum.accumulate(eqc) - eqc))
    dd = np.bincount(day, weights=pnls, minlength=day.max() + 1)
    ddd = float(-dd.min()) if dd.min() < 0 else 0.0
    nz = pnls[pnls != 0]; w = int((nz > 0).sum()); l = int((nz < 0).sum())
    return dict(wk=pnls.sum() / S["weeks"], wr=w / max(1, w + l) * 100,
                ddt=ddt, ddd=ddd, gate=ddt < 350 and ddd < 200,
                sig_d=taken / (S["weeks"] * 5))


STRUCTS = {
    "A) SINGLE (100% @ k:1, hard SL)":      dict(tgts=lambda k: [k],        alloc=[1.0]),
    "B) TP1+FINAL (50% @0.5R, 50% @k:1)":   dict(tgts=lambda k: [0.5, k],   alloc=[0.5, 0.5]),
    "C) TP2+FINAL (50% @1.0R, 50% @k:1)":   dict(tgts=lambda k: [1.0, k],   alloc=[0.5, 0.5]),
}
BASE_SL = 1.0  # keeps SL ~1x ATR — a realistic gold stop


def main():
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    print(f"SL geometry: {BASE_SL} x clip(vol_ratio,{VLO},{VHI}) x ATR(15m)  (sane, ~1x ATR)\n")
    for name, sc in STRUCTS.items():
        print("=" * 92)
        print(name)
        print(f"{'k(R:R)':>7} | {'medSL$':>6} {'medTP$':>6} | "
              f"{'23-24 $/wk':>10} {'WR':>4} {'DDt':>5} {'g':>3} | "
              f"{'25-26 $/wk':>10} {'WR':>4} {'DDt':>5} {'g':>3} | {'sig/d':>5}")
        for k in [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
            tg = sc["tgts"](k)
            if any(tg[i] >= tg[i + 1] for i in range(len(tg) - 1)):
                continue
            r_tr, sld_tr = ladder_R(tr["c"], BASE_SL, tg, sc["alloc"])
            r_te, sld_te = ladder_R(te["c"], BASE_SL, tg, sc["alloc"])
            a = cb_sim(tr, r_tr, sld_tr); b = cb_sim(te, r_te, sld_te)
            medsl = float(np.median(sld_te)); medtp = medsl * k
            print(f"  {k:>4.2f} | ${medsl:>5.1f} ${medtp:>5.1f} | "
                  f"${a['wk']:>+9.1f} {a['wr']:>3.0f}% ${a['ddt']:>4.0f} {'P' if a['gate'] else 'F':>3} | "
                  f"${b['wk']:>+9.1f} {b['wr']:>3.0f}% ${b['ddt']:>4.0f} {'P' if b['gate'] else 'F':>3} | "
                  f"{b['sig_d']:>4.1f}")
    print("=" * 92)
    print("Reference — shipped phase-15 (6R runner): 25-26 +$54/wk WR70% | 23-24 +$55/wk WR64%")


if __name__ == "__main__":
    main()
