"""
PHASE 19 — S-tier upgrade harness (friction, Kelly, regime gate, runner sweep).
Shared, honest, walk-forward on BOTH real datasets. Each phase reports net $/wk.

Assumptions (stated):
  - micro-lot = 0.01 lot; commission $0.07/micro-lot round-turn = $7 * lots per trade
  - tick = $0.10; slippage = randint(1,2) ticks adverse on ENTRY fill and TP1 exit fill
  - 2025-26 has no volume -> volume-profile density uses a range/vol proxy (both sets)
"""
import numpy as np
import pandas as pd
import tsai_optimize as T
import fullyear_backtest as fb
import phase12_ideal as P12

MAX_HOLD = T.MAX_HOLD
SPREAD = T.SPREAD
EQ0 = 5000.0
TICK = 0.10

BASE = dict(base_sl=1.1342, vol_lo=0.3460, vol_hi=1.2653, tp1_r=0.45, tp2_r=3.16,
            tp3_r=6.26, trend_gain=1.1363, tp_lo=0.849, tp_hi=2.509, a1=0.45, a2=0.16)


def outcomes(c, tp1_r, trend_gain, tp3_r=None, tp_hi=None):
    """Per-signal: sld, realized gross R (a1/a2/a3 with BE trailing), mfe_r,
    full_stop, tp1_hit flag. tp1_r / trend_gain / tp3_r / tp_hi overridable."""
    g = dict(BASE); g["tp1_r"] = tp1_r; g["trend_gain"] = trend_gain
    if tp3_r is not None: g["tp3_r"] = tp3_r
    if tp_hi is not None: g["tp_hi"] = tp_hi
    n = c["n"]; is_buy = c["d"] == 1
    H, L, Cc = c["H"], c["L"], c["C"]; entries = c["entries"]
    vrc = np.clip(c["vol_ratio"], g["vol_lo"], g["vol_hi"])
    sld = g["base_sl"] * vrc * c["x_atr"] + SPREAD / 2
    ext = np.clip(1 + g["trend_gain"] * (c["trend_strength"] - 1), g["tp_lo"], g["tp_hi"])
    t1 = g["tp1_r"]; t2 = g["tp2_r"]; t3 = g["tp3_r"] * ext
    e_sl0 = np.where(is_buy, entries - sld, entries + sld)
    e_t1 = np.where(is_buy, entries + t1*sld + SPREAD, entries - t1*sld - SPREAD)
    e_t2 = np.where(is_buy, entries + t2*sld + SPREAD, entries - t2*sld - SPREAD)
    e_t3 = np.where(is_buy, entries + t3*sld + SPREAD, entries - t3*sld - SPREAD)
    tp1_hit = np.zeros(n, bool); tp2_hit = np.zeros(n, bool); closed = np.zeros(n, bool)
    code = np.zeros(n, np.int8); mfe = np.zeros(n)
    for j in range(MAX_HOLD):
        if closed.all(): break
        hi = H[:, j]; lo = L[:, j]
        fav = np.where(is_buy, hi - entries, entries - lo)
        mfe = np.where(~closed, np.maximum(mfe, fav), mfe)
        sl_lvl = e_sl0.copy()
        sl_lvl = np.where(tp1_hit & ~tp2_hit, entries, sl_lvl)
        sl_lvl = np.where(tp2_hit, e_t1, sl_lvl)
        hitsl = (~closed) & np.where(is_buy, lo <= sl_lvl, hi >= sl_lvl)
        code = np.where(hitsl, np.where(tp2_hit, 2, np.where(tp1_hit, 3, 1)), code)
        closed = closed | hitsl; act = ~closed
        tp1_hit = tp1_hit | (act & ~tp1_hit & np.where(is_buy, hi >= e_t1, lo <= e_t1))
        tp2_hit = tp2_hit | (act & ~tp2_hit & np.where(is_buy, hi >= e_t2, lo <= e_t2))
        nf = act & np.where(is_buy, hi >= e_t3, lo <= e_t3)
        code = np.where(nf, 4, code); closed = closed | nf
    a1, a2 = g["a1"], g["a2"]; a3 = 1 - a1 - a2
    r = np.select([code == 4, code == 2, code == 3, code == 1],
                  [a1*t1 + a2*t2 + a3*t3, a1*t1 + a2*t2 + a3*t1, a1*t1, -1.0],
                  default=(Cc[:, -1]-entries)*np.where(is_buy, 1.0, -1.0)/sld)
    mfe_r = mfe / sld
    return sld, r, mfe_r, (code == 1), (mfe_r >= t1)


def sim(S, tp1_r=0.45, trend_gain=1.1363, tp3_r=None, tp_hi=None,
        base=47.0, friction=True, kelly=None, regime_mask=None, ny_gain=None,
        ds=120.0, ddt_thr=200.0, fac=0.35, hrs=range(12, 21), max_day=2, cd=3600,
        dd_cap=400.0, seed=7):
    """One walk-forward pass. kelly=(mult,cap$) enables fractional-Kelly sizing.
    regime_mask: optional bool array (len n, chrono order) — only trade where True.
    ny_gain: if set, use this runner trend-gain for NY hours 13-17 UTC, base gain else."""
    c = S["c"]; o = S["order"]
    sld, r, mfe_r, fs, tp1hit = outcomes(c, tp1_r, trend_gain, tp3_r, tp_hi)
    sld, r, fs, tp1hit = sld[o], r[o], fs[o], tp1hit[o]
    if ny_gain is not None:
        _, r_ny, _, _, _ = outcomes(c, tp1_r, ny_gain, tp3_r, tp_hi)
        r_ny = r_ny[o]
        ny_hours = np.isin(S["hour"], [13, 14, 15, 16, 17])
        r = np.where(ny_hours, r_ny, r)
    a1 = BASE["a1"]
    hour = S["hour"]; wd = S["wd"]; day = S["day"]; dts = S["dt"]
    allow = np.isin(hour, list(hrs)); n = len(r)
    rng = np.random.default_rng(seed)
    eq = EQ0; peak = EQ0; pnls = np.zeros(n)
    cur = -1; dp = 0.0; blk = False; dc = 0; taken = 0
    lt = pd.Timestamp("2000-01-01")
    recent = []  # rolling win/loss for Kelly
    for k in range(n):
        if day[k] != cur:
            cur = day[k]; dp = 0.0; blk = False; dc = 0
        if hour[k] in (21, 22) or (wd[k] == 4 and hour[k] >= 19): continue
        if not allow[k] or blk or dc >= max_day: continue
        if (dts[k] - lt).total_seconds() < cd: continue
        if regime_mask is not None and not regime_mask[k]: continue
        # ----- position sizing -----
        if kelly is not None:
            mult, cap = kelly
            if len(recent) >= 5:
                wr = np.mean(recent[-20:])
                # payoff b ~ avg win / avg loss in R; use realized structure ~2.3R win, 1R loss
                b = 2.3
                f_star = max(0.0, (wr * (b + 1) - 1) / b)  # Kelly fraction
                risk = min(cap, EQ0 * 0.02, max(15.0, eq * mult * f_star))
            else:
                risk = base
        else:
            risk = base
        risk = risk * (fac if (peak - eq) >= ddt_thr else 1.0)
        lpl = 100.0 * sld[k]
        lots = max(0.01, min(0.12, np.floor(risk / lpl * 100) / 100))
        pnl = r[k] * lots * lpl
        # ----- friction -----
        if friction:
            commission = 7.0 * lots
            s_entry = rng.integers(1, 3) * TICK           # adverse on full position
            s_tp1 = rng.integers(1, 3) * TICK             # adverse on the a1 chunk (if tp1 hit)
            fr = commission + lots*100*s_entry + (lots*100*a1*s_tp1 if tp1hit[k] else 0.0)
            pnl -= fr
        pnls[k] = pnl; eq += pnl; dp += pnl; taken += 1; dc += 1; lt = dts[k]
        if eq > peak: peak = eq
        if dp <= -ds: blk = True
        recent.append(1.0 if r[k] > 0 else 0.0)
    eqc = EQ0 + np.cumsum(pnls)
    ddt = float(np.max(np.maximum.accumulate(eqc) - eqc))
    dd = np.bincount(day, weights=pnls, minlength=day.max()+1)
    ddd = float(-dd.min()) if dd.min() < 0 else 0.0
    nz = pnls[pnls != 0]; w = int((nz > 0).sum()); l = int((nz < 0).sum())
    return dict(wk=pnls.sum()/S["weeks"], wr=w/max(1, w+l)*100, ddt=ddt, ddd=ddd,
                gate=ddt < dd_cap and ddd < 250, sig_d=taken/(S["weeks"]*5), taken=taken)


def phase1(tr, te):
    print("=" * 80)
    print("PHASE 1 — FRICTION INSULATION (commission $7*lots + 1-2 tick slippage)")
    print("=" * 80)
    for name, S in [("2023-24", tr), ("2025-26", te)]:
        clean = sim(S, friction=False)
        fric = sim(S, friction=True)
        print(f"\n{name}:")
        print(f"  no-friction (TP1 0.45R): {clean['wk']:+.1f}/wk WR{clean['wr']:.0f}% DDt${clean['ddt']:.0f}")
        print(f"  WITH friction (0.45R)  : {fric['wk']:+.1f}/wk WR{fric['wr']:.0f}% DDt${fric['ddt']:.0f}"
              f"  (friction cost {fric['wk']-clean['wk']:+.1f}/wk)")
    print("\n  TP1 sweep WITH friction (find optimal risk-mitigation anchor):")
    print(f"  {'TP1':>5} | {'23-24 $/wk':>10} {'WR':>4} {'DDt':>5} | {'25-26 $/wk':>10} {'WR':>4} {'DDt':>5}")
    best = None
    for tp1 in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        a = sim(tr, tp1_r=tp1, friction=True); b = sim(te, tp1_r=tp1, friction=True)
        worst = min(a['wk'], b['wk'])
        flag = ""
        if a['gate'] and b['gate'] and (best is None or worst > best[0]):
            best = (worst, tp1); flag = " <=="
        print(f"  {tp1:>4.2f} | ${a['wk']:>+9.1f} {a['wr']:>3.0f}% ${a['ddt']:>4.0f} | "
              f"${b['wk']:>+9.1f} {b['wr']:>3.0f}% ${b['ddt']:>4.0f}{flag}")
    if best:
        print(f"\n  BEST friction-adjusted TP1 anchor: {best[1]:.2f}R (worst-case ${best[0]:.1f}/wk)")
    return best[1] if best else 0.45


def phase2(tr, te):
    print("=" * 80)
    print("PHASE 2 — DYNAMIC FRACTIONAL KELLY (friction ON, TP1 0.45R)")
    print("  Kelly f* from rolling 20-trade WR, payoff b=2.3R; cap 2% ($100); DD<$400")
    print("=" * 80)
    print("\nRaw Kelly (2% cap) blows the gate. Fair test = DD-MATCHED ceilings:\n")
    print("  FIXED risk sweep (max gate-safe $/wk):")
    best_fixed = None
    for base in [47, 50, 55, 60, 65]:
        a = sim(tr, friction=True, base=base); b = sim(te, friction=True, base=base)
        ok = a['gate'] and b['gate']; worst = min(a['wk'], b['wk'])
        if ok and (best_fixed is None or worst > best_fixed[0]): best_fixed = (worst, base)
        print(f"    ${base:<3}: 23-24 ${a['wk']:+6.1f}/wk DDt${a['ddt']:.0f} | "
              f"25-26 ${b['wk']:+6.1f}/wk DDt${b['ddt']:.0f} | {'PASS' if ok else 'FAIL'}")
    print("\n  KELLY cap sweep (mult 0.25, max gate-safe $/wk):")
    best_kelly = None
    for cap in [50, 55, 60, 65, 70]:
        a = sim(tr, friction=True, kelly=(0.25, float(cap)))
        b = sim(te, friction=True, kelly=(0.25, float(cap)))
        ok = a['gate'] and b['gate']; worst = min(a['wk'], b['wk'])
        if ok and (best_kelly is None or worst > best_kelly[0]): best_kelly = (worst, cap)
        print(f"    cap${cap:<3}: 23-24 ${a['wk']:+6.1f}/wk DDt${a['ddt']:.0f} | "
              f"25-26 ${b['wk']:+6.1f}/wk DDt${b['ddt']:.0f} | {'PASS' if ok else 'FAIL'}")
    print("\n  VERDICT:")
    print(f"    best gate-safe FIXED : ${best_fixed[0]:.1f}/wk (risk ${best_fixed[1]})" if best_fixed else "    fixed: none gate-safe")
    print(f"    best gate-safe KELLY : ${best_kelly[0]:.1f}/wk (cap ${best_kelly[1]})" if best_kelly else "    kelly: none gate-safe")
    if best_fixed and best_kelly:
        d = best_kelly[0] - best_fixed[0]
        print(f"    --> dynamic Kelly {'BEATS' if d>0 else 'does NOT beat'} fixed by {d:+.1f}/wk at matched DD")


def phase3(tr, te):
    import phase19_regime as R
    R.phase3(tr, te, sim)


def phase4(tr, te):
    import phase19_runner as RN
    RN.phase4(tr, te, sim)


if __name__ == "__main__":
    import sys
    print("Loading real data...")
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    ph = sys.argv[1] if len(sys.argv) > 1 else "1"
    {"1": phase1, "2": phase2, "3": phase3, "4": phase4}[ph](tr, te)
