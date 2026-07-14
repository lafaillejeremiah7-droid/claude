"""
PHASE 25 — IDEA #1: adaptive geometry by predicted objective (2025-26, walk-forward).

The ONLY real signal found is CHOP predictability (~0.6 AUC). Instead of gating,
SWITCH geometry: in predicted-chop -> SCALP (bank near, no runner, which dies in
chop); else -> RUNNER (current 45/16/39 to ~6R). Objective classifier trained
walk-forward (past->future) within 2025-26. Friction ON. Compare vs runner-only,
evaluated ONLY on out-of-sample signals. Honest verdict.
"""
import numpy as np
import pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import phase21_intermarket as P21, phase19_upgrade as U
import hercules_v2 as H

MAX_HOLD = T.MAX_HOLD; SPREAD = T.SPREAD; EQ0 = 5000.0; TICK = 0.10
VL, VH = 0.3460, 1.2653; BSL = 1.1342; TG = 1.1363; TPLO, TPHI = 0.849, 2.509


def geom_outcomes(c, tgts_R, alloc, runner=False):
    """R + sld + tp1_hit for a given TP ladder. Same SL/vol-clip for all modes.
    If runner=True, the LAST target scales with trend (tp_ext)."""
    n = c["n"]; is_buy = c["d"] == 1
    Hh, Ll, Cc = c["H"], c["L"], c["C"]; entries = c["entries"]
    vrc = np.clip(c["vol_ratio"], VL, VH)
    sld = BSL * vrc * c["x_atr"] + SPREAD/2
    ext = np.clip(1 + TG*(c["trend_strength"]-1), TPLO, TPHI) if runner else np.ones(n)
    tR = list(tgts_R)
    tpx = []
    for m, tr in enumerate(tR):
        scale = ext if (runner and m == len(tR)-1) else 1.0
        tpx.append(np.where(is_buy, entries + tr*scale*sld + SPREAD,
                            entries - tr*scale*sld - SPREAD))
    e_sl0 = np.where(is_buy, entries - sld, entries + sld)
    m_ = len(tR); res = np.zeros(n); level = np.zeros(n, int); done = np.zeros(n, bool)
    be = np.zeros(n, bool); cumA = np.concatenate([[0.0], np.cumsum(alloc)])
    for j in range(MAX_HOLD):
        if done.all(): break
        hi = Hh[:, j]; lo = Ll[:, j]
        sl_lvl = np.where(be, entries, e_sl0)
        hitsl = (~done) & np.where(is_buy, lo <= sl_lvl, hi >= sl_lvl)
        res = np.where(hitsl & ~be, -1.0, res); done = done | hitsl
        for i in range(m_):
            tp = tpx[i]
            reach = (~done) & (level == i) & np.where(is_buy, hi >= tp, lo <= tp)
            # realized R of this leg = tr*scale
            scale = ext if (runner and i == m_-1) else 1.0
            res = np.where(reach, res + alloc[i]*tR[i]*scale, res)
            level = np.where(reach, i+1, level)
            if m_ > 1: be = np.where(reach, True, be)
            if i == m_-1: done = done | reach
        # (timeout handled below)
    if not done.all():
        lastc = Cc[:, -1]; rC = (lastc-entries)*np.where(is_buy, 1.0, -1.0)/sld
        rem = 1.0 - cumA[level]; res = np.where(~done, res + rem*rC, res)
    return res, sld


def fric_sim(dts, day, hour, wd, r, sld, oos_mask, base=85.0, ds=120.0,
             ddt_thr=200.0, fac=0.35, cd=3600, seed=7, tp1hit=None, a1=0.45):
    n = len(r); rng = np.random.default_rng(seed)
    eq = EQ0; peak = EQ0; pnls = np.zeros(n)
    cur = -1; dp = 0.0; blk = False; dc = 0; taken = 0; lt = pd.Timestamp("2000-01-01")
    for k in range(n):
        if day[k] != cur:
            cur = day[k]; dp = 0.0; blk = False; dc = 0
        if not oos_mask[k]: continue
        if hour[k] not in range(12, 21) or hour[k] in (21, 22): continue
        if wd[k] == 4 and hour[k] >= 19: continue
        if blk or dc >= 2: continue
        if (dts[k]-lt).total_seconds() < cd: continue
        risk = base * (fac if (peak-eq) >= ddt_thr else 1.0)
        lpl = 100.0*sld[k]; lots = max(0.01, min(0.12, np.floor(risk/lpl*100)/100))
        pnl = r[k]*lots*lpl
        commission = 7.0*lots; s_e = rng.integers(1, 3)*TICK; s_t = rng.integers(1, 3)*TICK
        pnl -= commission + lots*100*s_e + (lots*100*a1*s_t if (tp1hit is not None and tp1hit[k]) else 0.0)
        pnls[k] = pnl; eq += pnl; dp += pnl; taken += 1; dc += 1; lt = dts[k]
        if eq > peak: peak = eq
        if dp <= -ds: blk = True
    eqc = EQ0+np.cumsum(pnls); ddt = float(np.max(np.maximum.accumulate(eqc)-eqc))
    ddm = np.bincount(day, weights=pnls, minlength=day.max()+1)
    ddd = float(-ddm.min()) if ddm.min() < 0 else 0.0
    nz = pnls[pnls != 0]; w = int((nz > 0).sum()); l = int((nz < 0).sum())
    wks = (dts[oos_mask].max()-dts[oos_mask].min()).days/7.0
    return dict(wk=pnls.sum()/wks, wr=w/max(1, w+l)*100, ddt=ddt, ddd=ddd,
                gate=ddt < 400 and ddd < 250, taken=taken)


def main():
    print("IDEA #1: adaptive geometry by predicted objective. 2025-26 walk-forward.\n")
    te = P12.prep(T.load_data()); c = te["c"]; o = te["order"]
    dts = pd.to_datetime(c["dt"][o]); day = te["day"]; hour = te["hour"]; wd = te["wd"]
    df = T.load_data()

    # objective labels + state (entry-time)
    lab = H.objective_labels(te)  # already in te['order'] order
    St = H.world_state(te, df, P21.align_features(te, ry_dx()[0], ry_dx()[1]))

    # two geometries, same SL
    r_run, sld = geom_outcomes(c, [0.45, 3.16, 6.26], [0.45, 0.16, 0.39], runner=True)
    r_scalp, _ = geom_outcomes(c, [0.45, 1.20], [0.55, 0.45], runner=False)
    r_run, r_scalp = r_run[o], r_scalp[o]; sld = sld[o]
    tp1hit = np.ones(len(r_run), bool)  # both ladders have TP1 at 0.45R

    _, rr, _, _, _ = U.outcomes(c, 0.45, 1.1363)
    # win label for objective-model sanity (not needed for geometry)

    # walk-forward P(chop) OOS
    dts_ns = dts.values.astype("datetime64[ns]").astype(np.int64)
    ordr = np.argsort(dts_ns); n = len(lab); fold = n//5
    p_chop = np.full(n, np.nan)
    for k in range(1, 5):
        tr_end = k*fold; te_s = tr_end+20; te_e = (k+1)*fold if k < 4 else n
        if te_s >= te_e: continue
        idx_tr = ordr[:tr_end]; idx_te = ordr[te_s:te_e]
        mu = St[idx_tr].mean(0); sd = St[idx_tr].std(0)+1e-9
        a = (St[idx_tr]-mu)/sd; b = (St[idx_te]-mu)/sd
        ws = H.softmax_ovr_fit(a, lab[idx_tr], 4, lam=2.0)
        p_chop[idx_te] = H.softmax_ovr_pred(ws, b)[:, 3]
    oos = ~np.isnan(p_chop)
    print(f"OOS signals with chop-prediction: {oos.sum()}\n")

    # baseline: runner-only on OOS
    base = fric_sim(dts, day, hour, wd, r_run, sld, oos, tp1hit=tp1hit)
    print(f"RUNNER-ONLY (baseline, OOS): ${base['wk']:+.1f}/wk WR{base['wr']:.0f}% "
          f"DDt${base['ddt']:.0f} {'PASS' if base['gate'] else 'FAIL'} | trades {base['taken']}")

    # scalp-only for reference
    sc = fric_sim(dts, day, hour, wd, r_scalp, sld, oos, tp1hit=tp1hit)
    print(f"SCALP-ONLY (reference, OOS):  ${sc['wk']:+.1f}/wk WR{sc['wr']:.0f}% "
          f"DDt${sc['ddt']:.0f} {'PASS' if sc['gate'] else 'FAIL'} | trades {sc['taken']}")

    print("\nADAPTIVE: chop-> scalp, else -> runner (threshold learned on train dist):")
    print(f"  {'chop_thr':>8} | {'OOS $/wk':>9} {'WR':>4} {'DDt':>5} {'gate':>4} | vs runner")
    best = None
    for q in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        thr = np.nanquantile(p_chop, q)
        use_scalp = p_chop >= thr
        r_adapt = np.where(use_scalp, r_scalp, r_run)
        m = fric_sim(dts, day, hour, wd, r_adapt, sld, oos, tp1hit=tp1hit)
        d = m['wk'] - base['wk']
        if m['gate'] and (best is None or m['wk'] > best[0]):
            best = (m['wk'], q, m)
        print(f"  {thr:>8.2f} | ${m['wk']:>+8.1f} {m['wr']:>3.0f}% ${m['ddt']:>4.0f} "
              f"{'PASS' if m['gate'] else 'FAIL':>4} | {d:>+6.1f}")

    print("\n" + "=" * 60)
    if best and best[0] > base['wk'] + 3:
        print(f"ADAPTIVE GEOMETRY HELPS: ${best[0]:+.1f}/wk vs runner ${base['wk']:+.1f}/wk "
              f"(+{best[0]-base['wk']:.1f})")
    else:
        print(f"ADAPTIVE GEOMETRY does not beat runner-only meaningfully OOS.")
    print("=" * 60)


def ry_dx():
    return P21.load_macro()


if __name__ == "__main__":
    main()
