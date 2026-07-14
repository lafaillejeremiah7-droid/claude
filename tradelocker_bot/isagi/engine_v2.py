"""
ISAGI v2 — pre-train -> LOCK -> OOS, profit-maximizing, EGOLESS, shock-aware.

S1/S2: PRE-TRAIN a fluid-timeframe + geometry param set that maximizes NET PROFIT
       (risk-adjusted, gate-safe) on the pre-train slice; LOCK the single best set;
       apply UNCHANGED to the OOS backtest (no in-backtest adaptation).
S3:    SHOCK REFLEX learned in pre-train (skip/reduce when volatility-state spikes);
       measured in the OOS backtest during shocks.
S4:    EGOLESS sizing — risk scales with MARKET STATE (volatility), NOT win streak.
       Objective = max net profit / risk-adjusted return, not win rate.
S5:    BASE EDGE = the payoff-structure + management edge (the only real edge proven
       to exist); the loop refines geometry around it.

Real 2025-26 data + real Forex Factory calendar. Accurate bar-by-bar outcomes,
friction, gate. Signal-only.
"""
import time
import numpy as np, pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import phase28c_accurate as ACC
from isagi.calendar_feed import backtest_blackouts, NewsDecoupler

SPREAD = T.SPREAD


def prep_all():
    te = P12.prep(T.load_data()); c = te["c"]; o = te["order"]
    dt = pd.to_datetime(c["dt"][o])
    hour = te["hour"]; day = te["day"]; wd = dt.weekday.values
    vr = c["vol_ratio"][o]                      # market-state (volatility) signal
    news = NewsDecoupler(backtest_blackouts())
    blackout = np.array([news.is_blackout(t) for t in dt])
    # session 12-17 UTC per ISAGI, weekday, not Fri-late, not news-blackout
    session = np.isin(hour, [12, 13, 14, 15, 16]) & (~((wd == 4) & (hour >= 19))) & (~blackout)
    # time split: pre-train = first 60%, OOS = last 40%
    order = np.argsort(dt.values.astype("datetime64[ns]").astype(np.int64))
    cut_t = dt[order[int(len(order)*0.6)]]
    pretrain = np.asarray(dt < cut_t) & session
    oos = np.asarray(dt >= cut_t) & session
    return dict(c=c, o=o, dt=dt, hour=hour, day=day, wd=wd, vr=vr,
                pretrain=pretrain, oos=oos, split=cut_t, n=c["n"])


def sim(S, r, sld, mask, base, ego_k, shock_z, ds=120.0, ddt_thr=200.0, fac=0.35, cd=3600):
    """EGOLESS state-based sizing + shock reflex. Signal-only economics."""
    dt = S["dt"]; day = S["day"]; hour = S["hour"]; wd = S["wd"]; vr = S["vr"]; n = len(r)
    eq = 5000.0; peak = 5000.0; pnls = np.zeros(n)
    cur = -1; dp = 0.0; blk = False; dc = 0; taken = 0; shocks = 0
    lt = pd.Timestamp("2000-01-01")
    for k in range(n):
        if day[k] != cur:
            cur = day[k]; dp = 0.0; blk = False; dc = 0
        if not mask[k] or blk or dc >= 2:
            continue
        if (dt[k] - lt).total_seconds() < cd:
            continue
        # SHOCK REFLEX: skip entry when volatility-state spikes above learned threshold
        if vr[k] >= shock_z:
            shocks += 1
            continue
        # EGOLESS sizing: scale by market state (vol), NOT win streak. High vol -> smaller.
        risk = base * np.clip(1.0 - ego_k * (vr[k] - 1.0), 0.4, 1.6)
        risk = risk * (fac if (peak - eq) >= ddt_thr else 1.0)
        risk = min(100.0, max(15.0, risk))
        lpl = 100.0 * sld[k]
        lots = max(0.01, min(0.12, np.floor(risk / lpl * 100) / 100))
        pnl = r[k] * lots * lpl - (7.0*lots + lots*100*0.15 + (lots*100*0.45*0.15 if r[k] > 0 else 0))
        pnls[k] = pnl; eq += pnl; dp += pnl; taken += 1; dc += 1; lt = dt[k]
        if eq > peak: peak = eq
        if dp <= -ds: blk = True
    eqc = 5000 + np.cumsum(pnls); ddt = float(np.max(np.maximum.accumulate(eqc) - eqc))
    dd = np.bincount(day, weights=pnls, minlength=day.max()+1)
    ddd = float(-dd.min()) if dd.min() < 0 else 0.0
    nz = pnls[pnls != 0]; w = int((nz > 0).sum()); l = int((nz < 0).sum())
    net = float(pnls.sum())
    weeks = (dt[mask].max() - dt[mask].min()).days / 7.0 if mask.sum() else 1
    return dict(net=net, wk=net/weeks, wr=w/max(1, w+l)*100, ddt=ddt, ddd=ddd,
                gate=ddt < 400 and ddd < 250, taken=taken, shocks=shocks,
                risk_adj=net/max(50.0, ddt))   # profit per unit drawdown


def main():
    print("ISAGI v2 — pre-train -> lock -> OOS | profit-max | EGOLESS | shock-aware\n")
    S = prep_all(); c = S["c"]; o = S["o"]
    print(f"split @ {S['split']} | pre-train signals {S['pretrain'].sum()} | OOS {S['oos'].sum()}\n")

    # cache accurate outcomes per geometry (recomputed per config)
    def outcomes(base_sl, vlo, vhi, t1, t2, t3, tg, tphi):
        r, sld, h = ACC.accurate_outcomes(c, base_sl, vlo, vhi, [t1, t2, t3],
                                          [A1, A2, 1-A1-A2], tg, 0.5, tphi)
        return r[o], sld[o]

    global A1, A2
    rng = np.random.default_rng(2)
    best = None; N = 4000; t0 = time.time(); trials = 0
    print(f"PRE-TRAIN search (objective = NET PROFIT, gate-safe):")
    while trials < N and time.time() - t0 < 1200:
        trials += 1
        base_sl = rng.uniform(0.7, 1.5); vlo = rng.uniform(0.3, 0.9); vhi = rng.uniform(1.05, 1.4)
        t1 = rng.uniform(0.4, 1.6); t2 = t1 + rng.uniform(0.5, 2.5); t3 = t2 + rng.uniform(0.5, 4.0)
        tg = rng.uniform(0.5, 3.0); tphi = rng.uniform(1.2, 2.8)
        A1 = rng.uniform(0.2, 0.6); A2 = rng.uniform(0.1, 0.4)
        if A1 + A2 > 0.9: continue
        base = rng.uniform(35, 90); ego_k = rng.uniform(0.0, 1.0); shock = rng.uniform(1.4, 3.0)
        r, sld = outcomes(base_sl, vlo, vhi, t1, t2, t3, tg, tphi)
        m = sim(S, r, sld, S["pretrain"], base, ego_k, shock)
        if m["gate"] and (best is None or m["net"] > best[0]["net"]):
            best = (m, dict(base_sl=base_sl, vlo=vlo, vhi=vhi, t1=t1, t2=t2, t3=t3, tg=tg,
                            tphi=tphi, A1=A1, A2=A2, base=base, ego_k=ego_k, shock=shock))
    pm, cfg = best
    print(f"  trials {trials} | LOCKED config net ${pm['net']:.0f} pre-train "
          f"(${pm['wk']:+.1f}/wk WR{pm['wr']:.0f}% DDt${pm['ddt']:.0f})")
    print(f"  cfg: SL{cfg['base_sl']:.2f} vclip[{cfg['vlo']:.2f},{cfg['vhi']:.2f}] "
          f"TP {cfg['t1']:.2f}/{cfg['t2']:.2f}/{cfg['t3']:.2f} alloc {cfg['A1']:.2f}/{cfg['A2']:.2f} "
          f"tg{cfg['tg']:.2f} base${cfg['base']:.0f} ego_k{cfg['ego_k']:.2f} shock{cfg['shock']:.2f}")

    # LOCK and apply UNCHANGED to OOS
    A1, A2 = cfg["A1"], cfg["A2"]
    r, sld = outcomes(cfg["base_sl"], cfg["vlo"], cfg["vhi"], cfg["t1"], cfg["t2"], cfg["t3"], cfg["tg"], cfg["tphi"])
    oos = sim(S, r, sld, S["oos"], cfg["base"], cfg["ego_k"], cfg["shock"])
    print(f"\n  >>> OOS (locked, no adaptation): ${oos['wk']:+.1f}/wk WR{oos['wr']:.0f}% "
          f"DDt${oos['ddt']:.0f} DDd${oos['ddd']:.0f} shocks-skipped {oos['shocks']} "
          f"{'PASS' if oos['gate'] else 'FAIL-GATE'}")

    # baseline: current phase-15 on OOS, same split
    A1, A2 = 0.45, 0.16
    rb, sb = outcomes(1.1342, 0.346, 1.2653, 0.45, 3.16, 6.26, 1.1363, 2.509)
    base = sim(S, rb, sb, S["oos"], 47.0, 0.0, 99.0)
    print(f"  baseline phase-15 OOS: ${base['wk']:+.1f}/wk WR{base['wr']:.0f}% DDt${base['ddt']:.0f} "
          f"{'PASS' if base['gate'] else 'FAIL-GATE'}")
    print("\n" + "=" * 60)
    verdict = oos["gate"] and oos["wk"] > base["wk"]
    print(f"ISAGI v2 {'BEATS' if verdict else 'does NOT beat'} baseline OOS, gate-safe.")
    print("=" * 60)


if __name__ == "__main__":
    A1, A2 = 0.45, 0.16
    main()
