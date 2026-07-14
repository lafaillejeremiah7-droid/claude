"""
ISAGI v4 — DEEP pre-train with per-trade adaptation, optimized for ROBUSTNESS.

Fixes v3 (whose fixed adaptation rules hurt): here the ADAPTATION PARAMETERS
themselves are searched in pre-train, and every config must hold across 3
chronological pre-train sub-windows (score = WORST fold), so it generalizes.
Adaptation is causal ("adapt after every single trade" using only past closes).
LOCK the best robust config, run OOS ONCE. Profit objective, EGOLESS, shock-aware.

Fast first-hit resolver (searchsorted on cummax excursions) enables a deep search.
"""
import time
import numpy as np, pandas as pd
import phase12_ideal as P12, tsai_optimize as T
from isagi.calendar_feed import backtest_blackouts, NewsDecoupler

SPREAD = T.SPREAD; MAXH = 200; NB = 200


def precompute(te):
    c = te["c"]; o = te["order"]
    is_buy = (c["d"][o] == 1)
    H = c["H"][o][:, :NB]; L = c["L"][o][:, :NB]; C = c["C"][o][:, :NB]
    entry = c["entries"][o]; atr = c["x_atr"][o]
    fav = np.where(is_buy[:, None], H - entry[:, None], entry[:, None] - L) / atr[:, None]
    adv = np.where(is_buy[:, None], entry[:, None] - L, H - entry[:, None]) / atr[:, None]
    fav = np.nan_to_num(fav, nan=-9); adv = np.nan_to_num(adv, nan=-9)
    cmf = np.maximum.accumulate(fav, 1)      # cummax favorable (ATR), monotone
    cma = np.maximum.accumulate(adv, 1)      # cummax adverse (ATR), monotone
    # BE return: first bar where price back to entry (adv>=0), as next-index table
    back = adv >= 0.0
    be_next = np.full((len(entry), NB + 1), NB, np.int32)
    for i in range(len(entry)):
        nx = NB
        row = back[i]
        for b in range(NB - 1, -1, -1):
            if row[b]:
                nx = b
            be_next[i, b] = nx
    return dict(cmf=cmf, cma=cma, be=be_next, atr=atr, vr=c["vol_ratio"][o],
                ts=c["trend_strength"][o], hour=te["hour"], day=te["day"],
                dt=pd.to_datetime(c["dt"][o]), n=len(entry), lastc_R=None)


def fh(cmrow, lvl):
    b = int(np.searchsorted(cmrow, lvl, side="left"))
    return b if b < NB else NB


def resolve(P, k, sld_atr, t1, t2, t3, a1, a2):
    a3 = 1 - a1 - a2
    cmf = P["cmf"][k]; cma = P["cma"][k]
    b_sl = fh(cma, sld_atr)
    b_t1 = fh(cmf, t1 * sld_atr)
    if b_sl <= b_t1:                    # full stop before TP1
        b_t3 = fh(cmf, t3 * sld_atr)
        sweep = b_t3 < NB and b_t3 > b_sl
        return -1.0, 1, sweep, cmf[max(0, b_sl - 1)] / sld_atr
    R = a1 * t1
    be_stop = P["be"][k, min(b_t1, NB)]
    b_t2 = fh(cmf, t2 * sld_atr)
    if b_t2 < be_stop:
        R += a2 * t2
        b_t3 = fh(cmf, t3 * sld_atr)
        # runner: reach t3 before falling back to t1 level (approx via be after t2)
        fall = P["be"][k, min(b_t2, NB)]
        R += a3 * t3 if b_t3 <= fall else a3 * t1
        return R, (4 if b_t3 <= fall else 2), False, cmf[-1] / sld_atr
    return R, 3, False, cmf[-1] / sld_atr   # tp1 only (BE scratch)


def run(P, idxs, cfg, learn=True):
    (bsl, vlo, vhi, t1, t2, t3, tg, tphi, a1, a2, base, egok, shock,
     swg, wcut, ecut, dec) = cfg
    sl_adj = 1.0; tp_adj = 1.0; sz_adj = 1.0
    eq = 5000.0; peak = 5000.0; pnls = []
    cur = -1; dc = 0; blk = False; dp = 0.0; lt = pd.Timestamp("2000-01-01")
    day = P["day"]; dt = P["dt"]; vr = P["vr"]; ts = P["ts"]; hour = P["hour"]; atr = P["atr"]
    daypnl = {}
    for k in idxs:
        if day[k] != cur:
            cur = day[k]; dc = 0; blk = False; dp = 0.0
        if blk or dc >= 2:
            continue
        if (dt[k] - lt).total_seconds() < 3600:
            continue
        if vr[k] >= shock:                      # shock reflex: skip
            continue
        vrc = min(vhi, max(vlo, vr[k]))
        sld_atr = bsl * vrc * (sl_adj if learn else 1.0)
        t3e = t3 * np.clip(1 + tg * (ts[k] - 1), 0.5, tphi) * (tp_adj if learn else 1.0)
        t2e = t2 * (tp_adj if learn else 1.0)
        R, code, sweep, maxfav = resolve(P, k, sld_atr, t1, t2e, t3e, a1, a2)
        sld_price = sld_atr * atr[k] + SPREAD / 2
        risk = base * np.clip(1.0 - egok * (vr[k] - 1.0), 0.4, 1.6) * (sz_adj if learn else 1.0)
        risk = risk * (0.35 if (peak - eq) >= 200 else 1.0)
        risk = min(100.0, max(15.0, risk))
        lpl = 100.0 * sld_price
        lots = max(0.01, min(0.12, np.floor(risk / lpl * 100) / 100))
        pnl = R * lots * lpl - (7.0 * lots + lots * 100 * 0.15 + (lots * 100 * a1 * 0.15 if R > 0 else 0))
        pnls.append(pnl); eq += pnl; peak = max(peak, eq); dp += pnl; dc += 1; lt = dt[k]
        daypnl[cur] = daypnl.get(cur, 0) + pnl
        if dp <= -120: blk = True
        if learn:                               # CAUSAL adapt after this close
            if code == 1:
                if sweep: sl_adj = min(1.8, sl_adj * swg)
                elif maxfav >= 0.7 * t2: tp_adj = max(0.6, tp_adj * wcut)
                elif hour[k] >= 16 or ts[k] < 0.8: sz_adj = max(0.5, sz_adj * ecut)
            elif R > 0:
                sl_adj = 1 + (sl_adj - 1) * dec
                tp_adj = 1 + (tp_adj - 1) * dec
                sz_adj = 1 + (sz_adj - 1) * dec
    pnls = np.array(pnls)
    if len(pnls) < 10:
        return None
    eqc = 5000 + np.cumsum(pnls); ddt = float(np.max(np.maximum.accumulate(eqc) - eqc))
    ddd = float(-min(daypnl.values())) if daypnl and min(daypnl.values()) < 0 else 0.0
    w = int((pnls > 0).sum()); l = int((pnls < 0).sum())
    dts = P["dt"][idxs]; wks = max(1, (dts.max() - dts.min()).days / 7.0)
    return dict(net=float(pnls.sum()), wk=pnls.sum() / wks, wr=w / max(1, w + l) * 100,
                ddt=ddt, ddd=ddd, gate=ddt < 400 and ddd < 250, n=len(pnls))


def main():
    print("ISAGI v4 — DEEP robust pre-train (per-trade adaptation optimized) -> OOS once\n")
    te = P12.prep(T.load_data()); P = precompute(te)
    dt = P["dt"]; hour = P["hour"]; wd = dt.weekday.values
    news = NewsDecoupler(backtest_blackouts())
    blackout = np.array([news.is_blackout(t) for t in dt])
    session = np.isin(hour, [12, 13, 14, 15, 16]) & (~((wd == 4) & (hour >= 19))) & (~blackout)
    order = np.argsort(dt.values.astype("datetime64[ns]").astype(np.int64))
    order = order[session[order]]
    cut = int(len(order) * 0.6)
    pre, oos = order[:cut], order[cut:]
    # 3 robustness sub-folds within pre-train
    f = len(pre) // 3
    folds = [pre[:f], pre[f:2*f], pre[2*f:]]
    print(f"pre-train {len(pre)} (3 folds ~{f}) | OOS {len(oos)}\n")

    rng = np.random.default_rng(4); best = None; N = 60000; t0 = time.time(); tr = 0
    while tr < N and time.time() - t0 < 1300:
        tr += 1
        a1 = rng.uniform(0.2, 0.55); a2 = rng.uniform(0.1, 0.4)
        if a1 + a2 > 0.9: continue
        t1 = rng.uniform(0.4, 1.4); t2 = t1 + rng.uniform(0.5, 2.5); t3 = t2 + rng.uniform(0.5, 4)
        cfg = (rng.uniform(0.8, 1.5), rng.uniform(0.4, 0.9), rng.uniform(1.05, 1.35),
               t1, t2, t3, rng.uniform(0.5, 3), rng.uniform(1.2, 2.6), a1, a2,
               rng.uniform(40, 85), rng.uniform(0, 0.6), rng.uniform(1.5, 3.0),
               rng.uniform(1.0, 1.12), rng.uniform(0.88, 1.0), rng.uniform(0.8, 1.0),
               rng.uniform(0.9, 0.99))
        fold_nets = []
        okall = True
        for fd in folds:
            m = run(P, fd, cfg, learn=True)
            if m is None or not m["gate"]:
                okall = False; break
            fold_nets.append(m["net"])
        if not okall:
            continue
        robust = min(fold_nets)          # worst-fold profit (robustness)
        if best is None or robust > best[0]:
            best = (robust, cfg)
    if best is None:
        print("no gate-safe robust config found"); return
    robust, cfg = best
    import json
    json.dump(list(cfg), open("isagi/v4_locked_cfg.json", "w"))
    print(f"searched {tr} | LOCKED robust config (worst-fold net ${robust:.0f})")
    print(f"  cfg saved. SL{cfg[0]:.2f} vclip[{cfg[1]:.2f},{cfg[2]:.2f}] TP{cfg[3]:.2f}/{cfg[4]:.2f}/{cfg[5]:.2f} "
          f"alloc{cfg[8]:.2f}/{cfg[9]:.2f} base${cfg[10]:.0f} shock{cfg[12]:.2f}")
    # full pre-train metric
    mp = run(P, pre, cfg, learn=True)
    print(f"  pre-train (full): ${mp['wk']:+.1f}/wk WR{mp['wr']:.0f}% DDt${mp['ddt']:.0f}")
    # OOS ONCE
    mo = run(P, oos, cfg, learn=True)
    print(f"\n  >>> OOS (locked, adapt-after-every-trade active): ${mo['wk']:+.1f}/wk "
          f"WR{mo['wr']:.0f}% DDt${mo['ddt']:.0f} DDd${mo['ddd']:.0f} n={mo['n']} "
          f"{'PASS' if mo['gate'] else 'FAIL-GATE'}")
    print(f"  (v2 OOS was -$1.5/wk; v3 OOS was -$5.5/wk)")


if __name__ == "__main__":
    main()
