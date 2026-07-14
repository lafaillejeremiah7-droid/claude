"""
Validate the v4 locked config with an ACCURATE bar-by-bar resolver (proper
SL->BE->TP1 trail after TP2, real timeout), same causal adaptation. If v4's
+$103/wk OOS survives here, it's real. If it collapses, the fast cummax resolver
over-credited the runner (approximation artifact).
"""
import json
import numpy as np, pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import isagi.engine_v4 as V4
from isagi.calendar_feed import backtest_blackouts, NewsDecoupler

SPREAD = T.SPREAD; NB = 200


def resolve_accurate(fav, adv, sld_atr, t1, t2, t3, a1, a2):
    """True bar-by-bar: SL@-sld; after TP1 SL->BE; after TP2 SL->TP1 level.
    fav/adv are per-bar excursions in ATR units."""
    a3 = 1 - a1 - a2
    tp1 = tp2 = False; sl_level = -sld_atr  # in fav-space (fav=price-entry in ATR for buy)
    sweep = False; maxfav = 0.0; stop_bar = -1
    for j in range(len(fav)):
        f = fav[j]; adr = adv[j]
        maxfav = max(maxfav, f)
        # SL check: price fell to sl_level (in fav-space, fav <= sl_level)
        # fav can go negative; adverse = -fav roughly. Use fav min via -adv.
        low_fav = -adr   # most adverse fav this bar (price low)
        if low_fav <= sl_level:
            if tp2:
                return a1*t1 + a2*t2 + a3*t1, 2, False, maxfav/sld_atr
            if tp1:
                return a1*t1, 3, False, maxfav/sld_atr
            # full stop: sweep if fav later reached t3
            later = fav[j:]
            sweep = (later >= t3*sld_atr).any()
            return -1.0, 1, sweep, maxfav/sld_atr
        if not tp1 and f >= t1*sld_atr:
            tp1 = True; sl_level = 0.0            # BE
        if tp1 and not tp2 and f >= t2*sld_atr:
            tp2 = True; sl_level = t1*sld_atr     # trail to TP1 level
        if tp2 and f >= t3*sld_atr:
            return a1*t1 + a2*t2 + a3*t3, 4, False, maxfav/sld_atr
    # timeout: mark open portion at last fav
    return a1*t1 if tp1 else fav[-1]/sld_atr, 0, False, maxfav/sld_atr


def run_accurate(P, fav, adv, idxs, cfg, learn=True):
    (bsl, vlo, vhi, t1, t2, t3, tg, tphi, a1, a2, base, egok, shock,
     swg, wcut, ecut, dec) = cfg
    sl_adj = tp_adj = sz_adj = 1.0
    eq = 5000.0; peak = 5000.0; pnls = []
    cur = -1; dc = 0; blk = False; dp = 0.0; lt = pd.Timestamp("2000-01-01")
    day = P["day"]; dt = P["dt"]; vr = P["vr"]; ts = P["ts"]; hour = P["hour"]; atr = P["atr"]
    daypnl = {}
    for k in idxs:
        if day[k] != cur:
            cur = day[k]; dc = 0; blk = False; dp = 0.0
        if blk or dc >= 2 or (dt[k]-lt).total_seconds() < 3600 or vr[k] >= shock:
            continue
        vrc = min(vhi, max(vlo, vr[k])); sld_atr = bsl*vrc*(sl_adj if learn else 1)
        t3e = t3*np.clip(1+tg*(ts[k]-1), 0.5, tphi)*(tp_adj if learn else 1)
        t2e = t2*(tp_adj if learn else 1)
        R, code, sweep, maxfav = resolve_accurate(fav[k, :NB], adv[k, :NB], sld_atr, t1, t2e, t3e, a1, a2)
        sld_price = sld_atr*atr[k] + SPREAD/2
        risk = base*np.clip(1-egok*(vr[k]-1), 0.4, 1.6)*(sz_adj if learn else 1)
        risk = risk*(0.35 if (peak-eq) >= 200 else 1); risk = min(100, max(15, risk))
        lpl = 100*sld_price; lots = max(0.01, min(0.12, np.floor(risk/lpl*100)/100))
        pnl = R*lots*lpl - (7*lots + lots*100*0.15 + (lots*100*a1*0.15 if R > 0 else 0))
        pnls.append(pnl); eq += pnl; peak = max(peak, eq); dp += pnl; dc += 1; lt = dt[k]
        daypnl[cur] = daypnl.get(cur, 0)+pnl
        if dp <= -120: blk = True
        if learn:
            if code == 1:
                if sweep: sl_adj = min(1.8, sl_adj*swg)
                elif maxfav >= 0.7*t2: tp_adj = max(0.6, tp_adj*wcut)
                elif hour[k] >= 16 or ts[k] < 0.8: sz_adj = max(0.5, sz_adj*ecut)
            elif R > 0:
                sl_adj = 1+(sl_adj-1)*dec; tp_adj = 1+(tp_adj-1)*dec; sz_adj = 1+(sz_adj-1)*dec
    pnls = np.array(pnls)
    if len(pnls) < 10: return None
    eqc = 5000+np.cumsum(pnls); ddt = float(np.max(np.maximum.accumulate(eqc)-eqc))
    ddd = float(-min(daypnl.values())) if daypnl and min(daypnl.values()) < 0 else 0.0
    w = int((pnls > 0).sum()); l = int((pnls < 0).sum())
    dts = P["dt"][idxs]; wks = max(1, (dts.max()-dts.min()).days/7.0)
    return dict(wk=pnls.sum()/wks, wr=w/max(1, w+l)*100, ddt=ddt, ddd=ddd,
                gate=ddt < 400 and ddd < 250, n=len(pnls))


def main():
    cfg = tuple(json.load(open("isagi/v4_locked_cfg.json")))
    te = P12.prep(T.load_data()); P = V4.precompute(te)
    # raw per-bar fav/adv for accurate resolver
    c = te["c"]; o = te["order"]; is_buy = (c["d"][o] == 1)
    H = c["H"][o][:, :NB]; L = c["L"][o][:, :NB]; entry = c["entries"][o]; atr = c["atr5"][o] if False else c["x_atr"][o]
    fav = np.nan_to_num(np.where(is_buy[:, None], H-entry[:, None], entry[:, None]-L)/atr[:, None], nan=-9)
    adv = np.nan_to_num(np.where(is_buy[:, None], entry[:, None]-L, H-entry[:, None])/atr[:, None], nan=-9)
    dt = P["dt"]; hour = P["hour"]; wd = dt.weekday.values
    news = NewsDecoupler(backtest_blackouts()); bl = np.array([news.is_blackout(t) for t in dt])
    session = np.isin(hour, [12, 13, 14, 15, 16]) & (~((wd == 4) & (hour >= 19))) & (~bl)
    order = np.argsort(dt.values.astype("datetime64[ns]").astype(np.int64)); order = order[session[order]]
    cut = int(len(order)*0.6); pre, oos = order[:cut], order[cut:]
    print("ACCURATE re-validation of v4 locked config (proper SL->BE->TP1 trail):")
    mp = run_accurate(P, fav, adv, pre, cfg); mo = run_accurate(P, fav, adv, oos, cfg)
    print(f"  pre-train: ${mp['wk']:+.1f}/wk WR{mp['wr']:.0f}% DDt${mp['ddt']:.0f} {'PASS' if mp['gate'] else 'FAIL'}")
    print(f"  OOS      : ${mo['wk']:+.1f}/wk WR{mo['wr']:.0f}% DDt${mo['ddt']:.0f} DDd${mo['ddd']:.0f} n={mo['n']} {'PASS' if mo['gate'] else 'FAIL'}")
    print(f"\n  v4 fast-resolver claimed OOS +$103.2/wk. Accurate says: ${mo['wk']:+.1f}/wk")
    print("  If accurate << fast -> the runner trail was over-credited (artifact).")


if __name__ == "__main__":
    main()
