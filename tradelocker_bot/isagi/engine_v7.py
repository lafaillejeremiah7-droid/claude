"""
ISAGI v7 — LEARN THE RIGHT WAY: distributional, not anecdotal.

Prior loops reacted to individual losses (fit noise -> hurt). This learns from the
DISTRIBUTION of outcomes per volatility regime, causally (past closed trades only,
no look-ahead):
  - SL placed at the recent adverse-excursion percentile in this regime (just past
    where trades actually got swept) instead of a fixed multiple or a per-loss nudge.
  - TP placed where price STATISTICALLY tends to reach (recent MFE distribution),
    capped at 3R, chosen to maximize expected R.
So geometry adapts to the measured behavior of the current regime, not to the last
mistake. Pre-train 2025 builds the regime distributions; held-out 2026 uses them
(warm) and keeps updating. Compared head-to-head vs the static config.
"""
import numpy as np, pandas as pd
import phase12_ideal as P12, tsai_optimize as T
from isagi.calendar_feed import backtest_blackouts, NewsDecoupler
import isagi.engine_v6 as V6

SPREAD = T.SPREAD


def resolve(fav, adv, sl_atr, tp1_R, tgt_R, a1):
    a2 = 1 - a1; tp1 = False; sl = -sl_atr; trail = None; maxf = 0.0; mae = 0.0
    for j in range(len(fav)):
        f = fav[j]; low_f = -adv[j]; maxf = max(maxf, f); mae = max(mae, adv[j])
        if low_f <= sl:
            if tp1:
                ex = max(sl, 0.0); return a1*tp1_R + a2*ex, maxf, mae
            return -1.0, maxf, mae
        if not tp1 and f >= tp1_R:
            tp1 = True; sl = 0.0; trail = 0.0
        if tp1 and f >= tgt_R:
            return a1*tp1_R + a2*tgt_R, maxf, mae
        if tp1:
            trail = max(trail, f - 1.0); sl = max(sl, trail)
    return (a1*tp1_R + a2*max(sl, 0.0)) if tp1 else fav[-1], maxf, mae


def bucket(vr):
    if vr < 0.8: return 0
    if vr < 1.15: return 1
    if vr < 1.6: return 2
    return 3


def walk(P, idxs, news, hist, learn_dist=True):
    """hist[bucket] = list of (mae_atr, mfe_atr) from PAST closed trades (causal)."""
    eq = 5000.0; peak = 5000.0; pnls = []; cur = -1; blk = False; dp = 0.0
    lt = pd.Timestamp("2000-01-01"); daypnl = {}; wins = []
    vr = P["vr"]; ts = P["ts"]; hour = P["hour"]; day = P["day"]; dt = P["dt"]; atr = P["atr"]
    for k in idxs:
        if day[k] != cur:
            cur = day[k]; blk = False; dp = 0.0
        if blk or (dt[k]-lt).total_seconds() < 3600 or news.is_blackout(dt[k]):
            continue
        b = bucket(vr[k]); H = hist[b]
        if learn_dist and len(H) >= 15:
            maes = np.array([x[0] for x in H[-40:]]); mfes = np.array([x[1] for x in H[-40:]])
            sl_atr = float(np.percentile(maes, 75)) + 0.1          # just past typical sweep
            sl_atr = min(2.5, max(0.6, sl_atr))
            # TP where price statistically reaches: median MFE in R, capped 3R
            tgt_R = float(np.median(mfes) / sl_atr); tgt_R = min(3.0, max(1.5, tgt_R))
        else:
            sl_atr = 1.2; tgt_R = 2.0                                # default until enough data
        tp1_R = 0.35 * tgt_R
        R, mfe, mae = resolve(P["fav"][k], P["adv"][k], sl_atr, tp1_R, tgt_R, 0.45)
        sld_price = sl_atr*atr[k] + SPREAD/2
        p = np.mean(wins[-20:]) if len(wins) >= 5 else 0.5
        fstar = max(0.0, (p*(tgt_R+1)-1)/tgt_R)
        risk = min(70.0, max(17.5, 70.0*min(1.0, fstar/0.3)))
        if (peak-eq) >= 200: risk = 17.5
        lpl = 100.0*sld_price; lots = max(0.01, min(0.12, np.floor(risk/lpl*100)/100))
        pnl = R*lots*lpl - 7.0*lots
        pnls.append(pnl); eq += pnl; peak = max(peak, eq); dp += pnl; lt = dt[k]
        daypnl[cur] = daypnl.get(cur, 0)+pnl; wins.append(1.0 if R > 0 else 0.0)
        if dp <= -120: blk = True
        hist[b].append((mae, mfe))    # record THIS trade's distribution (causal, after close)
    pnls = np.array(pnls)
    if len(pnls) < 5: return dict(wk=0, wr=0, ddt=0, n=0, gate=False), hist
    eqc = 5000+np.cumsum(pnls); ddt = float(np.max(np.maximum.accumulate(eqc)-eqc))
    ddd = float(-min(daypnl.values())) if daypnl and min(daypnl.values()) < 0 else 0.0
    w = int((pnls > 0).sum()); l = int((pnls < 0).sum())
    dts = P["dt"][idxs]; wks = max(1, (dts.max()-dts.min()).days/7.0)
    return dict(wk=pnls.sum()/wks, wr=w/max(1, w+l)*100, ddt=ddt, ddd=ddd,
                gate=ddt < 400 and ddd < 250, n=len(pnls)), hist


def main():
    print("ISAGI v7 — DISTRIBUTIONAL learning (learn the RIGHT way). Pre-train 2025 -> OOS 2026.\n")
    P = V6.prep()
    dt = P["dt"]; hour = P["hour"]; wd = P["wd"]
    news = NewsDecoupler(backtest_blackouts())
    session = np.isin(hour, [12, 13, 14, 15, 16]) & (~((wd == 4) & (hour >= 19)))
    order = np.argsort(dt.values.astype("datetime64[ns]").astype(np.int64)); order = order[session[order]]
    yr = dt[order].year; pre = order[yr == 2025]; bt = order[yr == 2026]
    print(f"pre-train 2025 {len(pre)} | OOS 2026 {len(bt)}\n")

    import copy
    hist = {0: [], 1: [], 2: [], 3: []}
    mp, hist = walk(P, pre, news, hist, learn_dist=True)
    print(f"PRE-TRAIN 2025 (distributions building): ${mp['wk']:+.1f}/wk WR{mp['wr']:.0f}% DDt${mp['ddt']:.0f}")
    warm = copy.deepcopy(hist)
    mo, _ = walk(P, bt, news, copy.deepcopy(warm), learn_dist=True)
    print(f"\n  >>> OOS 2026 (distributional, warm): ${mo['wk']:+.1f}/wk WR{mo['wr']:.0f}% "
          f"DDt${mo['ddt']:.0f} DDd${mo['ddd']:.0f} n={mo['n']} {'PASS' if mo['gate'] else 'FAIL'}")
    ms, _ = walk(P, bt, news, {0: [], 1: [], 2: [], 3: []}, learn_dist=False)
    print(f"      static (fixed geometry): ${ms['wk']:+.1f}/wk WR{ms['wr']:.0f}% DDt${ms['ddt']:.0f} {'PASS' if ms['gate'] else 'FAIL'}")
    print(f"\n  distributional-learning contribution: ${mo['wk']-ms['wk']:+.1f}/wk")


if __name__ == "__main__":
    main()
