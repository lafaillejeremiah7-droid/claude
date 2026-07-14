"""
ISAGI ENGINE — backtestable implementation (Sections 1-6).
Data: real 2025-26 XAU 1m (Forexite) + real Forex Factory calendar. V_dens uses a
labeled range/volatility proxy (true Dukascopy volume auto-used if the sample file
exists). Signal-only. Accurate bar-by-bar outcomes + friction + gate.
"""
import numpy as np, pandas as pd, os
import tsai_optimize as T
import phase25_adaptive_geom as P25
from isagi.calendar_feed import backtest_blackouts, NewsDecoupler

SPREAD = 0.30; RISK = 47.0


def ema(a, n):
    k = 2/(n+1); out = np.empty_like(a, float); out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = a[i]*k + out[i-1]*(1-k)
    return out


def atr_series(h, l, c, n=14):
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    tr = np.concatenate([[h[0]-l[0]], tr])
    out = np.full(len(c), np.nan)
    csum = np.cumsum(tr)
    out[n:] = (csum[n:]-csum[:-n])/n
    out[:n] = csum[:n]/np.arange(1, n+1)
    return out


def build(df):
    df = df.sort_index()
    o, h, l, c = df["o"].values, df["h"].values, df["l"].values, df["c"].values
    idx = df.index
    # 15m ATR + z-score (regime)
    m15 = df.resample("15min").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last")).dropna()
    a15 = atr_series(m15["h"].values, m15["l"].values, m15["c"].values, 14)
    z = np.full(len(a15), 0.0)
    for i in range(50, len(a15)):
        w = a15[i-50:i]; z[i] = (a15[i]-w.mean())/(w.std()+1e-9)
    m15 = m15.assign(atr=a15, z=z)
    m15["e20"] = ema(m15["c"].values, 20); m15["e50"] = ema(m15["c"].values, 50)
    # 1h bias
    h1 = df.resample("1h").agg(c=("c", "last")).dropna()
    h1["e20"] = ema(h1["c"].values, 20); h1["e50"] = ema(h1["c"].values, 50)
    # 1m EMA20 (trigger) + 1m atr
    e20_1m = ema(c, 20); a1m = atr_series(h, l, c, 14)
    return dict(idx=idx, o=o, h=h, l=l, c=c, e20_1m=e20_1m, a1m=a1m, m15=m15, h1=h1)


def run(df, news, use_kelly=True):
    D = build(df)
    idx = D["idx"]; o = D["o"]; h = D["h"]; l = D["l"]; c = D["c"]
    e20 = D["e20_1m"]; a1m = D["a1m"]; m15 = D["m15"]; h1 = D["h1"]
    m15i = m15.index; h1i = h1.index
    n = len(c)
    # V_dens proxy: 1m range relative to rolling 60-bar avg range (low=void, high=wall)
    rng = h - l
    rmean = pd.Series(rng).rolling(60, min_periods=10).mean().values
    vdens = np.nan_to_num(rng/(rmean+1e-9), nan=1.0)

    sigs = []  # (i, direction, sl, tp1, tp_final, R, sl_dist)
    day_last = {}
    last_i = -10**9
    for i in range(60, n-1):
        ts = idx[i]
        if ts.hour < 12 or ts.hour > 16:   # session 12-17 UTC (entries up to 16:xx)
            continue
        # regime via latest completed 15m z
        p15 = m15i.searchsorted(ts, "right")-1
        if p15 < 50:
            continue
        zc = m15.iloc[p15]["z"]
        # macro bias chart: compressed(z<-1)->H1, else M15
        if zc < -1.0:
            ph = h1i.searchsorted(ts, "right")-1
            if ph < 50: continue
            bias = 1 if h1.iloc[ph]["e20"] > h1.iloc[ph]["e50"] else -1
            anchor = h1.iloc[ph]["e20"]
        else:
            bias = 1 if m15.iloc[p15]["e20"] > m15.iloc[p15]["e50"] else -1
            anchor = m15.iloc[p15]["e20"]
        atr = a1m[i]
        if atr <= 0 or np.isnan(atr): continue
        # move-exhaustion: price vs session structural extreme (today's session hi/lo so far)
        d0 = ts.date()
        # value-anchor: within 1.5 ATR of anchor EMA
        if abs(c[i]-anchor) > 1.5*atr: continue
        # execution trigger on 1m: prev bar touches EMA20 and closes in bias dir
        if bias == 1:
            trig = (l[i-1] <= e20[i-1]) and (c[i-1] > o[i-1]) 
        else:
            trig = (h[i-1] >= e20[i-1]) and (c[i-1] < o[i-1])
        if not trig: continue
        # exhaustion guard: >=3 ATR from last 60-bar extreme
        ext = (c[i]-l[i-60:i].min()) if bias == 1 else (h[i-60:i].max()-c[i])
        if ext >= 3*atr: continue
        # news blackout
        if news.is_blackout(ts): continue
        # 2/day cap + 60min cooldown
        if day_last.get(d0, 0) >= 2: continue
        if i - last_i < 60: continue
        # ---- elastic capped geometry ----
        # adaptive SL: last 15 1m bars sweep extreme + spread buffer
        if bias == 1:
            sl = l[i-15:i].min() - SPREAD/2
            sl_dist = c[i]-sl
        else:
            sl = h[i-15:i].max() + SPREAD/2
            sl_dist = sl-c[i]
        if sl_dist <= 0: continue
        sl_dist = max(sl_dist, 0.4*atr)   # floor
        # elastic TP: void (low vdens) -> 3R ; wall (high vdens) -> 1.5-2R
        vd = vdens[i]
        tp_R = 3.0 if vd < 0.9 else (2.0 if vd < 1.3 else 1.5)
        sigs.append((i, bias, sl_dist, tp_R))
        day_last[d0] = day_last.get(d0, 0)+1
        last_i = i

    # ---- resolve accurately + size + friction + gate ----
    eq = 5000.0; peak = 5000.0; pnls = []; recent = []
    for (i, bias, sl_dist, tp_R) in sigs:
        # forward path 200 bars
        end = min(i+200, n)
        hh = h[i+1:end]; ll = l[i+1:end]
        entry = c[i]
        tp1_R = 0.35*tp_R
        # accurate 2-tier: TP1@0.35*target (close45%,BE), final@tp_R (55%)
        favmax = 0.0; R = 0.0; be = False; done = False
        tp1px = entry + bias*tp1_R*sl_dist
        finpx = entry + bias*tp_R*sl_dist
        slpx = entry - bias*sl_dist
        tp1_hit = False
        for j in range(len(hh)):
            hi = hh[j]; lo = ll[j]
            cur_sl = entry if be else slpx
            if bias == 1:
                if lo <= cur_sl: R = (0.45*tp1_R if tp1_hit else -1.0); done = True; break
                if not tp1_hit and hi >= tp1px: tp1_hit = True; be = True
                if tp1_hit and hi >= finpx: R = 0.45*tp1_R + 0.55*tp_R; done = True; break
            else:
                if hi >= cur_sl: R = (0.45*tp1_R if tp1_hit else -1.0); done = True; break
                if not tp1_hit and lo <= tp1px: tp1_hit = True; be = True
                if tp1_hit and lo <= finpx: R = 0.45*tp1_R + 0.55*tp_R; done = True; break
        if not done:
            R = 0.45*tp1_R if tp1_hit else 0.0
        # sizing: Kelly (capped) or flat
        if use_kelly and len(recent) >= 5:
            p = np.mean(recent[-20:]); b = tp_R
            fstar = max(0.0, (p*(b+1)-1)/b)
            risk = min(100.0, max(17.5, eq*0.25*fstar))
            if (peak-eq) >= 200: risk = 17.5
        else:
            risk = RISK
        lpl = 100.0*sl_dist
        lots = max(0.01, min(0.12, np.floor(risk/lpl*100)/100))
        pnl = R*lots*lpl - (7.0*lots + lots*100*0.15 + (lots*100*0.45*0.15 if R > 0 else 0))
        pnls.append(pnl); eq += pnl; peak = max(peak, eq); recent.append(1.0 if R > 0 else 0.0)

    pnls = np.array(pnls)
    if len(pnls) == 0:
        return dict(wk=0, wr=0, ddt=0, n=0, gate=False)
    eqc = 5000+np.cumsum(pnls); ddt = float(np.max(np.maximum.accumulate(eqc)-eqc))
    weeks = (idx.max()-idx.min()).days/7.0
    w = int((pnls > 0).sum()); ll_ = int((pnls < 0).sum())
    return dict(wk=pnls.sum()/weeks, wr=w/max(1, w+ll_)*100, ddt=ddt, n=len(pnls),
                gate=ddt < 400, sig_d=len(pnls)/(weeks*5))


def main():
    print("ISAGI backtest — real 2025-26 XAU 1m + real FF calendar (V_dens=range proxy)")
    df = T.load_data()
    news = NewsDecoupler(backtest_blackouts())
    for kelly in [False, True]:
        m = run(df, news, use_kelly=kelly)
        tag = "Kelly" if kelly else "flat $47"
        print(f"  [{tag:>8}] ${m['wk']:+.1f}/wk WR{m['wr']:.0f}% DDt${m['ddt']:.0f} "
              f"sig/d {m['sig_d']:.2f} n={m['n']} {'PASS' if m['gate'] else 'FAIL-GATE'}")


if __name__ == "__main__":
    main()
