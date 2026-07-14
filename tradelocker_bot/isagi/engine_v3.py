"""
ISAGI v3 — CAUSAL learning loop active in BOTH pre-train AND the real backtest.

Legitimacy rule (no look-ahead): the loop adjusts trade[i] using ONLY trades that
CLOSED before trade[i] opened (trades are sequential: max-1-position + cooldown).
It never re-tunes against the whole backtest. This is exactly how a live adaptive
bot behaves.

Learning (Section 5): after each closed LOSS, classify the root cause from the
trade's own path, then nudge SL/TP for FUTURE trades:
  - VOLATILITY SWEEP  (stopped, then price reached the target) -> widen SL a bit.
  - VOLUME WALL       (ran most of the way, reversed before target) -> trim TP.
  - TREND EXHAUSTION  (late-session/low-momentum fade) -> shrink size next similar.
Adjustments decay back toward neutral on wins (regime-adaptive, not permanent).

Compares, on the SAME OOS slice: static locked config vs the causal learning loop.
"""
import numpy as np, pandas as pd
import phase12_ideal as P12, tsai_optimize as T
from isagi.calendar_feed import backtest_blackouts, NewsDecoupler

SPREAD = T.SPREAD; MAXH = 200
# locked base geometry (from v2 pre-train)
BASE_SL, VLO, VHI = 1.38, 0.62, 1.18
T1R, T2R, T3R = 0.41, 2.86, 6.23
TG, TP_LO, TP_HI = 2.38, 0.5, 2.6
A1, A2 = 0.22, 0.16; A3 = 1 - A1 - A2
BASE_RISK = 61.0


def resolve(is_buy, entry, H, L, C, sld, t1r, t2r, t3r):
    """Accurate per-trade resolution with adjustable SL/TP. Returns (R, code, diag).
    code: 1 full-stop, 3 tp1-only, 2 tp2, 4 final, 0 timeout.
    diag: (reached_final_after_stop, max_fav_R_before_stop)."""
    e_sl = entry - sld if is_buy else entry + sld
    e_t1 = entry + t1r*sld if is_buy else entry - t1r*sld
    e_t2 = entry + t2r*sld if is_buy else entry - t2r*sld
    e_t3 = entry + t3r*sld if is_buy else entry - t3r*sld
    tp1 = tp2 = False; be = False; stop_bar = -1
    maxfav = 0.0
    for j in range(len(H)):
        hi = H[j]; lo = L[j]
        fav = (hi-entry) if is_buy else (entry-lo)
        maxfav = max(maxfav, fav/sld)
        sl_lvl = entry if be else e_sl
        if (is_buy and lo <= sl_lvl) or ((not is_buy) and hi >= sl_lvl):
            if tp2:
                R = A1*t1r + A2*t2r + A3*t1r; code = 2
            elif tp1:
                R = A1*t1r; code = 3
            else:
                R = -1.0; code = 1; stop_bar = j
            # post-stop: did price later reach final target? (sweep detection)
            reached_final = False
            if code == 1:
                seg_h = H[j:]; seg_l = L[j:]
                reached_final = (seg_h >= e_t3).any() if is_buy else (seg_l <= e_t3).any()
            return R, code, (reached_final, maxfav)
        if not tp1 and ((is_buy and hi >= e_t1) or ((not is_buy) and lo <= e_t1)):
            tp1 = True; be = True
        if tp1 and not tp2 and ((is_buy and hi >= e_t2) or ((not is_buy) and lo <= e_t2)):
            tp2 = True
        if tp2 and ((is_buy and hi >= e_t3) or ((not is_buy) and lo <= e_t3)):
            R = A1*t1r + A2*t2r + A3*t3r; return R, 4, (False, maxfav)
    # timeout
    last = C[-1]; R = (last-entry)/sld*(1 if is_buy else -1)
    return (A1*t1r + (R-A1*t1r) if tp1 else R), 0, (False, maxfav)


def backtest(S, idxs, learn):
    """Sequential causal backtest over signal indices idxs (already time-sorted).
    learn=True activates the loop. Returns metrics + carries learning state out."""
    c = S["c"]; o = S["order"]
    H = c["H"][o]; L = c["L"][o]; C = c["C"][o]
    entry = c["entries"][o]; atr = c["x_atr"][o]; vr = c["vol_ratio"][o]
    ts = c["trend_strength"][o]; hour = S["hour"]; is_buy = (c["d"][o] == 1)
    day = S["day"]; dt = S["dt"]
    # learning state (regime-adaptive multipliers, decay to 1.0)
    sl_adj = 1.0; tp_adj = 1.0; size_adj = 1.0
    DECAY = 0.94
    eq = 5000.0; peak = 5000.0; pnls = []; cls = {"sweep": 0, "wall": 0, "exh": 0}
    lt = pd.Timestamp("2000-01-01"); cur = -1; dc = 0; blk = False; dp = 0.0
    day_pnl = {}
    for k in idxs:
        if day[k] != cur:
            cur = day[k]; dc = 0; blk = False; dp = 0.0
        if blk or dc >= 2:
            continue
        if (dt[k]-lt).total_seconds() < 3600:
            continue
        a = atr[k]
        if a <= 0:
            continue
        vrc = min(VHI, max(VLO, vr[k]))
        sld = BASE_SL*vrc*a + SPREAD/2
        if learn:
            sld *= sl_adj
        t3 = T3R*np.clip(1+TG*(ts[k]-1), TP_LO, TP_HI)
        t2 = T2R*(tp_adj if learn else 1.0)
        t3 = t3*(tp_adj if learn else 1.0)
        R, code, (reached_final, maxfav) = resolve(is_buy[k], entry[k],
                                                   H[k, :MAXH], L[k, :MAXH], C[k, :MAXH],
                                                   sld, T1R, t2, t3)
        # EGOLESS sizing (state) * learned size_adj
        risk = BASE_RISK*np.clip(1.0-0.03*(vr[k]-1.0), 0.4, 1.6)*(size_adj if learn else 1.0)
        risk = risk*(0.35 if (peak-eq) >= 200 else 1.0)
        risk = min(100.0, max(15.0, risk))
        lpl = 100.0*sld
        lots = max(0.01, min(0.12, np.floor(risk/lpl*100)/100))
        pnl = R*lots*lpl - (7.0*lots + lots*100*0.15 + (lots*100*A1*0.15 if R > 0 else 0))
        pnls.append(pnl); eq += pnl; peak = max(peak, eq); dp += pnl; dc += 1; lt = dt[k]
        day_pnl[cur] = day_pnl.get(cur, 0)+pnl
        if dp <= -120: blk = True
        # ---- CAUSAL LEARNING: update AFTER this trade closes ----
        if learn:
            if code == 1:      # full-stop loss -> classify + correct future
                if reached_final:                 # SWEEP: SL too tight
                    sl_adj = min(1.8, sl_adj*1.06); cls["sweep"] += 1
                elif maxfav >= 0.7*T2R:            # WALL: ran most way, reversed
                    tp_adj = max(0.6, tp_adj*0.94); cls["wall"] += 1
                elif hour[k] >= 16 or ts[k] < 0.8: # EXHAUSTION: late/low-momentum
                    size_adj = max(0.5, size_adj*0.9); cls["exh"] += 1
            elif R > 0:        # win -> decay adjustments back toward neutral
                sl_adj = 1.0 + (sl_adj-1.0)*DECAY
                tp_adj = 1.0 + (tp_adj-1.0)*DECAY
                size_adj = 1.0 + (size_adj-1.0)*DECAY
    pnls = np.array(pnls)
    if len(pnls) == 0:
        return dict(wk=0, wr=0, ddt=0, n=0, gate=False, cls=cls)
    eqc = 5000+np.cumsum(pnls); ddt = float(np.max(np.maximum.accumulate(eqc)-eqc))
    ddd = float(-min(day_pnl.values())) if day_pnl and min(day_pnl.values()) < 0 else 0.0
    w = int((pnls > 0).sum()); l = int((pnls < 0).sum())
    dts = S["dt"][idxs]; weeks = (dts.max()-dts.min()).days/7.0
    return dict(wk=pnls.sum()/weeks, wr=w/max(1, w+l)*100, ddt=ddt, ddd=ddd,
                gate=ddt < 400 and ddd < 250, n=len(pnls), cls=cls)


def main():
    print("ISAGI v3 — CAUSAL learning loop in pre-train AND real backtest (no look-ahead)\n")
    te = P12.prep(T.load_data())
    dt = pd.to_datetime(te["c"]["dt"][te["order"]])
    hour = te["hour"]; wd = dt.weekday.values
    news = NewsDecoupler(backtest_blackouts())
    blackout = np.array([news.is_blackout(t) for t in dt])
    session = np.isin(hour, [12, 13, 14, 15, 16]) & (~((wd == 4) & (hour >= 19))) & (~blackout)
    S = dict(c=te["c"], order=te["order"], dt=dt, hour=hour, day=te["day"])
    order = np.argsort(dt.values.astype("datetime64[ns]").astype(np.int64))
    order = order[session[order]]
    cut = int(len(order)*0.6)
    pre, oos = order[:cut], order[cut:]
    print(f"pre-train {len(pre)} | OOS {len(oos)}\n")

    print("PRE-TRAIN:")
    for lr in [False, True]:
        m = backtest(S, pre, learn=lr)
        print(f"  {'learning' if lr else 'static  '}: ${m['wk']:+.1f}/wk WR{m['wr']:.0f}% "
              f"DDt${m['ddt']:.0f} n={m['n']} {'PASS' if m['gate'] else 'FAIL'} cls={m['cls']}")
    print("\nREAL BACKTEST (OOS):")
    for lr in [False, True]:
        m = backtest(S, oos, learn=lr)
        print(f"  {'learning' if lr else 'static  '}: ${m['wk']:+.1f}/wk WR{m['wr']:.0f}% "
              f"DDt${m['ddt']:.0f} DDd${m['ddd']:.0f} n={m['n']} {'PASS' if m['gate'] else 'FAIL'} cls={m['cls']}")


if __name__ == "__main__":
    main()
