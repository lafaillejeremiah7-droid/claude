"""
PHASE 18 — the backtest you actually asked for, EVERY time:
the ASWP BRAIN in the loop + the ideal phase-15 XAUUSD strategy.

For every candidate signal, chronologically:
  1. build the feature vector {rsi, align, hour}  (align=0: no FRED yields in
     backtest, so the brain effectively learns on rsi+hour+outcome)
  2. ASK THE BRAIN for P(reach 0.45R), P(reach 3.16R), P(reach 6.26R), P(fullstop)
  3. GATE on EV >= 0.55  (skip the trade if the brain says it's not worth it)
  4. if taken: simulate the real outcome (phase-15 geometry), size on the $5k
     account with the circuit-breaker, update equity/drawdown
  5. FEED the outcome back so the brain ADAPTS (recency + similarity weighted)
Brain warm-started with the same 100-memory prior as the live bot.

Head-to-head: ASWP-gated  vs  take-every-signal (no brain).  Validated on BOTH
2023-24 and 2025-26. Honest verdict on whether the brain earns its name.
"""
import random as _rnd
import numpy as np
import pandas as pd
import tsai_optimize as T
import fullyear_backtest as fb
import phase12_ideal as P12
from modules.xauusd_aswp_engine import ASWP, EngineConfig, TradeMemory

MAX_HOLD = T.MAX_HOLD
SPREAD = T.SPREAD
EQ0 = 5000.0
# phase-15 ideal geometry
G = dict(base_sl=1.1342, vol_lo=0.3460, vol_hi=1.2653, tp1_r=0.45, tp2_r=3.16,
         tp3_r=6.26, trend_gain=1.1363, tp_lo=0.849, tp_hi=2.509, a1=0.45, a2=0.16)
MIN_EV = 0.55


def outcomes(c):
    """Per-signal: sl_dist, realized R (45/16/39 with BE trailing), mfe_r (to
    close bar), full_stop. Uses the phase-15 geometry with trend-extended final."""
    n = c["n"]; is_buy = c["d"] == 1
    H, L, Cc = c["H"], c["L"], c["C"]; entries = c["entries"]
    vrc = np.clip(c["vol_ratio"], G["vol_lo"], G["vol_hi"])
    sld = G["base_sl"] * vrc * c["x_atr"] + SPREAD / 2
    ext = np.clip(1 + G["trend_gain"] * (c["trend_strength"] - 1), G["tp_lo"], G["tp_hi"])
    t1 = G["tp1_r"]; t2 = G["tp2_r"]; t3 = G["tp3_r"] * ext
    e_sl0 = np.where(is_buy, entries - sld, entries + sld)
    e_t1 = np.where(is_buy, entries + t1 * sld + SPREAD, entries - t1 * sld - SPREAD)
    e_t2 = np.where(is_buy, entries + t2 * sld + SPREAD, entries - t2 * sld - SPREAD)
    e_t3 = np.where(is_buy, entries + t3 * sld + SPREAD, entries - t3 * sld - SPREAD)
    tp1_hit = np.zeros(n, bool); tp2_hit = np.zeros(n, bool); closed = np.zeros(n, bool)
    code = np.zeros(n, np.int8); mfe = np.zeros(n)
    tp1_as_sl = e_t1.copy()
    for j in range(MAX_HOLD):
        if closed.all():
            break
        hi = H[:, j]; lo = L[:, j]
        fav = np.where(is_buy, hi - entries, entries - lo)
        mfe = np.where(~closed, np.maximum(mfe, fav), mfe)
        sl_lvl = e_sl0.copy()
        sl_lvl = np.where(tp1_hit & ~tp2_hit, entries, sl_lvl)
        sl_lvl = np.where(tp2_hit, tp1_as_sl, sl_lvl)
        hitsl = (~closed) & np.where(is_buy, lo <= sl_lvl, hi >= sl_lvl)
        code = np.where(hitsl, np.where(tp2_hit, 2, np.where(tp1_hit, 3, 1)), code)
        closed = closed | hitsl
        act = ~closed
        tp1_hit = tp1_hit | (act & ~tp1_hit & np.where(is_buy, hi >= e_t1, lo <= e_t1))
        tp2_hit = tp2_hit | (act & ~tp2_hit & np.where(is_buy, hi >= e_t2, lo <= e_t2))
        nf = act & np.where(is_buy, hi >= e_t3, lo <= e_t3)
        code = np.where(nf, 4, code); closed = closed | nf
    a1, a2 = G["a1"], G["a2"]; a3 = 1 - a1 - a2
    t3R = t3  # realized final R multiple (already ext-scaled)
    r = np.select([code == 4, code == 2, code == 3, code == 1],
                  [a1*t1 + a2*t2 + a3*t3R, a1*t1 + a2*t2 + a3*t1, a1*t1, -1.0],
                  default=(Cc[:, -1] - entries) * np.where(is_buy, 1.0, -1.0) / sld)
    mfe_r = mfe / sld
    full_stop = code == 1
    return sld, r, mfe_r, full_stop


def seed_brain(mfe_pool, fs_pool):
    """Seed from the REAL phase-15 outcome distribution (aggregate mfe/full-stop
    of this geometry) so the brain's probabilities are calibrated to the strategy
    it is actually gating. Uses aggregate base rates only (no per-trade look-ahead)."""
    cfg = EngineConfig(tp1_r=G["tp1_r"], tp2_r=G["tp2_r"], tp3_r=G["tp3_r"],
                       tp1_close=G["a1"], tp2_close=G["a2"], tp3_close=1-G["a1"]-G["a2"],
                       feature_keys=("rsi", "align", "hour"))
    b = ASWP(cfg)
    rng = _rnd.Random(42)
    idx = list(range(len(mfe_pool)))
    pick = [rng.choice(idx) for _ in range(120)]
    seed = [TradeMemory(features={"rsi": rng.uniform(30, 70), "align": 0.0,
                                  "hour": float(rng.randint(12, 20))},
                        mfe_r=float(mfe_pool[i]), full_stop=bool(fs_pool[i])) for i in pick]
    b.seed(seed)
    return b, cfg


def ev_of(b, cfg, feats):
    p1 = b.prob_reach(feats, cfg.tp1_r); p2 = b.prob_reach(feats, cfg.tp2_r)
    p3 = b.prob_reach(feats, cfg.tp3_r); pfs = b.prob_full_stop(feats)
    e_win = cfg.tp1_close*cfg.tp1_r*p1 + cfg.tp2_close*cfg.tp2_r*p2 + cfg.tp3_close*cfg.tp3_r*p3
    p_scratch = max(0.0, 1 - p1 - pfs)
    return e_win - (pfs*1.0 + p_scratch*0.05)


def run(S, min_ev, base=45.0, ds=120.0, ddt_thr=200.0, fac=0.35,
        hrs=range(12, 21), max_day=2, cd=3600):
    """min_ev = None -> take-all (no brain). Otherwise brain gates on EV>=min_ev."""
    c = S["c"]; o = S["order"]
    sld, r, mfe_r, fs = outcomes(c)
    sld, r, mfe_r, fs = sld[o], r[o], mfe_r[o], fs[o]
    rsi = c["rsi"][o]; hour = S["hour"]; wd = S["wd"]; day = S["day"]; dts = S["dt"]
    allow = np.isin(hour, list(hrs)); n = len(r)
    use_brain = min_ev is not None
    brain, cfg = seed_brain(mfe_r[allow], fs[allow]) if use_brain else (None, None)
    eq = EQ0; peak = EQ0; pnls = np.zeros(n)
    cur = -1; dp = 0.0; blk = False; dc = 0; taken = 0; gated = 0
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
        if use_brain:
            feats = {"rsi": float(rsi[k]), "align": 0.0, "hour": float(hour[k])}
            if ev_of(brain, cfg, feats) < min_ev:
                gated += 1
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
        if use_brain:
            brain.add(TradeMemory(features=feats, mfe_r=float(mfe_r[k]), full_stop=bool(fs[k])))
    eqc = EQ0 + np.cumsum(pnls)
    ddt = float(np.max(np.maximum.accumulate(eqc) - eqc))
    dd = np.bincount(day, weights=pnls, minlength=day.max()+1)
    ddd = float(-dd.min()) if dd.min() < 0 else 0.0
    nz = pnls[pnls != 0]; w = int((nz > 0).sum()); l = int((nz < 0).sum())
    return dict(wk=pnls.sum()/S["weeks"], wr=w/max(1, w+l)*100, ddt=ddt, ddd=ddd,
                gate=ddt < 350 and ddd < 200, sig_d=taken/(S["weeks"]*5),
                taken=taken, gated=gated)


def main():
    print("Loading real data (2023-24 + 2025-26)...")
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    print("\nASWP brain seeded from the REAL phase-15 outcome distribution, then")
    print("walk-forward learning. Sweeping the EV gate to see if the brain helps.")
    print("(min_ev = 'off' means take every signal — no brain.)\n")
    for name, S in [("2023-24", tr), ("2025-26", te)]:
        print(f"=== {name} ===")
        base = run(S, None)
        print(f"  {'off (take-all)':>16}: {base['wk']:+6.1f}/wk WR{base['wr']:.0f}% "
              f"DDt${base['ddt']:.0f} trades {base['taken']}")
        for mev in [-0.40, -0.30, -0.20, -0.10, -0.05, 0.0, 0.10]:
            x = run(S, mev)
            d = x['wk'] - base['wk']
            print(f"  {('EV>=%.2f' % mev):>16}: {x['wk']:+6.1f}/wk WR{x['wr']:.0f}% "
                  f"DDt${x['ddt']:.0f} trades {x['taken']} (skip {x['gated']}) "
                  f"| vs take-all {d:+.1f}/wk")
        print()


if __name__ == "__main__":
    main()
