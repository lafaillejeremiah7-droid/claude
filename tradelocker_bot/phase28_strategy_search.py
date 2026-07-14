"""
PHASE 28 — 500k-trial creative search to improve the STRATEGY (geometry + mgmt).

Same honest 3-way discipline as the brain search, but the target is gate-safe
$/wk (the strategy's REAL edge = management, not prediction).
  SEARCH signals (first 2/3 by time) -> decide improvements (compounding)
  CONFIRM signals (last 1/3)          -> untouched honesty check

Fast evaluator: precompute per-signal first-hit bars (favorable/adverse, ATR
grid) + breakeven-return, so each of 500k geometry configs is a vectorized
lookup (2-leg ladder: near TP1 -> breakeven -> runner; the dominant dynamics).
The final winner is re-checked with the full friction sim.
Reports: trials, # SEARCH improvements, # CONFIRMED improvements out of 500k.
"""
import time
import numpy as np
import pandas as pd
import phase12_ideal as P12, tsai_optimize as T

SPREAD = T.SPREAD; STEP = 0.05; GMAX = 8.0; NB = 200
RISK = 85.0; GATE = 400.0


def precompute(te):
    c = te["c"]; o = te["order"]
    is_buy = (c["d"] == 1)[o]
    H = c["H"][o][:, :NB]; L = c["L"][o][:, :NB]; C = c["C"][o][:, :NB]
    entry = c["entries"][o]; atr = c["x_atr"][o]
    vr = c["vol_ratio"][o]; ts = c["trend_strength"][o]
    fav = np.where(is_buy[:, None], H - entry[:, None], entry[:, None] - L) / atr[:, None]
    adv = np.where(is_buy[:, None], entry[:, None] - L, H - entry[:, None]) / atr[:, None]
    cmf = np.maximum.accumulate(np.nan_to_num(fav, nan=-9), axis=1)
    cma = np.maximum.accumulate(np.nan_to_num(adv, nan=-9), axis=1)
    n = len(entry); G = int(GMAX / STEP) + 1; levels = np.arange(G) * STEP
    ff = np.full((n, G), NB, np.int32); fa = np.full((n, G), NB, np.int32)
    for i in range(n):
        ff[i] = np.searchsorted(cmf[i], levels, side="left")
        fa[i] = np.searchsorted(cma[i], levels, side="left")
    ff = np.minimum(ff, NB); fa = np.minimum(fa, NB)
    # breakeven return: first bar j>=b where price back to entry (adverse-from-entry>=0)
    back = np.where(is_buy[:, None], L <= entry[:, None], H >= entry[:, None])
    be_ret = np.full((n, NB + 1), NB, np.int32)
    for i in range(n):
        nxt = NB
        for b in range(NB - 1, -1, -1):
            if back[i, b]:
                nxt = b
            be_ret[i, b] = nxt
    dt = pd.to_datetime(c["dt"][o]); hour = te["hour"]; day = te["day"]
    return dict(ff=ff, fa=fa, be=be_ret, vr=vr, ts=ts, atr=atr, entry=entry,
                dt=dt, hour=hour, day=day, n=n, G=G)


def li(x):  # level -> grid index
    return np.clip(np.round(x / STEP).astype(np.int32), 0, int(GMAX / STEP))


def fast_R(P, base_sl, vlo, vhi, t1r, t3r, a1, tgain, tp_hi, tp_lo=0.5):
    n = P["n"]; ar = np.arange(n)
    sl_atr = base_sl * np.clip(P["vr"], vlo, vhi)
    ext = np.clip(1 + tgain * (P["ts"] - 1), tp_lo, tp_hi)
    t3 = t3r * ext
    t_sl = P["fa"][ar, li(sl_atr)]
    t_1 = P["ff"][ar, li(np.full(n, t1r))]
    t_3 = P["ff"][ar, li(t3)]
    r = np.zeros(n)
    stop_first = t_sl <= t_1
    r[stop_first] = -1.0
    reached1 = ~stop_first & (t_1 < NB)
    idx = np.where(reached1)[0]
    b1 = np.clip(t_1[idx], 0, NB)
    be_stop = P["be"][idx, b1]
    got3 = t_3[idx] < be_stop
    rr = np.where(got3, a1 * t1r + (1 - a1) * t3[idx], a1 * t1r)
    r[idx] = rr
    # neither stop nor TP1 within window -> timeout ~ 0 (flat)
    return r, sl_atr


def score(r, sld_atr, mask, weeks):
    # pnl ~ r * RISK (lots*100*sld ~= RISK, ignoring lot cap); friction ~ flat small
    pnl = r[mask] * RISK - 0.9  # ~$0.9 avg friction/trade
    if len(pnl) == 0:
        return -1e9, 1e9
    eq = np.cumsum(pnl)
    dd = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
    return float(pnl.sum() / weeks), dd


def main():
    print("Precomputing fast first-hit tables (one-time)...")
    te = P12.prep(T.load_data())
    P = precompute(te)
    # session + first-2-per-day mask (config-independent)
    insess = np.isin(P["hour"], list(range(12, 21)))
    day = P["day"]; take = np.zeros(P["n"], bool); cnt = {}
    order = np.argsort(P["dt"].values.astype("datetime64[ns]").astype(np.int64))
    for k in order:
        if not insess[k]:
            continue
        d = day[k]; cnt[d] = cnt.get(d, 0)
        if cnt[d] < 2:
            take[k] = True; cnt[d] += 1
    idx_sorted = order[take[order]]
    n_take = len(idx_sorted)
    cut = int(n_take * 2 / 3)
    search_idx = idx_sorted[:cut]; confirm_idx = idx_sorted[cut:]
    dtv = P["dt"]
    wk_s = (dtv[search_idx].max() - dtv[search_idx].min()).days / 7.0
    wk_c = (dtv[confirm_idx].max() - dtv[confirm_idx].min()).days / 7.0
    smask = np.zeros(P["n"], bool); smask[search_idx] = True
    cmask = np.zeros(P["n"], bool); cmask[confirm_idx] = True
    print(f"tradeable signals: {n_take} | SEARCH {len(search_idx)} CONFIRM {len(confirm_idx)}\n")

    # baseline = shipped phase-15 geometry
    def evalcfg(cfg):
        r, sld = fast_R(P, *cfg)
        ws, dds = score(r, sld, smask, wk_s)
        return ws, dds, r, sld
    base_cfg = (1.1342, 0.346, 1.2653, 0.45, 6.26, 0.45, 1.1363, 2.509)
    b_ws, b_dds, br, bsld = evalcfg(base_cfg)
    b_wc, b_ddc = score(br, bsld, cmask, wk_c)
    print(f"BASELINE (phase-15): SEARCH ${b_ws:+.1f}/wk (DD${b_dds:.0f}) | CONFIRM ${b_wc:+.1f}/wk (DD${b_ddc:.0f})\n")

    rng = np.random.default_rng(28)
    best_ws = b_ws if b_dds < GATE else -1e9
    best_cfg = base_cfg
    improvements = 0; confirmed = 0; TARGET = 500_000; t0 = time.time(); trial = 0
    trace = []
    while trial < TARGET:
        trial += 1
        cfg = (
            rng.uniform(0.6, 1.6),      # base_sl
            rng.uniform(0.2, 0.9),      # vlo
            rng.uniform(1.0, 1.6),      # vhi
            rng.uniform(0.3, 1.5),      # t1r
            rng.uniform(2.0, 8.0),      # t3r
            rng.uniform(0.25, 0.75),    # a1
            rng.uniform(0.5, 3.0),      # tgain
            rng.uniform(1.2, 3.0),      # tp_hi
        )
        r, sld = fast_R(P, *cfg)
        ws, dds = score(r, sld, smask, wk_s)
        if dds < GATE and ws > best_ws + 0.05:
            # confirm on untouched set
            wc, ddc = score(r, sld, cmask, wk_c)
            best_ws = ws; best_cfg = cfg; improvements += 1
            if wc > b_wc and ddc < GATE:
                confirmed += 1
            trace.append((improvements, ws, wc, confirmed))
        if trial % 100000 == 0:
            print(f"  trial {trial:>7}/{TARGET} | SEARCH-improvements {improvements} | "
                  f"confirmed {confirmed} | best SEARCH ${best_ws:.1f}/wk | {time.time()-t0:.0f}s")
        if time.time() - t0 > 900:
            print(f"  [wall cap at trial {trial}]"); break

    r, sld = fast_R(P, *best_cfg)
    fin_wc, fin_ddc = score(r, sld, cmask, wk_c)

    # ---- REALITY CHECK: validate winner + baseline in the FULL friction sim ----
    import phase25_adaptive_geom as P25
    wd = P["dt"].weekday.values
    def real_validate(cfg, mask, tag):
        rr, sl_atr = fast_R(P, *cfg)
        sldp = sl_atr * P["atr"] + SPREAD / 2
        a1 = cfg[5]
        res = P25.fric_sim(P["dt"], P["day"], P["hour"], wd, rr, sldp, mask,
                           base=RISK, tp1hit=(rr > 0), a1=a1)
        print(f"    {tag}: ${res['wk']:+.1f}/wk WR{res['wr']:.0f}% "
              f"DDt${res['ddt']:.0f} DDd${res['ddd']:.0f} {'PASS' if res['gate'] else 'FAIL-GATE'}")
        return res
    print("\n  >>> REALITY CHECK (full friction sim: lot-cap, circuit-breaker, cooldown):")
    print("  CONFIRM period:")
    rb = real_validate(base_cfg, cmask, "baseline (phase-15)")
    rw = real_validate(best_cfg, cmask, "search winner      ")
    real_ok = rw["gate"] and rw["wk"] > rb["wk"]
    print(f"  -> winner beats baseline in REAL sim & gate-safe? {real_ok}")

    print("\n" + "=" * 68)
    print(f"STRATEGY SEARCH DONE: {trial} trials")
    print(f"  SEARCH improvements (beat running best gate-safe): {improvements}")
    print(f"  CONFIRMED improvements (also beat baseline on untouched CONFIRM): {confirmed}")
    print(f"  best SEARCH ${best_ws:+.1f}/wk -> its CONFIRM ${fin_wc:+.1f}/wk "
          f"(baseline CONFIRM ${b_wc:+.1f}/wk)")
    print("=" * 68)
    if trace:
        print("  overfit check (SEARCH vs CONFIRM as improvements accrue):")
        print(f"  {'#impr':>6} {'SEARCH$/wk':>10} {'CONFIRM$/wk':>11}")
        for imp, ws, wc, cf in trace[::max(1, len(trace)//8)]:
            print(f"  {imp:>6} {ws:>10.1f} {wc:>11.1f}")
    print()
    print(f"  best_cfg = {tuple(round(x,4) for x in best_cfg)}")
    print(f"\nHONEST COUNT (fast proxy): brain 0/300k confirmed | strategy {confirmed}/{trial} confirmed.")
    print(f"REAL-SIM VERDICT: winner {'IS' if real_ok else 'is NOT'} a genuine, gate-safe")
    print(f"improvement over baseline once lot-cap + circuit-breaker + friction are enforced.")


if __name__ == "__main__":
    main()
