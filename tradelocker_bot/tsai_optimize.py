"""
TSAI-scored adaptive SL/TP1/TP2/Final optimizer on REAL Jan 2025 -> Jul 2026
1-minute XAUUSD data.

Objective: maximize total profit (=> weekly profit) SUBJECT TO the three
structural security gates passing (account-safe). Reports the user's TSAI
(Gated Harmonic Mean) score for old vs new config.

Parameter space (9 dims) — makes SL, TP1, TP2 and Final all adaptive:
  base_sl, vol_lo, vol_hi         -> SL width = base_sl*clip(vol_ratio,lo,hi)*ATR15
  tp1_r, tp2_r                    -> TP1/TP2 as multiples of the (adaptive) SL dist
  tp3_r, trend_gain, tp_lo, tp_hi -> Final = tp3_r*sl_dist*clip(1+gain*(trend-1),lo,hi)
Allocation stays 10% / 20% / 70% (TP1 / TP2 / Final).
"""
import sys
import json
import time
import numpy as np
import pandas as pd

import fullyear_backtest as fb
import adaptive_backtest as ab

SPREAD = 0.30
COMMISSION = 0.0            # TradeLocker/AquaFunded XAUUSD is spread-only
MAX_HOLD = 200
COOLDOWN_MIN = 180
RISK_CAP = 60.0
CONTRACT = 100.0
LEVERAGE = 10.0
EQUITY0 = 5000.0
DATA_PATH = "data/xau_2025_2026.parquet"

# TSAI gate thresholds (from the user's framework)
G_ATR_MULT = 1.2           # g1: SL > 1.2 * ATR(5m)
G_DD_TOTAL = 350.0         # g2: total max DD < $350
G_DD_DAY = 200.0           # g3: worst-day DD < $200
K_SCALE = 5.0              # C scaling constant


def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df.set_index("dt").sort_index()
    return df[["o", "h", "l", "c"]]


def build_candidates(df1m):
    m, e = ab.compute_features(df1m)
    cand = m[m["signal"]].copy()
    e_h = e["h"].values; e_l = e["l"].values; e_c = e["c"].values
    e_idx = {ts: i for i, ts in enumerate(e.index)}
    cooldown = max(1, COOLDOWN_MIN // 1)
    last_i = -10**9
    rows = []
    for _, r in cand.iterrows():
        i = e_idx.get(r["dt"])
        if i is None or i < 1:
            continue
        if i - last_i < cooldown:
            continue
        rows.append((r["dt"], i, int(r["dir"]), e_c[i], r["x_atr"],
                     r["vol_ratio"], r["trend_strength"], r["atr5"],
                     r.get("rsi", 50.0), r.get("pb_dist", 0.0)))
        last_i = i
    n = len(rows)
    dts = np.array([r[0] for r in rows])
    idxs = np.array([r[1] for r in rows])
    d = np.array([r[2] for r in rows], dtype=np.int8)
    entries = np.array([r[3] for r in rows])
    x_atr = np.array([r[4] for r in rows])
    vol_ratio = np.array([r[5] for r in rows])
    trend_strength = np.array([r[6] for r in rows])
    atr5 = np.array([r[7] for r in rows])
    rsi = np.array([r[8] for r in rows])
    pb_dist = np.array([r[9] for r in rows])
    nb = len(e_c)
    H = np.empty((n, MAX_HOLD)); L = np.empty((n, MAX_HOLD)); C = np.empty((n, MAX_HOLD))
    for k, i in enumerate(idxs):
        end = min(i + 1 + MAX_HOLD, nb)
        sh, sl, sc = e_h[i+1:end], e_l[i+1:end], e_c[i+1:end]
        got = len(sh)
        H[k, :got] = sh; L[k, :got] = sl; C[k, :got] = sc
        if got < MAX_HOLD:
            H[k, got:] = sh[-1] if got else entries[k]
            L[k, got:] = sl[-1] if got else entries[k]
            C[k, got:] = sc[-1] if got else entries[k]
    hours = np.array([pd.Timestamp(t).hour for t in dts])

    # Precompute chronological order + integer day-ids (for fast worst-day DD)
    order = np.argsort(dts)
    dts_sorted = dts[order]
    day_codes = pd.factorize(pd.to_datetime(dts_sorted).date)[0]
    return {"dt": dts, "d": d, "entries": entries, "x_atr": x_atr,
            "vol_ratio": vol_ratio, "trend_strength": trend_strength, "atr5": atr5,
            "rsi": rsi, "pb_dist": pb_dist, "hours": hours,
            "H": H, "L": L, "C": C, "n": n,
            "order": order, "day_codes": day_codes.astype(np.int64),
            "n_days": int(day_codes.max()) + 1 if len(day_codes) else 0}


def simulate(c, base_sl, vol_lo, vol_hi, tp1_r, tp2_r, tp3_r, trend_gain, tp_lo, tp_hi):
    d = c["d"]; entries = c["entries"]; x_atr = c["x_atr"]
    H, L, Cc = c["H"], c["L"], c["C"]; n = c["n"]
    is_buy = d == 1

    vr = np.clip(c["vol_ratio"], vol_lo, vol_hi)
    tp_ext = np.clip(1 + trend_gain * (c["trend_strength"] - 1), tp_lo, tp_hi)
    sl_dist = base_sl * vr * x_atr + SPREAD / 2
    tp1 = tp1_r * sl_dist + SPREAD
    tp2 = tp2_r * sl_dist + SPREAD
    tp3 = tp3_r * sl_dist * tp_ext + SPREAD

    tp1_hit = np.zeros(n, bool); tp2_hit = np.zeros(n, bool); closed = np.zeros(n, bool)
    code = np.zeros(n, np.int8)  # 1=sl,2=final_be,3=tp1_be,4=final
    e_sl = np.where(is_buy, entries - sl_dist, entries + sl_dist)
    e_tp1 = np.where(is_buy, entries + tp1, entries - tp1)
    e_tp2 = np.where(is_buy, entries + tp2, entries - tp2)
    e_tp3 = np.where(is_buy, entries + tp3, entries - tp3)
    be = entries
    tp1_as_sl = np.where(is_buy, entries + tp1, entries - tp1)

    for j in range(MAX_HOLD):
        active = ~closed
        if not active.any():
            break
        hi = H[:, j]; lo = L[:, j]
        sl_lvl = e_sl.copy()
        sl_lvl = np.where(tp1_hit & ~tp2_hit, be, sl_lvl)
        sl_lvl = np.where(tp2_hit, tp1_as_sl, sl_lvl)
        hit_sl = np.where(is_buy, lo <= sl_lvl, hi >= sl_lvl)
        newly = active & hit_sl
        code = np.where(newly, np.where(tp2_hit, 2, np.where(tp1_hit, 3, 1)), code)
        closed = closed | newly
        active = ~closed
        t1 = np.where(is_buy, hi >= e_tp1, lo <= e_tp1)
        tp1_hit = tp1_hit | (active & ~tp1_hit & t1)
        t2 = np.where(is_buy, hi >= e_tp2, lo <= e_tp2)
        tp2_hit = tp2_hit | (active & ~tp2_hit & t2)
        t3 = np.where(is_buy, hi >= e_tp3, lo <= e_tp3)
        nf = active & t3
        code = np.where(nf, 4, code)
        closed = closed | nf

    last_c = Cc[:, -1]
    r_to = (last_c - entries) * d / sl_dist
    r = np.where(code == 4, (0.10*tp1 + 0.20*tp2 + 0.70*tp3) / sl_dist,
        np.where(code == 2, (0.10*tp1 + 0.20*tp2 + 0.70*tp1) / sl_dist,
        np.where(code == 3, (0.10*tp1) / sl_dist,
        np.where(code == 1, -1.0, r_to))))
    return r, sl_dist


def metrics(c, r, sl_dist, risk_cap=RISK_CAP):
    entries = c["entries"]
    lpl = CONTRACT * sl_dist
    lots = np.minimum(0.12, risk_cap / lpl)
    lots = np.maximum(0.01, np.floor(lots * 100) / 100)
    margin = lots * CONTRACT * entries / LEVERAGE
    over = margin > EQUITY0 * 0.95
    if over.any():
        alt = np.maximum(0.01, np.floor((EQUITY0 * 0.95 * LEVERAGE / (CONTRACT * entries)) * 100) / 100)
        lots = np.where(over, alt, lots)
    risk = lots * lpl
    pnl = r * risk

    order = c["order"]
    pnl_sorted = pnl[order]
    eq = EQUITY0 + np.cumsum(pnl_sorted)
    peak = np.maximum.accumulate(eq)
    dd_total = float(np.max(peak - eq)) if len(eq) else 0.0

    # worst-day drawdown via fast bincount over precomputed day ids
    day_sums = np.bincount(c["day_codes"], weights=pnl_sorted, minlength=c["n_days"])
    dd_day = float(-day_sums.min()) if len(day_sums) and day_sums.min() < 0 else 0.0

    total_d = float(pnl.sum())
    n = len(pnl)
    ev = float(pnl.mean()) if n else 0.0
    avg_sl = float(sl_dist.mean()) if n else 0.0
    avg_atr5 = float(c["atr5"].mean()) if n else 0.0
    avg_lots = float(lots.mean()) if n else 0.0
    wins = int((r > 0).sum()); losses = int((r < 0).sum())
    return {"total_d": total_d, "ev": ev, "avg_sl": avg_sl, "avg_atr5": avg_atr5,
            "avg_lots": avg_lots, "dd_total": dd_total, "dd_day": dd_day,
            "wins": wins, "losses": losses, "n": n}


def tsai_score(mt):
    # Gates
    g1 = 1.0 if mt["avg_sl"] > G_ATR_MULT * mt["avg_atr5"] else 0.0
    g2 = 1.0 if mt["dd_total"] < G_DD_TOTAL else 0.0
    g3 = 1.0 if mt["dd_day"] < G_DD_DAY else 0.0
    G = g1 * g2 * g3
    # Compatibility
    spread_comm = mt["avg_lots"] * CONTRACT * SPREAD + COMMISSION
    friction = mt["ev"] / spread_comm if spread_comm > 0 else 0.0
    noise = mt["avg_sl"] / mt["avg_atr5"] if mt["avg_atr5"] > 0 else 0.0
    C = max(0.0, K_SCALE * friction * noise)
    # Suitability (nested harmonic of daily & total safety distance)
    s_day = 250.0 / mt["dd_day"] if mt["dd_day"] > 0 else 1e6
    s_total = 400.0 / mt["dd_total"] if mt["dd_total"] > 0 else 1e6
    S = (2 * s_day * s_total) / (s_day + s_total) if (s_day + s_total) > 0 else 0.0
    TSAI = G * ((2 * C * S) / (C + S)) if (C + S) > 0 else 0.0
    return {"TSAI": TSAI, "G": G, "g1": g1, "g2": g2, "g3": g3, "C": C, "S": S,
            "friction": friction, "noise": noise, "s_day": s_day, "s_total": s_total}


OLD = dict(base_sl=0.8939, vol_lo=0.8559, vol_hi=1.0723, tp1_r=1.0, tp2_r=2.0,
           tp3_r=3.0, trend_gain=1.7875, tp_lo=0.3295, tp_hi=2.8771)


def evaluate(c, params, risk_cap=RISK_CAP):
    sim_params = {k: v for k, v in params.items() if k != "risk_cap"}
    r, sl_dist = simulate(c, **sim_params)
    mt = metrics(c, r, sl_dist, risk_cap=params.get("risk_cap", risk_cap))
    ts = tsai_score(mt)
    return mt, ts


def fmt(params, mt, ts, weeks):
    return (f"total=${mt['total_d']:+.2f} | ${mt['total_d']/weeks:+.2f}/wk | "
            f"WR={mt['wins']/max(1,mt['wins']+mt['losses'])*100:.1f}% | "
            f"DDtot=${mt['dd_total']:.0f} DDday=${mt['dd_day']:.0f} | "
            f"TSAI={ts['TSAI']:.3f} (G={ts['G']:.0f} C={ts['C']:.2f} S={ts['S']:.2f})")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    print("Loading REAL Jan 2025 -> Jul 2026 XAUUSD 1m data...")
    df1m = load_data()
    span_days = (df1m.index.max() - df1m.index.min()).days
    weeks = span_days / 7.0
    print(f"  {len(df1m)} bars | {df1m.index.min()} -> {df1m.index.max()} | ~{weeks:.1f} weeks")
    print("Building candidate signals...")
    c = build_candidates(df1m)
    print(f"  {c['n']} candidate signals\n")

    # Score OLD config (at the live $60 risk cap)
    old_mt, old_ts = evaluate(c, OLD, risk_cap=60.0)
    print("OLD (current live adaptive config, $60 risk):")
    print("  " + fmt(OLD, old_mt, old_ts, weeks))
    print()

    # Random search maximizing total_$ subject to gates (G==1)
    rng = np.random.default_rng(2025)
    best_val = -1e18
    best_params = None
    best_mt = best_ts = None
    peaks = 0
    t0 = time.time()
    for trial in range(1, N + 1):
        tp1_r = rng.uniform(0.5, 2.0)
        tp2_r = tp1_r + rng.uniform(0.3, 1.8)     # guarantees tp1<tp2<tp3, no wasted trials
        tp3_r = tp2_r + rng.uniform(0.4, 3.0)
        p = dict(
            base_sl=rng.uniform(0.5, 2.0),
            vol_lo=rng.uniform(0.3, 1.0),
            vol_hi=rng.uniform(1.0, 2.5),
            tp1_r=tp1_r,
            tp2_r=tp2_r,
            tp3_r=tp3_r,
            trend_gain=rng.uniform(-0.5, 3.0),
            tp_lo=rng.uniform(0.2, 1.0),
            tp_hi=rng.uniform(1.0, 4.0),
        )
        risk_cap = rng.uniform(8.0, 60.0)   # position risk size is the main DD lever
        r, sl_dist = simulate(c, **p)
        mt = metrics(c, r, sl_dist, risk_cap=risk_cap)
        ts = tsai_score(mt)
        if ts["G"] < 1:      # must pass all security gates to be valid
            continue
        if mt["total_d"] > best_val:
            best_val = mt["total_d"]
            p["risk_cap"] = risk_cap
            best_params = p; best_mt = mt; best_ts = ts
            peaks += 1
        if trial % 25000 == 0:
            el = time.time() - t0
            print(f"  trial {trial:>7}/{N} | {trial/el:.0f}/s | peaks={peaks} | "
                  f"best=${best_val:+.2f} (${best_val/weeks:+.2f}/wk)")

    el = time.time() - t0
    print()
    print("=" * 92)
    print(f"SEARCH DONE: {N} trials in {el/60:.1f} min | REAL peak-surpassing progressions found: {peaks}")
    print()
    if best_params is None:
        print("NO gate-passing config found. The security gates (DD_total<$350, "
              "DD_day<$200, SL>1.2*ATR5) could not be satisfied on this data even "
              "with reduced risk. Widen the search or relax constraints.")
        print("OLD:  " + fmt(OLD, old_mt, old_ts, weeks))
        sys.exit(0)
    print("OLD:  " + fmt(OLD, old_mt, old_ts, weeks))
    print("BEST: " + fmt(best_params, best_mt, best_ts, weeks))
    imp = best_mt["total_d"] - old_mt["total_d"]
    print(f"IMPROVEMENT: ${imp:+.2f} total ({imp/weeks:+.2f}/wk)")
    print()
    print("BEST PARAMS:")
    for k, v in best_params.items():
        print(f"  {k} = {v:.4f}")
    print("=" * 92)

    json.dump({"params": best_params, "metrics": best_mt, "tsai": best_ts,
               "old_params": OLD, "old_metrics": old_mt, "old_tsai": old_ts,
               "weeks": weeks, "n_trials": N, "peaks": peaks},
              open("tsai_best.json", "w"), indent=2, default=float)
    print("Saved tsai_best.json")
