"""
PHASE 5 — Hybrid strategy search.

Graft the best logical component from each major trading-strategy family onto
the current ASWP engine, then test ALL 2^10 = 1024 combinations against real
XAU/USD data to find the best final strategy. Validate out-of-sample and
recheck 5x. Signal-only throughout.

Components (each a boolean keep-filter on the current signal set):
  C1 ADX(1h) trend-strength gate        (trend-following)
  C2 MACD histogram momentum confirm    (momentum)
  C3 Bollinger %B pullback zone (15m)   (mean-reversion)
  C4 Donchian(1h,20) breakout confirm   (breakout)
  C5 London+NY session filter           (session / news-fade)
  C6 Anchored-VWAP side (5m proxy)      (VWAP-bounce)
  C7 prior-day pivot proximity skip     (support/resistance)
  C8 strong trigger candle (15m)        (price action)
  C9 Ichimoku cloud side (1h)           (Ichimoku)
  C10 Bollinger squeeze regime (15m)    (volatility)
"""
import json
import time
import itertools
import numpy as np
import pandas as pd

import tsai_optimize as T
import phase4_usage as P4
import fullyear_backtest as fb

BIG = 1e9
# Current LIVE usage config to compare against (adaptive SL/TP already in the
# outcome codes; allocation 10/20/70, risk $26 = what you run now).
LIVE = dict(a1=0.10, a2=0.20, tilt=0.0, risk=26.0)
# Phase-4 validated allocation (also tested as the sizing layer)
P4CFG = dict(a1=0.30, a2=0.40, tilt=0.19, risk=25.0)

COMPONENTS = ["ADX", "MACD", "BB%B", "Donchian", "Session", "VWAP", "Pivot",
              "Candle", "Ichimoku", "Squeeze"]


# ---------- indicator helpers on a resampled OHLC frame ----------
def _adx(df, n=14):
    h, l, cl = df["h"], df["l"], df["c"]
    up = h.diff(); dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(h - l), (h - cl.shift()).abs(), (l - cl.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def _macd_hist(close):
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    return macd - sig


def _bollinger(close, n=20, k=2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = mid + k * sd; lower = mid - k * sd
    pctB = (close - lower) / (upper - lower).replace(0, np.nan)
    bw = (upper - lower) / mid.replace(0, np.nan)
    return pctB, bw


def _ichimoku(df):
    h, l = df["h"], df["l"]
    ten = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kij = (h.rolling(26).max() + l.rolling(26).min()) / 2
    spanA = ((ten + kij) / 2).shift(26)
    spanB = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    top = pd.concat([spanA, spanB], axis=1).max(axis=1)
    bot = pd.concat([spanA, spanB], axis=1).min(axis=1)
    return top, bot


def _merge(sig_dt, series):
    s = pd.DataFrame({"dt": series.index, "v": series.values}).dropna().sort_values("dt")
    q = pd.DataFrame({"dt": sig_dt}).sort_values("dt")
    m = pd.merge_asof(q, s, on="dt", direction="backward")
    return m.set_index("dt").reindex(sig_dt)["v"].values


def build_features(df1m, c):
    """Compute all component features aligned to the signal times c['dt']."""
    dt = pd.to_datetime(c["dt"])
    d = c["d"]; price = c["entries"]
    b1h = fb.resample_tf(df1m, 60)
    b15 = fb.resample_tf(df1m, 15)
    b5 = fb.resample_tf(df1m, 5)
    bd = fb.resample_tf(df1m, 1440)

    feats = {}
    # C1 ADX 1h
    feats["adx"] = _merge(dt, _adx(b1h))
    # C2 MACD hist 15m
    feats["macd"] = _merge(dt, _macd_hist(b15["c"]))
    # C3 & C10 Bollinger 15m
    pctB, bw = _bollinger(b15["c"])
    feats["pctB"] = _merge(dt, pctB)
    feats["bw"] = _merge(dt, bw)
    feats["bw_med"] = _merge(dt, bw.rolling(100).median())
    # C4 Donchian 1h (prior 20 high/low, shifted to avoid look-ahead)
    feats["don_hi"] = _merge(dt, b1h["h"].rolling(20).max().shift(1))
    feats["don_lo"] = _merge(dt, b1h["l"].rolling(20).min().shift(1))
    # C6 anchored session-VWAP proxy on 5m (cumulative typical price mean per UTC day)
    tp = (b5["h"] + b5["l"] + b5["c"]) / 3
    vwap = tp.groupby(b5.index.date).expanding().mean().reset_index(level=0, drop=True)
    feats["vwap"] = _merge(dt, vwap)
    # C7 prior-day pivot (prior day high/low)
    feats["pd_hi"] = _merge(dt, bd["h"].shift(1))
    feats["pd_lo"] = _merge(dt, bd["l"].shift(1))
    # C8 strong candle: 15m body vs 15m ATR
    body15 = (b15["c"] - b15["o"]).abs()
    atr15 = fb.atr(b15)
    feats["body_ratio"] = _merge(dt, body15 / atr15.replace(0, np.nan))
    # C9 Ichimoku 1h cloud
    top, bot = _ichimoku(b1h)
    feats["cloud_top"] = _merge(dt, top)
    feats["cloud_bot"] = _merge(dt, bot)
    # atr15 at signal (for pivot proximity)
    feats["atr15"] = _merge(dt, atr15)
    feats["hour"] = c["hours"]
    feats["price"] = price
    feats["dir"] = d
    return feats


def component_masks(f):
    """Boolean keep-mask for each component (True = signal passes the filter)."""
    d = f["dir"]; price = f["price"]; buy = d == 1
    m = {}
    m["ADX"] = np.nan_to_num(f["adx"], nan=0.0) >= 20.0
    m["MACD"] = np.where(buy, np.nan_to_num(f["macd"]) > 0, np.nan_to_num(f["macd"]) < 0)
    # mean-reversion favorable: buy on lower half, sell on upper half of BB
    m["BB%B"] = np.where(buy, np.nan_to_num(f["pctB"], nan=0.5) < 0.6,
                         np.nan_to_num(f["pctB"], nan=0.5) > 0.4)
    # breakout confirm: near/through prior Donchian extreme in direction
    m["Donchian"] = np.where(buy, price >= np.nan_to_num(f["don_hi"], nan=1e12) * 0.999,
                             price <= np.nan_to_num(f["don_lo"], nan=0.0) * 1.001)
    m["Session"] = (f["hour"] >= 7) & (f["hour"] <= 21)
    m["VWAP"] = np.where(buy, price >= np.nan_to_num(f["vwap"], nan=price),
                         price <= np.nan_to_num(f["vwap"], nan=price))
    # pivot proximity: skip if entry within 0.5*ATR of overhead resistance (buy) / support (sell)
    near_res = (np.nan_to_num(f["pd_hi"], nan=1e12) - price) < 0.5 * f["atr15"]
    near_res &= price < np.nan_to_num(f["pd_hi"], nan=1e12)
    near_sup = (price - np.nan_to_num(f["pd_lo"], nan=-1e12)) < 0.5 * f["atr15"]
    near_sup &= price > np.nan_to_num(f["pd_lo"], nan=-1e12)
    m["Pivot"] = np.where(buy, ~near_res, ~near_sup)
    m["Candle"] = np.nan_to_num(f["body_ratio"], nan=0.0) >= 0.3
    m["Ichimoku"] = np.where(buy, price > np.nan_to_num(f["cloud_top"], nan=1e12),
                             price < np.nan_to_num(f["cloud_bot"], nan=-1e12))
    m["Squeeze"] = np.nan_to_num(f["bw"], nan=1e9) <= np.nan_to_num(f["bw_med"], nan=0.0)
    return m


def eval_mask(pc, keep, usage, weeks):
    idx = np.where(keep)[0]
    if len(idx) < 50:
        return None
    import walk_forward as W
    p, dts = W.pnl_idx(pc, idx, usage["a1"], usage["a2"], usage["tilt"], usage["risk"])
    ddt, ddd = W.stitched_dd(p, dts)
    r_pos = int((p > 0).sum()); r_neg = int((p < 0).sum())
    return {"total": float(p.sum()), "wk": float(p.sum())/weeks, "n": len(idx),
            "wr": r_pos/max(1, r_pos+r_neg), "dd_total": ddt, "dd_day": ddd,
            "gate": ddt < 350 and ddd < 200}


def run(df1m, label):
    c = T.build_candidates(df1m)
    pc = P4.precompute(c); pc["dt_sorted"] = c["dt"][pc["order"]]
    weeks = (df1m.index.max() - df1m.index.min()).days / 7.0
    f = build_features(df1m, c)
    masks_sig = component_masks(f)     # aligned to signal order (c order)
    # reorder masks to pc order (pc uses order = argsort(dt))
    order = pc["order"]
    masks = {k: v[order] for k, v in masks_sig.items()}
    return pc, weeks, masks


def main():
    print("Loading REAL 2025-2026 data + building signals + features...")
    df1m = T.load_data()
    pc, weeks, masks = run(df1m, "IS")
    n = pc["n"]

    def base_eval(usage):
        return eval_mask(pc, np.ones(n, bool), usage, weeks)

    live_base = base_eval(LIVE)
    p4_base = base_eval(P4CFG)
    print(f"\nBASELINE current-live (10/20/70,$26): +${live_base['wk']:.2f}/wk | WR={live_base['wr']*100:.1f}% "
          f"| DDtot=${live_base['dd_total']:.0f} DDday=${live_base['dd_day']:.0f} | gate={live_base['gate']}")
    print(f"BASELINE Phase-4 alloc (30/40/31,$25): +${p4_base['wk']:.2f}/wk | WR={p4_base['wr']*100:.1f}% "
          f"| DDtot=${p4_base['dd_total']:.0f} DDday=${p4_base['dd_day']:.0f} | gate={p4_base['gate']}")

    # Individual component effect (each alone, on top of current-live usage)
    print("\n--- Each component ALONE (on live 10/20/70 usage) ---")
    for name in COMPONENTS:
        m = eval_mask(pc, masks[name], LIVE, weeks)
        if m:
            print(f"  {name:10} kept {m['n']:5}/{n} | +${m['wk']:7.2f}/wk | WR={m['wr']*100:4.1f}% "
                  f"| DDt=${m['dd_total']:.0f} DDd=${m['dd_day']:.0f} | gate={m['gate']}")

    # Full 1024-combination search, over BOTH usage layers, maximize $/wk s.t. gate
    print(f"\nSearching all 1024 component combinations x 2 usage layers...")
    best = None; best_wk = -1e18; best_desc = None
    for r in range(0, len(COMPONENTS) + 1):
        for combo in itertools.combinations(COMPONENTS, r):
            keep = np.ones(n, bool)
            for name in combo:
                keep &= masks[name]
            for uname, usage in [("live", LIVE), ("p4", P4CFG)]:
                m = eval_mask(pc, keep, usage, weeks)
                if m is None or not m["gate"]:
                    continue
                if m["wk"] > best_wk:
                    best_wk = m["wk"]; best = m; best_desc = (combo, uname)
    combo, uname = best_desc
    print(f"\nBEST COMBO (gate-safe, max $/wk): usage={uname} | components={list(combo) if combo else 'NONE'}")
    print(f"  +${best['wk']:.2f}/wk | WR={best['wr']*100:.1f}% | n={best['n']} | "
          f"DDtot=${best['dd_total']:.0f} DDday=${best['dd_day']:.0f}")
    print(f"\nvs current-live +${live_base['wk']:.2f}/wk  ->  "
          f"{'SURPASSES' if best['wk'] > live_base['wk'] else 'does NOT surpass'} "
          f"(${best['wk']-live_base['wk']:+.2f}/wk)")
    print(f"vs Phase-4 alloc +${p4_base['wk']:.2f}/wk  ->  "
          f"{'SURPASSES' if best['wk'] > p4_base['wk'] else 'does NOT surpass'} "
          f"(${best['wk']-p4_base['wk']:+.2f}/wk)")

    json.dump({"best_combo": list(combo), "best_usage": uname, "best": best,
               "live_base": live_base, "p4_base": p4_base},
              open("phase5_result.json", "w"), indent=2, default=float)
    print("\nSaved phase5_result.json")


if __name__ == "__main__":
    main()
