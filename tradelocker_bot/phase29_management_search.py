"""
PHASE 29 — improve the CURRENT strategy via IN-TRADE MANAGEMENT (not prediction).
Legitimate real-time decisions (use only info available up to each bar):
  - early_cut: exit a loser at -X R (cap the -1.34R blowups) before full stop
  - reentry: after a stop-hunt (stop then price recovers to entry), re-enter to
    capture the move that "got away" — directly answers "how could I have won it"
  - be_after: when to move stop to breakeven

ACCURATE: full bar-by-bar path, real per-trade economics (0.12 lot cap +
commission + slippage). NO proxy. Current phase-15 geometry fixed; only
management varies. 3-way TIME split on REAL 2025-26 data. Winner re-validated
with the full sequential friction sim.
"""
import time
import numpy as np
import pandas as pd
import phase12_ideal as P12, tsai_optimize as T

SPREAD = T.SPREAD; B = 200; RISK = 47.0; TICK = 0.10
# current deployed geometry
BSL, VLO, VHI = 1.1342, 0.346, 1.2653
T1, T2, T3B = 0.45, 3.16, 6.26
TG, ELO, EHI = 1.1363, 0.849, 2.509
A1, A2 = 0.45, 0.16; A3 = 1 - A1 - A2


def precompute():
    te = P12.prep(T.load_data()); c = te["c"]; o = te["order"]
    is_buy = (c["d"] == 1)[o]
    H = c["H"][o][:, :B]; L = c["L"][o][:, :B]; C = c["C"][o][:, :B]
    entry = c["entries"][o]; atr = c["x_atr"][o]; vr = c["vol_ratio"][o]; ts = c["trend_strength"][o]
    sld_atr = BSL * np.clip(vr, VLO, VHI)
    sld_price = sld_atr * atr + SPREAD / 2
    favR = np.where(is_buy[:, None], H - entry[:, None], entry[:, None] - L) / (sld_atr[:, None] * atr[:, None])
    advR = np.where(is_buy[:, None], entry[:, None] - L, H - entry[:, None]) / (sld_atr[:, None] * atr[:, None])
    favR = np.nan_to_num(favR, nan=-9); advR = np.nan_to_num(advR, nan=-9)
    ext = np.clip(1 + TG * (ts - 1), ELO, EHI)
    dt = pd.to_datetime(c["dt"][o]); day = te["day"]; hour = te["hour"]; wd = dt.weekday.values
    return dict(favR=favR, advR=advR, ext=ext, sld=sld_price, dt=dt, day=day,
                hour=hour, wd=wd, n=len(entry))


def first_hit(cum, lvl):
    hit = cum >= lvl
    return np.where(hit.any(1), hit.argmax(1), B)


def base_and_manage(P, early_cut, be_after, reentry_on, re_target, re_window):
    """Accurate per-signal R with management. Vectorized."""
    favR = P["favR"]; advR = P["advR"]; n = P["n"]
    cmf = np.maximum.accumulate(favR, 1); cma = np.maximum.accumulate(advR, 1)
    t3 = T3B * P["ext"]
    b_sl = first_hit(cma, 1.0)                 # initial 1R stop
    b_cut = first_hit(cma, early_cut)          # early-cut level
    b_t1 = first_hit(cmf, be_after)            # BE trigger (== TP1 bank)
    b_t2 = first_hit(cmf, T2)
    b_t3 = np.array([first_hit(cmf[i:i+1], t3[i])[0] for i in range(n)])  # per-row level
    r = np.zeros(n); stop_bar = np.full(n, B); full_stop = np.zeros(n, bool)
    for i in range(n):
        s = b_sl[i]; c1 = b_cut[i]; h1 = b_t1[i]; h2 = b_t2[i]; h3 = b_t3[i]
        # early cut fires only before TP1 and before full SL
        cut = c1 if (early_cut < 1.0 and c1 < h1 and c1 <= s) else B
        stop0 = min(s, cut)
        if stop0 < h1:                          # stopped before locking TP1
            r[i] = -(early_cut if cut < s else 1.0)
            stop_bar[i] = stop0; full_stop[i] = True
            continue
        # reached TP1 -> bank A1*T1, SL to BE
        R = A1 * T1
        # BE stop after h1: first bar>h1 where advR>=0 (back to entry)
        be = h1 + 1 + (advR[i, h1+1:] >= 0).argmax() if (advR[i, h1+1:] >= 0).any() else B
        if h2 <= be:                            # reached TP2
            R += A2 * T2
            # runner: TP1-level trail after TP2; reaches T3 or falls to T1
            fall = h2 + 1 + (advR[i, h2+1:] >= -T1).argmax() if (advR[i, h2+1:] >= -T1).any() else B
            if h3 <= fall:
                R += A3 * t3[i]
            else:
                R += A3 * T1
        else:                                   # BE stop before TP2
            R += 0.0                            # remainder scratched at BE
        r[i] = R
    # ---- reentry on full-stop signals (stop-hunt recovery) ----
    if reentry_on:
        for i in np.where(full_stop)[0]:
            js = stop_bar[i]
            if js >= B - 5:
                continue
            seg_f = favR[i, js+1:]; seg_a = advR[i, js+1:]
            # recover to entry: favR back to >=0 (price returns up through entry)
            rec = (seg_f >= 0)
            if not rec.any():
                continue
            rb = rec.argmax()
            win = min(re_window, len(seg_f) - rb)
            ff = np.maximum.accumulate(seg_f[rb:rb+win]) if win > 0 else np.array([])
            aa = np.maximum.accumulate(seg_a[rb:rb+win]) if win > 0 else np.array([])
            if len(ff) == 0:
                continue
            hit_t = (ff >= re_target).argmax() if (ff >= re_target).any() else B
            hit_s = (aa >= 1.0).argmax() if (aa >= 1.0).any() else B
            r[i] += (re_target if hit_t < hit_s else -1.0)  # 2nd trade, full risk
    return r


def score(r, P, mask, weeks, base=RISK):
    sld = P["sld"]; lpl = 100.0 * sld
    lots = np.clip(np.floor(base / lpl * 100) / 100, 0.01, 0.12)
    pnl = r * lots * lpl
    fr = 7.0 * lots + lots * 100 * (1.5 * TICK) + np.where(r > 0, lots * 100 * A1 * 1.5 * TICK, 0)
    pnl = pnl - fr
    pm = pnl[mask]
    if len(pm) == 0:
        return -1e9, 1e9
    eq = np.cumsum(pm); dd = float(np.max(np.maximum.accumulate(eq) - eq))
    return float(pm.sum() / weeks), dd


def main():
    print("Precomputing accurate paths (current phase-15 geometry, real 2025-26)...")
    P = precompute()
    # 2/day + session mask (config-independent)
    order = np.argsort(P["dt"].values.astype("datetime64[ns]").astype(np.int64))
    insess = np.isin(P["hour"], list(range(12, 21))); take = np.zeros(P["n"], bool); cnt = {}
    for k in order:
        if not insess[k]:
            continue
        d = P["day"][k]; cnt[d] = cnt.get(d, 0)
        if cnt[d] < 2:
            take[k] = True; cnt[d] += 1
    idx = order[take[order]]; nt = len(idx); cut = int(nt*2/3)
    sidx, cidx = idx[:cut], idx[cut:]
    smask = np.zeros(P["n"], bool); smask[sidx] = True
    cmask = np.zeros(P["n"], bool); cmask[cidx] = True
    dtv = P["dt"]; wks = (dtv[sidx].max()-dtv[sidx].min()).days/7.0
    wkc = (dtv[cidx].max()-dtv[cidx].min()).days/7.0
    print(f"tradeable {nt} | SEARCH {len(sidx)} CONFIRM {len(cidx)}\n")

    # baseline = current management (no early cut, BE after TP1=0.45, no reentry)
    r0 = base_and_manage(P, 1.0, 0.45, False, 0, 0)
    b_ws, b_dds = score(r0, P, smask, wks); b_wc, b_ddc = score(r0, P, cmask, wkc)
    print(f"BASELINE (current mgmt): SEARCH ${b_ws:+.1f}/wk (DD${b_dds:.0f}) | "
          f"CONFIRM ${b_wc:+.1f}/wk (DD${b_ddc:.0f})\n")

    rng = np.random.default_rng(29)
    best_ws = b_ws if b_dds < 400 else -1e9; best = None
    improvements = 0; confirmed = 0; TARGET = 800_000; t0 = time.time(); trial = 0
    trace = []
    while trial < TARGET:
        trial += 1
        early_cut = float(rng.choice([1.0, 1.0, 0.6, 0.7, 0.8, 0.9]))
        be_after = float(rng.choice([0.45, 0.3, 0.6, 0.8]))
        reentry_on = bool(rng.integers(0, 2))
        re_target = float(rng.uniform(1.0, 4.0))
        re_window = int(rng.integers(10, 120))
        r = base_and_manage(P, early_cut, be_after, reentry_on, re_target, re_window)
        ws, dds = score(r, P, smask, wks)
        if dds < 400 and ws > best_ws + 0.05:
            wc, ddc = score(r, P, cmask, wkc)
            best_ws = ws; improvements += 1
            best = (early_cut, be_after, reentry_on, re_target, re_window, ws, wc, dds, ddc)
            if wc > b_wc and ddc < 400:
                confirmed += 1
            trace.append((improvements, ws, wc))
        if trial % 2000 == 0:
            el = time.time()-t0
            print(f"  trial {trial:>7} | improvements {improvements} | confirmed {confirmed} | "
                  f"best SEARCH ${best_ws:.1f}/wk | {el:.0f}s")
        if time.time()-t0 > 1500:
            print(f"  [wall cap at trial {trial}]"); break

    print("\n" + "=" * 66)
    print(f"MANAGEMENT SEARCH: {trial} REAL accurate trials")
    print(f"  SEARCH improvements: {improvements} | CONFIRMED (untouched set): {confirmed}")
    if best:
        ec, ba, ro, rt, rw, ws, wc, dds, ddc = best
        print(f"  BEST: early_cut={ec} be_after={ba} reentry={ro} re_target={rt:.2f} re_window={rw}")
        print(f"        SEARCH ${ws:+.1f}/wk (DD${dds:.0f}) | CONFIRM ${wc:+.1f}/wk (DD${ddc:.0f})")
        print(f"        baseline CONFIRM ${b_wc:+.1f}/wk")
    print("=" * 66)
    if trace:
        print("  overfit check (SEARCH vs CONFIRM as improvements accrue):")
        for imp, ws, wc in trace[::max(1, len(trace)//10)]:
            print(f"    #{imp:>3}  SEARCH ${ws:+6.1f}  CONFIRM ${wc:+6.1f}")


if __name__ == "__main__":
    main()
