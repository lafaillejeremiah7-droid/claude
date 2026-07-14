"""
ISAGI v5 — DEEP pre-train using the ACCURATE resolver for scoring (no simulator
overfit), per-trade causal adaptation, robust across 3 pre-train folds, OOS once.
This is the honest version of v4: the search cannot exploit resolver flaws because
it IS the accurate resolver. Slower -> fewer configs, but every number is real.
"""
import time, json
import numpy as np, pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import isagi.engine_v4 as V4
import isagi.v4_validate as VV
from isagi.calendar_feed import backtest_blackouts, NewsDecoupler

NB = 200


def main():
    te = P12.prep(T.load_data()); P = V4.precompute(te)
    c = te["c"]; o = te["order"]; is_buy = (c["d"][o] == 1)
    H = c["H"][o][:, :NB]; L = c["L"][o][:, :NB]; entry = c["entries"][o]; atr = c["x_atr"][o]
    fav = np.nan_to_num(np.where(is_buy[:, None], H-entry[:, None], entry[:, None]-L)/atr[:, None], nan=-9)
    adv = np.nan_to_num(np.where(is_buy[:, None], entry[:, None]-L, H-entry[:, None])/atr[:, None], nan=-9)
    dt = P["dt"]; hour = P["hour"]; wd = dt.weekday.values
    news = NewsDecoupler(backtest_blackouts()); bl = np.array([news.is_blackout(t) for t in dt])
    session = np.isin(hour, [12, 13, 14, 15, 16]) & (~((wd == 4) & (hour >= 19))) & (~bl)
    order = np.argsort(dt.values.astype("datetime64[ns]").astype(np.int64)); order = order[session[order]]
    cut = int(len(order)*0.6); pre, oos = order[:cut], order[cut:]
    fld = len(pre)//3; folds = [pre[:fld], pre[fld:2*fld], pre[2*fld:]]
    print(f"ACCURATE deep pre-train | pre {len(pre)} (3 folds) | OOS {len(oos)}\n")

    rng = np.random.default_rng(5); best = None; N = 20000; t0 = time.time(); tr = 0
    while tr < N and time.time()-t0 < 1300:
        tr += 1
        a1 = rng.uniform(0.2, 0.55); a2 = rng.uniform(0.1, 0.4)
        if a1+a2 > 0.9: continue
        t1 = rng.uniform(0.4, 1.4); t2 = t1+rng.uniform(0.5, 2.5); t3 = t2+rng.uniform(0.5, 4)
        cfg = (rng.uniform(0.8, 1.5), rng.uniform(0.4, 0.9), rng.uniform(1.05, 1.35),
               t1, t2, t3, rng.uniform(0.5, 3), rng.uniform(1.2, 2.6), a1, a2,
               rng.uniform(40, 85), rng.uniform(0, 0.6), rng.uniform(1.5, 3.0),
               rng.uniform(1.0, 1.12), rng.uniform(0.88, 1.0), rng.uniform(0.8, 1.0),
               rng.uniform(0.9, 0.99))
        nets = []; ok = True
        for fd in folds:
            m = VV.run_accurate(P, fav, adv, fd, cfg, learn=True)
            if m is None or not m["gate"]:
                ok = False; break
            nets.append(m["wk"])
        if not ok: continue
        robust = min(nets)
        if best is None or robust > best[0]:
            best = (robust, cfg)
        if tr % 3000 == 0:
            print(f"  searched {tr} | best worst-fold ${best[0]:.1f}/wk | {time.time()-t0:.0f}s", flush=True)
    if best is None:
        print("no gate-safe robust config (accurate)"); return
    robust, cfg = best
    json.dump(list(cfg), open("isagi/v5_locked_cfg.json", "w"))
    print(f"\nsearched {tr} | LOCKED (accurate) worst-fold ${robust:+.1f}/wk")
    mp = VV.run_accurate(P, fav, adv, pre, cfg); mo = VV.run_accurate(P, fav, adv, oos, cfg)
    print(f"  pre-train full: ${mp['wk']:+.1f}/wk WR{mp['wr']:.0f}% DDt${mp['ddt']:.0f} {'PASS' if mp['gate'] else 'FAIL'}")
    print(f"  >>> OOS (once): ${mo['wk']:+.1f}/wk WR{mo['wr']:.0f}% DDt${mo['ddt']:.0f} DDd${mo['ddd']:.0f} "
          f"n={mo['n']} {'PASS' if mo['gate'] else 'FAIL'}")
    print(f"  reference: v2 OOS -$1.5/wk | v4 fake +$103 (accurate -$19.8)")


if __name__ == "__main__":
    main()
