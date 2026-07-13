"""Fine confirmation: lock the safe risk for Overlap+NY (12-21 UTC), 2/day."""
import numpy as np
import pandas as pd
import tsai_optimize as T
import fullyear_backtest as fb
import rebuild_v2 as R

tr = R.prep(fb.load_data())   # 2023-24
te = R.prep(T.load_data())    # 2025-26
HRS = set(range(12, 21))

print("Baseline (current live logic proxy: ALL 24h, $20):")
a = R.sim(tr, set(range(24)), 2, 20); b = R.sim(te, set(range(24)), 2, 20)
print(f"  2023-24: ${a['wk']:+.1f}/wk WR{a['wr']:.0f}% DDt${a['ddt']:.0f} {'PASS' if a['gate'] else 'FAIL'}")
print(f"  2025-26: ${b['wk']:+.1f}/wk WR{b['wr']:.0f}% DDt${b['ddt']:.0f} {'PASS' if b['gate'] else 'FAIL'}")

print("\nOverlap+NY (12-21 UTC), fine risk sweep:")
print(f"{'risk':>5} | {'23-24 $/wk':>10} {'DDt':>5} {'DDd':>5} {'g':>4} "
      f"| {'25-26 $/wk':>10} {'DDt':>5} {'DDd':>5} {'g':>4} | {'sig/d':>6} {'WR':>4}")
for risk in [24, 26, 28, 30, 32, 34]:
    a = R.sim(tr, HRS, 2, risk); b = R.sim(te, HRS, 2, risk)
    both = a['gate'] and b['gate']
    flag = " <== safe both" if both else ""
    print(f"  ${risk:<3} | ${a['wk']:>+9.1f} ${a['ddt']:>4.0f} ${a['ddd']:>4.0f} "
          f"{'P' if a['gate'] else 'F':>4} | ${b['wk']:>+9.1f} ${b['ddt']:>4.0f} ${b['ddd']:>4.0f} "
          f"{'P' if b['gate'] else 'F':>4} | {b['sig_d']:>4.1f}/d {b['wr']:>3.0f}%{flag}")

print("\nFull TRAIN 2023-24 per-hour table (confirm 12-21 window is genuinely strong):")
R.hour_table(tr, "TRAIN 2023-24")
