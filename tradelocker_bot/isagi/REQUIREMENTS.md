# ISAGI ENGINE — Requirements (Spec)

Signal-only XAU/USD engine per the ISAGI production blueprint. The engine emits
signals/alerts; the user executes. (Standing rule: the bot NEVER places, modifies,
or closes orders automatically.)

## R1 — Fluid Focus Timeframe (Section 1A)
- WHEN the rolling 15-minute ATR Z-score changes state, the engine SHALL switch its
  scan/entry layers:
  - Z > 2.0 (Hyper): structural bias on M15, entry trigger on 30s/15s.
  - Z in [-1.0, 2.0] (Balanced): bias M15, trigger M1.
  - Z < -1.0 (Compressed): bias H1, trigger M5.

## R2 — Structural Entry Pipeline (Section 1B) — all 5 must pass
- R2.1 Directional bias: EMA20 vs EMA50 on the macro chart.
- R2.2 Move-exhaustion guard: block if price >= 3x ATR from session structural extreme.
- R2.3 Value-anchor zone: price within 1.5x ATR of anchor EMA.
- R2.4 Execution trigger: touch trigger-TF EMA20 and close back in trend direction.
- R2.5 Session boundary: entries only 12:00-17:00 UTC.

## R3 — Metavision Level Automation (Section 2) — capped elastic, 3.0R ceiling
- R3.1 Environment array: volatility Z-score, Volume Profile Density (V_dens),
  Trend Velocity T_vel = |EMA20-EMA50|/ATR.
- R3.2 Adaptive SL: scan last 15 micro-bars, place SL 1 tick past local sweep
  extreme + spread buffer; if V_dens hyper-low, collapse SL behind entry candle node.
- R3.3 Elastic TP: expansion/void -> 3.0R ceiling; chop/dense -> compress to 1.5-2.0R.
- R3.4 Management: TP1 at 35% of target distance; close 45% + SL->breakeven;
  ride 55% to final.

## R4 — Decouplers (Section 3)
- R4.1 News blackout: disable entries from 2 min before to 15 min after high-impact
  USD events (CPI/NFP/FOMC).
- R4.2 Exploitation state: minute 16-120 post-release, IF 15m ATR Z-score stays
  positive-expansion, lock final target to 3.0R and scale size to peak.
- R4.3 Shock reflex: if tick velocity > 400% of rolling 5-min avg, flag Unexpected
  Shock -> emit FLATTEN alert, cancel-orders alert, stall trigger 5 min until a new
  structural range forms.

## R5 — Ego Sizing / Fractional Kelly (Section 4)
- f* = Multiplier * (p(R+1)-1)/R ; p = rolling 20-trade WR, R = elastic R:R.
- Cap 2% ($100 on $5k) on win streaks; force 0.35% ($17.50) if WR degrades OR
  rolling drawdown pool < -$200.

## R6 — Learning Loop / Cognitive Post-Mortem (Section 5)
- On loss/suboptimal scratch: 60-min cooldown; extract trade path.
- Root-cause classify: Volatility Sweep | Volume Wall | Trend Exhaustion.
- Counterfactual sim: compute the exact adjustment (SL buffer / TP reduction) that
  would have saved the trade.
- ASWP injection: store corrective spatial offset; apply to future similar setups.

## R7 — Offline Synthesis (Section 6)
- Pre-train over 2024-mid2026 XAU history with the post-mortem loop active; compile
  weights + spatial corrections into the core memory so Day-1 is not cold-start.

## Non-functional
- Signal-only (no auto-execution). Every "action" is an emitted alert.
- Honest validation: backtest on real 2025-26 data; no look-ahead; report true tiers.
- Telegram notifications on completion and key events.
