# ISAGI ENGINE — Implementation Plan (Tasks)

Legend: [now] buildable immediately on 1m data | [data] needs data decision |
[live] live-only (can't backtest without tick/sub-minute)

1. [now] `regime.py` — 15m ATR Z-score state machine (Balanced M1 / Compressed M5,H1).
   Hyper 30s/15s stubbed to M1 until sub-minute data (Q2).
2. [now] `pipeline.py` — 5-filter entry (bias, exhaustion 3xATR, value 1.5xATR,
   EMA20 trigger, session 12-17 UTC). Multi-TF via 1m resample.
3. [now/data] `levels.py` — adaptive SL from 15-bar sweep + spread; 3R ceiling;
   elastic TP. V_dens via chosen option (Q3: proxy / data / volatility-only).
   Management TP1@35% -> close 45% + BE -> 55% runner.
4. [now/data/live] `decouplers.py` — news blackout via chosen calendar (Q4);
   exploitation state; shock reflex emitted as LIVE-only alert (Q2).
5. [now] `sizing.py` — Fractional Kelly with 2% cap / 0.35% defensive.
6. [now] `learning.py` — post-mortem: root-cause classify (sweep/wall/exhaustion),
   counterfactual (SL buffer / TP cut that saves it), ASWP corrective offset store.
7. [now] `synthesis.py` — offline pre-train over 2025-26 (+2024 if data), populate
   ASWP memory + spatial corrections.
8. [now] `engine.py` — orchestrate; emit SIGNAL / FLATTEN / SIZE alerts.
9. [now] `backtest.py` — accurate walk-forward on real 2025-26; friction; gate;
   report tiers vs current phase-15 baseline.
10. [now] Wire the validated engine into `live_terminal.py` (signal-only) if it
    beats baseline gate-safe; Telegram on completion.

## Order of execution
regime -> pipeline -> levels -> sizing -> learning -> engine -> backtest -> (ship) ->
decouplers(live) -> synthesis -> Telegram.
