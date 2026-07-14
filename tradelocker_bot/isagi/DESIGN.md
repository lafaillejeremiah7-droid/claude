# ISAGI ENGINE — Design + Data-Availability Map

## Architecture (modules)
```
isagi/
  regime.py       # R1  Fluid timeframe state machine (ATR Z-score -> TF set)
  pipeline.py     # R2  5-filter structural entry pipeline
  levels.py       # R3  Metavision elastic level engine (SL/TP/3R cap/mgmt)
  decouplers.py   # R4  News blackout + exploitation + shock reflex
  sizing.py       # R5  Fractional-Kelly Ego sizing with caps
  learning.py     # R6  Cognitive post-mortem + ASWP corrective offsets
  synthesis.py    # R7  Offline pre-train over history
  engine.py       # orchestrator: state -> pipeline -> levels -> sizing -> signal
  backtest.py     # honest walk-forward validation on real 2025-26 data
```

## DATA-AVAILABILITY MAP (what I have vs what each requirement needs)
| Requirement | Needs | Have? | Plan |
|---|---|---|---|
| R1 Balanced (M1) / Compressed (M5,H1) | 1m OHLC | YES | build + backtest |
| R1 Hyper (30s/15s) | sub-minute bars | NO | LIVE-only OR needs data |
| R2 full pipeline | 1m/5m/15m/1h OHLC | YES (resample from 1m) | build + backtest |
| R3.1 volatility Z / T_vel | 1m OHLC | YES | build |
| R3.1 V_dens (Volume Profile Density) | volume | NO (2025-26 has none) | proxy / data / drop |
| R3.2-3.4 SL/TP/mgmt | 1m OHLC | YES | build + backtest |
| R4.1 news blackout | econ calendar | NO | static schedule / API |
| R4.3 tick-velocity shock | tick feed | NO | LIVE-only |
| R5 Kelly sizing | trade history | YES | build (note: gate risk) |
| R6 learning loop | trade path | YES (1m) | build + backtest |
| R7 offline synthesis | 2024-2026 history | 2025-26 YES; 2024 partial | build on available |

## Honest engineering notes
- The PREDICTIVE core (does a setup win?) remains ~0.50 AUC on entry-time data
  (proven ~11x). ISAGI's genuine value is in EXECUTION + MANAGEMENT + ADAPTATION
  (regime-aware timeframe, elastic capped geometry, learning loop) — real levers,
  not predictive alpha. Backtested $/wk is bounded by the $5k lot-cap.
- Every "auto" action in the blueprint is implemented as an EMITTED ALERT.
- All backtests use the accurate bar-by-bar sim + real friction + gate checks;
  no proxy scoring will be trusted without accurate-sim confirmation.

## Validation protocol
- Walk-forward on real 2025-26 (train-past -> test-future), embargo, friction ON,
  gate DDt<400 / DDd<250, via the existing CV harness + accurate sim.
- Report tiers (Regime, Pipeline, Levels, Decouplers, Sizing, Learning, Overall).
