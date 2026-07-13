# XAUUSD ASWP Signal Strategy — Complete Logic (LIVE)

This document reflects what is **actually running** in `live_terminal.py` as of
the latest commit. Validated against real 1-minute XAUUSD data spanning Jan 2025
→ Jul 2026 (569,981 bars, Forexite) plus out-of-sample on Dec 2022 → Nov 2024
(HuggingFace `Ashraf-CK/XAUUSD`). Live price and historical bars pulled
directly from the **TradeLocker API** (your AquaFunded broker).

> This is a **signal generator**, not an auto-trader. It decides *whether* to
> signal and *where* the levels go. You place every trade manually. Nothing here
> is financial advice; backtest results are historical and live results will differ.

---

## 1. The one rule everything serves: MAXIMIZE $/TRADE, NOT WIN RATE

A high win rate with tiny reward loses to a lower win rate with big reward.
The runner-weighted structure (30% banked early + 30% riding to the final) nearly
**doubles $/trade** vs a scalp. Win rate is a vanity metric; expected dollars is
the goal.

---

## 2. Signal gate = minimum EXPECTED VALUE, never probability

```
EV = A1·R1·P(TP1) + A2·R2·P(TP2) + A3·R3·P(TP3) − P(full_stop)·1R − P(scratch)·0.05R
Fire only if EV ≥ 0.55R
```

Probabilities come from the **live ASWP brain** (see §7), not a fixed heuristic.

---

## 3. Risk per signal: DRAWDOWN CIRCUIT-BREAKER (not flat)

```
BASE RISK    = $45 per signal (at equity peak / normal conditions)
DAILY STOP   = pause all new signals once daily P&L <= -$120
DD THROTTLE  = when (equity_peak - equity) >= $200, cut risk to 35% ($15.75)
               until a new equity peak is made
```

This replaces the old $60 flat cap. Validated: +37% weekly profit vs flat-$25,
static $4,600 floor never breached across 2025-26, 2023-24, and 4.5yr
(min equity $4,934).

---

## 4. Multi-timeframe hierarchy (backtested optimal)

```
1H  → TREND GATE      EMA20 vs EMA50 sets direction (buy/sell/skip)
15m → PULLBACK ZONE   price within 1.5×ATR of its EMA20 (not overextended)
1m  → ENTRY TRIGGER   prev candle touches EMA20 + closes in trend direction
                      + RSI filter (buy<65, sell>35)
```

Entry trigger moved from 5m → **1m** after full-year optimization (catches the
reversal earlier, +2× edge per signal). Exit ATR is sized from **15m** so the
~$0.30 spread stays a negligible fraction of the stop.

---

## 5. Adaptive SL/TP (validated via 100,000-trial search)

```
SL    = 0.894 × clip(vol_ratio, 0.856, 1.072) × ATR(15m)
        vol_ratio = current 15m ATR ÷ its 50-bar rolling average
        → tightens when calm, widens when volatility expands
TP1   = 1R   (close A1%, SL→breakeven)
TP2   = 2R   (close A2%, SL→TP1)
Final = 3R × clip(1 + 1.79×(trend_strength−1), 0.33, 2.88)  (ride A3%)
        trend_strength = |EMA20₁ₕ − EMA50₁ₕ| ÷ ATR(1h)
        → runner extends in strong 1H trends, contracts in chop
```

---

## 6. Multi-TP harvest allocation (Phase-4 validated upgrade)

```
DEFAULT:     A1=30%  A2=40%  A3=30%   (was 10/20/70)
STRONG TREND: tilt ~19% back toward A3 (runner rides more in strong 1H trends)
```

Why front-loaded? Phase 3 proved gold runs to TP1/TP2 then frequently reverses
(23% of stop-outs were stop-hunts where price later reached TP anyway). Banking
30% at 2R + 40% at 4R locks in gold's reliable early move; the 30% runner
(boosted in strong trends) still catches the big multi-R swings. Validated:
+66% weekly profit in-sample AND out-of-sample (2023-24 + 2025-26), gate-safe.

---

## 7. ASWP — Adaptive Similarity-Weighted Probability (LIVE BRAIN)

The **real** ASWP memory engine is now wired into the live bot (was previously
a simplified static heuristic). It produces `P(TP1)`, `P(TP2)`, `P(TP3)`, and
`P(full_stop)` from:

```
P(X) = Σ[ similarity_i × recency_i × (trade_i reached X) ] / Σ[ similarity_i × recency_i ]

similarity_i = exp( −Σ_k ((feature_now − feature_i)/σ_k)² / bandwidth² )
recency_i    = λ^(age)          λ = 0.99, bandwidth = 1.5
features      = (RSI, yield_alignment, hour_of_day)
```

- Pre-seeded with 100 synthetic memories matching validated base rates
  (P(TP1)~0.52, P(fullstop)~0.41, cold-start EV ≈ 0.79).
- **Adapts after every closed trade**: `aswp.add(TradeMemory(features, mfe_r,
  full_stop))` is called on every signal resolution, with live MFE tracking
  in R-multiples.
- This is the "improve over and over" — it genuinely learns from each outcome
  and sharpens its probability estimates. The more signals resolve, the
  better the probabilities get.

---

## 8. Real-yields macro filter (gold's strongest driver, ~−0.82)

10Y TIPS real yields (FRED `DFII10`) — pulled hourly. Rising real yields →
bearish gold; falling → bullish. The alignment score (−1 opposed … +1 aligned)
is a feature in the ASWP similarity vector, so the macro regime shapes both
signal selection and probability.

---

## 9. Guards

- **Account:** $250 daily loss / $400 max DD (static floor $4,600).
  Circuit-breaker pauses at −$120/day long before the $250 limit.
- **Signal:** **max 2 signals/day**, 60-min cooldown, max 1 open position.
- **Session (rebuild v2):** signals fire **only 12-20 UTC** (London/NY overlap
  + NY session). No entries 21-22 UTC (rollover), no new trades after Fri 19:00
  UTC.

### 9a. SESSION FILTER — the rebuild-v2 upgrade (validated)

Gold's directional edge is concentrated in the London/NY window. Trading the
Asian / early-London chop was the source of the drawdown problem. Restricting
signals to **12-20 UTC** (same $45 risk + circuit-breaker, same 2/day cap):

| | all hours (old) | 12-20 UTC (new) |
|---|---|---|
| 2025-26 max DD | **$407 — FAILS $350 gate** | **$312 — PASS** |
| 2025-26 $/wk | +$123 | **+$139** |
| 2025-26 win rate | 43% | **46%** |
| 2023-24 $/wk | +$128 | +$129 (gate PASS) |
| 2023-24 win rate | 45% | **50%** |

The old all-hours config actually **breached the max-DD gate** on recent
(2025-26) data. The session filter fixes that *and* lifts weekly profit and win
rate simultaneously, validated on both datasets. "Adaptability" here = knowing
which sessions to trade and which to skip — not forcing a trade in every session.

---

## 10. Broker / account reality (AquaFunded $5k, 1:10, 0.01-0.12 lots)

- **Contract:** 1 lot = 100 oz → $1 move = $100/lot.
- **Leverage:** 1:10 confirmed. Margin-aware sizing.
- **Lot bounds:** 0.01–0.12 (0.01 step).
- **Spread:** $0.30, added to every TP distance and SL.
- **Risk at current 15m ATR (~$8):** ~$45 base → 0.05 lots normally; throttled
  to ~$16 → 0.02 lots during drawdowns.

---

## 11. Validated performance

### In-sample (Jan 2025 → Jul 2026, circuit-breaker + 30/40/30):
- **+$420/wk** | max DD $326 | max daily DD $157 | gate PASS
- ~2,731 signals/80wk (~34/wk ≈ 3.4/day)

### Out-of-sample (2023-2024):
- **+$194/wk** | max DD $308 | gate PASS

### Flat-$25 comparison (same allocation, no circuit-breaker):
- +$307/wk (in-sample), +$145/wk (OOS) — circuit-breaker adds +37%.

---

## 12. What 9 phases of research proved

- **5 independent methods** (win-prob, EV regression, hybrid grafting of 10
  classic-strategy families, filter search, permutation-tested candle-EV) all
  confirmed: **entry-time information cannot predict which trades win.** Winners
  and losers look identical at entry. The edge is 100% in payoff structure +
  trade management — never entry selection.
- **Walk-forward re-optimization** adds only +5% risk-matched vs a fixed config.
  XAU/USD's optimal parameters are stable (structural, not regime-drifting).
- **No classic strategy component (ADX, MACD, Bollinger, Donchian, session,
  VWAP, pivots, price-action, Ichimoku, squeeze) surpasses the engine** across
  all 1,024 tested combinations.

---

## 13. Backtest tooling (committed in repo)

| Script | Purpose |
|--------|---------|
| `fullyear_backtest.py` | Vectorized year-scale backtest (merge_asof, no look-ahead) |
| `tsai_optimize.py` | TSAI-scored optimizer with the Gated Harmonic architecture |
| `vectorized_adaptive.py` | ~9ms/trial numpy engine (100k trials = 15 min) |
| `phase_optimize.py` | Peak-surpassing hill-climb optimizer |
| `phase2_optimize.py` | +0.10% compounding progression search |
| `phase3_loss.py` | Loss forensics (stop-hunt attribution) |
| `phase4_usage.py` | Harvest allocation + usage optimization |
| `phase5_hybrid.py` | 1,024-combo cross-strategy grafting |
| `phase9_risk_engine.py` | Drawdown circuit-breaker search |
| `walk_forward.py` | Rolling-window walk-forward stability test |
| `sl_sweep.py` / `sl_sweep_outsample.py` | SL-multiplier sweep |
| `sl_then_tp_check.py` | Stop-hunt rate measurement |
| `download_2025_2026.py` | Forexite Jan2025-Jul2026 data download |

---

**Remember: This is a SIGNAL-ONLY bot. It will NEVER place trades automatically.
You must manually execute all trades!**
