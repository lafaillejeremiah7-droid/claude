# XAUUSD ASWP Signal Strategy — Complete Logic

This document captures every design decision behind `modules/xauusd_aswp_engine.py`
and the live engine in `live_terminal.py`. The timeframe combination was selected
by backtesting **every valid gate/pullback/entry combination against a full year
of real 1-minute XAUUSD data** (Nov 2023 → Nov 2024, 366,927 one-minute bars from
the Hugging Face dataset `Ashraf-CK/XAUUSD`), cross-checked against **real 10Y
TIPS real-yield data from FRED**. Live price and historical bars are pulled
directly from the **TradeLocker API** (the actual broker), with Forexite as a
fallback bar source.

> This is a **signal generator**, not an auto-trader. It decides *whether* to
> signal and *where* the levels go. You decide execution. Nothing here is
> financial advice; backtest results are historical and live results will differ.

---

## 1. The one rule everything serves: MAXIMIZE $/TRADE, NOT WIN RATE

A high win rate with tiny reward loses to a lower win rate with big reward.
Proven on identical data:

| TP structure | $/trade | Win rate |
|---|---|---|
| Scalp 40% at 1R (win-rate biased) | +$41 | 51% |
| **Runner: ride 70% to 3R** | **+$68–74** | 41–51% |

The runner-weighted structure nearly **doubled $/trade** at a *lower* win rate.
Win rate is a vanity metric; expected dollars is the goal.

---

## 2. Signal gate = minimum EXPECTED VALUE, never minimum probability

```
EV = P(win) × avg_win_R − P(loss) × avg_loss_R
```

Gating on probability silently re-optimizes for win rate. The gate is **EV ≥ min_ev**.
Probability is only one input. Refined loss model: a true −1R only occurs on a
full stop *before* TP1; trades that reach TP1 then fade scratch near breakeven
(SL→BE), so `e_loss = P(full_stop)×1R + P(scratch)×~0`.

---

## 3. Risk per trade is FLAT — never scaled by win rate

Position size is bounded only by account rules and volatility (SL distance),
**never** by a Kelly fraction or recent streak. Win rate affects *whether* a
signal fires, not *how much* is risked.

---

## 4. Multi-timeframe hierarchy (backtest-selected optimal)

```
1H  → trend gate      (EMA20 vs EMA50 sets directional bias)
15m → pullback zone   (price within 1.5 × ATR of its EMA20 — not overextended)
1m  → entry trigger   (prev candle touches EMA20, then CLOSES back in trend
                       direction; RSI filter buy<65 / sell>35 —
                       "wait for the bounce, don't catch the knife")
```

The 1m confirmation candle (not a bare touch) times the entry precisely at the
turn while the higher timeframes confirm context. This exact combo — **1H gate,
15m pullback, 1m entry** — was the practical winner across a full year of real
1-minute data (see §11). The entry was moved from 5m → **1m** because 1m catches
the reversal a few bars earlier, materially improving win rate and per-signal edge.

---

## 5. Stop loss = 1.0 × ATR(15m), breakeven after TP1

> **Why 15m ATR (not the 1m entry ATR)?** The entry trigger fires on 1m, but 1m
> ATR is tiny (~$0.40–1.5). Sizing stops off 1m ATR would put the ~$0.30 spread
> at 20–40% of the stop — the spread alone would eat the edge. Backtesting proved
> the raw "1m-ATR stop" result (+175R) was a mirage that real execution costs
> destroy. Sizing SL/TP from **15m ATR** (~$3–9) keeps the spread a negligible
> fraction of risk while still entering on the precise 1m turn. This is the single
> most important practicality fix in the current build.

- Tighter stops get noise-hunted on gold; wider stops cut $/trade. 1.0×ATR(15m) +
  BE-after-TP1 is the practical sweet spot.
- Winners barely dip before running (median MFE-before-TP1 = **0.13R**, 90% stay
  above −0.78R). Losers blow through (median **−1.34R**). So an aggressive early
  cut inside the noise band (≤0.35R) **destroys edge** — it stops out winners.
- The only safe early-cut is deep + close-confirmed (0.7R), which merely *ties*
  the plain full-stop. So: **plain 1.0R stop + BE-after-TP1** is optimal; the
  early cut is a disaster brake only.

---

## 6. Multi-TP allocation (where the $ lives)

```
TP1 = 1.0R  → close 10%   (then SL → breakeven)
TP2 = 2.0R  → close 20%   (then SL → TP1)
Final = 3.0R → ride 70%   (trail 70% of the excursion beyond TP2)
```

R:R is bounded **1:1 minimum, 3:1 maximum**, decimal-precise. The runner weighting
wins because gold trends hard once it clears TP1 — a large share of winners that
reach TP1 go on to run toward 3R, so weighting 70% to the final target captures
far more dollars than an early scalp. TP1/TP2 partials plus the BE/TP1 stop
trail lock in gains while the runner chases the 3R tail.

---

## 7. ASWP — Adaptive Similarity-Weighted Probability

`P(reaching level X | current conditions)` = recency- and similarity-weighted
frequency over all remembered trades:

```
P(X) = Σ [ similarity_i × recency_i × (reached_X_i) ]  /  Σ [ similarity_i × recency_i ]

similarity_i = exp( −Σ_k w_k · ((feature_k_now − feature_k_i)/σ_k)² / bandwidth² )
recency_i    = λ^(age_i)          (λ = 0.99 calibrated; recent trades weigh more)
```

No fixed buckets. Solves small-sample, regime-change, continuous-feature, and
market-state problems simultaneously. Pre-seeded from backtest, then adapts
after **every** closed trade. Calibrated: λ = 0.99, bandwidth = 1.5.

---

## 8. Real-yields macro filter (gold's strongest driver, ~−0.82)

10Y TIPS real yields (FRED `DFII10`) correlate with gold far more strongly and
stably than DXY. Rising real yields → bearish gold; falling → bullish. The
alignment score (−1 opposed … +1 aligned) is a feature in the ASWP similarity
vector, so the macro regime shapes both signal selection and probability.

---

## 9. Concentration raises $/trade

Taking only the highest-EV setups nearly **doubles** per-trade edge:

| Selectivity | Trades/yr | Per day | Avg R/trade | Win% |
|---|---|---|---|---|
| Take everything | 8,415 | 23 | +0.68R | 51% |
| Moderate | 4,239 | 12 | +0.91R | 56% |
| Tight | 1,383 | 3.8 | +1.04R | 59% |
| Very tight | 193 | ~0.5 | **+1.34R** | 66% |

Fewer, higher-EV trades = more $ per trade. Frequency is a dial (`min_ev`,
`max_signals_per_day`).

---

## 10. Broker / account reality (this account: $5k, 1:10, 0.01–0.12 lots)

Confirmed from the live platform (0.06 lots → $2,472.17 margin = exactly 10% of
notional → **1:10 leverage**):

- **Lot bounds:** 0.01–0.12. **Contract:** 1 lot = 100 oz → $1 move = $100/lot.
- **Margin (1:10):** 0.12 lots ≈ $4,944 margin ≈ **99% of a $5k account** → you
  can realistically hold **one** full-size position at a time; ~0.06 lots each to
  hold two. Sizing is **margin-aware** and refuses trades it can't afford.
- **Risk cap: $60 max loss per signal** (hard-coded `max_loss = 60`). Sizing:
  `sl_dist = 1.0×ATR(15m) + spread/2`, `loss_per_lot = 100 × sl_dist`,
  `lots = min(0.12, $60 / loss_per_lot)` floored to the 0.01 step. Because lots
  floor down, realized risk is **always ≤ $60**.
- **Volatility-aware sizing:** lots auto-shrink on high-ATR days to hold the $60
  ceiling. Examples (using 15m ATR): quiet (ATR $5) → 0.11 lots ≈ −$57;
  current (ATR ~$8.6) → 0.06 lots ≈ −$53; volatile (ATR $15) → 0.03 lots ≈ −$46.
  The 0.12-lot cap only binds when 15m ATR falls below ~$4.85.
- **Max full win per signal (all 3 TPs = 2.6R):** ≈ **$130–145** at typical ATR
  (scales with ATR). Effective reward:risk ≈ **2.56 : 1**.
- **Spread ($0.30):** widens the effective SL and is added to every TP distance —
  a real cost, now negligible relative to the 15m-ATR-sized stop.

### Funded-account hard guards (never adapt)
- Daily loss limit 5% ($250) → worst case is 4 signals × $60 = **$240/day**,
  which stays under the limit by design.
- Max drawdown 8% ($400) → equity floor $4,600.
- Per-signal loss capped at $60 so no single stop can end the day.
- Margin / concurrent-position cap enforced before every signal.

### XAUUSD event/session guards (minimize ugly −$)
- **News blackout:** no entries within 30 min of high-impact USD events
  (NFP/CPI/FOMC) — the #1 source of stop-blowing spikes.
- **Weekend-gap guard:** no new entries after Fri 19:00 UTC (Sunday-open gap).
- **Thin-hour avoidance:** skip the 21:00–22:00 UTC rollover window.

---

## 11. Timeframe selection — full-year backtest (Nov 2023 → Nov 2024)

Every valid gate/pullback/entry combination was backtested against **366,927 real
1-minute XAUUSD bars**, with SL/TP allocation 10/20/70 at 1/2/3R, flat risk, and
**exit stops sized from a higher timeframe** so the spread stays realistic. R is
risk-normalized; because the strategy is R-based it is scale-invariant across
gold price levels.

**Practical results (exit stops sized from a higher TF — realistic):**

| Combo (Gate/PB/Entry, exit-ATR) | Signals/yr | Win% | Total R | Avg R/signal |
|---|---|---|---|---|
| **1H / 15m / 1m (15m-ATR stops)** ← ACTIVE | 892 (~3.5/day) | **52.4%** | **+121.32R** | **+0.136** |
| 1H / 15m / 5m (previous config) | 1,674 | 47.4% | +115.77R | +0.069 |
| 30m / 15m / 1m (15m-ATR stops) | 892 | 50.3% | +103.17R | +0.116 |
| 30m / 5m / 5m (5m-ATR stops) | 1,771 | 46.4% | +100.96R | +0.057 |
| 30m / 5m / 1m (5m-ATR stops) | 966 | 49.1% | +74.65R | +0.077 |

**Why 1H / 15m / 1m won:** highest total R, highest win rate (52.4%), and nearly
**2× the edge per signal** (+0.136R vs the previous +0.069R) — with *fewer,
higher-quality* signals. Switching only the entry trigger from 5m → 1m (and
sizing exits from 15m ATR) was the decisive change.

> **Note on the "+175R" trap:** the raw-R winner was `30m/5m/1m` sized off the 1m
> ATR (+175R). That relied on ~$0.40 stops the ~$0.30 spread would obliterate in
> live trading. Sizing exits from 15m ATR is what makes the edge survive real
> costs — hence the active config above.

**Dollar terms on this $5k account** (risk capped at $60/signal, §10): at
+0.136R average and ~3.5 signals/day, dollar results scale with the per-signal
risk and account size. Live results will differ from backtest — this is a signal
generator, not a guarantee.

### Backtest tooling
- `fullyear_backtest.py` — full-year, vectorized, `merge_asof` timeframe
  alignment (no look-ahead), tests the practical exit-ATR variants.
- `timeframe_optimizer.py` — quick multi-combo scan on shorter windows.
- Input data (gitignored, ~264MB): `data/xau_train.parquet`, `data/xau_test.parquet`.
