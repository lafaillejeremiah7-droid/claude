# XAUUSD ASWP Signal Strategy — Complete Logic

This document captures every design decision behind `modules/xauusd_aswp_engine.py`,
all learned iteratively and validated against **one year of real 1-minute XAUUSD
data** (Jul 2025 → Jul 2026, Forexite, resampled to 5m/15m/1H) plus **real 10Y
TIPS real-yield data from FRED**.

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

## 4. Multi-timeframe hierarchy

```
1H  → trend gate      (EMA20 vs EMA50 sets directional bias)
15m → pullback zone   (price within ±0.5 ATR of its EMA20)
5m  → entry trigger   (touch EMA20, then a candle CLOSES back in trend
                       direction — "wait for the bounce, don't catch the knife")
```

The 5m confirmation candle (not a bare touch) is what avoids the whipsaw that
otherwise inflates the stop-out rate.

---

## 5. Stop loss = 1.0 × ATR(5m), breakeven after TP1

- Tighter stops get noise-hunted on gold; wider stops cut $/trade. 1.0×ATR +
  BE-after-TP1 sits at the **+0.53R/trade ceiling**.
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

R:R is bounded **1:1 minimum, 3:1 maximum**, decimal-precise. On the test year,
**~44% of all trades hit the full 3R final TP**, and **82% of winners ran all
the way to 3R** — gold trends hard once it clears TP1, which is exactly why the
runner weighting wins.

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
- **Volatility-aware sizing:** lots auto-shrink on high-ATR days so a single
  stop-out never exceeds ~45% of the daily limit (≈$112). Median day (ATR $3.86)
  = 0.12 lots ≈ −$48 risk; volatile day (ATR $11) = 0.10 lots ≈ −$111.
- **Spread ($0.30):** widens the effective SL and is added to every TP distance —
  a real cost, especially on tight median-ATR setups.

### Funded-account hard guards (never adapt)
- Daily loss limit 5% ($250) → auto-stop the day at 90% ($225).
- Max drawdown 8% ($400) → equity floor $4,600.
- One stop-out capped at ~45% of the daily limit so it can't end the day.
- Margin / concurrent-position cap enforced before every signal.

### XAUUSD event/session guards (minimize ugly −$)
- **News blackout:** no entries within 30 min of high-impact USD events
  (NFP/CPI/FOMC) — the #1 source of stop-blowing spikes.
- **Weekend-gap guard:** no new entries after Fri 19:00 UTC (Sunday-open gap).
- **Thin-hour avoidance:** skip the 21:00–22:00 UTC rollover window.

---

## 11. Validated performance (out-of-sample Nov 2025 → Jul 2026)

Config: 1H gate → 15m pullback → 5m trigger, SL 1.0×ATR, BE-after-TP1, multi-TP
10/20/70 at 1/2/3R, EV-gated, flat risk.

| Frequency | Win% | $/trade | $/5-days ($5k) | Max DD |
|---|---|---|---|---|
| ~324/yr (minEV 0.9, ~1/day) | 58–60% | +$81 | **+$359** | 5.4% |
| ~670/yr (~2.6/day) | 58% | +$68 | **+$977** | 5.4% |
| ~2/day cap, minEV 0.9 | 58% | +$81 | +$359 | 5.4% |

Dollar results scale linearly with account size (the strategy produces a **%
return**). $200–$1k risk per trade requires a proportionally larger account
(~$10k for $200, ~$25–50k for $500–$1k); on $5k the safe max is ~$100–130/trade,
further capped to ~$48–111 by the 0.12-lot ceiling and 1:10 margin.
