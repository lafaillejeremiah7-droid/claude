# LOOK-AHEAD AUDIT — Retraction of Phase 21-23 "predictive brain" results

## What happened
Phases 22-23 reported a brain with OOS AUC 0.594 and a HIGH-CONF gate producing
$52-56/wk at 70-73% WR. **These results are INVALID due to look-ahead bias.**

## The leak
`phase22_pathshape.path_features()` computed "path shape" from `H[i, :10]` /
`L[i, :10]` / `C[i, :10]` — the FIRST 10 BARS OF THE FORWARD (post-entry) WINDOW.
Those are bars that occur AFTER the trade is entered. A trade already moving
favorably in its first 10 minutes is obviously more likely to win, so these
features "predicted" the outcome by peeking at it. The backtest gate then used
these contaminated predictions, inflating $/wk.

## The audit (real data, both datasets, OOS = 2025-26)
| Feature set | OOS AUC |
|---|---|
| WITH forward-bar path features (contaminated) | 0.594 |
| NO path features (macro + intrinsic + interactions, entry-time only) | 0.515 |
| PRE-ENTRY path (10 bars BEFORE signal, no leak) + macro + intrinsic | 0.506 |

## Honest conclusion
With clean, entry-time-only information, out-of-sample AUC is ~0.51 — no
predictive power. This is the 7th independent confirmation that XAU/USD trade
outcomes are NOT predictable at entry from available features. The apparent
"breakthrough" was leakage.

## Corrected status
- Brain (ASWP / logistic / HERCULES Phase-5 historical edge): **F** — no real edge.
- Strategy (session + geometry + management): honest net after friction **~$29-33/wk**,
  64-70% WR. This edge is REAL (payoff structure + management), not prediction.
- Retracted: the $52-56/wk and A- tier claims from phases 22-23.

## What is still true and valuable
- The geometry/management/session edge is real and positive-expectancy.
- HERCULES as an ARCHITECTURE (state estimation, EV>0 gating, execution-quality
  checks, no-overfit promotion) improves ROBUSTNESS and honesty — but cannot
  manufacture predictive alpha that is not present in the data.
