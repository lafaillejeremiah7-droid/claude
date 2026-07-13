"""
TSAI - Trading Strategy Adaptability Index
==========================================

A Gated Harmonic Mean scoring framework for judging whether a backtested
XAUUSD configuration is actually fit to run on the AquaFunded prop-firm
account - not just "profitable on paper".

    TSAI = G * HM(C, S)                        HM(a, b) = 2ab / (a + b)

The harmonic mean deliberately anchors the final score to the WEAKER of the
two components. A brilliant compatibility score cannot paper over a fatal
suitability (survival-distance) problem: if C = 30 but S = 1, the blended
score collapses to ~1.93 rather than averaging to a comfortable 15.5.

--------------------------------------------------------------------------
Component 1 - Structural Security Gate  (G in {0, 1})
--------------------------------------------------------------------------
Three hard binary gates. Fail ANY one and G = 0, which zeroes the entire
index regardless of how good C and S look.

    G = g1 * g2 * g3

    g1  Volatility Guard    : SL > 1.2 * ATR
        Ultra-tight stops on Gold get hunted by ordinary noise + spread
        widening. If the optimizer tries to buy a huge R:R with a razor
        stop, this trips.

    g2  Total Drawdown Shield : DD_total_max < $350
        AquaFunded terminates the account at a $400 total loss. Stay clear.

    g3  Daily Safety Valve    : DD_day_max < $200
        The hard daily loss limit is $250. One catastrophic day trips this.

--------------------------------------------------------------------------
Component 2 - Normalized Compatibility Score  (C >= 0)
--------------------------------------------------------------------------
How cleanly the strategy extracts profit from the asset, insulated against
negative-EV math errors and pip-scale ceilings.

    C = max(0, k * (EV / (Spread + Commission)) * (SL / ATR))

      Friction ratio  EV / (Spread + Commission) : edge vs broker fees.
      Noise cushion   SL / ATR                    : mathematical breathing room.
      Scaling const   k = 5                        : normalizes the fractional
                                                     metal-pip scale so C does
                                                     not create a glass ceiling.
    The max(0, ...) floor stops a negative-EV regime from producing a
    negative C that would crash the harmonic mean.

--------------------------------------------------------------------------
Component 3 - Nested Harmonic Suitability Score  (S >= 0)
--------------------------------------------------------------------------
Statistical distance from the two liquidation walls, itself harmonically
blended so the closer wall dominates.

    S       = HM(S_day, S_total)
    S_day   = $250 / DD_day_max
    S_total = $400 / DD_total_max
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd


# --- prop-firm / structural constants (AquaFunded) -------------------------
DAILY_LOSS_LIMIT = 250.0        # hard daily loss wall (used by S_day)
TOTAL_LOSS_LIMIT = 400.0        # hard total loss wall (used by S_total)
DAILY_GATE = 200.0              # g3 must stay strictly below this
TOTAL_GATE = 350.0              # g2 must stay strictly below this
VOL_GATE_MULT = 1.2             # g1: SL must exceed 1.2 * ATR
K_SCALE = 5.0                   # compatibility scaling constant

# A sentinel used when a drawdown is exactly 0 (perfect survival). Avoids a
# 1/0 -> inf. A giant-but-finite distance keeps the harmonic mean well-defined
# while still expressing "effectively no risk on this wall".
_ZERO_DD_DISTANCE = 1e6


def _harmonic_mean(a: float, b: float) -> float:
    """2ab / (a + b), guarded against the a + b == 0 division-by-zero."""
    denom = a + b
    if denom <= 0:
        return 0.0
    return (2.0 * a * b) / denom


@dataclass
class TSAIResult:
    tsai: float

    # gate
    G: int
    g1_volatility: int
    g2_total_dd: int
    g3_daily_dd: int

    # compatibility
    C: float
    friction_ratio: float
    noise_cushion: float

    # suitability
    S: float
    S_day: float
    S_total: float

    # raw inputs echoed back for transparency / audit
    inputs: dict = field(default_factory=dict)

    def verdict(self) -> str:
        if self.G == 0:
            fails = []
            if not self.g1_volatility:
                fails.append("g1 volatility guard (SL too tight vs ATR)")
            if not self.g2_total_dd:
                fails.append("g2 total-drawdown shield (>= $350)")
            if not self.g3_daily_dd:
                fails.append("g3 daily safety valve (>= $200)")
            return "REJECTED by security gate: " + "; ".join(fails)
        if self.tsai >= 15:
            band = "STRONG"
        elif self.tsai >= 8:
            band = "ACCEPTABLE"
        elif self.tsai >= 4:
            band = "MARGINAL"
        else:
            band = "WEAK"
        return f"{band} (TSAI = {self.tsai:.2f})"

    def report(self) -> str:
        lines = [
            "=" * 60,
            "  TSAI - Trading Strategy Adaptability Index",
            "=" * 60,
            f"  Structural Security Gate  G = {self.G}",
            f"      g1 volatility guard  (SL > 1.2*ATR)   : {self.g1_volatility}",
            f"      g2 total DD shield   (< ${TOTAL_GATE:.0f})    : {self.g2_total_dd}",
            f"      g3 daily DD valve    (< ${DAILY_GATE:.0f})    : {self.g3_daily_dd}",
            "-" * 60,
            f"  Compatibility  C = {self.C:.3f}",
            f"      friction ratio  EV/(spread+comm)      : {self.friction_ratio:.3f}",
            f"      noise cushion   SL/ATR                : {self.noise_cushion:.3f}",
            "-" * 60,
            f"  Suitability    S = {self.S:.3f}",
            f"      S_day    ($250 / daily DD)            : {self.S_day:.3f}",
            f"      S_total  ($400 / total DD)            : {self.S_total:.3f}",
            "-" * 60,
            f"  TSAI = G * HM(C, S) = {self.tsai:.3f}",
            f"  Verdict: {self.verdict()}",
            "=" * 60,
        ]
        return "\n".join(lines)


def compute_tsai(
    *,
    sl: float,
    atr: float,
    dd_total_max: float,
    dd_day_max: float,
    ev: float,
    spread: float,
    commission: float = 0.0,
    k: float = K_SCALE,
) -> TSAIResult:
    """Evaluate the full TSAI index from already-consistent numeric inputs.

    All monetary inputs (ev, spread, commission, dd_*) MUST be in the same
    currency unit (dollars). sl and atr must be in the same price unit as
    each other (their ratio is dimensionless).

    Parameters
    ----------
    sl            representative stop-loss distance (price units)
    atr           representative ATR over the same window (price units)
    dd_total_max  worst peak-to-trough equity drop over the full history ($)
    dd_day_max    worst single-day equity drop ($)
    ev            expected value per trade, net of costs ($)
    spread        per-trade spread cost ($)
    commission    per-trade commission cost ($), default 0
    k             compatibility scaling constant, default 5
    """
    # ---- Component 1: Structural Security Gate --------------------------
    g1 = int(sl > VOL_GATE_MULT * atr)
    g2 = int(dd_total_max < TOTAL_GATE)
    g3 = int(dd_day_max < DAILY_GATE)
    G = g1 * g2 * g3

    # ---- Component 2: Normalized Compatibility Score --------------------
    friction_cost = spread + commission
    if friction_cost <= 0:
        # No modelled friction -> undefined ratio. Treat as no edge amplification.
        friction_ratio = 0.0
    else:
        friction_ratio = ev / friction_cost
    noise_cushion = sl / atr if atr > 0 else 0.0
    C = max(0.0, k * friction_ratio * noise_cushion)

    # ---- Component 3: Nested Harmonic Suitability Score -----------------
    day_dd = dd_day_max if dd_day_max > 0 else 1.0 / _ZERO_DD_DISTANCE
    tot_dd = dd_total_max if dd_total_max > 0 else 1.0 / _ZERO_DD_DISTANCE
    S_day = DAILY_LOSS_LIMIT / day_dd
    S_total = TOTAL_LOSS_LIMIT / tot_dd
    S = _harmonic_mean(S_day, S_total)

    # ---- Master equation ------------------------------------------------
    tsai = G * _harmonic_mean(C, S)

    return TSAIResult(
        tsai=round(float(tsai), 4),
        G=G, g1_volatility=g1, g2_total_dd=g2, g3_daily_dd=g3,
        C=round(float(C), 4), friction_ratio=round(float(friction_ratio), 4),
        noise_cushion=round(float(noise_cushion), 4),
        S=round(float(S), 4), S_day=round(float(S_day), 4), S_total=round(float(S_total), 4),
        inputs={
            "sl": sl, "atr": atr, "dd_total_max": dd_total_max,
            "dd_day_max": dd_day_max, "ev": ev, "spread": spread,
            "commission": commission, "k": k,
        },
    )


# ---------------------------------------------------------------------------
# Metric extraction from a backtest run
# ---------------------------------------------------------------------------
# The vectorized engine (vectorized_adaptive.py) hands back per-signal R,
# per-signal sl_dist, plus the candidate metadata (entries, x_atr, dt). From
# those we reconstruct dollar PnL exactly the way dollar_stats_vectorized does,
# then derive the seven TSAI inputs - including the DAILY drawdown, which the
# existing stats helper does not compute.

STARTING_EQUITY = 5000.0
CONTRACT = 100.0
LEVERAGE = 10.0
SPREAD_PRICE = 0.30            # gold points, matches the engine's SPREAD


def _lots_and_risk(sl_dist: np.ndarray, entries: np.ndarray, equity0: float):
    """Reproduce the engine's position sizing to get per-trade lots + $ risk."""
    loss_per_lot = CONTRACT * sl_dist
    lots = np.minimum(0.12, 60.0 / loss_per_lot)
    lots = np.maximum(0.01, np.floor(lots * 100) / 100)
    margin = lots * CONTRACT * entries / LEVERAGE
    over = margin > equity0 * 0.95
    if over.any():
        alt = np.maximum(0.01, np.floor((equity0 * 0.95 * LEVERAGE / (CONTRACT * entries)) * 100) / 100)
        lots = np.where(over, alt, lots)
    risk = lots * loss_per_lot
    return lots, risk


def daily_drawdown(dt: np.ndarray, pnl: np.ndarray, equity0: float = STARTING_EQUITY) -> float:
    """Worst single-day equity decline, measured from each day's OPENING
    equity down to its intraday low (the way a prop-firm daily loss limit is
    enforced: it resets at the start of every trading day)."""
    if len(pnl) == 0:
        return 0.0
    s = pd.Series(pnl, index=pd.to_datetime(dt)).sort_index()
    eq = equity0 + s.cumsum()
    day = eq.index.normalize()

    worst = 0.0
    # equity at the close of the previous day = opening equity of this day
    prev_close = equity0
    for _, grp in eq.groupby(day):
        start_equity = prev_close
        intraday_low = grp.min()
        loss = start_equity - intraday_low        # positive => a loss on the day
        worst = max(worst, float(loss))
        prev_close = grp.iloc[-1]
    return round(worst, 2)


def tsai_from_backtest(
    cands: dict,
    r: np.ndarray,
    sl_dist: np.ndarray,
    *,
    commission_per_lot: float = 0.0,
    equity0: float = STARTING_EQUITY,
    k: float = K_SCALE,
) -> TSAIResult:
    """Derive the seven TSAI inputs from a vectorized backtest trial and score it.

    Parameters
    ----------
    cands   the candidate dict from vectorized_adaptive.get_candidates()
    r       per-signal R multiple (from simulate_trial)
    sl_dist per-signal stop distance in price units (from simulate_trial)
    commission_per_lot  round-turn commission in $ per lot (0 if not modelled)
    """
    entries = cands["entries"]
    x_atr = cands["x_atr"]
    dt = cands["dt"]

    lots, risk = _lots_and_risk(sl_dist, entries, equity0)
    pnl = r * risk

    # --- drawdowns (dollars) ---
    order = np.argsort(dt)
    pnl_sorted = pnl[order]
    eq = equity0 + np.cumsum(pnl_sorted)
    peak = np.maximum.accumulate(eq)
    dd_total_max = round(float(np.max(peak - eq)) if len(eq) else 0.0, 2)
    dd_day_max = daily_drawdown(dt, pnl, equity0)

    # --- expected value per trade (dollars, net of the modelled outcome) ---
    ev = float(pnl.mean()) if len(pnl) else 0.0

    # --- per-trade friction in dollars ---
    #   spread cost  = spread(price) * contract * avg lots
    #   commission   = commission_per_lot * avg lots (round-turn)
    avg_lots = float(lots.mean()) if len(lots) else 0.0
    spread_cost = SPREAD_PRICE * CONTRACT * avg_lots
    commission_cost = commission_per_lot * avg_lots

    # --- representative SL / ATR (price units) ---
    sl_rep = float(np.mean(sl_dist)) if len(sl_dist) else 0.0
    atr_rep = float(np.mean(x_atr)) if len(x_atr) else 0.0

    return compute_tsai(
        sl=sl_rep, atr=atr_rep,
        dd_total_max=dd_total_max, dd_day_max=dd_day_max,
        ev=ev, spread=spread_cost, commission=commission_cost, k=k,
    )


# ---------------------------------------------------------------------------
# Demo / self-check: score the saved best-adaptive params against the baseline.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import fullyear_backtest as fb
    import vectorized_adaptive as va

    print("Loading full-year data + building candidate signal set...\n")
    df1m = fb.load_data()
    cands = va.get_candidates(df1m)
    print(f"Candidates: {cands['n']} signals\n")

    # Static baseline
    r0, sl0, _ = va.simulate_trial(cands, base_sl=1.0, vol_lo=1.0, vol_hi=1.0,
                                   trend_gain=0.0, tp_lo=1.0, tp_hi=1.0)
    base = tsai_from_backtest(cands, r0, sl0)
    print("STATIC BASELINE (1.0x ATR)")
    print(base.report())
    print()

    # Saved best-adaptive params
    try:
        with open("best_adaptive_params.json") as f:
            p = json.load(f)
        r1, sl1, _ = va.simulate_trial(
            cands, base_sl=p["base_sl"], vol_lo=p["vol_lo"], vol_hi=p["vol_hi"],
            trend_gain=p["trend_gain"], tp_lo=p["tp_lo"], tp_hi=p["tp_hi"],
        )
        best = tsai_from_backtest(cands, r1, sl1)
        print("BEST ADAPTIVE (best_adaptive_params.json)")
        print(best.report())
    except FileNotFoundError:
        print("best_adaptive_params.json not found - skipping adaptive score.")
