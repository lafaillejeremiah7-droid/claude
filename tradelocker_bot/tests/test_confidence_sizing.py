"""
Tests for confidence-scaled position sizing (Feature 1).

Covers:
- confidence_to_risk_pct mapping (8.0 -> 1%, 9.0 -> 2%, 10.0 -> 3%) and clamping.
- Scaled create_trade_setup risk amount on a $10k account.
"""
import numpy as np
import pandas as pd
import pytest

from modules.risk_management import (
    RiskManager,
    confidence_to_risk_pct,
)


# ============================================================
# confidence_to_risk_pct
# ============================================================

def test_gate_maps_to_min_pct():
    assert confidence_to_risk_pct(8.0) == pytest.approx(1.0)


def test_midpoint_maps_to_two_pct():
    assert confidence_to_risk_pct(9.0) == pytest.approx(2.0)


def test_perfect_maps_to_max_pct():
    assert confidence_to_risk_pct(10.0) == pytest.approx(3.0)


def test_clamps_below_gate_to_min():
    # Anything at/below the gate maps to the minimum, never lower.
    assert confidence_to_risk_pct(7.0) == pytest.approx(1.0)
    assert confidence_to_risk_pct(0.0) == pytest.approx(1.0)
    assert confidence_to_risk_pct(-5.0) == pytest.approx(1.0)


def test_clamps_above_ten_to_max():
    assert confidence_to_risk_pct(11.0) == pytest.approx(3.0)
    assert confidence_to_risk_pct(100.0) == pytest.approx(3.0)


def test_monotonic_increase_across_range():
    vals = [confidence_to_risk_pct(c) for c in np.linspace(8.0, 10.0, 21)]
    for earlier, later in zip(vals, vals[1:]):
        assert later >= earlier
    # Quarter point: conf 8.5 -> 1.5%
    assert confidence_to_risk_pct(8.5) == pytest.approx(1.5)


def test_custom_bounds_and_gate():
    # gate=5, min=2, max=4 -> conf 5 => 2, conf 10 => 4, conf 7.5 => 3
    assert confidence_to_risk_pct(5.0, gate=5.0, min_pct=2.0, max_pct=4.0) == pytest.approx(2.0)
    assert confidence_to_risk_pct(10.0, gate=5.0, min_pct=2.0, max_pct=4.0) == pytest.approx(4.0)
    assert confidence_to_risk_pct(7.5, gate=5.0, min_pct=2.0, max_pct=4.0) == pytest.approx(3.0)


# ============================================================
# Scaled create_trade_setup on $10k
# ============================================================

def _make_uptrend_df(n=80, start=100.0, step=0.5):
    """Build a simple rising OHLCV frame suitable for a 'buy' setup."""
    closes = np.array([start + i * step for i in range(n)], dtype=float)
    opens = closes - step * 0.5
    highs = closes + step * 0.5
    lows = closes - step * 0.75
    vol = np.full(n, 1000.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vol,
        }
    )


@pytest.fixture
def rm(tmp_path):
    # Isolated stats file so the test never touches live stats.
    return RiskManager(stats_file=tmp_path / "daily_stats.json")


def _risk_for_confidence(rm, df, equity, confidence):
    setup = rm.create_trade_setup(
        symbol="BTCUSD",
        direction="buy",
        entry_price=df["close"].iloc[-1],
        df_5m=df,
        account_equity=equity,
        trend_confidence=0.8,
        pip_size=0.01,
        lot_size=1.0,
        min_lot=0.0001,
        lot_step=0.0001,
        confidence=confidence,
    )
    return setup


def test_scaled_risk_amounts_on_10k(rm):
    df = _make_uptrend_df()
    equity = 10_000.0

    for confidence, expected_pct in [(8.0, 1.0), (9.0, 2.0), (10.0, 3.0)]:
        setup = _risk_for_confidence(rm, df, equity, confidence)
        assert setup.valid
        target = equity * (expected_pct / 100.0)  # 100, 200, 300
        # Position size rounds DOWN to lot_step, so actual risk is <= target but
        # within one lot-step of SL distance of it.
        tolerance = 0.0001 * setup.sl_distance + 1e-6
        assert setup.risk_amount <= target + 1e-6
        assert setup.risk_amount >= target - tolerance


def test_scaled_risk_is_monotonic(rm):
    df = _make_uptrend_df()
    equity = 10_000.0
    r8 = _risk_for_confidence(rm, df, equity, 8.0).risk_amount
    r9 = _risk_for_confidence(rm, df, equity, 9.0).risk_amount
    r10 = _risk_for_confidence(rm, df, equity, 10.0).risk_amount
    assert r8 < r9 < r10
    # ~1% / ~2% / ~3% => roughly doubling / tripling relative to r8
    assert r9 == pytest.approx(2 * r8, rel=0.02)
    assert r10 == pytest.approx(3 * r8, rel=0.02)


def test_no_confidence_falls_back_to_fixed(rm):
    """Without a confidence score, sizing uses the fixed RISK_PERCENT (2%)."""
    df = _make_uptrend_df()
    equity = 10_000.0
    setup = rm.create_trade_setup(
        symbol="BTCUSD",
        direction="buy",
        entry_price=df["close"].iloc[-1],
        df_5m=df,
        account_equity=equity,
        trend_confidence=0.8,
        min_lot=0.0001,
        lot_step=0.0001,
    )
    assert setup.valid
    target = equity * (rm.risk_percent / 100.0)  # 2% -> 200
    tolerance = 0.0001 * setup.sl_distance + 1e-6
    assert target - tolerance <= setup.risk_amount <= target + 1e-6
