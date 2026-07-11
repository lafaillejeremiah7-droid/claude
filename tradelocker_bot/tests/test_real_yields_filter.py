"""
Hermetic tests for modules.real_yields_filter.

Covers:
- CSV parsing (DFII10's missing-value convention: blank or ".")
- Trend computation (rising / falling / flat, insufficient data)
- Direction-aware bias translation (buy/sell aliases, saturation, flat=neutral)
- RealYieldsClient caching + graceful degradation (fail-open, disk cache,
  injected fetch_fn -- no real network calls in any test)
"""
import json
import time

import pytest

from modules.real_yields_filter import (
    parse_dfii10_csv,
    compute_real_yield_trend,
    real_yield_bias_for_direction,
    RealYieldTrend,
    RealYieldsClient,
)


# ============================================================
# parse_dfii10_csv
# ============================================================

def test_parse_dfii10_csv_basic():
    csv_text = (
        "observation_date,DFII10\n"
        "2026-07-06,2.24\n"
        "2026-07-07,2.30\n"
        "2026-07-08,2.31\n"
    )
    rows = parse_dfii10_csv(csv_text)
    assert rows == [
        ("2026-07-06", 2.24),
        ("2026-07-07", 2.30),
        ("2026-07-08", 2.31),
    ]


def test_parse_dfii10_csv_skips_missing_values():
    # FRED uses a blank field (or literal ".") for holidays/missing days.
    csv_text = (
        "observation_date,DFII10\n"
        "2026-07-06,2.24\n"
        "2026-07-07,\n"
        "2026-07-08,.\n"
        "2026-07-09,2.31\n"
    )
    rows = parse_dfii10_csv(csv_text)
    assert rows == [("2026-07-06", 2.24), ("2026-07-09", 2.31)]


def test_parse_dfii10_csv_empty_input():
    assert parse_dfii10_csv("") == []
    assert parse_dfii10_csv("observation_date,DFII10\n") == []


def test_parse_dfii10_csv_malformed_lines_ignored():
    csv_text = "observation_date,DFII10\nnot,a,valid,row\n2026-07-08,2.31\n"
    rows = parse_dfii10_csv(csv_text)
    assert rows == [("2026-07-08", 2.31)]


# ============================================================
# compute_real_yield_trend
# ============================================================

def test_trend_rising():
    values = [2.00] * 5 + [2.20]  # +0.20 over the lookback window
    trend = compute_real_yield_trend(values, lookback_days=5, threshold=0.05)
    assert trend.direction == "rising"
    assert trend.change == pytest.approx(0.20)


def test_trend_falling():
    values = [2.20] * 5 + [2.00]  # -0.20 over the lookback window
    trend = compute_real_yield_trend(values, lookback_days=5, threshold=0.05)
    assert trend.direction == "falling"
    assert trend.change == pytest.approx(-0.20)


def test_trend_flat_within_threshold():
    values = [2.20, 2.21, 2.19, 2.20, 2.22, 2.21]  # tiny drift < threshold
    trend = compute_real_yield_trend(values, lookback_days=5, threshold=0.05)
    assert trend.direction == "flat"


def test_trend_no_data():
    trend = compute_real_yield_trend([], lookback_days=10)
    assert trend.direction == "flat"
    assert trend.latest_value is None
    assert trend.sample_count == 0


def test_trend_single_data_point():
    trend = compute_real_yield_trend([2.15], lookback_days=10)
    assert trend.direction == "flat"
    assert trend.latest_value == 2.15
    assert trend.sample_count == 1


def test_trend_lookback_longer_than_history_clamps_to_start():
    # Only 3 points but lookback_days=10 -> reference clamps to index 0.
    values = [2.00, 2.10, 2.30]
    trend = compute_real_yield_trend(values, lookback_days=10, threshold=0.05)
    assert trend.change == pytest.approx(0.30)
    assert trend.direction == "rising"


# ============================================================
# real_yield_bias_for_direction
# ============================================================

@pytest.mark.parametrize("direction", ["buy", "bullish", "long", "BUY", "Long"])
def test_rising_yields_oppose_gold_buys(direction):
    trend = RealYieldTrend(latest_value=2.30, change=0.30, direction="rising",
                            lookback_days=10, sample_count=20)
    label, score = real_yield_bias_for_direction(trend, direction, full_scale_change=0.30)
    assert label == "opposed"
    assert score < 0


@pytest.mark.parametrize("direction", ["sell", "bearish", "short", "SELL"])
def test_rising_yields_align_with_gold_sells(direction):
    trend = RealYieldTrend(latest_value=2.30, change=0.30, direction="rising",
                            lookback_days=10, sample_count=20)
    label, score = real_yield_bias_for_direction(trend, direction, full_scale_change=0.30)
    assert label == "aligned"
    assert score > 0


def test_falling_yields_align_with_gold_buys():
    trend = RealYieldTrend(latest_value=1.90, change=-0.30, direction="falling",
                            lookback_days=10, sample_count=20)
    label, score = real_yield_bias_for_direction(trend, "buy", full_scale_change=0.30)
    assert label == "aligned"
    assert score > 0


def test_falling_yields_oppose_gold_sells():
    trend = RealYieldTrend(latest_value=1.90, change=-0.30, direction="falling",
                            lookback_days=10, sample_count=20)
    label, score = real_yield_bias_for_direction(trend, "sell", full_scale_change=0.30)
    assert label == "opposed"
    assert score < 0


def test_flat_trend_is_always_neutral():
    trend = RealYieldTrend(latest_value=2.20, change=0.01, direction="flat",
                            lookback_days=10, sample_count=20)
    label, score = real_yield_bias_for_direction(trend, "buy")
    assert label == "neutral"
    assert score == 0.0


def test_unknown_direction_is_neutral():
    trend = RealYieldTrend(latest_value=2.30, change=0.30, direction="rising",
                            lookback_days=10, sample_count=20)
    label, score = real_yield_bias_for_direction(trend, "sideways")
    assert label == "neutral"
    assert score == 0.0


def test_score_saturates_at_full_scale_change():
    # A move far beyond full_scale_change should still clamp to magnitude 1.0.
    trend = RealYieldTrend(latest_value=3.00, change=1.00, direction="rising",
                            lookback_days=10, sample_count=20)
    label, score = real_yield_bias_for_direction(trend, "sell", full_scale_change=0.30)
    assert label == "aligned"
    assert score == pytest.approx(1.0)


def test_score_scales_linearly_below_full_scale():
    trend = RealYieldTrend(latest_value=2.15, change=0.15, direction="rising",
                            lookback_days=10, sample_count=20)
    _label, score = real_yield_bias_for_direction(trend, "sell", full_scale_change=0.30)
    assert score == pytest.approx(0.5, abs=1e-6)


# ============================================================
# RealYieldsClient
# ============================================================

def _fake_series():
    return [
        ("2026-06-25", 2.00),
        ("2026-06-26", 2.05),
        ("2026-06-29", 2.10),
        ("2026-06-30", 2.15),
        ("2026-07-01", 2.20),
        ("2026-07-02", 2.25),
        ("2026-07-06", 2.30),
    ]


def test_client_get_bias_uses_injected_fetch_fn(tmp_path):
    client = RealYieldsClient(
        cache_file=tmp_path / "cache.json",
        fetch_fn=_fake_series,
        lookback_days=6,
        threshold=0.05,
        full_scale_change=0.30,
    )
    label, score, meta = client.get_bias("buy")
    assert meta["trend_direction"] == "rising"
    assert label == "opposed"  # rising real yields oppose a gold buy
    assert score < 0
    assert meta["latest_value"] == pytest.approx(2.30)


def test_client_caches_within_ttl_and_does_not_refetch(tmp_path):
    call_count = {"n": 0}

    def counting_fetch():
        call_count["n"] += 1
        return _fake_series()

    client = RealYieldsClient(
        cache_file=tmp_path / "cache.json",
        fetch_fn=counting_fetch,
        cache_ttl_seconds=3600.0,
    )
    client.get_bias("buy")
    client.get_bias("buy")
    client.get_bias("sell")
    assert call_count["n"] == 1  # only fetched once; subsequent calls hit cache


def test_client_fails_open_when_fetch_raises_and_no_cache(tmp_path):
    def broken_fetch():
        raise RuntimeError("network down")

    client = RealYieldsClient(
        cache_file=tmp_path / "does_not_exist.json",
        fetch_fn=broken_fetch,
    )
    label, score, meta = client.get_bias("buy")
    assert label == "neutral"
    assert score == 0.0
    assert meta["reason"] == "no_data"


def test_client_falls_back_to_disk_cache_on_fetch_failure(tmp_path):
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(json.dumps({
        "fetched_at": time.time() - 100000,  # stale by wall-clock, but present
        "series": _fake_series(),
    }))

    def broken_fetch():
        raise RuntimeError("network down")

    client = RealYieldsClient(
        cache_file=cache_file,
        fetch_fn=broken_fetch,
        cache_ttl_seconds=1.0,  # force a refresh attempt immediately
    )
    label, score, meta = client.get_bias("sell")
    # Fetch failed, but the on-disk cache should be used instead of going neutral.
    assert meta.get("reason") != "no_data"
    assert meta["series_points"] == len(_fake_series())


def test_client_persists_fetched_series_to_disk(tmp_path):
    cache_file = tmp_path / "cache.json"
    client = RealYieldsClient(cache_file=cache_file, fetch_fn=_fake_series)
    client.get_bias("buy")

    assert cache_file.exists()
    saved = json.loads(cache_file.read_text())
    assert len(saved["series"]) == len(_fake_series())


def test_client_empty_fetch_result_falls_back_to_no_data(tmp_path):
    client = RealYieldsClient(cache_file=tmp_path / "cache.json", fetch_fn=lambda: [])
    label, score, meta = client.get_bias("buy")
    assert label == "neutral"
    assert meta["reason"] == "no_data"
