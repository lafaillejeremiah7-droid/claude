"""
Real Interest Rate ("Real Yields") Correlation Filter for XAUUSD

Real yields (the 10-Year Treasury Inflation-Protected Security yield, FRED
series ``DFII10``) are gold's single strongest macro driver -- empirically
stronger and far more stable than the US Dollar Index (DXY). Gold pays no
yield, so when inflation-adjusted yields rise, the opportunity cost of
holding gold rises and gold tends to sell off; when real yields fall (or go
negative), gold becomes relatively more attractive.

Historical inverse correlation: real yields vs. gold ~ -0.82, versus DXY's
much weaker and far less stable correlation (commonly cited as inverse, but
documented to swing positive for extended stretches, and to fully decouple
from gold for a year or more -- e.g. 2023-2024 when gold and DXY both rallied
simultaneously).

This module replaces the placeholder "DXY correlation filter" concept with a
real-yields-based filter:

1. Fetches the DFII10 series from FRED's public CSV endpoint (no API key
   required), caches it on disk, and refreshes on a TTL (the underlying data
   is daily-resolution, so refreshing every scan cycle is unnecessary).
2. Computes a real-yield MOMENTUM/trend over a lookback window: rising,
   falling, or flat.
3. Translates that trend into a directional bias for a gold trade: rising
   real yields => bearish gold bias (opposes buys, aligns with sells);
   falling real yields => bullish gold bias (aligns with buys, opposes
   sells).
4. Exposes pure, hermetically-testable functions (`parse_dfii10_csv`,
   `compute_real_yield_trend`, `real_yield_bias_for_direction`) plus a
   `RealYieldsClient` that handles the network fetch + caching + graceful
   degradation.

FAIL-OPEN: if FRED is unreachable and no cached data exists, the filter
returns a NEUTRAL bias and never blocks trading -- this is a confirmation
layer on top of the core technical strategy, not a hard dependency on network
availability.
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# FRED's public "graph csv" endpoint requires no API key and returns a plain
# two-column CSV: observation_date,DFII10
FRED_DFII10_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"

DEFAULT_CACHE_FILE = Path(__file__).parent.parent / "logs" / "real_yields_cache.json"

_BUY_ALIASES = {"buy", "bullish", "long"}
_SELL_ALIASES = {"sell", "bearish", "short"}


# ============================================================
# PURE FUNCTIONS (hermetic, no I/O)
# ============================================================

def parse_dfii10_csv(csv_text: str) -> List[Tuple[str, float]]:
    """
    Parse FRED's DFII10 CSV text into an ordered list of (date, value) pairs.

    Rows with missing values (FRED uses a blank field or ".") are skipped.
    The input is assumed to already be in ascending date order (as FRED
    returns it); this function does not re-sort.
    """
    rows: List[Tuple[str, float]] = []
    lines = csv_text.strip().splitlines()
    if len(lines) < 2:
        return rows

    for line in lines[1:]:  # skip header row
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue
        date_str, value_str = parts[0].strip(), parts[1].strip()
        if not date_str or not value_str or value_str == ".":
            continue
        try:
            value = float(value_str)
        except ValueError:
            continue
        rows.append((date_str, value))

    return rows


@dataclass
class RealYieldTrend:
    """Computed momentum of the real-yield series over a lookback window."""
    latest_value: Optional[float]
    change: float  # latest - reference, in percentage points
    direction: str  # 'rising' | 'falling' | 'flat'
    lookback_days: int
    sample_count: int


def compute_real_yield_trend(
    values: List[float],
    lookback_days: int = 10,
    threshold: float = 0.05,
) -> RealYieldTrend:
    """
    Compute the real-yield trend from an ordered (oldest -> newest) list of
    daily values.

    ``change`` is the difference (in percentage points) between the latest
    value and the value ``lookback_days`` observations back. Classified as
    'rising' if change >= threshold, 'falling' if change <= -threshold,
    otherwise 'flat'.

    Degrades gracefully on insufficient data (0 or 1 samples) -> 'flat'.
    """
    if not values:
        return RealYieldTrend(None, 0.0, "flat", lookback_days, 0)

    latest = values[-1]

    if len(values) < 2:
        return RealYieldTrend(latest, 0.0, "flat", lookback_days, len(values))

    ref_idx = max(0, len(values) - 1 - lookback_days)
    reference = values[ref_idx]
    change = round(latest - reference, 4)

    if change >= threshold:
        direction = "rising"
    elif change <= -threshold:
        direction = "falling"
    else:
        direction = "flat"

    return RealYieldTrend(latest, change, direction, lookback_days, len(values))


def real_yield_bias_for_direction(
    trend: RealYieldTrend,
    direction: str,
    full_scale_change: float = 0.30,
) -> Tuple[str, float]:
    """
    Translate a real-yield trend into a bias label + continuous alignment
    score (-1.0 fully opposed .. 0.0 neutral .. +1.0 fully aligned) for a
    proposed trade ``direction`` ('buy'/'bullish'/'long' or
    'sell'/'bearish'/'short').

    Rising real yields -> bearish for gold (opposes buys, aligns with sells).
    Falling real yields -> bullish for gold (aligns with buys, opposes sells).
    Flat -> always neutral (0.0).

    The magnitude scales linearly with the size of the real-yield move,
    saturating at +/-1.0 once |change| >= full_scale_change (percentage
    points).
    """
    direction_norm = (direction or "").lower()
    is_buy = direction_norm in _BUY_ALIASES
    is_sell = direction_norm in _SELL_ALIASES

    if not (is_buy or is_sell) or trend.direction == "flat" or full_scale_change <= 0:
        return "neutral", 0.0

    magnitude = min(1.0, abs(trend.change) / full_scale_change)
    # +1 = bullish-for-gold real-yield regime (falling), -1 = bearish (rising)
    gold_bias_sign = 1.0 if trend.direction == "falling" else -1.0
    directional_sign = 1.0 if is_buy else -1.0
    alignment_score = round(gold_bias_sign * directional_sign * magnitude, 4)

    if alignment_score > 0.05:
        label = "aligned"
    elif alignment_score < -0.05:
        label = "opposed"
    else:
        label = "neutral"

    return label, alignment_score


# ============================================================
# CLIENT (network I/O + caching + graceful degradation)
# ============================================================

class RealYieldsClient:
    """
    Fetches, caches, and interprets the 10Y TIPS real-yield series to produce
    a directional bias for gold trades.

    ``fetch_fn`` can be injected (e.g. in tests) to avoid any network call;
    it must return a list of ``(date_str, value)`` tuples in ascending date
    order, matching ``parse_dfii10_csv``'s output shape.
    """

    def __init__(
        self,
        cache_ttl_seconds: float = 6 * 3600.0,
        lookback_days: int = 10,
        threshold: float = 0.05,
        full_scale_change: float = 0.30,
        fetch_timeout: float = 10.0,
        cache_file: Optional[Path] = None,
        fetch_fn: Optional[Callable[[], List[Tuple[str, float]]]] = None,
    ):
        self.cache_ttl_seconds = cache_ttl_seconds
        self.lookback_days = lookback_days
        self.threshold = threshold
        self.full_scale_change = full_scale_change
        self.fetch_timeout = fetch_timeout
        self.cache_file = cache_file or DEFAULT_CACHE_FILE
        self._fetch_fn = fetch_fn or self._fetch_series_from_fred

        self._cached_series: Optional[List[Tuple[str, float]]] = None
        self._cache_fetched_at: Optional[float] = None

    # ---- network ----

    def _fetch_series_from_fred(self) -> List[Tuple[str, float]]:
        resp = requests.get(FRED_DFII10_URL, timeout=self.fetch_timeout)
        resp.raise_for_status()
        return parse_dfii10_csv(resp.text)

    # ---- disk cache (survives restarts; daily data doesn't need more) ----

    def _load_disk_cache(self) -> Optional[List[Tuple[str, float]]]:
        try:
            if self.cache_file.exists():
                data = json.loads(self.cache_file.read_text())
                series = [(row[0], float(row[1])) for row in data.get("series", [])]
                return series or None
        except Exception as e:
            logger.debug(f"REAL_YIELDS: Failed to load disk cache: {e}")
        return None

    def _save_disk_cache(self, series: List[Tuple[str, float]]) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps({
                "fetched_at": time.time(),
                "series": series,
            }))
        except Exception as e:
            logger.debug(f"REAL_YIELDS: Failed to save disk cache: {e}")

    # ---- refresh logic ----

    def _ensure_fresh(self) -> None:
        now = time.time()
        if (
            self._cached_series is not None
            and self._cache_fetched_at is not None
            and (now - self._cache_fetched_at) < self.cache_ttl_seconds
        ):
            return

        try:
            series = self._fetch_fn()
            if series:
                self._cached_series = series
                self._cache_fetched_at = now
                self._save_disk_cache(series)
                logger.info(f"REAL_YIELDS: Refreshed DFII10 series ({len(series)} points)")
                return
        except Exception as e:
            logger.warning(f"REAL_YIELDS: Fetch failed ({e}); falling back to cache if available")

        if self._cached_series is None:
            disk_series = self._load_disk_cache()
            if disk_series:
                self._cached_series = disk_series
                # Treat disk cache as "fresh enough" to avoid hammering FRED
                # on every scan cycle while the API stays unreachable.
                self._cache_fetched_at = now
                logger.info("REAL_YIELDS: Using on-disk cached series (network unavailable)")

    # ---- public API ----

    def get_bias(self, direction: str) -> Tuple[str, float, dict]:
        """
        Get the real-yield bias ('aligned' | 'opposed' | 'neutral'), a
        continuous alignment score (-1.0..+1.0), and metadata for a proposed
        trade direction. Fails open to ('neutral', 0.0, {'reason': 'no_data'})
        if no data is available at all.
        """
        try:
            self._ensure_fresh()
        except Exception as e:
            logger.debug(f"REAL_YIELDS: Unexpected error during refresh: {e}")

        if not self._cached_series:
            return "neutral", 0.0, {"reason": "no_data"}

        values = [v for _, v in self._cached_series]
        trend = compute_real_yield_trend(values, self.lookback_days, self.threshold)
        label, score = real_yield_bias_for_direction(trend, direction, self.full_scale_change)

        meta = {
            "latest_value": trend.latest_value,
            "change": trend.change,
            "trend_direction": trend.direction,
            "lookback_days": trend.lookback_days,
            "series_points": trend.sample_count,
        }
        return label, score, meta
