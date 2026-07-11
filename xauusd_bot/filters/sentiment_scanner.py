"""
X/Twitter Sentiment Scanner for XAUUSD Bot

Scans for keywords from accounts that move gold prices.
In backtesting mode, simulates sentiment based on price action
(big moves = news event detected, trending = sentiment shift).

Systems 5 & 6:
5. X/Twitter Sentiment Scanner
6. Influencer Flow Tracking

Key Accounts:
    @federalreserve, @GoldTelegraph, @zaborhedge, @unusual_whales,
    @ForexLive, @RANSquawk, geopolitical accounts, fund managers

Signal Categories:
    BULLISH: rate cut, dovish, war, crisis, inflation -> bias buy, stretch TP
    BEARISH: rate hike, hawkish, strong dollar, risk on -> bias sell, reduce size
    DANGER: FOMC leaked, flash crash, manipulation, liquidation -> DON'T TRADE
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta


@dataclass
class SentimentSignal:
    """A single sentiment signal from X/Twitter."""
    source: str = ""
    account: str = ""
    keyword: str = ""
    category: str = "NEUTRAL"    # "BULLISH", "BEARISH", "DANGER", "NEUTRAL"
    strength: float = 0.0        # 0-1
    timestamp: Optional[datetime] = None
    raw_text: str = ""


@dataclass
class InfluencerFlow:
    """Tracked influencer position/bias."""
    account: str = ""
    bias: str = "NEUTRAL"        # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float = 0.0
    last_signal_time: Optional[datetime] = None
    signal_count: int = 0


# Keywords that move gold
BULLISH_KEYWORDS = [
    "rate cut", "dovish", "war", "crisis", "inflation rising",
    "safe haven", "gold rally", "buy gold", "bullion demand",
    "geopolitical risk", "fed pivot", "stagflation", "debt ceiling",
    "bank failure", "recession fears", "yields falling",
    "dollar weakness", "central bank buying", "de-dollarization",
]

BEARISH_KEYWORDS = [
    "rate hike", "hawkish", "strong dollar", "risk on",
    "gold sell", "yields rising", "taper", "quantitative tightening",
    "inflation cooling", "soft landing", "equity rally",
    "dollar strength", "bond yields up", "risk appetite",
    "gold overvalued", "profit taking gold",
]

DANGER_KEYWORDS = [
    "fomc leaked", "flash crash", "manipulation", "liquidation",
    "circuit breaker", "black swan", "margin call cascade",
    "market halt", "emergency meeting", "systemic risk",
    "breaking: fed", "surprise rate", "unexpected",
]

# Key accounts that move gold
TRACKED_ACCOUNTS = {
    "tier1_institutional": [
        "@federalreserve", "@ecaborshoff", "@ABORSHOFF",
        "@IMFNews", "@WorldBank",
    ],
    "tier2_gold_focused": [
        "@GoldTelegraph", "@zaborhedge", "@PeterSchiff",
        "@GoldSilverPros", "@goldaborshoff",
    ],
    "tier3_market_intel": [
        "@unusual_whales", "@ForexLive", "@RANSquawk",
        "@zaborhedge", "@DeItaone", "@FirstSquawk",
    ],
    "tier4_geopolitical": [
        "@BNONews", "@IntelCrab", "@sentdefender",
        "@WarMonitor3", "@Global_Mil_Info",
    ],
    "tier5_fund_managers": [
        "@RayDalio", "@MarkMinervini", "@jimcramer",
        "@Michael_Burry", "@ResijAborshoff",
    ],
}

# Account influence weights (higher = more market-moving)
ACCOUNT_WEIGHTS = {
    "tier1_institutional": 1.0,
    "tier2_gold_focused": 0.8,
    "tier3_market_intel": 0.7,
    "tier4_geopolitical": 0.6,
    "tier5_fund_managers": 0.5,
}


class SentimentScanner:
    """
    X/Twitter Sentiment Scanner for gold trading signals.

    In LIVE mode: connects to Twitter API, monitors key accounts and keywords.
    In BACKTEST mode: simulates sentiment from price action patterns.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.mode = config.get("mode", "backtest")  # "live" or "backtest"

        # Sentiment decay (signals fade over time)
        self.signal_decay_minutes = config.get("signal_decay_minutes", 60)
        self.danger_decay_minutes = config.get("danger_decay_minutes", 120)

        # Thresholds
        self.bullish_threshold = config.get("bullish_threshold", 0.3)
        self.bearish_threshold = config.get("bearish_threshold", -0.3)
        self.danger_threshold = config.get("danger_threshold", 0.5)

        # State
        self._active_signals: list[SentimentSignal] = []
        self._influencer_flows: dict[str, InfluencerFlow] = {}
        self._last_big_move_time: Optional[datetime] = None
        self._price_history: list[float] = []
        self._price_timestamps: list[datetime] = []

        # Backtest simulation state
        self._sim_sentiment: float = 0.0  # -1 to 1
        self._sim_danger: bool = False
        self._sim_last_update: Optional[datetime] = None

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def update(self, timestamp: datetime, close: float,
               high: float = 0.0, low: float = 0.0,
               atr: float = 0.0) -> None:
        """
        Update scanner with new price data.
        In backtest mode, this simulates sentiment from price action.
        Only does full simulation every 4th call for performance.
        """
        self._price_history.append(close)
        self._price_timestamps.append(timestamp)

        # Keep last 200 bars
        if len(self._price_history) > 200:
            self._price_history = self._price_history[-200:]
            self._price_timestamps = self._price_timestamps[-200:]

        if self.mode == "backtest":
            # Only simulate every 4th bar for performance
            if len(self._price_history) % 4 == 0:
                self._simulate_sentiment(timestamp, close, high, low, atr)
            else:
                # Still apply decay
                self._sim_sentiment *= 0.995

        # Decay old signals periodically
        if len(self._price_history) % 8 == 0:
            self._decay_signals(timestamp)

    def get_sentiment(self, timestamp: datetime,
                      direction: str = "") -> 'SentimentBias':
        """
        Get current sentiment bias for trading decisions.

        Returns SentimentBias with:
            - bias: BULLISH/BEARISH/NEUTRAL/DANGER
            - strength: 0-1
            - tp_stretch: multiplier for TP
            - size_modifier: multiplier for position size
            - should_avoid: True if DANGER signal active
        """
        from ..core.adaptive_engine import SentimentBias

        result = SentimentBias()

        if self.mode == "backtest":
            return self._get_simulated_sentiment(timestamp, direction)

        # LIVE mode: aggregate active signals
        if not self._active_signals:
            result.bias = "NEUTRAL"
            result.reason = "No active signals"
            return result

        # Score signals
        bull_score = 0.0
        bear_score = 0.0
        danger_score = 0.0

        for sig in self._active_signals:
            # Apply time decay
            age_minutes = (timestamp - sig.timestamp).total_seconds() / 60
            if sig.category == "DANGER":
                decay = max(0, 1.0 - age_minutes / self.danger_decay_minutes)
            else:
                decay = max(0, 1.0 - age_minutes / self.signal_decay_minutes)

            weighted_strength = sig.strength * decay

            if sig.category == "BULLISH":
                bull_score += weighted_strength
            elif sig.category == "BEARISH":
                bear_score += weighted_strength
            elif sig.category == "DANGER":
                danger_score += weighted_strength

        # Determine bias
        if danger_score >= self.danger_threshold:
            result.bias = "DANGER"
            result.should_avoid = True
            result.strength = min(danger_score, 1.0)
            result.reason = "DANGER signals active - avoid trading"
            return result

        net_score = bull_score - bear_score

        if net_score >= self.bullish_threshold:
            result.bias = "BULLISH"
            result.strength = min(net_score, 1.0)
            result.tp_stretch = 1.0 + result.strength * 0.3  # Up to 1.3x TP
            if direction == "BUY":
                result.size_modifier = 1.0 + result.strength * 0.2
            elif direction == "SELL":
                result.size_modifier = max(0.7, 1.0 - result.strength * 0.3)
            result.reason = f"Bullish sentiment (score={net_score:.2f})"

        elif net_score <= self.bearish_threshold:
            result.bias = "BEARISH"
            result.strength = min(abs(net_score), 1.0)
            result.tp_stretch = 1.0 + result.strength * 0.3
            if direction == "SELL":
                result.size_modifier = 1.0 + result.strength * 0.2
            elif direction == "BUY":
                result.size_modifier = max(0.7, 1.0 - result.strength * 0.3)
            result.reason = f"Bearish sentiment (score={net_score:.2f})"

        else:
            result.bias = "NEUTRAL"
            result.strength = 0.0
            result.reason = "Neutral sentiment"

        result.sources = [s.account for s in self._active_signals[:5]]
        return result

    # ==================================================================
    # INFLUENCER FLOW TRACKING (System 6)
    # ==================================================================

    def get_influencer_flow(self) -> dict:
        """
        Get aggregated influencer flow direction.
        Tracks what big accounts are saying about gold direction.
        """
        if not self._influencer_flows:
            return {
                "net_bias": "NEUTRAL",
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "confidence": 0.0,
                "top_signals": [],
            }

        bullish = sum(1 for f in self._influencer_flows.values() if f.bias == "BULLISH")
        bearish = sum(1 for f in self._influencer_flows.values() if f.bias == "BEARISH")
        total = len(self._influencer_flows)

        if bullish > bearish + 2:
            net_bias = "BULLISH"
        elif bearish > bullish + 2:
            net_bias = "BEARISH"
        else:
            net_bias = "NEUTRAL"

        confidence = abs(bullish - bearish) / total if total > 0 else 0.0

        return {
            "net_bias": net_bias,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": total - bullish - bearish,
            "confidence": min(confidence, 1.0),
            "top_signals": list(self._influencer_flows.keys())[:5],
        }

    def record_influencer_signal(self, account: str, bias: str,
                                  timestamp: datetime):
        """Record a signal from a tracked influencer."""
        if account not in self._influencer_flows:
            self._influencer_flows[account] = InfluencerFlow(account=account)

        flow = self._influencer_flows[account]
        flow.bias = bias
        flow.last_signal_time = timestamp
        flow.signal_count += 1
        flow.confidence = min(flow.signal_count / 5.0, 1.0)

    # ==================================================================
    # LIVE MODE: Twitter API Integration (stub for real deployment)
    # ==================================================================

    def connect_twitter_api(self, bearer_token: str):
        """
        Connect to Twitter API v2 for real-time monitoring.
        In production, this would set up a filtered stream.
        """
        # This would be implemented with tweepy or httpx
        # For now, this is the interface that would work with real API
        self._bearer_token = bearer_token
        self._connected = True

    def add_live_signal(self, account: str, text: str, timestamp: datetime):
        """
        Process a live tweet from a monitored account.
        Categorizes it and adds to active signals.
        """
        # Determine category from keywords
        text_lower = text.lower()
        category = "NEUTRAL"
        strength = 0.0
        matched_keyword = ""

        # Check danger first (highest priority)
        for kw in DANGER_KEYWORDS:
            if kw in text_lower:
                category = "DANGER"
                strength = 0.9
                matched_keyword = kw
                break

        if category == "NEUTRAL":
            for kw in BULLISH_KEYWORDS:
                if kw in text_lower:
                    category = "BULLISH"
                    strength = 0.6
                    matched_keyword = kw
                    break

        if category == "NEUTRAL":
            for kw in BEARISH_KEYWORDS:
                if kw in text_lower:
                    category = "BEARISH"
                    strength = 0.6
                    matched_keyword = kw
                    break

        # Weight by account tier
        for tier, accounts in TRACKED_ACCOUNTS.items():
            if account in accounts:
                strength *= ACCOUNT_WEIGHTS[tier]
                break

        if category != "NEUTRAL":
            signal = SentimentSignal(
                source="twitter",
                account=account,
                keyword=matched_keyword,
                category=category,
                strength=strength,
                timestamp=timestamp,
                raw_text=text[:200],
            )
            self._active_signals.append(signal)

            # Update influencer flow
            self.record_influencer_signal(account, category, timestamp)

    # ==================================================================
    # BACKTEST MODE: Simulate sentiment from price action
    # ==================================================================

    def _simulate_sentiment(self, timestamp: datetime, close: float,
                            high: float, low: float, atr: float):
        """
        Simulate sentiment signals based on price action patterns.

        Logic:
        - Big sudden moves (>2x ATR in short time) = news event detected
        - Sustained trend = sentiment shift in that direction
        - Spike + reversal = danger/manipulation detected
        - Quiet periods = neutral sentiment
        """
        if len(self._price_history) < 20:
            return

        # Calculate recent move magnitude
        prices = self._price_history
        current = prices[-1]
        price_5_bars_ago = prices[-5] if len(prices) >= 5 else prices[0]
        price_10_bars_ago = prices[-10] if len(prices) >= 10 else prices[0]
        price_20_bars_ago = prices[-20] if len(prices) >= 20 else prices[0]

        # Move sizes
        move_5 = abs(current - price_5_bars_ago)
        move_10 = abs(current - price_10_bars_ago)
        move_20 = abs(current - price_20_bars_ago)

        effective_atr = atr if atr > 0 else 2.0  # Default ATR for gold

        # --- Detect big move (news-like event) ---
        if move_5 > effective_atr * 3.5:
            # Big sudden move = news event
            if current > price_5_bars_ago:
                # Up move = bullish news (crisis, rate cut expectation)
                self._sim_sentiment = min(self._sim_sentiment + 0.4, 1.0)
                self._inject_simulated_signal(timestamp, "BULLISH", 0.7,
                                             "Simulated: large bullish move detected")
            else:
                # Down move = bearish news (hawkish, dollar strength)
                self._sim_sentiment = max(self._sim_sentiment - 0.4, -1.0)
                self._inject_simulated_signal(timestamp, "BEARISH", 0.7,
                                             "Simulated: large bearish move detected")
            self._last_big_move_time = timestamp

        # --- Detect spike + reversal (manipulation/flash crash) ---
        if len(prices) >= 10:
            move_first_half = abs(prices[-5] - prices[-10])
            move_second_half = abs(prices[-1] - prices[-5])
            # If first half was big AND second half reversed most of it
            if (move_first_half > effective_atr * 3.0 and
                move_second_half > effective_atr * 2.5):
                direction_first = 1 if prices[-5] > prices[-10] else -1
                direction_second = 1 if prices[-1] > prices[-5] else -1
                if direction_first != direction_second:
                    # Spike and reversal = danger
                    self._sim_danger = True
                    self._inject_simulated_signal(timestamp, "DANGER", 0.8,
                                                 "Simulated: spike reversal (manipulation risk)")

        # --- Sustained trend detection ---
        if len(prices) >= 20:
            trend_direction = current - price_20_bars_ago
            if abs(trend_direction) > effective_atr * 3:
                if trend_direction > 0:
                    self._sim_sentiment = min(self._sim_sentiment + 0.1, 0.8)
                else:
                    self._sim_sentiment = max(self._sim_sentiment - 0.1, -0.8)

        # --- Natural decay ---
        self._sim_sentiment *= 0.98  # Decay toward neutral
        if self._sim_danger and self._last_big_move_time:
            time_since = (timestamp - self._last_big_move_time).total_seconds() / 60
            if time_since > 30:  # Danger clears after 30 minutes
                self._sim_danger = False

        # Simulate influencer flow from sustained sentiment
        hour = timestamp.hour
        if hour == 12 and timestamp.minute == 0:
            # Daily influencer update
            if self._sim_sentiment > 0.3:
                self.record_influencer_signal("@sim_GoldTelegraph", "BULLISH", timestamp)
                self.record_influencer_signal("@sim_ForexLive", "BULLISH", timestamp)
            elif self._sim_sentiment < -0.3:
                self.record_influencer_signal("@sim_GoldTelegraph", "BEARISH", timestamp)
                self.record_influencer_signal("@sim_ForexLive", "BEARISH", timestamp)

    def _inject_simulated_signal(self, timestamp: datetime, category: str,
                                  strength: float, reason: str):
        """Inject a simulated sentiment signal for backtesting."""
        signal = SentimentSignal(
            source="simulated",
            account="@sim_price_action",
            keyword=reason,
            category=category,
            strength=strength,
            timestamp=timestamp,
            raw_text=reason,
        )
        self._active_signals.append(signal)

    def _get_simulated_sentiment(self, timestamp: datetime,
                                  direction: str) -> 'SentimentBias':
        """Get sentiment bias from simulated data."""
        from ..core.adaptive_engine import SentimentBias

        result = SentimentBias()

        # Check danger first
        if self._sim_danger:
            result.bias = "DANGER"
            result.should_avoid = True
            result.strength = 0.8
            result.reason = "DANGER: Abnormal price action detected"
            return result

        # Net sentiment
        sentiment = self._sim_sentiment

        if sentiment > self.bullish_threshold:
            result.bias = "BULLISH"
            result.strength = min(abs(sentiment), 1.0)
            result.tp_stretch = 1.0 + result.strength * 0.25
            if direction == "BUY":
                result.size_modifier = 1.0 + result.strength * 0.15
            elif direction == "SELL":
                result.size_modifier = max(0.7, 1.0 - result.strength * 0.2)
            result.reason = f"Bullish sentiment (sim={sentiment:.2f})"

        elif sentiment < self.bearish_threshold:
            result.bias = "BEARISH"
            result.strength = min(abs(sentiment), 1.0)
            result.tp_stretch = 1.0 + result.strength * 0.25
            if direction == "SELL":
                result.size_modifier = 1.0 + result.strength * 0.15
            elif direction == "BUY":
                result.size_modifier = max(0.7, 1.0 - result.strength * 0.2)
            result.reason = f"Bearish sentiment (sim={sentiment:.2f})"

        else:
            result.bias = "NEUTRAL"
            result.strength = 0.0
            result.tp_stretch = 1.0
            result.size_modifier = 1.0
            result.reason = "Neutral sentiment"

        # Add influencer flow as additional source
        flow = self.get_influencer_flow()
        result.sources = flow.get("top_signals", [])

        return result

    # ==================================================================
    # SIGNAL MANAGEMENT
    # ==================================================================

    def _decay_signals(self, current_time: datetime):
        """Remove expired signals."""
        cutoff_normal = current_time - timedelta(minutes=self.signal_decay_minutes)
        cutoff_danger = current_time - timedelta(minutes=self.danger_decay_minutes)

        self._active_signals = [
            s for s in self._active_signals
            if s.timestamp and (
                (s.category == "DANGER" and s.timestamp > cutoff_danger) or
                (s.category != "DANGER" and s.timestamp > cutoff_normal)
            )
        ]

    def get_active_signal_count(self) -> dict:
        """Get count of active signals by category."""
        counts = {"BULLISH": 0, "BEARISH": 0, "DANGER": 0, "NEUTRAL": 0}
        for s in self._active_signals:
            counts[s.category] = counts.get(s.category, 0) + 1
        return counts

    def clear(self):
        """Clear all signals and state."""
        self._active_signals.clear()
        self._influencer_flows.clear()
        self._sim_sentiment = 0.0
        self._sim_danger = False
