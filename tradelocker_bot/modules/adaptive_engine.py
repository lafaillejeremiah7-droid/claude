"""
Adaptive Learning Engine

Self-learning system that analyzes the bot's own trade history to:
1. Score every trade with features (ATR, hour, slope, pattern, volume, etc.)
2. Identify which feature combinations produce winners vs losers
3. Dynamically adjust strategy parameters within safety bounds
4. Assign a confidence score (0-10) to each potential trade
5. Only allow trades with confidence >= 8/10

The engine operates on a rolling window of the last 50-200 trades,
recalculates optimal parameters after every N trades, and writes
the adjusted config to disk so it persists across restarts.

NO CURVE-FITTING PROTECTION:
- Parameters can only move within bounded ranges (min/max)
- Changes are gradual (max 10% shift per optimization cycle)
- Minimum sample size of 30 trades required before adapting
- Out-of-sample holdout: uses first 70% to learn, last 30% to validate
"""
import json
import logging
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "logs"
ADAPTIVE_CONFIG_FILE = DATA_DIR / "adaptive_config.json"
TRADE_FEATURES_FILE = DATA_DIR / "trade_features.jsonl"
LEARNING_LOG_FILE = DATA_DIR / "learning_log.jsonl"


# ============================================================
# FEATURE VECTOR - What we tag every trade with
# ============================================================
@dataclass
class TradeFeatures:
    """Feature vector for a single trade - used for learning."""
    # Trade identifiers
    trade_id: str = ""
    symbol: str = ""
    direction: str = ""
    timestamp: str = ""
    
    # Outcome
    result: str = ""  # 'win', 'loss', 'breakeven'
    pnl_r: float = 0.0
    max_favorable_r: float = 0.0  # Future use: max favorable excursion tracking
    
    # Entry features (what conditions existed at entry)
    hour_utc: int = 0
    session: str = ""  # 'london', 'ny_morning', 'ny_afternoon'
    atr_percentile: float = 0.0  # 0-1
    atr_value: float = 0.0
    rsi_at_entry: float = 0.0
    volume_ratio: float = 0.0  # vs 20-period avg
    ema20_distance_pct: float = 0.0  # how far from EMA20 at entry
    ema20_slope_strength: float = 0.0  # slope normalized by price
    slope_4h_strength: float = 0.0  # 4H EMA slope strength
    slope_30m_gap_pct: float = 0.0  # 30M EMA50 vs EMA200 gap
    candle_pattern: str = ""
    candle_body_ratio: float = 0.0  # body/range of entry candle
    # Future use: populated when pullback-depth and structure-break analysis is
    # integrated with the feature extraction pipeline.
    pullback_depth_pct: float = 0.0  # how deep was the pullback
    structure_break_strength: float = 0.0  # how far past the structure level
    had_liquidity_sweep: bool = False
    bars_since_last_trade: int = 0  # Future use: avoid clustering
    
    # Derived features
    trend_alignment_score: float = 0.0  # 0-1 how well all TFs agree
    momentum_score: float = 0.0  # Future use: composite momentum measure


# ============================================================
# ADAPTIVE PARAMETERS - What the engine can tune
# ============================================================
@dataclass
class AdaptiveParams:
    """Parameters that the engine can dynamically adjust."""
    # RSI boundaries
    rsi_long_min: float = 40.0
    rsi_long_max: float = 65.0
    rsi_short_min: float = 35.0
    rsi_short_max: float = 60.0
    
    # Volume threshold
    volume_min_ratio: float = 1.0
    
    # Trailing stop
    trail_trigger_r: float = 1.0
    trail_distance_r: float = 0.4
    
    # Session hours to avoid (list of UTC hours)
    avoid_hours: List[int] = field(default_factory=lambda: [15, 16, 17])
    
    # ATR filter
    max_atr_percentile: float = 0.80
    
    # EMA20 slope minimum strength (normalized)
    min_ema20_slope: float = 0.0
    
    # Pullback threshold
    pullback_threshold_pct: float = 0.002
    
    # Minimum confidence to take trade (0-10)
    min_confidence: float = 8.0
    
    # Pattern weights (which patterns perform best)
    pattern_weights: Dict[str, float] = field(default_factory=lambda: {
        'bullish_engulfing': 1.0, 'hammer': 1.0,
        'bullish_rejection': 1.0, 'strong_bullish': 1.0,
        'bearish_engulfing': 1.0, 'shooting_star': 1.0,
        'bearish_rejection': 1.0, 'strong_bearish': 1.0,
    })
    
    # Feature importance weights for confidence scoring
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'atr_percentile': 1.0,
        'volume_ratio': 1.0,
        'ema20_slope_strength': 1.0,
        'slope_4h_strength': 1.0,
        'candle_body_ratio': 1.0,
        'rsi_zone_quality': 1.0,
        'session_quality': 1.0,
        'pattern_quality': 1.0,
    })
    
    # Meta
    last_updated: str = ""
    trades_analyzed: int = 0
    optimization_cycles: int = 0
    current_win_rate: float = 0.0
    current_avg_r: float = 0.0


# ============================================================
# PARAMETER BOUNDS - Safety rails
# ============================================================
PARAM_BOUNDS = {
    'rsi_long_min': (35.0, 50.0),
    'rsi_long_max': (55.0, 70.0),
    'rsi_short_min': (30.0, 45.0),
    'rsi_short_max': (50.0, 65.0),
    'volume_min_ratio': (0.8, 1.8),
    'trail_trigger_r': (0.6, 1.4),
    'trail_distance_r': (0.2, 0.6),
    'max_atr_percentile': (0.60, 0.95),
    'pullback_threshold_pct': (0.001, 0.005),
    'min_confidence': (7.0, 9.5),
}

# Maximum parameter shift per optimization cycle (prevents wild swings)
MAX_SHIFT_PCT = 0.10  # 10% max change per cycle


# ============================================================
# ADAPTIVE ENGINE
# ============================================================
class AdaptiveEngine:
    """
    Self-learning engine that evolves strategy parameters based on results.
    
    Flow:
    1. Every trade gets tagged with a feature vector
    2. After N trades (default 20), engine runs an optimization cycle
    3. Optimization: look at winning vs losing trade features, adjust params
    4. New params are applied to next trade's confidence scoring
    5. Trades below confidence threshold are rejected
    
    Anti-overfitting:
    - Rolling window (not all history)
    - Bounded parameter ranges
    - Gradual shifts only
    - Validates on holdout subset before applying
    """
    
    def __init__(self, optimize_every_n: int = 20, min_trades_to_learn: int = 30):
        self.optimize_every_n = optimize_every_n
        self.min_trades_to_learn = min_trades_to_learn
        self.params = AdaptiveParams()
        self.trade_history: List[TradeFeatures] = []
        self.trades_since_last_optimize = 0
        
        self._load_params()
        self._load_trade_history()
    
    # ========================================
    # CONFIDENCE SCORING (the 8/10 gate)
    # ========================================
    
    def score_trade_confidence(self, features: TradeFeatures) -> float:
        """
        Score a potential trade from 0 to 10 based on how similar it is
        to historically winning trades.
        
        Only trades scoring >= 8.0 are taken.
        
        Scoring dimensions:
        1. ATR Zone (is volatility in a productive range?)
        2. Volume Strength (above average = institutional participation)
        3. EMA20 Momentum (slope confirming direction)
        4. 4H Trend Strength (strong trend = higher conviction)
        5. Candle Quality (pattern win rate from history)
        6. RSI Position (center of zone = highest quality)
        7. Session Quality (which hours produce winners?)
        8. Pullback Quality (how clean was the pullback?)
        9. Pattern Win Rate (historical performance of this pattern)
        10. Composite Trend Alignment
        
        Returns:
            Confidence score 0-10 (need 8+ to trade)
        """
        scores = {}
        weights = self.params.feature_weights
        
        # 1. ATR Zone Score (0-10)
        # Sweet spot: 20th-70th percentile (not too calm, not too wild)
        atr_p = features.atr_percentile
        if 0.20 <= atr_p <= 0.70:
            scores['atr_percentile'] = 10.0
        elif 0.10 <= atr_p <= 0.80:
            scores['atr_percentile'] = 7.0
        elif atr_p > self.params.max_atr_percentile:
            scores['atr_percentile'] = 2.0  # Filtered out anyway
        else:
            scores['atr_percentile'] = 5.0
        
        # 2. Volume Score (0-10)
        vol_r = features.volume_ratio
        if vol_r >= 1.5:
            scores['volume_ratio'] = 10.0
        elif vol_r >= 1.2:
            scores['volume_ratio'] = 8.5
        elif vol_r >= 1.0:
            scores['volume_ratio'] = 7.0
        else:
            scores['volume_ratio'] = 4.0
        
        # 3. EMA20 Slope Score (0-10)
        slope = features.ema20_slope_strength
        if slope > 0.0005:
            scores['ema20_slope_strength'] = 10.0
        elif slope > 0.0002:
            scores['ema20_slope_strength'] = 8.0
        elif slope > 0:
            scores['ema20_slope_strength'] = 6.0
        else:
            scores['ema20_slope_strength'] = 2.0  # Wrong direction
        
        # 4. 4H Trend Strength (0-10)
        slope_4h = features.slope_4h_strength
        if slope_4h > 0.002:
            scores['slope_4h_strength'] = 10.0
        elif slope_4h > 0.001:
            scores['slope_4h_strength'] = 8.0
        elif slope_4h > 0.0005:
            scores['slope_4h_strength'] = 6.5
        else:
            scores['slope_4h_strength'] = 4.0
        
        # 5. Candle Body Ratio (0-10)
        # Strong body = strong conviction
        br = features.candle_body_ratio
        if br >= 0.70:
            scores['candle_body_ratio'] = 10.0
        elif br >= 0.55:
            scores['candle_body_ratio'] = 8.0
        elif br >= 0.40:
            scores['candle_body_ratio'] = 6.5
        else:
            scores['candle_body_ratio'] = 5.0
        
        # 6. RSI Zone Quality (0-10)
        # Best when RSI is near center of allowed zone
        rsi = features.rsi_at_entry
        if features.direction == 'bullish':
            center = (self.params.rsi_long_min + self.params.rsi_long_max) / 2
            zone_width = (self.params.rsi_long_max - self.params.rsi_long_min) / 2
        else:
            center = (self.params.rsi_short_min + self.params.rsi_short_max) / 2
            zone_width = (self.params.rsi_short_max - self.params.rsi_short_min) / 2
        
        distance_from_center = abs(rsi - center) / zone_width if zone_width > 0 else 1
        scores['rsi_zone_quality'] = max(0, 10.0 - distance_from_center * 5)
        
        # 7. Session Quality (0-10)
        hour = features.hour_utc
        if hour in self.params.avoid_hours:
            scores['session_quality'] = 0.0  # Should be filtered already
        elif hour in [12, 13, 14]:  # London-NY overlap
            scores['session_quality'] = 10.0
        elif hour in [7, 8, 9, 10, 11]:  # London session
            scores['session_quality'] = 8.5
        elif hour in [18, 19, 20, 21]:  # NY afternoon
            scores['session_quality'] = 7.5
        else:
            scores['session_quality'] = 5.0
        
        # 8. Pattern Historical Win Rate (0-10)
        pattern = features.candle_pattern
        pattern_weight = self.params.pattern_weights.get(pattern, 0.5)
        scores['pattern_quality'] = min(10.0, pattern_weight * 10.0)
        
        # WEIGHTED COMPOSITE SCORE
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return 5.0
        
        weighted_sum = sum(
            scores.get(key, 5.0) * weight
            for key, weight in weights.items()
        )
        
        confidence = weighted_sum / total_weight
        
        # Bonus for liquidity sweep (rare but powerful)
        if features.had_liquidity_sweep:
            confidence = min(10.0, confidence + 0.5)
        
        return round(confidence, 2)
    
    def should_take_trade(self, features: TradeFeatures) -> Tuple[bool, float, str]:
        """
        Final gate: should we take this trade?
        
        Returns:
            (should_trade, confidence_score, reason)
        """
        confidence = self.score_trade_confidence(features)
        
        if confidence >= self.params.min_confidence:
            return True, confidence, f"Confidence {confidence:.1f}/10 >= {self.params.min_confidence:.1f} threshold"
        else:
            return False, confidence, f"Confidence {confidence:.1f}/10 < {self.params.min_confidence:.1f} threshold (rejected)"
    
    # ========================================
    # TRADE RECORDING
    # ========================================
    
    def record_trade(self, features: TradeFeatures):
        """
        Record a completed trade for learning.
        Triggers optimization cycle after N trades.
        """
        self.trade_history.append(features)
        self.trades_since_last_optimize += 1
        self._save_trade_features(features)
        
        logger.info(
            f"ADAPTIVE: Recorded trade | {features.symbol} {features.direction} | "
            f"Result: {features.result} ({features.pnl_r:+.2f}R) | "
            f"History: {len(self.trade_history)} trades"
        )
        
        # Check if it's time to optimize
        if (self.trades_since_last_optimize >= self.optimize_every_n and
                len(self.trade_history) >= self.min_trades_to_learn):
            self.run_optimization_cycle()
    
    # ========================================
    # OPTIMIZATION CYCLE
    # ========================================
    
    def run_optimization_cycle(self):
        """
        Analyze recent trade history and adjust parameters.
        
        Process:
        1. Split history into train (70%) and validation (30%)
        2. Analyze winning vs losing trade features on training set
        3. Propose parameter adjustments
        4. Validate: do proposed params improve validation set score?
        5. If yes, apply (gradually). If no, keep current.
        """
        logger.info("=" * 50)
        logger.info("ADAPTIVE: Running optimization cycle...")
        logger.info("=" * 50)
        
        # Use rolling window (last 100 trades max)
        window = self.trade_history[-100:]
        
        if len(window) < self.min_trades_to_learn:
            logger.info(f"ADAPTIVE: Not enough trades ({len(window)} < {self.min_trades_to_learn})")
            return
        
        # Split: 70% train, 30% validate
        split_idx = int(len(window) * 0.7)
        train = window[:split_idx]
        validate = window[split_idx:]
        
        # Analyze training set
        winners = [t for t in train if t.result == 'win']
        losers = [t for t in train if t.result == 'loss']
        
        if not winners or not losers:
            logger.info("ADAPTIVE: Need both winners and losers to learn")
            return
        
        # ---- LEARN FROM WINNERS VS LOSERS ----
        proposed = self._propose_adjustments(winners, losers)
        
        # ---- VALIDATE ----
        current_score = self._score_params_on_trades(self.params, validate)
        proposed_score = self._score_params_on_trades(proposed, validate)
        
        logger.info(f"ADAPTIVE: Current params score: {current_score:.3f}")
        logger.info(f"ADAPTIVE: Proposed params score: {proposed_score:.3f}")
        
        # Only apply if improvement is meaningful (>5%)
        if proposed_score > current_score * 1.05:
            # Gradually blend: 70% current + 30% proposed
            self._blend_params(proposed, blend_factor=0.3)
            logger.info("ADAPTIVE: Parameters UPDATED (blended 30% toward proposed)")
        else:
            logger.info("ADAPTIVE: Proposed params did not improve enough. Keeping current.")
        
        # Update meta
        all_recent = window
        self.params.last_updated = datetime.now(timezone.utc).isoformat()
        self.params.trades_analyzed = len(self.trade_history)
        self.params.optimization_cycles += 1
        self.params.current_win_rate = len([t for t in all_recent if t.result == 'win']) / len(all_recent) * 100
        self.params.current_avg_r = np.mean([t.pnl_r for t in all_recent])
        
        self.trades_since_last_optimize = 0
        self._save_params()
        self._log_optimization(current_score, proposed_score)
        
        logger.info(
            f"ADAPTIVE: Cycle #{self.params.optimization_cycles} complete | "
            f"WR: {self.params.current_win_rate:.1f}% | "
            f"Avg R: {self.params.current_avg_r:+.3f}"
        )
    
    def _propose_adjustments(self, winners: List[TradeFeatures], losers: List[TradeFeatures]) -> AdaptiveParams:
        """
        Compare winner features vs loser features and propose parameter shifts.
        """
        proposed = AdaptiveParams(
            rsi_long_min=self.params.rsi_long_min,
            rsi_long_max=self.params.rsi_long_max,
            rsi_short_min=self.params.rsi_short_min,
            rsi_short_max=self.params.rsi_short_max,
            volume_min_ratio=self.params.volume_min_ratio,
            trail_trigger_r=self.params.trail_trigger_r,
            trail_distance_r=self.params.trail_distance_r,
            max_atr_percentile=self.params.max_atr_percentile,
            pullback_threshold_pct=self.params.pullback_threshold_pct,
            min_confidence=self.params.min_confidence,
            avoid_hours=list(self.params.avoid_hours),
            pattern_weights=dict(self.params.pattern_weights),
            feature_weights=dict(self.params.feature_weights),
        )
        
        # --- VOLUME: Winners tend to have higher volume ---
        avg_vol_winners = np.mean([t.volume_ratio for t in winners])
        avg_vol_losers = np.mean([t.volume_ratio for t in losers])
        if avg_vol_winners > avg_vol_losers * 1.1:
            # Raise volume threshold toward winner average
            proposed.volume_min_ratio = min(
                PARAM_BOUNDS['volume_min_ratio'][1],
                self.params.volume_min_ratio + 0.05
            )
        
        # --- ATR: Find the sweet spot ---
        avg_atr_winners = np.mean([t.atr_percentile for t in winners])
        avg_atr_losers = np.mean([t.atr_percentile for t in losers])
        if avg_atr_losers > avg_atr_winners + 0.05:
            # Losers cluster at higher ATR, tighten the filter
            proposed.max_atr_percentile = max(
                PARAM_BOUNDS['max_atr_percentile'][0],
                self.params.max_atr_percentile - 0.02
            )
        elif avg_atr_winners > avg_atr_losers + 0.05:
            # Winners at higher ATR, loosen filter
            proposed.max_atr_percentile = min(
                PARAM_BOUNDS['max_atr_percentile'][1],
                self.params.max_atr_percentile + 0.02
            )
        
        # --- RSI: Tighten zones around where winners cluster ---
        long_winners = [t for t in winners if t.direction == 'bullish']
        long_losers = [t for t in losers if t.direction == 'bullish']
        short_winners = [t for t in winners if t.direction == 'bearish']
        short_losers = [t for t in losers if t.direction == 'bearish']
        
        if len(long_winners) >= 5:
            rsi_wins = [t.rsi_at_entry for t in long_winners]
            win_center = np.median(rsi_wins)
            proposed.rsi_long_min = np.clip(
                win_center - 12, *PARAM_BOUNDS['rsi_long_min']
            )
            proposed.rsi_long_max = np.clip(
                win_center + 12, *PARAM_BOUNDS['rsi_long_max']
            )
        
        if len(short_winners) >= 5:
            rsi_wins = [t.rsi_at_entry for t in short_winners]
            win_center = np.median(rsi_wins)
            proposed.rsi_short_min = np.clip(
                win_center - 12, *PARAM_BOUNDS['rsi_short_min']
            )
            proposed.rsi_short_max = np.clip(
                win_center + 12, *PARAM_BOUNDS['rsi_short_max']
            )
        
        # --- TRAILING STOP: Optimize trigger and distance ---
        # Check average max favorable excursion of winners
        avg_mfe_winners = np.mean([t.max_favorable_r for t in winners if t.max_favorable_r > 0])
        if avg_mfe_winners > 1.5:
            # Winners run further, we can afford later trail trigger
            proposed.trail_trigger_r = min(
                PARAM_BOUNDS['trail_trigger_r'][1],
                self.params.trail_trigger_r + 0.05
            )
        elif avg_mfe_winners < 1.2:
            # Winners don't run far, trigger trail sooner
            proposed.trail_trigger_r = max(
                PARAM_BOUNDS['trail_trigger_r'][0],
                self.params.trail_trigger_r - 0.05
            )
        
        # --- PATTERN WEIGHTS: Update based on win rates ---
        pattern_results = defaultdict(lambda: {'wins': 0, 'total': 0})
        for t in winners + losers:
            pattern_results[t.candle_pattern]['total'] += 1
            if t.result == 'win':
                pattern_results[t.candle_pattern]['wins'] += 1
        
        for pattern, data in pattern_results.items():
            if data['total'] >= 5:  # Minimum sample
                win_rate = data['wins'] / data['total']
                proposed.pattern_weights[pattern] = np.clip(win_rate, 0.2, 1.0)
        
        # --- SESSION: Identify bad hours ---
        hour_results = defaultdict(lambda: {'wins': 0, 'total': 0})
        for t in winners + losers:
            hour_results[t.hour_utc]['total'] += 1
            if t.result == 'win':
                hour_results[t.hour_utc]['wins'] += 1
        
        new_avoid = []
        for hour, data in hour_results.items():
            if data['total'] >= 5:
                wr = data['wins'] / data['total']
                if wr < 0.35:  # Less than 35% win rate = bad hour
                    new_avoid.append(hour)
        
        if new_avoid:
            # Only add hours that consistently lose, keep existing avoids
            proposed.avoid_hours = list(set(self.params.avoid_hours + new_avoid))
        
        # --- FEATURE WEIGHTS: Boost features that differentiate W from L ---
        for feature_name in proposed.feature_weights:
            w_scores = self._get_feature_scores(winners, feature_name)
            l_scores = self._get_feature_scores(losers, feature_name)
            
            if w_scores and l_scores:
                w_avg = np.mean(w_scores)
                l_avg = np.mean(l_scores)
                separation = abs(w_avg - l_avg) / max(abs(w_avg), abs(l_avg), 0.001)
                
                # Higher separation = more important feature
                proposed.feature_weights[feature_name] = np.clip(
                    0.5 + separation * 2, 0.3, 2.0
                )
        
        return proposed
    
    def _get_feature_scores(self, trades: List[TradeFeatures], feature_name: str) -> List[float]:
        """Get the raw feature values for scoring comparison."""
        values = []
        for t in trades:
            if feature_name == 'atr_percentile':
                values.append(t.atr_percentile)
            elif feature_name == 'volume_ratio':
                values.append(t.volume_ratio)
            elif feature_name == 'ema20_slope_strength':
                values.append(t.ema20_slope_strength)
            elif feature_name == 'slope_4h_strength':
                values.append(t.slope_4h_strength)
            elif feature_name == 'candle_body_ratio':
                values.append(t.candle_body_ratio)
            elif feature_name == 'rsi_zone_quality':
                values.append(t.rsi_at_entry)
            elif feature_name == 'session_quality':
                values.append(t.hour_utc)
            elif feature_name == 'pattern_quality':
                values.append(self.params.pattern_weights.get(t.candle_pattern, 0.5))
        return values
    
    def _score_params_on_trades(self, params: AdaptiveParams, trades: List[TradeFeatures]) -> float:
        """
        Score how well a param set would have performed on a set of trades.
        Higher = better (we want high win rate + high avg R).
        """
        if not trades:
            return 0.0
        
        # Temporarily apply params for scoring
        original_params = self.params
        self.params = params
        
        would_take = []
        for t in trades:
            should, confidence, _ = self.should_take_trade(t)
            if should:
                would_take.append(t)
        
        self.params = original_params
        
        if not would_take:
            return 0.0
        
        # Score = win_rate * avg_positive_r * sqrt(trade_count)
        # This rewards: high win rate, high profit, reasonable frequency
        wins = sum(1 for t in would_take if t.result == 'win')
        wr = wins / len(would_take)
        avg_r = np.mean([t.pnl_r for t in would_take])
        frequency = len(would_take) / len(trades)  # What % of trades pass
        
        # Penalize if too few trades pass (too restrictive)
        freq_bonus = min(1.0, frequency / 0.3)  # Full bonus if 30%+ pass
        
        score = (wr * 2 + max(0, avg_r)) * freq_bonus
        return score
    
    def _blend_params(self, proposed: AdaptiveParams, blend_factor: float = 0.3):
        """Gradually blend proposed params into current (prevents wild swings).

        Special case: if the current value is 0 and the proposed value is
        non-zero, multiplicative blending would keep it at 0 forever. In that
        case we adopt the proposed value directly (additive blending).
        """

        def blend(current, proposed_val, bounds_key=None):
            # Handle zero-valued parameters: multiplicative blending can't
            # escape zero, so adopt the proposed value directly.
            if current == 0 and proposed_val != 0:
                blended = proposed_val * blend_factor
            else:
                blended = current * (1 - blend_factor) + proposed_val * blend_factor
            if bounds_key and bounds_key in PARAM_BOUNDS:
                blended = np.clip(blended, *PARAM_BOUNDS[bounds_key])
            # Enforce max shift (skip if current is 0 — no meaningful % shift)
            if current != 0:
                max_shift = abs(current) * MAX_SHIFT_PCT
                blended = np.clip(blended, current - max_shift, current + max_shift)
            return round(blended, 4)
        
        self.params.rsi_long_min = blend(self.params.rsi_long_min, proposed.rsi_long_min, 'rsi_long_min')
        self.params.rsi_long_max = blend(self.params.rsi_long_max, proposed.rsi_long_max, 'rsi_long_max')
        self.params.rsi_short_min = blend(self.params.rsi_short_min, proposed.rsi_short_min, 'rsi_short_min')
        self.params.rsi_short_max = blend(self.params.rsi_short_max, proposed.rsi_short_max, 'rsi_short_max')
        self.params.volume_min_ratio = blend(self.params.volume_min_ratio, proposed.volume_min_ratio, 'volume_min_ratio')
        self.params.trail_trigger_r = blend(self.params.trail_trigger_r, proposed.trail_trigger_r, 'trail_trigger_r')
        self.params.trail_distance_r = blend(self.params.trail_distance_r, proposed.trail_distance_r, 'trail_distance_r')
        self.params.max_atr_percentile = blend(self.params.max_atr_percentile, proposed.max_atr_percentile, 'max_atr_percentile')
        self.params.pullback_threshold_pct = blend(self.params.pullback_threshold_pct, proposed.pullback_threshold_pct, 'pullback_threshold_pct')
        
        # Discrete params: adopt directly
        self.params.avoid_hours = proposed.avoid_hours
        self.params.pattern_weights = proposed.pattern_weights
        self.params.feature_weights = proposed.feature_weights
    
    # ========================================
    # GET CURRENT OPTIMAL PARAMS (for bot to use)
    # ========================================
    
    def get_rsi_bounds(self, direction: str) -> Tuple[float, float]:
        """Get current optimal RSI bounds for a direction."""
        if direction in ('bullish', 'buy'):
            return self.params.rsi_long_min, self.params.rsi_long_max
        else:
            return self.params.rsi_short_min, self.params.rsi_short_max
    
    def get_volume_threshold(self) -> float:
        """Get current optimal volume threshold."""
        return self.params.volume_min_ratio
    
    def get_trailing_params(self) -> Tuple[float, float]:
        """Get current trailing stop parameters (trigger_r, distance_r)."""
        return self.params.trail_trigger_r, self.params.trail_distance_r
    
    def get_avoid_hours(self) -> List[int]:
        """Get hours to avoid trading."""
        return self.params.avoid_hours
    
    def get_atr_filter(self) -> float:
        """Get max ATR percentile allowed."""
        return self.params.max_atr_percentile
    
    def is_pattern_allowed(self, pattern: str) -> bool:
        """Check if a candle pattern has sufficient historical win rate."""
        weight = self.params.pattern_weights.get(pattern, 0.5)
        return weight >= 0.35  # Minimum 35% historical WR for pattern
    
    # ========================================
    # PERSISTENCE
    # ========================================
    
    def _save_params(self):
        """Save current adaptive params to disk."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = asdict(self.params)
            with open(ADAPTIVE_CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("ADAPTIVE: Params saved to disk")
        except Exception as e:
            logger.warning(f"ADAPTIVE: Failed to save params: {e}")
    
    def _load_params(self):
        """Load adaptive params from disk."""
        try:
            if ADAPTIVE_CONFIG_FILE.exists():
                with open(ADAPTIVE_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                # Filter to known fields only to avoid TypeError on extra keys
                # (forward-compatibility with newer config versions).
                known_fields = {f.name for f in AdaptiveParams.__dataclass_fields__.values()}
                filtered = {k: v for k, v in data.items() if k in known_fields}
                self.params = AdaptiveParams(**filtered)
                logger.info(
                    f"ADAPTIVE: Loaded params | Cycles: {self.params.optimization_cycles} | "
                    f"WR: {self.params.current_win_rate:.1f}% | "
                    f"Trades analyzed: {self.params.trades_analyzed}"
                )
        except Exception as e:
            logger.warning(f"ADAPTIVE: Failed to load params (using defaults): {e}")
            self.params = AdaptiveParams()
    
    def _save_trade_features(self, features: TradeFeatures):
        """Append trade features to JSONL file."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(TRADE_FEATURES_FILE, 'a') as f:
                f.write(json.dumps(asdict(features)) + '\n')
        except Exception as e:
            logger.warning(f"ADAPTIVE: Failed to save trade features: {e}")
    
    def _load_trade_history(self):
        """Load trade feature history from disk."""
        try:
            if TRADE_FEATURES_FILE.exists():
                with open(TRADE_FEATURES_FILE, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self.trade_history.append(TradeFeatures(**data))
                
                if self.trade_history:
                    logger.info(f"ADAPTIVE: Loaded {len(self.trade_history)} historical trades")
        except Exception as e:
            logger.warning(f"ADAPTIVE: Failed to load history (starting fresh): {e}")
            self.trade_history = []
    
    def _log_optimization(self, old_score: float, new_score: float):
        """Log optimization cycle results."""
        try:
            entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cycle': self.params.optimization_cycles,
                'old_score': old_score,
                'new_score': new_score,
                'improved': new_score > old_score * 1.05,
                'trades_in_window': len(self.trade_history[-100:]),
                'current_wr': self.params.current_win_rate,
                'current_avg_r': self.params.current_avg_r,
                'params_snapshot': {
                    'rsi_long': (self.params.rsi_long_min, self.params.rsi_long_max),
                    'rsi_short': (self.params.rsi_short_min, self.params.rsi_short_max),
                    'volume': self.params.volume_min_ratio,
                    'trail': (self.params.trail_trigger_r, self.params.trail_distance_r),
                    'atr_max': self.params.max_atr_percentile,
                    'avoid_hours': self.params.avoid_hours,
                },
            }
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(LEARNING_LOG_FILE, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.warning(f"ADAPTIVE: Failed to log optimization: {e}")
    
    # ========================================
    # STATUS
    # ========================================
    
    def get_status(self) -> dict:
        """Get current adaptive engine status."""
        return {
            'optimization_cycles': self.params.optimization_cycles,
            'trades_in_history': len(self.trade_history),
            'trades_until_next_optimize': max(0, self.optimize_every_n - self.trades_since_last_optimize),
            'current_win_rate': f"{self.params.current_win_rate:.1f}%",
            'current_avg_r': f"{self.params.current_avg_r:+.3f}R",
            'min_confidence_threshold': self.params.min_confidence,
            'last_updated': self.params.last_updated or 'Never',
            'rsi_long_zone': f"{self.params.rsi_long_min:.0f}-{self.params.rsi_long_max:.0f}",
            'rsi_short_zone': f"{self.params.rsi_short_min:.0f}-{self.params.rsi_short_max:.0f}",
            'volume_threshold': f"{self.params.volume_min_ratio:.2f}x",
            'trailing_stop': f"trigger={self.params.trail_trigger_r:.2f}R, trail={self.params.trail_distance_r:.2f}R",
            'atr_max': f"{self.params.max_atr_percentile:.0%}",
            'avoid_hours': self.params.avoid_hours,
            'top_patterns': {k: f"{v:.0%}" for k, v in 
                           sorted(self.params.pattern_weights.items(), key=lambda x: x[1], reverse=True)[:5]},
        }
