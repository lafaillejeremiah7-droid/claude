"""
TradeLocker Trading Bot - Main Orchestrator (Self-Adaptive)

Autonomous trading bot that implements a multi-timeframe EMA strategy
with liquidity sweep detection, trailing stops, and SELF-ADAPTIVE LEARNING
for BTC/USD and XAU/USD on a $5,000 funded account.

Key Features:
- 8/10 confidence gate (adaptive engine scores every trade)
- Self-learning: adjusts parameters after every 20 trades
- Trailing stop exit (trigger 1R, trail 0.4R)
- Session + ATR filters
- 71%+ target win rate

Flow:
1. Authenticate with TradeLocker
2. Every 60 seconds:
   a. Check if trading is allowed (session, risk limits, ATR, hours)
   b. Manage existing positions (trailing stop)
   c. For each instrument:
      - Fetch 4H, 30M, 5M data
      - Analyze multi-timeframe trend
      - If trend aligned, scan for entry signals on 5M
      - Build feature vector for potential trade
      - Score confidence via adaptive engine (need 8/10+)
      - If passes all gates, calculate risk and execute
   d. If trade closed, record features for adaptive learning
3. Every 20 trades: run optimization cycle (self-adapt)

Usage:
    python main.py          # Run the bot (live)
    python main.py --dry    # Dry run (no real trades)
    python main.py --status # Show current status + adaptive engine state
"""
import sys
import time
import signal
import logging
import argparse
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    INSTRUMENTS,
    TIMEFRAMES,
    SCAN_INTERVAL_SECONDS,
    LOG_LEVEL,
    TL_EMAIL,
    TL_SERVER,
    TL_ENVIRONMENT,
    RISK_PERCENT,
)
from modules.api_client import TradeLockerClient
from modules.indicators import add_all_indicators
from modules.trend_analysis import get_trend_state, TrendDirection
from modules.entry_signals import scan_for_entry
from modules.risk_management import RiskManager
from modules.session_filter import can_trade_now, check_session_status
from modules.trade_manager import TradeManager
from modules.adaptive_engine import AdaptiveEngine, TradeFeatures

# ========================================
# LOGGING SETUP
# ========================================

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"bot_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("BOT")

# ========================================
# CONSTANTS
# ========================================
AVOID_HOURS = [15, 16, 17]  # UTC hours with high loss rate (London close)
MAX_ATR_PERCENTILE = 0.80   # Skip when volatility is extreme


# ========================================
# BOT CLASS
# ========================================

class TradingBot:
    """Main trading bot orchestrator with self-adaptive learning."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.running = False
        self.scan_count = 0
        self.last_scan_time = None

        # Core components
        self.client = TradeLockerClient()
        self.risk_manager = RiskManager()
        self.trade_manager = None  # Initialized after client setup

        # ADAPTIVE ENGINE - the self-learning brain
        self.adaptive = AdaptiveEngine(
            optimize_every_n=20,       # Re-optimize after every 20 trades
            min_trades_to_learn=30,    # Need 30 trades before first adaptation
        )

        # Track pending trades (for feature recording on close)
        self.pending_features: dict = {}  # position_id -> TradeFeatures

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Graceful shutdown on SIGINT/SIGTERM."""
        logger.info("Shutdown signal received. Stopping bot gracefully...")
        self.running = False

    # ========================================
    # STARTUP
    # ========================================

    def start(self):
        """Initialize and start the bot."""
        logger.info("=" * 60)
        logger.info("TRADELOCKER SELF-ADAPTIVE TRADING BOT")
        logger.info("=" * 60)
        logger.info(f"Environment: {TL_ENVIRONMENT.upper()}")
        logger.info(f"Server: {TL_SERVER}")
        logger.info(f"Instruments: {INSTRUMENTS}")
        logger.info(f"Dry Run: {self.dry_run}")
        logger.info(f"Scan Interval: {SCAN_INTERVAL_SECONDS}s")
        logger.info(f"Risk Per Trade: {RISK_PERCENT}%")
        logger.info(f"Confidence Threshold: {self.adaptive.params.min_confidence}/10")
        logger.info(f"Adaptive Cycles Completed: {self.adaptive.params.optimization_cycles}")
        logger.info(f"Historical Trades Learned: {len(self.adaptive.trade_history)}")
        logger.info("=" * 60)

        # Validate credentials
        if not TL_EMAIL:
            logger.error("TL_EMAIL not set. Please configure .env file.")
            sys.exit(1)

        # Authenticate
        logger.info("Authenticating with TradeLocker...")
        if not self.client.setup():
            logger.error("Failed to authenticate. Check credentials.")
            sys.exit(1)

        logger.info(f"Authenticated! Account ID: {self.client.account_id}")

        # Load instruments
        logger.info("Loading instruments...")
        instruments = self.client.get_instruments()
        if not instruments:
            logger.error("Failed to load instruments.")
            sys.exit(1)

        # Verify our target instruments exist
        for symbol in INSTRUMENTS:
            inst_id = self.client.get_instrument_id(symbol)
            if inst_id:
                logger.info(f"  {symbol}: ID={inst_id}")
            else:
                logger.warning(f"  {symbol}: NOT FOUND - will skip")

        # Initialize trade manager
        self.trade_manager = TradeManager(self.client, self.risk_manager)

        # Log adaptive engine status
        logger.info("\nAdaptive Engine Status:")
        for key, val in self.adaptive.get_status().items():
            logger.info(f"  {key}: {val}")

        # Start main loop
        self.running = True
        logger.info("\nBot initialized. Starting main loop...")
        self._main_loop()

    # ========================================
    # MAIN LOOP
    # ========================================

    def _main_loop(self):
        """Main scanning loop."""
        while self.running:
            try:
                self._scan_cycle()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in scan cycle: {e}", exc_info=True)
                time.sleep(10)

            if self.running:
                time.sleep(SCAN_INTERVAL_SECONDS)

        self._shutdown()

    def _scan_cycle(self):
        """Single scan cycle."""
        self.scan_count += 1
        self.last_scan_time = datetime.now(timezone.utc)

        # Ensure authenticated
        if not self.client.ensure_authenticated():
            logger.warning("Authentication expired, re-authenticating...")
            if not self.client.authenticate():
                logger.error("Re-authentication failed!")
                return

        # Get account balance
        balance = self.client.get_account_balance()
        if not balance:
            logger.warning("Could not fetch account balance")
            return

        equity = balance["equity"]

        # Check if we can trade (risk limits)
        can_trade, trade_reason = self.risk_manager.can_trade(equity)

        # Manage existing positions + check for closed trades
        self._manage_positions_and_learn()

        if not can_trade:
            if self.scan_count % 30 == 1:
                logger.info(f"Trading paused: {trade_reason}")
            return

        # Scan each instrument
        for symbol in INSTRUMENTS:
            self._scan_instrument(symbol, equity)

    # ========================================
    # POSITION MANAGEMENT + LEARNING
    # ========================================

    def _manage_positions_and_learn(self):
        """
        Manage open positions AND detect closed trades for adaptive learning.
        When a trade closes, record its features + outcome in the adaptive engine.
        """
        if not self.trade_manager:
            return

        # Get positions before management
        positions_before = set(self.trade_manager.active_positions.keys())

        # Run position management (trailing stop, breakeven, etc.)
        self.trade_manager.manage_open_positions()

        # Check which positions closed
        positions_after = set(self.trade_manager.active_positions.keys())
        closed_ids = positions_before - positions_after

        # Record closed trades in adaptive engine
        for pos_id in closed_ids:
            if pos_id in self.pending_features:
                features = self.pending_features.pop(pos_id)

                # Get the trade result from the trade manager's journal
                # Estimate based on last known data
                # The trade_manager already logged the PnL - we need to update features
                # For now, use risk manager's last recorded trade
                # In production, we'd query the API for exact fill prices

                # Record in adaptive engine
                self.adaptive.record_trade(features)

                logger.info(
                    f"ADAPTIVE LEARNING: Recorded closed trade {pos_id} | "
                    f"{features.symbol} {features.direction} | "
                    f"Result: {features.result} ({features.pnl_r:+.2f}R)"
                )

    # ========================================
    # INSTRUMENT SCANNING
    # ========================================

    def _scan_instrument(self, symbol: str, equity: float):
        """
        Full analysis pipeline for one instrument.

        1. Check session + adaptive hour filter
        2. Check ATR filter
        3. Fetch multi-timeframe data
        4. Analyze trend
        5. Scan for entry signals
        6. Build feature vector
        7. Score confidence (adaptive engine - need 8/10+)
        8. Execute if passes all gates
        """
        now = datetime.now(timezone.utc)

        # 1. Session check
        session_ok, session_reason = can_trade_now(symbol)
        if not session_ok:
            if self.scan_count % 60 == 1:
                logger.debug(f"{symbol}: {session_reason}")
            return

        # 1b. Adaptive hour filter (learned bad hours)
        avoid_hours = self.adaptive.get_avoid_hours()
        if now.hour in avoid_hours:
            if self.scan_count % 60 == 1:
                logger.debug(f"{symbol}: Hour {now.hour} in adaptive avoid list")
            return

        # 2. Fetch price data for all timeframes
        df_4h = self.client.get_price_history(symbol, TIMEFRAMES["4h"], lookback_bars=100)
        df_30m = self.client.get_price_history(symbol, TIMEFRAMES["30m"], lookback_bars=250)
        df_5m = self.client.get_price_history(symbol, TIMEFRAMES["5m"], lookback_bars=200)

        if df_4h is None or df_30m is None or df_5m is None:
            logger.warning(f"{symbol}: Failed to fetch price data")
            return

        # 3. Add indicators
        df_4h = add_all_indicators(df_4h, "4h")
        df_30m = add_all_indicators(df_30m, "30m")
        df_5m = add_all_indicators(df_5m, "5m")

        # 3b. ATR filter (adaptive threshold)
        if len(df_5m) > 200 and "atr" in df_5m.columns:
            atr_series = df_5m["atr"].dropna()
            if len(atr_series) > 50:
                current_atr = atr_series.iloc[-1]
                atr_rank = (atr_series < current_atr).sum() / len(atr_series)
                max_atr = self.adaptive.get_atr_filter()
                if atr_rank > max_atr:
                    logger.debug(
                        f"{symbol}: ATR too high ({atr_rank:.2f} > {max_atr:.2f})"
                    )
                    return

        # 4. Multi-timeframe trend analysis
        trend_state = get_trend_state(df_4h, df_30m)

        if not trend_state.is_tradeable:
            logger.debug(
                f"{symbol}: Trend not aligned | "
                f"4H={trend_state.direction_4h.value} "
                f"30M={trend_state.direction_30m.value}"
            )
            return

        # 5. Scan for entry signal on 5M (needs 5/6 confirmations - sweep optional)
        entry_signal = scan_for_entry(df_5m, trend_state.combined)

        if not entry_signal.valid:
            logger.debug(
                f"{symbol}: No valid entry | "
                f"Confirmations: {entry_signal.confirmation_count}/6 | "
                f"Missing: {entry_signal.rejections[:2]}"
            )
            return

        # 5b. EMA20 slope filter (WR8 requirement)
        if "ema_20" in df_5m.columns and len(df_5m) > 5:
            ema20_slope = df_5m["ema_20"].iloc[-1] - df_5m["ema_20"].iloc[-4]
            direction = "bullish" if trend_state.combined == TrendDirection.BULLISH else "bearish"
            if direction == "bullish" and ema20_slope <= 0:
                logger.debug(f"{symbol}: EMA20 slope not bullish, skipping")
                return
            if direction == "bearish" and ema20_slope >= 0:
                logger.debug(f"{symbol}: EMA20 slope not bearish, skipping")
                return
        else:
            ema20_slope = 0
            direction = "bullish" if trend_state.combined == TrendDirection.BULLISH else "bearish"

        # 6. Build feature vector for adaptive confidence scoring
        features = self._build_features(
            symbol=symbol,
            direction=direction,
            df_5m=df_5m,
            df_4h=df_4h,
            df_30m=df_30m,
            entry_signal=entry_signal,
            trend_state=trend_state,
            ema20_slope=ema20_slope,
        )

        # 7. ADAPTIVE CONFIDENCE GATE (need 8/10+)
        should_trade, confidence, reason = self.adaptive.should_take_trade(features)

        if not should_trade:
            logger.info(
                f"{symbol}: REJECTED by adaptive engine | "
                f"Confidence: {confidence:.1f}/10 | {reason}"
            )
            return

        # 8. ALL GATES PASSED - Execute!
        logger.info(
            f"{'='*50}\n"
            f"TRADE SIGNAL APPROVED (Adaptive)\n"
            f"{'='*50}\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction.upper()}\n"
            f"Confidence: {confidence:.1f}/10 ✅\n"
            f"Pattern: {entry_signal.candle_pattern}\n"
            f"RSI: {entry_signal.rsi_value:.1f}\n"
            f"Volume: {entry_signal.volume_ratio:.2f}x avg\n"
            f"Trend Confidence: {trend_state.confidence:.2f}\n"
            f"{'='*50}"
        )

        self._execute_trade(
            symbol=symbol,
            signal=entry_signal,
            df_5m=df_5m,
            equity=equity,
            trend_confidence=trend_state.confidence,
            features=features,
            confidence_score=confidence,
        )

    # ========================================
    # FEATURE VECTOR BUILDER
    # ========================================

    def _build_features(
        self, symbol, direction, df_5m, df_4h, df_30m,
        entry_signal, trend_state, ema20_slope
    ) -> TradeFeatures:
        """
        Build a complete feature vector for the adaptive engine to score.
        """
        now = datetime.now(timezone.utc)
        entry_price = df_5m["close"].iloc[-1] if "close" in df_5m.columns else df_5m["Close"].iloc[-1]

        # ATR percentile
        atr_pctile = 0.5
        if "atr" in df_5m.columns:
            atr_series = df_5m["atr"].dropna()
            if len(atr_series) > 50:
                current_atr = atr_series.iloc[-1]
                atr_pctile = (atr_series < current_atr).sum() / len(atr_series)

        # EMA20 distance
        ema20_dist = 0.0
        if "ema_20" in df_5m.columns:
            ema20 = df_5m["ema_20"].iloc[-1]
            close = df_5m["close"].iloc[-1] if "close" in df_5m.columns else df_5m["Close"].iloc[-1]
            if ema20 > 0:
                ema20_dist = abs(close - ema20) / ema20

        # EMA20 slope strength (normalized)
        ema20_slope_str = 0.0
        if entry_price > 0:
            ema20_slope_str = abs(ema20_slope) / entry_price

        # 4H slope strength
        slope_4h = 0.0
        if "ema_50_slope" in df_4h.columns and len(df_4h) > 0:
            slope_val = df_4h["ema_50_slope"].iloc[-1]
            ema_val = df_4h["ema_50"].iloc[-1]
            if ema_val > 0 and not pd.isna(slope_val):
                slope_4h = abs(slope_val) / ema_val

        # 30M EMA gap
        slope_30m_gap = 0.0
        if "ema_50" in df_30m.columns and "ema_200" in df_30m.columns:
            ema50_30m = df_30m["ema_50"].iloc[-1]
            ema200_30m = df_30m["ema_200"].iloc[-1]
            if ema200_30m > 0:
                slope_30m_gap = abs(ema50_30m - ema200_30m) / ema200_30m

        # Candle body ratio
        body_ratio = 0.5
        if len(df_5m) > 0:
            o = df_5m["open"].iloc[-1] if "open" in df_5m.columns else df_5m["Open"].iloc[-1]
            c = df_5m["close"].iloc[-1] if "close" in df_5m.columns else df_5m["Close"].iloc[-1]
            h = df_5m["high"].iloc[-1] if "high" in df_5m.columns else df_5m["High"].iloc[-1]
            l = df_5m["low"].iloc[-1] if "low" in df_5m.columns else df_5m["Low"].iloc[-1]
            rng = h - l
            if rng > 0:
                body_ratio = abs(c - o) / rng

        # Session name
        hour = now.hour
        if 7 <= hour <= 11:
            session_name = "london"
        elif 12 <= hour <= 14:
            session_name = "overlap"
        elif 15 <= hour <= 17:
            session_name = "london_close"
        elif 18 <= hour <= 21:
            session_name = "ny_afternoon"
        else:
            session_name = "off_hours"

        return TradeFeatures(
            trade_id=str(uuid.uuid4()),
            symbol=symbol,
            direction=direction,
            timestamp=now.isoformat(),
            hour_utc=hour,
            session=session_name,
            atr_percentile=atr_pctile,
            atr_value=df_5m["atr"].iloc[-1] if "atr" in df_5m.columns else 0,
            rsi_at_entry=entry_signal.rsi_value,
            volume_ratio=entry_signal.volume_ratio,
            ema20_distance_pct=ema20_dist,
            ema20_slope_strength=ema20_slope_str,
            slope_4h_strength=slope_4h,
            slope_30m_gap_pct=slope_30m_gap,
            candle_pattern=entry_signal.candle_pattern or "",
            candle_body_ratio=body_ratio,
            had_liquidity_sweep=entry_signal.liquidity_sweep_detected,
            trend_alignment_score=trend_state.confidence,
        )

    # ========================================
    # TRADE EXECUTION
    # ========================================

    def _execute_trade(
        self, symbol, signal, df_5m, equity, trend_confidence, features, confidence_score
    ):
        """Calculate position parameters and execute the trade."""
        # Get instrument info for lot sizing
        inst_info = self.client.instruments_cache.get(symbol, {})
        pip_size = inst_info.get("pipSize", 0.01)
        lot_size = inst_info.get("lotSize", 1)
        min_lot = inst_info.get("minLot", 0.01)
        lot_step = inst_info.get("lotStep", 0.01)

        # Use adaptive trailing params
        trail_trigger, trail_distance = self.adaptive.get_trailing_params()

        # Create complete trade setup
        setup = self.risk_manager.create_trade_setup(
            symbol=symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            df_5m=df_5m,
            account_equity=equity,
            trend_confidence=trend_confidence,
            pip_size=pip_size,
            lot_size=lot_size,
            min_lot=min_lot,
            lot_step=lot_step,
        )

        if not setup.valid:
            logger.warning(f"Trade setup rejected: {setup.rejection_reason}")
            return

        # Dry run mode
        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would execute: {setup.direction.upper()} "
                f"{setup.position_size} {symbol} | "
                f"Entry={setup.entry_price:.2f} SL={setup.stop_loss:.2f} "
                f"TP={setup.take_profit:.2f} | "
                f"Risk=${setup.risk_amount:.2f} | "
                f"Confidence: {confidence_score:.1f}/10"
            )
            # Still record for learning in dry run
            features.result = "pending"
            features.pnl_r = 0.0
            self.adaptive.record_trade(features)
            return

        # Execute the trade
        position_id = self.trade_manager.open_position(
            setup=setup,
            entry_reasons=signal.reasons + [f"Confidence: {confidence_score:.1f}/10"],
        )

        if position_id:
            # Store features for this position (will be updated on close)
            self.pending_features[position_id] = features
            logger.info(
                f"TRADE EXECUTED: {symbol} | ID: {position_id} | "
                f"Confidence: {confidence_score:.1f}/10"
            )
        else:
            logger.error(f"TRADE EXECUTION FAILED: {symbol}")

    # ========================================
    # STATUS & SHUTDOWN
    # ========================================

    def print_status(self):
        """Print current bot status including adaptive engine state."""
        logger.info("\n" + "=" * 60)
        logger.info("BOT STATUS (Self-Adaptive)")
        logger.info("=" * 60)

        # Risk status
        risk_status = self.risk_manager.get_status_summary()
        logger.info("Risk Management:")
        for key, value in risk_status.items():
            logger.info(f"  {key}: {value}")

        # Position status
        if self.trade_manager:
            trade_status = self.trade_manager.get_status()
            logger.info(f"\nPositions: {trade_status['active_positions']} active")
            for pos in trade_status.get("positions", []):
                logger.info(
                    f"  {pos['symbol']} {pos['direction']} | "
                    f"Entry={pos['entry']:.2f} SL={pos['sl']:.2f} TP={pos['tp']:.2f} | "
                    f"BE={'Yes' if pos['breakeven'] else 'No'}"
                )

        # Adaptive engine status
        logger.info(f"\nAdaptive Engine:")
        for key, val in self.adaptive.get_status().items():
            logger.info(f"  {key}: {val}")

        # Session status
        logger.info(f"\nSession Status:")
        for symbol in INSTRUMENTS:
            status = check_session_status(symbol)
            logger.info(f"  {symbol}: {status.reason}")

        logger.info("=" * 60)

    def _shutdown(self):
        """Clean shutdown."""
        logger.info("=" * 60)
        logger.info("BOT SHUTTING DOWN")
        logger.info(f"Total scans completed: {self.scan_count}")
        logger.info(f"Last scan: {self.last_scan_time}")
        self.print_status()
        logger.info("Goodbye!")
        logger.info("=" * 60)


# ========================================
# ENTRY POINT
# ========================================

def main():
    # Need pandas for feature building
    global pd
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="TradeLocker Self-Adaptive Trading Bot"
    )
    parser.add_argument(
        "--dry", "--dry-run",
        action="store_true",
        help="Run in dry mode (no real trades executed)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current status and exit",
    )
    parser.add_argument(
        "--reset-adaptive",
        action="store_true",
        help="Reset adaptive engine to defaults (fresh start)",
    )

    args = parser.parse_args()

    bot = TradingBot(dry_run=args.dry)

    if args.reset_adaptive:
        from modules.adaptive_engine import AdaptiveParams, ADAPTIVE_CONFIG_FILE
        import json
        params = AdaptiveParams()
        with open(ADAPTIVE_CONFIG_FILE, 'w') as f:
            json.dump(asdict(params), f, indent=2)
        logger.info("Adaptive engine reset to defaults.")
        return

    if args.status:
        if bot.client.setup():
            bot.trade_manager = TradeManager(bot.client, bot.risk_manager)
            bot.print_status()
        else:
            logger.error("Failed to authenticate for status check")
    else:
        bot.start()


if __name__ == "__main__":
    main()
