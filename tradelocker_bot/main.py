"""
TradeLocker Trading Bot - Main Orchestrator

Autonomous trading bot that implements a multi-timeframe EMA strategy
with liquidity sweep detection for BTC/USD and XAU/USD.

Flow:
1. Authenticate with TradeLocker
2. Every 60 seconds:
   a. Check if trading is allowed (session, risk limits)
   b. Manage existing positions (breakeven moves)
   c. For each instrument:
      - Fetch 4H, 30M, 5M data
      - Analyze multi-timeframe trend
      - If trend aligned, scan for entry signals on 5M
      - If valid signal, calculate risk and execute trade
3. Log everything for review

Usage:
    python main.py          # Run the bot
    python main.py --dry    # Dry run (no real trades)
    python main.py --status # Show current status
"""
import sys
import time
import signal
import logging
import argparse
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
)
from modules.api_client import TradeLockerClient
from modules.indicators import add_all_indicators
from modules.trend_analysis import get_trend_state, TrendDirection
from modules.entry_signals import scan_for_entry
from modules.risk_management import RiskManager
from modules.session_filter import can_trade_now, check_session_status
from modules.trade_manager import TradeManager

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
# BOT CLASS
# ========================================

class TradingBot:
    """Main trading bot orchestrator."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.running = False
        self.scan_count = 0
        self.last_scan_time = None

        # Core components
        self.client = TradeLockerClient()
        self.risk_manager = RiskManager()
        self.trade_manager = None  # Initialized after client setup

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
        logger.info("TRADELOCKER TRADING BOT STARTING")
        logger.info("=" * 60)
        logger.info(f"Environment: {TL_ENVIRONMENT.upper()}")
        logger.info(f"Server: {TL_SERVER}")
        logger.info(f"Instruments: {INSTRUMENTS}")
        logger.info(f"Dry Run: {self.dry_run}")
        logger.info(f"Scan Interval: {SCAN_INTERVAL_SECONDS}s")
        logger.info(f"Risk Per Trade: {self.risk_manager.risk_percent}%")
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

        # Start main loop
        self.running = True
        logger.info("Bot initialized successfully. Starting main loop...")
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
                time.sleep(10)  # Brief pause on error before retry

            # Wait for next scan
            if self.running:
                time.sleep(SCAN_INTERVAL_SECONDS)

        self._shutdown()

    def _scan_cycle(self):
        """Single scan cycle - runs every SCAN_INTERVAL_SECONDS."""
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

        # Manage existing positions (always, even if can't open new)
        self.trade_manager.manage_open_positions()

        if not can_trade:
            if self.scan_count % 30 == 1:  # Log every 30 scans (~30 min)
                logger.info(f"Trading paused: {trade_reason}")
            return

        # Scan each instrument
        for symbol in INSTRUMENTS:
            self._scan_instrument(symbol, equity)

    def _scan_instrument(self, symbol: str, equity: float):
        """
        Full analysis pipeline for one instrument.

        1. Check session
        2. Fetch multi-timeframe data
        3. Analyze trend
        4. Scan for entry
        5. Execute if valid
        """
        # 1. Session check
        session_ok, session_reason = can_trade_now(symbol)
        if not session_ok:
            if self.scan_count % 60 == 1:  # Log every hour
                logger.debug(f"{symbol}: {session_reason}")
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

        # 4. Multi-timeframe trend analysis
        trend_state = get_trend_state(df_4h, df_30m)

        if not trend_state.is_tradeable:
            logger.debug(
                f"{symbol}: Trend not aligned | "
                f"4H={trend_state.direction_4h.value} "
                f"30M={trend_state.direction_30m.value}"
            )
            return

        # 5. Scan for entry signal on 5M
        entry_signal = scan_for_entry(df_5m, trend_state.combined)

        if not entry_signal.valid:
            logger.debug(
                f"{symbol}: No valid entry | "
                f"Confirmations: {entry_signal.confirmation_count}/6 | "
                f"Missing: {entry_signal.rejections[:2]}"
            )
            return

        # 6. Valid signal! Calculate risk and execute
        logger.info(
            f"{'='*40}\n"
            f"ENTRY SIGNAL DETECTED: {symbol}\n"
            f"Direction: {entry_signal.direction.upper()}\n"
            f"Confirmations: {entry_signal.confirmation_count}/6\n"
            f"Pattern: {entry_signal.candle_pattern}\n"
            f"RSI: {entry_signal.rsi_value:.1f}\n"
            f"Volume: {entry_signal.volume_ratio:.2f}x avg\n"
            f"Trend Confidence: {trend_state.confidence:.2f}\n"
            f"{'='*40}"
        )

        self._execute_trade(symbol, entry_signal, df_5m, equity, trend_state.confidence)

    # ========================================
    # TRADE EXECUTION
    # ========================================

    def _execute_trade(
        self,
        symbol: str,
        signal,
        df_5m,
        equity: float,
        trend_confidence: float,
    ):
        """Calculate position parameters and execute the trade."""
        # Get instrument info for lot sizing
        inst_info = self.client.instruments_cache.get(symbol, {})
        pip_size = inst_info.get("pipSize", 0.01)
        lot_size = inst_info.get("lotSize", 1)
        min_lot = inst_info.get("minLot", 0.01)
        lot_step = inst_info.get("lotStep", 0.01)

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
            logger.warning(
                f"Trade setup rejected: {setup.rejection_reason}"
            )
            return

        # Dry run mode - log but don't execute
        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would execute: {setup.direction.upper()} "
                f"{setup.position_size} {symbol} | "
                f"Entry={setup.entry_price:.2f} SL={setup.stop_loss:.2f} "
                f"TP={setup.take_profit:.2f} | "
                f"Risk=${setup.risk_amount:.2f} R:R={setup.risk_reward_ratio:.2f}"
            )
            return

        # Execute the trade
        position_id = self.trade_manager.open_position(
            setup=setup,
            entry_reasons=signal.reasons,
        )

        if position_id:
            logger.info(f"TRADE EXECUTED: {symbol} | Position ID: {position_id}")
        else:
            logger.error(f"TRADE EXECUTION FAILED: {symbol}")

    # ========================================
    # STATUS & SHUTDOWN
    # ========================================

    def print_status(self):
        """Print current bot status."""
        logger.info("\n" + "=" * 50)
        logger.info("BOT STATUS")
        logger.info("=" * 50)

        # Risk status
        risk_status = self.risk_manager.get_status_summary()
        logger.info(f"Risk Management:")
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

        # Session status
        logger.info(f"\nSession Status:")
        for symbol in INSTRUMENTS:
            status = check_session_status(symbol)
            logger.info(f"  {symbol}: {status.reason}")

        logger.info("=" * 50)

    def _shutdown(self):
        """Clean shutdown."""
        logger.info("=" * 60)
        logger.info("BOT SHUTTING DOWN")
        logger.info(f"Total scans completed: {self.scan_count}")
        logger.info(f"Last scan: {self.last_scan_time}")

        # Print final status
        self.print_status()

        logger.info("Goodbye!")
        logger.info("=" * 60)


# ========================================
# ENTRY POINT
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="TradeLocker Multi-Timeframe Trading Bot"
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

    args = parser.parse_args()

    bot = TradingBot(dry_run=args.dry)

    if args.status:
        # Just authenticate and show status
        if bot.client.setup():
            bot.trade_manager = TradeManager(bot.client, bot.risk_manager)
            bot.print_status()
        else:
            logger.error("Failed to authenticate for status check")
    else:
        bot.start()


if __name__ == "__main__":
    main()
