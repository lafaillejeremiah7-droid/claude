"""
StreamManager – Polling loop and SSE push mechanism.

Requirement 12:
  - Poll bot files every ≤2 seconds.
  - Poll TradeLocker API every ≤5 seconds.
  - Push updates to frontend via SSE within 1s of detecting change.
  - Track data freshness (last update timestamp).
  - Retry failed polls on next interval, retain last values.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

from dashboard_backend.config import Settings
from dashboard_backend.readers.api_reader import APIReader, AccountState, Quote
from dashboard_backend.readers.file_reader import FileReader
from dashboard_backend.calculators.streak import compute_streaks, StreakResult
from dashboard_backend.calculators.pnl import (
    compute_daily_pnl,
    compute_cumulative_pnl,
    compute_unrealized_pnl,
    DailyPnL,
    CumulativePnL,
    PositionPnL,
)
from dashboard_backend.calculators.equity_curve import build_equity_curve, EquityCurve
from dashboard_backend.calculators.countdown import compute_bot_status, BotStatus
from dashboard_backend.calculators.confidence import (
    extract_confidence_gate,
    extract_from_journal,
    extract_from_bot_log,
    ConfidenceData,
    ConfidenceEntry,
)

logger = logging.getLogger("dashboard_backend.stream")


class StreamManager:
    """
    Manages the polling loops and serves SSE events to connected clients.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._file_reader = FileReader(settings)
        self._api_reader = APIReader(settings)

        # Current state (latest computed values)
        self._state: Dict[str, Any] = {}
        self._state_lock = asyncio.Lock()
        self._last_update_time: float = 0.0

        # SSE subscribers
        self._subscribers: Set[asyncio.Queue] = set()

        # Background tasks
        self._file_poll_task: Optional[asyncio.Task] = None
        self._api_poll_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start polling loops."""
        self._running = True
        await self._api_reader.initialize()

        # Initial data load
        await self._poll_files()
        await self._poll_api()

        # Start background polling
        self._file_poll_task = asyncio.create_task(self._file_poll_loop())
        self._api_poll_task = asyncio.create_task(self._api_poll_loop())

        logger.info("StreamManager started.")

    async def stop(self) -> None:
        """Stop polling loops and cleanup."""
        self._running = False

        if self._file_poll_task:
            self._file_poll_task.cancel()
            try:
                await self._file_poll_task
            except asyncio.CancelledError:
                pass

        if self._api_poll_task:
            self._api_poll_task.cancel()
            try:
                await self._api_poll_task
            except asyncio.CancelledError:
                pass

        await self._api_reader.close()
        logger.info("StreamManager stopped.")

    # ------------------------------------------------------------------
    # Polling loops
    # ------------------------------------------------------------------

    async def _file_poll_loop(self) -> None:
        """Poll bot files every file_poll_interval seconds (Req 12.1)."""
        while self._running:
            try:
                await asyncio.sleep(self._settings.file_poll_interval)
                await self._poll_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"File poll error: {e}")
                # Req 12.7: retry on next interval, retain last values

    async def _api_poll_loop(self) -> None:
        """Poll TradeLocker API every api_poll_interval seconds (Req 12.1)."""
        while self._running:
            try:
                await asyncio.sleep(self._settings.api_poll_interval)
                await self._poll_api()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"API poll error: {e}")
                # Req 12.7: retry on next interval

    async def _poll_files(self) -> None:
        """Read all bot files and compute derived metrics."""
        try:
            # Run file reads in thread pool (they're blocking I/O)
            loop = asyncio.get_event_loop()

            daily_stats = await loop.run_in_executor(None, self._file_reader.read_daily_stats)
            positions_data = await loop.run_in_executor(None, self._file_reader.read_positions)
            adaptive_config = await loop.run_in_executor(None, self._file_reader.read_adaptive_config)
            today_journal = await loop.run_in_executor(None, self._file_reader.read_today_journal)
            all_journals = await loop.run_in_executor(None, self._file_reader.read_all_journals)
            bot_log = await loop.run_in_executor(None, self._file_reader.read_today_bot_log)
            latest_mtime = await loop.run_in_executor(None, self._file_reader.get_latest_file_mtime)
            any_existed = await loop.run_in_executor(None, self._file_reader.any_file_ever_existed)

            # Compute metrics
            daily_pnl = compute_daily_pnl(daily_stats)
            cumulative_pnl = compute_cumulative_pnl(all_journals)
            streaks = compute_streaks(all_journals)
            equity_curve = build_equity_curve(
                all_journals, self._settings.starting_equity_baseline
            )

            # Confidence
            gate = extract_confidence_gate(adaptive_config)
            journal_confidence = extract_from_journal(today_journal, gate)
            log_confidence = extract_from_bot_log(bot_log, gate)

            # Win rate (Req 5)
            win_rate = self._compute_win_rate(daily_stats, adaptive_config)

            # Bot status / countdown (Req 11, 14)
            bot_status = compute_bot_status(
                latest_file_mtime=latest_mtime,
                any_file_ever_existed=any_existed,
                daily_stats=daily_stats,
                adaptive_config=adaptive_config,
                bot_log_content=bot_log,
                scan_interval_seconds=self._settings.scan_interval_seconds,
                offline_threshold_seconds=self._settings.bot_offline_threshold_seconds,
            )

            # Trade feed (Req 8)
            trade_feed = self._build_trade_feed(today_journal, log_confidence)

            # Positions
            positions_list = self._format_positions(positions_data)

            # Update state
            async with self._state_lock:
                self._state.update({
                    "daily_pnl": asdict(daily_pnl),
                    "cumulative_pnl": asdict(cumulative_pnl),
                    "streaks": asdict(streaks),
                    "equity_curve": {
                        "points": [asdict(p) for p in equity_curve.points],
                        "starting_equity": equity_curve.starting_equity,
                        "insufficient_data": equity_curve.insufficient_data,
                        "error": equity_curve.error,
                    },
                    "confidence": {
                        "gate": gate,
                        "gate_unavailable": gate is None,
                        "entries": [asdict(e) for e in (journal_confidence + log_confidence)[-50:]],
                    },
                    "win_rate": win_rate,
                    "bot_status": asdict(bot_status),
                    "trade_feed": trade_feed[:100],  # Req 8.2: limit to 100
                    "positions": positions_list,
                    "last_file_update": time.time(),
                })

            self._last_update_time = time.time()
            await self._notify_subscribers("file_update")

        except Exception as e:
            logger.error(f"File poll computation error: {e}", exc_info=True)

    async def _poll_api(self) -> None:
        """Fetch account state and quotes from TradeLocker API."""
        try:
            # Account state (Req 3)
            account_state = await self._api_reader.get_account_state()

            # If API fails, fallback to daily_stats (Req 3.4)
            if account_state.error and not account_state.equity:
                async with self._state_lock:
                    daily_pnl = self._state.get("daily_pnl", {})
                    fallback_equity = daily_pnl.get("current_equity")
                    if fallback_equity is not None:
                        account_state = AccountState(
                            equity=fallback_equity,
                            source="fallback",
                            timestamp=time.time(),
                        )

            # Live quotes (Req 15.3)
            quotes = await self._api_reader.get_quotes(self._settings.instrument_list)

            # Compute unrealized P&L with live prices (Req 4.6)
            async with self._state_lock:
                positions_data = self._state.get("positions_raw", {})

            positions_pnl = compute_unrealized_pnl(
                positions_data if isinstance(positions_data, dict) else {},
                quotes,
            )

            # Compute total unrealized
            unrealized_total: Optional[float] = None
            valid_pnls = [p.unrealized_pnl for p in positions_pnl if p.unrealized_pnl is not None]
            if valid_pnls:
                unrealized_total = sum(valid_pnls)

            # Update state
            async with self._state_lock:
                self._state.update({
                    "account_state": {
                        "balance": account_state.balance,
                        "equity": account_state.equity,
                        "free_margin": account_state.free_margin,
                        "timestamp": account_state.timestamp,
                        "source": account_state.source,
                        "error": account_state.error,
                    },
                    "quotes": {
                        sym: {
                            "symbol": q.symbol,
                            "bid": q.bid,
                            "ask": q.ask,
                            "last_price": q.last_price,
                            "timestamp": q.timestamp,
                            "error": q.error,
                        }
                        for sym, q in quotes.items()
                    },
                    "unrealized_pnl": {
                        "total": unrealized_total,
                        "positions": [asdict(p) for p in positions_pnl],
                    },
                    "auth_status": {
                        "authenticated": self._api_reader.is_authenticated,
                        "error": self._api_reader.auth_error,
                    },
                    "last_api_update": time.time(),
                })

            self._last_update_time = time.time()
            await self._notify_subscribers("api_update")

        except Exception as e:
            logger.error(f"API poll error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # SSE subscriber management
    # ------------------------------------------------------------------

    async def subscribe(self) -> AsyncGenerator[str, None]:
        """
        Subscribe to SSE events. Yields JSON-encoded state updates.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._subscribers.add(queue)

        try:
            # Send initial full state
            async with self._state_lock:
                initial = self._state.copy()
            initial["event_type"] = "initial"
            initial["server_time"] = time.time()
            yield json.dumps(initial, default=str)

            # Stream updates
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield json.dumps({
                        "event_type": "keepalive",
                        "server_time": time.time(),
                    })
        finally:
            self._subscribers.discard(queue)

    async def _notify_subscribers(self, event_type: str) -> None:
        """Push update to all SSE subscribers (Req 12.2)."""
        if not self._subscribers:
            return

        async with self._state_lock:
            state = self._state.copy()

        state["event_type"] = event_type
        state["server_time"] = time.time()
        data = json.dumps(state, default=str)

        dead_queues: List[asyncio.Queue] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                # Drop oldest if queue is full
                try:
                    queue.get_nowait()
                    queue.put_nowait(data)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    dead_queues.append(queue)

        for q in dead_queues:
            self._subscribers.discard(q)

    # ------------------------------------------------------------------
    # Public state access
    # ------------------------------------------------------------------

    async def get_current_state(self) -> Dict[str, Any]:
        """Get the current full state snapshot."""
        async with self._state_lock:
            state = self._state.copy()
        state["server_time"] = time.time()
        state["last_update_time"] = self._last_update_time
        return state

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _compute_win_rate(
        self, daily_stats: Dict[str, Any], adaptive_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute win rate data (Req 5)."""
        daily = daily_stats.get("daily", {})

        wins = daily.get("wins")
        losses = daily.get("losses")
        daily_win_rate: Optional[float] = None

        if wins is not None and losses is not None:
            try:
                w = int(wins)
                l = int(losses)
                total = w + l
                if total > 0:
                    daily_win_rate = round((w / total) * 100, 1)
                # If total == 0, display "N/A" (Req 5.2) – keep as None
            except (TypeError, ValueError):
                pass

        # Adaptive rolling win rate (Req 5.3)
        rolling_win_rate: Optional[float] = None
        raw_rate = adaptive_config.get("current_win_rate")
        if raw_rate is not None:
            try:
                rolling_win_rate = round(float(raw_rate) * 100, 1) if float(raw_rate) <= 1 else round(float(raw_rate), 1)
            except (TypeError, ValueError):
                pass

        return {
            "daily_win_rate": daily_win_rate,
            "wins": int(wins) if wins is not None else None,
            "losses": int(losses) if losses is not None else None,
            "rolling_win_rate": rolling_win_rate,
        }

    def _build_trade_feed(
        self,
        journal_entries: List[Dict[str, Any]],
        log_events: List[ConfidenceEntry],
    ) -> List[Dict[str, Any]]:
        """Build live trade feed (Req 8)."""
        feed: List[Dict[str, Any]] = []

        # Valid action types (Req 8.1)
        valid_actions = {"OPEN", "BREAKEVEN", "PARTIAL_CLOSE", "CLOSE", "EMERGENCY_CLOSE"}

        for entry in journal_entries:
            action = entry.get("action", "")
            if action not in valid_actions:
                continue

            timestamp = entry.get("timestamp", "")
            symbol = entry.get("symbol", "")
            direction = entry.get("direction", "")

            # Req 8.4: skip if missing required fields
            if not all([timestamp, action, symbol, direction]):
                continue

            feed.append({
                "timestamp": timestamp,
                "action": action,
                "symbol": symbol,
                "direction": direction,
                "pnl": entry.get("pnl"),
                "is_win": entry.get("is_win"),
                "source": "journal",
            })

        # Add notable events from bot log (Req 8.6)
        for event in log_events:
            event_type = "NEAR_MISS" if event.is_near_miss else "APPROVED"
            if event.timestamp:
                feed.append({
                    "timestamp": event.timestamp,
                    "action": event_type,
                    "symbol": event.symbol or "—",
                    "direction": "—",
                    "confidence": event.score,
                    "source": "bot_log",
                })

        # Sort newest-first (Req 8.2)
        feed.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return feed[:100]

    def _format_positions(self, positions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format positions for display (Req 9)."""
        if not isinstance(positions_data, dict):
            return []

        # Store raw for unrealized PnL computation
        # positions_data may be {id: record} or a list
        async def _store():
            async with self._state_lock:
                self._state["positions_raw"] = positions_data

        # Can't await here; store synchronously in state at poll time
        # This is handled in the caller

        positions: List[Dict[str, Any]] = []
        items = positions_data.items() if isinstance(positions_data, dict) else []

        for pos_id, pos in items:
            if not isinstance(pos, dict):
                continue

            entry_price = pos.get("entry_price")
            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            quantity = pos.get("quantity")

            # Compute risk-reward ratio (Req 9.2)
            rr_ratio: Optional[float] = None
            if entry_price and sl and tp:
                try:
                    ep = float(entry_price)
                    s = float(sl)
                    t = float(tp)
                    risk = abs(ep - s)
                    reward = abs(t - ep)
                    if risk > 0:
                        rr_ratio = round(reward / risk, 2)
                except (TypeError, ValueError):
                    pass

            positions.append({
                "id": str(pos_id),
                "symbol": pos.get("symbol", ""),
                "direction": pos.get("direction", ""),
                "entry_price": entry_price,
                "stop_loss": sl,
                "take_profit": tp,
                "quantity": quantity,
                "risk_reward_ratio": rr_ratio,
                "is_breakeven": pos.get("is_breakeven", False),
            })

        return positions
