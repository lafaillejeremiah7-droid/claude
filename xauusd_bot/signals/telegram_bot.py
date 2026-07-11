"""
Telegram Signal Delivery for XAUUSD Bot V2.

Sends trade signals, trade closures, and daily summaries
to a Telegram channel/group.

Setup:
    1. Create bot via @BotFather on Telegram
    2. Get bot token
    3. Get channel/group chat_id
    4. Set in environment variables or pass directly
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str = ""
    chat_id: str = ""
    parse_mode: str = "HTML"
    disable_notification: bool = False


class TelegramSignalBot:
    """
    Delivers XAUUSD signals to Telegram.

    Signal Format:
        XAUUSD BUY SIGNAL
        Entry: $4,086.05
        SL: $4,085.10
        TP: $4,087.00
        RR: 1:1
        Confidence: 4/6
        Condition: Range Reversal (Asian)
    """

    def __init__(self, config: Optional[TelegramConfig] = None):
        if config is None:
            config = TelegramConfig(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            )
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{config.bot_token}"

    # ------------------------------------------------------------------
    # SIGNAL MESSAGES
    # ------------------------------------------------------------------

    def format_signal(self, signal) -> str:
        """Format a trade signal for Telegram delivery."""
        direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
        arrow = "⬆️" if signal.direction == "BUY" else "⬇️"

        msg = (
            f"{direction_emoji} <b>XAUUSD {signal.direction} SIGNAL</b> {arrow}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 <b>Entry:</b> ${signal.entry_price:,.2f}\n"
            f"🛑 <b>Stop Loss:</b> ${signal.stop_loss:,.2f}\n"
            f"🎯 <b>Take Profit:</b> ${signal.take_profit:,.2f}\n"
            f"📊 <b>Risk:Reward:</b> {signal.risk_reward}\n"
            f"🔥 <b>Confidence:</b> {signal.confidence}\n"
            f"📋 <b>Condition:</b> {signal.condition}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ <i>Set & Forget — No management needed</i>\n"
            f"⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M UTC') if signal.timestamp else 'N/A'}"
        )
        return msg

    def format_close(self, trade: dict) -> str:
        """Format a trade closure message."""
        if trade["result"] == "WIN":
            emoji = "✅"
            result_text = f"+${trade['pnl_dollars']:.2f}"
        else:
            emoji = "❌"
            result_text = f"-${abs(trade['pnl_dollars']):.2f}"

        duration = ""
        if trade.get("entry_time") and trade.get("exit_time"):
            dur = (trade["exit_time"] - trade["entry_time"]).total_seconds() / 60
            duration = f"{int(dur)} min"

        msg = (
            f"{emoji} <b>TRADE CLOSED — {trade['result']}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Entry: ${trade['entry_price']:,.2f}\n"
            f"📍 Exit: ${trade['exit_price']:,.2f}\n"
            f"💰 P&L: <b>{result_text}</b> ({trade['pnl_pips']:.0f} pips)\n"
            f"📋 Reason: {trade['reason']}\n"
            f"⏱️ Duration: {duration}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )
        return msg

    def format_daily_summary(self, stats: dict) -> str:
        """Format daily performance summary."""
        msg = (
            f"📊 <b>DAILY SUMMARY — XAUUSD</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 Trades: {stats.get('total_trades', 0)}\n"
            f"✅ Wins: {stats.get('wins', 0)} | ❌ Losses: {stats.get('losses', 0)}\n"
            f"🎯 Win Rate: {stats.get('win_rate', '0%')}\n"
            f"💰 P&L: {stats.get('total_pnl', '$0')}\n"
            f"📉 Max DD: {stats.get('max_drawdown', '0%')}\n"
            f"💎 Equity: {stats.get('equity', '$0')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 Strategy: Range Reversals | 1:1 RR\n"
            f"📅 {datetime.utcnow().strftime('%Y-%m-%d')}"
        )
        return msg

    # ------------------------------------------------------------------
    # SENDING
    # ------------------------------------------------------------------

    async def send_message(self, text: str) -> bool:
        """Send a message to Telegram (async)."""
        if not HAS_AIOHTTP:
            print(f"[TELEGRAM] Would send:\n{text}")
            return True

        if not self.config.bot_token or not self.config.chat_id:
            print(f"[TELEGRAM] Not configured. Message:\n{text}")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": self.config.parse_mode,
            "disable_notification": self.config.disable_notification,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    return resp.status == 200
        except Exception as e:
            print(f"[TELEGRAM] Error: {e}")
            return False

    def send_message_sync(self, text: str) -> bool:
        """Synchronous wrapper for sending messages."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create task
                asyncio.ensure_future(self.send_message(text))
                return True
            else:
                return loop.run_until_complete(self.send_message(text))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.send_message(text))

    # ------------------------------------------------------------------
    # CONVENIENCE METHODS
    # ------------------------------------------------------------------

    def send_signal(self, signal) -> bool:
        """Send a trade signal."""
        msg = self.format_signal(signal)
        return self.send_message_sync(msg)

    def send_close(self, trade: dict) -> bool:
        """Send a trade closure notification."""
        msg = self.format_close(trade)
        return self.send_message_sync(msg)

    def send_daily_summary(self, stats: dict) -> bool:
        """Send daily performance summary."""
        msg = self.format_daily_summary(stats)
        return self.send_message_sync(msg)

    def send_alert(self, text: str) -> bool:
        """Send a custom alert."""
        msg = f"⚠️ <b>ALERT</b>\n{text}"
        return self.send_message_sync(msg)
