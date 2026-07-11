"""
Discord Signal Delivery for XAUUSD Bot V2.

Sends trade signals via Discord webhook.

Setup:
    1. Create a webhook in your Discord channel settings
    2. Copy the webhook URL
    3. Set as environment variable or pass directly
"""

import os
import json
from datetime import datetime
from typing import Optional

try:
    import aiohttp
    import asyncio
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class DiscordSignalBot:
    """Delivers XAUUSD signals to Discord via webhook."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")

    def format_signal_embed(self, signal) -> dict:
        """Format signal as Discord embed."""
        color = 0x3fb950 if signal.direction == "BUY" else 0xf85149
        arrow = "BUY" if signal.direction == "BUY" else "SELL"

        embed = {
            "title": f"XAUUSD {arrow} SIGNAL",
            "color": color,
            "fields": [
                {"name": "Entry", "value": f"${signal.entry_price:,.2f}", "inline": True},
                {"name": "Stop Loss", "value": f"${signal.stop_loss:,.2f}", "inline": True},
                {"name": "Take Profit", "value": f"${signal.take_profit:,.2f}", "inline": True},
                {"name": "Risk:Reward", "value": signal.risk_reward, "inline": True},
                {"name": "Confidence", "value": signal.confidence, "inline": True},
                {"name": "Condition", "value": signal.condition, "inline": True},
            ],
            "footer": {
                "text": "Set & Forget | No management needed"
            },
            "timestamp": signal.timestamp.isoformat() if signal.timestamp else datetime.utcnow().isoformat(),
        }
        return embed

    def format_close_embed(self, trade: dict) -> dict:
        """Format trade closure as Discord embed."""
        is_win = trade["result"] == "WIN"
        color = 0x3fb950 if is_win else 0xf85149

        pnl_text = f"+${trade['pnl_dollars']:.2f}" if is_win else f"-${abs(trade['pnl_dollars']):.2f}"

        embed = {
            "title": f"TRADE CLOSED — {'WIN' if is_win else 'LOSS'}",
            "color": color,
            "fields": [
                {"name": "Direction", "value": trade["direction"], "inline": True},
                {"name": "Entry", "value": f"${trade['entry_price']:,.2f}", "inline": True},
                {"name": "Exit", "value": f"${trade['exit_price']:,.2f}", "inline": True},
                {"name": "P&L", "value": f"**{pnl_text}** ({trade['pnl_pips']:.0f} pips)", "inline": True},
                {"name": "Reason", "value": trade["reason"], "inline": True},
            ],
            "timestamp": trade["exit_time"].isoformat() if trade.get("exit_time") else datetime.utcnow().isoformat(),
        }
        return embed

    def format_summary_embed(self, stats: dict) -> dict:
        """Format daily summary as Discord embed."""
        embed = {
            "title": "DAILY SUMMARY — XAUUSD",
            "color": 0xd29922,
            "fields": [
                {"name": "Trades", "value": str(stats.get("total_trades", 0)), "inline": True},
                {"name": "Win Rate", "value": stats.get("win_rate", "0%"), "inline": True},
                {"name": "P&L", "value": stats.get("total_pnl", "$0"), "inline": True},
                {"name": "Profit Factor", "value": stats.get("profit_factor", "0"), "inline": True},
                {"name": "Max Drawdown", "value": stats.get("max_drawdown", "0%"), "inline": True},
                {"name": "Equity", "value": stats.get("equity", "$0"), "inline": True},
            ],
            "footer": {"text": "Strategy: Range Reversals | 1:1 RR | Set & Forget"},
            "timestamp": datetime.utcnow().isoformat(),
        }
        return embed

    async def send_embed(self, embed: dict) -> bool:
        """Send an embed to Discord webhook."""
        if not HAS_AIOHTTP:
            print(f"[DISCORD] Would send embed: {embed['title']}")
            return True

        if not self.webhook_url:
            print(f"[DISCORD] No webhook configured. Embed: {embed['title']}")
            return False

        payload = {"embeds": [embed]}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    return resp.status in (200, 204)
        except Exception as e:
            print(f"[DISCORD] Error: {e}")
            return False

    def send_embed_sync(self, embed: dict) -> bool:
        """Synchronous wrapper."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.send_embed(embed))
                return True
            return loop.run_until_complete(self.send_embed(embed))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.send_embed(embed))

    def send_signal(self, signal) -> bool:
        embed = self.format_signal_embed(signal)
        return self.send_embed_sync(embed)

    def send_close(self, trade: dict) -> bool:
        embed = self.format_close_embed(trade)
        return self.send_embed_sync(embed)

    def send_daily_summary(self, stats: dict) -> bool:
        embed = self.format_summary_embed(stats)
        return self.send_embed_sync(embed)
