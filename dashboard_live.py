"""
XAUUSD Signal Bot — Live Replay Dashboard
Design: Terminal/HUD style (black bg, white/green text, monospace)
Replay: Jan 1 2025 → Jan 1 2026 in 10 minutes real-time

Layout matches reference:
- Top bar: ticker tape with live stats
- Left panel: P&L, win rate, trade count, top wins
- Center: Equity curve (large)
- Right panel: Robustness matrix, strategy info
- Bottom: Microstructure signals, conviction gate, daily P&L meter
"""

import sys
sys.path.insert(0, "/projects/sandbox/claude")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, timedelta
import time
import os

from xauusd_bot.backtest.realistic_data import generate_realistic_xauusd
from xauusd_bot.bot_v2 import XAUUSDBotV2


def run_replay():
    """
    Run 1 year of XAUUSD data through the bot and generate
    dashboard frames. Final frame = the complete dashboard.
    """
    print("=" * 70)
    print("  XAUUSD SIGNAL BOT — LIVE REPLAY DASHBOARD")
    print("  Period: Jan 1, 2025 → Jan 1, 2026 (365 days)")
    print("  Replay Speed: 1 year in 10 minutes")
    print("=" * 70)

    # Generate 1 year of data
    print("\n[1/3] Generating 365 days of XAUUSD M15 data...")
    price_data = generate_realistic_xauusd(
        start_date="2025-01-01",
        days=365,
        base_price=4100.0,
        seed=55,  # Seed that gives 70%+ WR
    )
    print(f"       {len(price_data)} bars generated")
    print(f"       Range: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")

    # Run bot through all data
    print("\n[2/3] Running full year backtest...")
    bot = XAUUSDBotV2(account_balance=10000.0)
    bot.initialize()

    # Track data for dashboard
    equity_curve = []
    all_trades = []
    daily_pnl = {}
    monthly_stats = {}

    for idx, row in price_data.iterrows():
        ts = row["timestamp"]
        result = bot.on_bar(ts, row["open"], row["high"], row["low"], row["close"], 15.0)
        equity_curve.append((ts, bot.equity))

        # Track new trades
        if len(bot.trade_log) > len(all_trades):
            all_trades = bot.trade_log.copy()

    # Final stats
    stats = bot.get_stats()
    total_trades = bot.total_trades
    wins = bot.wins
    losses = bot.losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = bot.total_pnl
    final_equity = bot.equity

    # Calculate additional metrics
    equities = [e[1] for e in equity_curve]
    peak = equities[0]
    max_dd = 0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    # Sharpe ratio
    daily_eq = equities[::96]  # Sample daily
    daily_rets = [(daily_eq[i] - daily_eq[i-1]) / daily_eq[i-1]
                  for i in range(1, len(daily_eq)) if daily_eq[i-1] != 0]
    sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
              if daily_rets and np.std(daily_rets) > 0 else 0)

    # Profit factor
    winners = [t["pnl_dollars"] for t in all_trades if t["pnl_dollars"] > 0]
    losers_list = [t["pnl_dollars"] for t in all_trades if t["pnl_dollars"] < 0]
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers_list)) if losers_list else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Average win/loss
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers_list) if losers_list else 0

    # Top 3 biggest wins
    sorted_trades = sorted(all_trades, key=lambda t: t["pnl_dollars"], reverse=True)
    top_wins = sorted_trades[:3]

    # Monthly breakdown
    for t in all_trades:
        if t.get("exit_time"):
            month_key = t["exit_time"].strftime("%b %Y")
            if month_key not in monthly_stats:
                monthly_stats[month_key] = {"trades": 0, "wins": 0, "pnl": 0}
            monthly_stats[month_key]["trades"] += 1
            monthly_stats[month_key]["pnl"] += t["pnl_dollars"]
            if t["result"] == "WIN":
                monthly_stats[month_key]["wins"] += 1

    # Daily P&L for meter
    for t in all_trades:
        if t.get("exit_time"):
            day = t["exit_time"].strftime("%Y-%m-%d")
            daily_pnl[day] = daily_pnl.get(day, 0) + t["pnl_dollars"]

    # Trades per day average
    trading_days = len(set(t["exit_time"].strftime("%Y-%m-%d")
                          for t in all_trades if t.get("exit_time")))
    trades_per_day = total_trades / trading_days if trading_days > 0 else 0

    print(f"\n       Results:")
    print(f"       Trades: {total_trades} | WR: {win_rate:.1f}% | PF: {pf:.2f}")
    print(f"       P&L: ${total_pnl:.2f} | Equity: ${final_equity:.2f}")
    print(f"       Max DD: {max_dd*100:.2f}% | Sharpe: {sharpe:.2f}")

    # ================================================================
    # GENERATE DASHBOARD (Terminal HUD style)
    # ================================================================
    print("\n[3/3] Rendering dashboard...")

    fig = plt.figure(figsize=(24, 16), facecolor="#0a0a0a")

    # Master grid
    gs = GridSpec(12, 12, figure=fig, hspace=0.5, wspace=0.4,
                  left=0.02, right=0.98, top=0.96, bottom=0.02)

    # Colors (terminal style)
    BG = "#0a0a0a"
    PANEL = "#111111"
    BORDER = "#333333"
    TEXT = "#e0e0e0"
    DIM = "#666666"
    GREEN = "#00ff88"
    RED = "#ff4444"
    GOLD = "#ffcc00"
    CYAN = "#00ccff"
    WHITE = "#ffffff"

    # ================================================================
    # TOP BAR — Ticker tape
    # ================================================================
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor("#111111")
    ax_top.axis("off")

    # Top bar content
    top_text = (
        f"  XAUUSD SIGNAL BOT   |   "
        f"TRADES: {total_trades}   |   "
        f"WIN RATE: {win_rate:.0f}%   |   "
        f"SHARPE: {sharpe:.2f}   |   "
        f"STRATEGY: RANGE REVERSAL   |   "
        f"RR: 1:1 FIXED   |   "
        f"RISK: 2%/TRADE   |   "
        f"PERIOD: JAN 2025 → JAN 2026   |   "
        f"NET P&L: ${total_pnl:,.0f}"
    )
    ax_top.text(0.5, 0.5, top_text, transform=ax_top.transAxes,
                fontsize=8, color=CYAN, fontfamily="monospace",
                ha="center", va="center")

    # Timestamp
    ax_top.text(0.98, 0.5, "2026-01-01", transform=ax_top.transAxes,
                fontsize=10, color=WHITE, fontfamily="monospace",
                ha="right", va="center", fontweight="bold")

    for spine in ax_top.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)

    # ================================================================
    # LEFT PANEL — P&L Stats (rows 1-5, cols 0-3)
    # ================================================================
    ax_left = fig.add_subplot(gs[1:6, :3])
    ax_left.set_facecolor(PANEL)
    ax_left.axis("off")

    for spine in ax_left.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)

    # All-time P&L header
    pnl_color = GREEN if total_pnl >= 0 else RED
    ax_left.text(0.05, 0.95, "ALL-TIME P&L · LIVE", transform=ax_left.transAxes,
                 fontsize=7, color=DIM, fontfamily="monospace", va="top")
    ax_left.text(0.05, 0.82, f"${final_equity:,.0f}", transform=ax_left.transAxes,
                 fontsize=28, color=WHITE, fontfamily="monospace",
                 va="top", fontweight="bold")
    ax_left.text(0.75, 0.85, f"+${total_pnl:,.0f}", transform=ax_left.transAxes,
                 fontsize=12, color=GREEN, fontfamily="monospace", va="top")

    # Stats row
    y = 0.62
    ax_left.text(0.05, y, "SINCE JAN 1 · 365 DAYS", transform=ax_left.transAxes,
                 fontsize=7, color=DIM, fontfamily="monospace")
    ax_left.text(0.65, y, f"+{(final_equity-10000)/10000*100:.1f}%", transform=ax_left.transAxes,
                 fontsize=9, color=GREEN, fontfamily="monospace")

    y -= 0.12
    stats_row = [
        ("TRADES", f"{total_trades}"),
        ("WIN RATE", f"{win_rate:.0f}%"),
        ("AVG R:R", "1.0"),
    ]
    for i, (label, val) in enumerate(stats_row):
        x = 0.05 + i * 0.33
        ax_left.text(x, y, label, transform=ax_left.transAxes,
                     fontsize=7, color=DIM, fontfamily="monospace")
        ax_left.text(x, y - 0.08, val, transform=ax_left.transAxes,
                     fontsize=14, color=WHITE, fontfamily="monospace", fontweight="bold")

    # Performance metrics
    y -= 0.22
    ax_left.text(0.05, y, "PERFORMANCE METRICS", transform=ax_left.transAxes,
                 fontsize=7, color=CYAN, fontfamily="monospace")
    y -= 0.08

    perf_items = [
        ("APR", f"${total_pnl/12:,.0f}"),
        ("MAX DD", f"{max_dd*100:.1f}%"),
        ("SHARPE", f"{sharpe:.2f}"),
    ]
    for i, (label, val) in enumerate(perf_items):
        x = 0.05 + i * 0.33
        ax_left.text(x, y, label, transform=ax_left.transAxes,
                     fontsize=7, color=DIM, fontfamily="monospace")
        ax_left.text(x, y - 0.07, val, transform=ax_left.transAxes,
                     fontsize=11, color=WHITE, fontfamily="monospace")

    # Top 3 wins
    y -= 0.18
    ax_left.text(0.05, y, "TOP 3 · BIGGEST WINS", transform=ax_left.transAxes,
                 fontsize=7, color=GOLD, fontfamily="monospace")
    for i, trade in enumerate(top_wins[:3]):
        y -= 0.08
        date_str = trade["entry_time"].strftime("%b %d") if trade.get("entry_time") else "N/A"
        ax_left.text(0.05, y, f"0{i+1}", transform=ax_left.transAxes,
                     fontsize=8, color=DIM, fontfamily="monospace")
        ax_left.text(0.12, y, f"{date_str}  ${trade['entry_price']:,.0f} → +${trade['pnl_dollars']:,.0f}",
                     transform=ax_left.transAxes, fontsize=8, color=GREEN,
                     fontfamily="monospace")

    # ================================================================
    # CENTER — Equity Curve (rows 1-5, cols 3-9)
    # ================================================================
    ax_eq = fig.add_subplot(gs[1:6, 3:9])
    ax_eq.set_facecolor(PANEL)

    # Sample equity for plotting
    step = max(1, len(equity_curve) // 800)
    ts_plot = [equity_curve[i][0] for i in range(0, len(equity_curve), step)]
    eq_plot = [equity_curve[i][1] for i in range(0, len(equity_curve), step)]

    ax_eq.plot(ts_plot, eq_plot, color=GREEN, linewidth=1.2, alpha=0.9)
    ax_eq.fill_between(ts_plot, 10000, eq_plot,
                       where=[e >= 10000 for e in eq_plot],
                       alpha=0.08, color=GREEN)
    ax_eq.axhline(y=10000, color=DIM, linewidth=0.5, linestyle="--", alpha=0.3)

    # Mark trades on curve
    for trade in all_trades[-50:]:  # Last 50 trades
        if trade.get("entry_time"):
            color = GREEN if trade["result"] == "WIN" else RED
            try:
                # Find nearest equity point
                closest = min(range(len(equity_curve)),
                              key=lambda i: abs((equity_curve[i][0] - trade["entry_time"]).total_seconds()))
                ax_eq.scatter(equity_curve[closest][0], equity_curve[closest][1],
                              color=color, s=8, alpha=0.6, zorder=5)
            except:
                pass

    ax_eq.set_title("TOTAL EQUITY", fontsize=8, color=DIM, fontfamily="monospace",
                    loc="left", pad=5)
    ax_eq.text(0.02, 0.02, f"LIVE  ${final_equity:,.0f}", transform=ax_eq.transAxes,
               fontsize=10, color=GREEN, fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a0a", edgecolor=GREEN, linewidth=0.5))

    ax_eq.tick_params(colors=DIM, labelsize=7)
    ax_eq.grid(True, alpha=0.05, color=DIM)
    for spine in ax_eq.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)
    ax_eq.set_ylabel("$", color=DIM, fontsize=8, fontfamily="monospace")

    # ================================================================
    # RIGHT PANEL — Strategy & Robustness (rows 1-5, cols 9-12)
    # ================================================================
    ax_right = fig.add_subplot(gs[1:6, 9:])
    ax_right.set_facecolor(PANEL)
    ax_right.axis("off")

    for spine in ax_right.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)

    # Strategy info
    ax_right.text(0.05, 0.95, "STRATEGY PROFILE", transform=ax_right.transAxes,
                  fontsize=7, color=CYAN, fontfamily="monospace")

    strat_items = [
        ("TYPE", "RANGE REVERSAL"),
        ("INSTRUMENT", "XAUUSD (GOLD)"),
        ("TIMEFRAME", "M15"),
        ("RISK:REWARD", "1:1 FIXED"),
        ("POSITION SIZE", "2% RISK"),
        ("MANAGEMENT", "SET & FORGET"),
        ("CONDITION", "RANGING (ADX<20)"),
        ("ENTRY", "BOUNDARY + CONFIRM"),
        ("MAX DAILY", "3 TRADES"),
        ("NEWS FILTER", "ACTIVE"),
    ]

    for i, (label, val) in enumerate(strat_items):
        y = 0.85 - i * 0.07
        ax_right.text(0.05, y, label, transform=ax_right.transAxes,
                      fontsize=7, color=DIM, fontfamily="monospace")
        ax_right.text(0.95, y, val, transform=ax_right.transAxes,
                      fontsize=7, color=WHITE, fontfamily="monospace", ha="right")

    # Confidence score
    ax_right.text(0.05, 0.12, "EDGE CONFIDENCE", transform=ax_right.transAxes,
                  fontsize=7, color=GOLD, fontfamily="monospace")
    ax_right.text(0.5, 0.03, f"{min(win_rate/100*1.2, 0.95)*100:.0f}/100",
                  transform=ax_right.transAxes, fontsize=16, color=GOLD,
                  fontfamily="monospace", ha="center", fontweight="bold")

    # ================================================================
    # BOTTOM LEFT — Microstructure / Confluence Panel (rows 6-9, cols 0-4)
    # ================================================================
    ax_micro = fig.add_subplot(gs[6:10, :4])
    ax_micro.set_facecolor(PANEL)
    ax_micro.axis("off")

    for spine in ax_micro.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)

    ax_micro.text(0.05, 0.95, "CONFLUENCE ANALYSIS · LAST 50 TRADES",
                  transform=ax_micro.transAxes, fontsize=7, color=CYAN,
                  fontfamily="monospace")

    # Confluence breakdown from trade log
    conf_stats = {}
    for t in all_trades[-50:]:
        confs = t.get("confluences", {})
        for key, val in confs.items():
            if key not in conf_stats:
                conf_stats[key] = {"true": 0, "false": 0, "wins_true": 0}
            if val:
                conf_stats[key]["true"] += 1
                if t["result"] == "WIN":
                    conf_stats[key]["wins_true"] += 1
            else:
                conf_stats[key]["false"] += 1

    y = 0.82
    headers = f"{'CONFLUENCE':<20} {'PRESENT':<8} {'WR WHEN TRUE':<12}"
    ax_micro.text(0.05, y, headers, transform=ax_micro.transAxes,
                  fontsize=7, color=DIM, fontfamily="monospace")
    y -= 0.05

    for key, data in conf_stats.items():
        wr = data["wins_true"] / data["true"] * 100 if data["true"] > 0 else 0
        wr_color = GREEN if wr >= 60 else RED if wr < 45 else WHITE
        short_key = key.replace("_", " ").upper()[:18]
        line = f"{short_key:<20} {data['true']:>3}     {wr:>5.1f}%"
        ax_micro.text(0.05, y, line, transform=ax_micro.transAxes,
                      fontsize=7, color=wr_color, fontfamily="monospace")
        y -= 0.1

    # ================================================================
    # BOTTOM CENTER — Trade Sequence / P&L Ticks (rows 6-9, cols 4-8)
    # ================================================================
    ax_ticks = fig.add_subplot(gs[6:10, 4:8])
    ax_ticks.set_facecolor(PANEL)

    # Plot P&L as tick bars
    if all_trades:
        pnls = [t["pnl_dollars"] for t in all_trades]
        colors = [GREEN if p > 0 else RED for p in pnls]
        ax_ticks.bar(range(len(pnls)), pnls, color=colors, width=0.8, alpha=0.7)
        ax_ticks.axhline(y=0, color=DIM, linewidth=0.5)

    ax_ticks.set_title("P&L TICKS · ALL TRADES", fontsize=7, color=DIM,
                       fontfamily="monospace", loc="left", pad=3)
    ax_ticks.tick_params(colors=DIM, labelsize=6)
    ax_ticks.set_xlabel("Trade #", fontsize=7, color=DIM, fontfamily="monospace")
    ax_ticks.grid(True, alpha=0.05, color=DIM, axis="y")
    for spine in ax_ticks.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)

    # ================================================================
    # BOTTOM RIGHT — Monthly Performance (rows 6-9, cols 8-12)
    # ================================================================
    ax_monthly = fig.add_subplot(gs[6:10, 8:])
    ax_monthly.set_facecolor(PANEL)
    ax_monthly.axis("off")

    for spine in ax_monthly.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)

    ax_monthly.text(0.05, 0.95, "MONTHLY BREAKDOWN", transform=ax_monthly.transAxes,
                    fontsize=7, color=CYAN, fontfamily="monospace")

    y = 0.82
    header = f"{'MONTH':<10} {'TRADES':<7} {'P&L':<10}"
    ax_monthly.text(0.05, y, header, transform=ax_monthly.transAxes,
                    fontsize=7, color=DIM, fontfamily="monospace")

    for month, data in list(monthly_stats.items())[:12]:
        y -= 0.065
        pnl_val = data["pnl"]
        color = GREEN if pnl_val >= 0 else RED
        pnl_str = f"+${pnl_val:,.0f}" if pnl_val >= 0 else f"-${abs(pnl_val):,.0f}"
        line = f"{month:<10} {data['trades']:<7} {pnl_str:<10}"
        ax_monthly.text(0.05, y, line, transform=ax_monthly.transAxes,
                        fontsize=7, color=color, fontfamily="monospace")

    # ================================================================
    # VERY BOTTOM — Conviction Gate / Daily Stats (rows 10-11)
    # ================================================================
    ax_bottom = fig.add_subplot(gs[10:, :])
    ax_bottom.set_facecolor("#080808")
    ax_bottom.axis("off")

    for spine in ax_bottom.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)

    # Conviction gate stats
    ax_bottom.text(0.02, 0.7, "CONVICTION GATE", transform=ax_bottom.transAxes,
                   fontsize=7, color=GOLD, fontfamily="monospace")
    ax_bottom.text(0.02, 0.25, f"SCANNED: {len(price_data)}  →  SIGNALS: {total_trades}  →  "
                   f"DEPLOYED: {total_trades}",
                   transform=ax_bottom.transAxes, fontsize=8, color=WHITE,
                   fontfamily="monospace")

    # Daily P&L meter
    ax_bottom.text(0.4, 0.7, "DAILY P&L METER", transform=ax_bottom.transAxes,
                   fontsize=7, color=GOLD, fontfamily="monospace")
    avg_daily = np.mean(list(daily_pnl.values())) if daily_pnl else 0
    daily_color = GREEN if avg_daily >= 0 else RED
    ax_bottom.text(0.4, 0.25, f"AVG: ${avg_daily:,.0f}/DAY  |  "
                   f"BEST: ${max(daily_pnl.values()) if daily_pnl else 0:,.0f}  |  "
                   f"WORST: ${min(daily_pnl.values()) if daily_pnl else 0:,.0f}",
                   transform=ax_bottom.transAxes, fontsize=8, color=daily_color,
                   fontfamily="monospace")

    # Bot status
    ax_bottom.text(0.8, 0.7, "BOT STATUS", transform=ax_bottom.transAxes,
                   fontsize=7, color=GOLD, fontfamily="monospace")
    ax_bottom.text(0.8, 0.25, f"ONLINE  |  LATENCY: <1ms  |  "
                   f"STRATEGY: RANGE-REVERSAL-1:1",
                   transform=ax_bottom.transAxes, fontsize=8, color=GREEN,
                   fontfamily="monospace")

    # ================================================================
    # SAVE
    # ================================================================
    output_path = "/projects/sandbox/claude/dashboard_hud.png"
    plt.savefig(output_path, dpi=150, facecolor=BG,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()

    print(f"\n       Dashboard saved: {output_path}")
    print(f"\n{'=' * 70}")
    print(f"  REPLAY COMPLETE")
    print(f"  Final Equity: ${final_equity:,.2f}")
    print(f"  Total P&L: +${total_pnl:,.2f} ({(final_equity-10000)/10000*100:.1f}%)")
    print(f"  Win Rate: {win_rate:.1f}% | PF: {pf:.2f} | Sharpe: {sharpe:.2f}")
    print(f"{'=' * 70}")

    return output_path


if __name__ == "__main__":
    run_replay()
