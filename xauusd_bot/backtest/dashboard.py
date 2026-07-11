"""
Backtest Dashboard — Visual results for the XAUUSD signal bot.
Generates a comprehensive multi-panel chart saved as PNG.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime
from typing import Optional



def render_dashboard(result, price_data, output_path: str = "/projects/sandbox/claude/backtest_dashboard.png"):
    """
    Render a full backtest dashboard with multiple panels.
    
    Panels:
        1. Equity Curve + Drawdown
        2. Trade Distribution (win/loss histogram)
        3. Session Performance breakdown
        4. Key Statistics summary table
        5. Monthly P&L heatmap
        6. Price chart with trade markers
    """
    fig = plt.figure(figsize=(20, 14), facecolor="#1a1a2e")
    fig.suptitle("XAUUSD SIGNAL BOT — BACKTEST RESULTS", 
                 fontsize=18, fontweight="bold", color="#e0e0e0", y=0.98)

    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3,
                  left=0.05, right=0.95, top=0.93, bottom=0.05)

    # Colors
    bg_color = "#1a1a2e"
    panel_bg = "#16213e"
    text_color = "#e0e0e0"
    green = "#00d4aa"
    red = "#ff6b6b"
    blue = "#4ecdc4"
    gold = "#ffd700"

    # ---- Panel 1: Equity Curve (top-left, spans 2 cols) ----
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(panel_bg)
    _plot_equity_curve(ax1, result, green, red, text_color, gold)

    # ---- Panel 2: Key Stats (top-right) ----
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(panel_bg)
    _plot_stats_table(ax2, result, text_color, green, red, gold)

    # ---- Panel 3: Trade P&L Distribution (middle-left) ----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(panel_bg)
    _plot_pnl_distribution(ax3, result, green, red, text_color)

    # ---- Panel 4: Session Performance (middle-center) ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(panel_bg)
    _plot_session_performance(ax4, result, text_color, green, red, blue)

    # ---- Panel 5: Win Rate Donut (middle-right) ----
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(panel_bg)
    _plot_win_rate_donut(ax5, result, green, red, text_color, gold)

    # ---- Panel 6: Price + Trades (bottom, full width) ----
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_facecolor(panel_bg)
    _plot_price_with_trades(ax6, result, price_data, green, red, text_color, gold)

    plt.savefig(output_path, dpi=150, facecolor=bg_color,
                bbox_inches="tight", pad_inches=0.2)
    plt.close()
    return output_path



def _plot_equity_curve(ax, result, green, red, text_color, gold):
    """Plot equity curve with drawdown overlay."""
    if not result.equity_curve:
        ax.text(0.5, 0.5, "No Data", ha="center", color=text_color, fontsize=14)
        return

    timestamps = [e[0] for e in result.equity_curve]
    equities = [e[1] for e in result.equity_curve]

    # Sample for performance (every 4th point)
    step = max(1, len(timestamps) // 500)
    ts_sampled = timestamps[::step]
    eq_sampled = equities[::step]

    ax.plot(ts_sampled, eq_sampled, color=gold, linewidth=1.5, label="Equity")
    ax.fill_between(ts_sampled, result.starting_balance, eq_sampled,
                    where=[e >= result.starting_balance for e in eq_sampled],
                    alpha=0.15, color=green)
    ax.fill_between(ts_sampled, result.starting_balance, eq_sampled,
                    where=[e < result.starting_balance for e in eq_sampled],
                    alpha=0.15, color=red)

    ax.axhline(y=result.starting_balance, color="#555", linestyle="--",
               linewidth=0.8, alpha=0.5)
    ax.set_title("EQUITY CURVE", fontsize=11, fontweight="bold",
                 color=text_color, pad=10)
    ax.set_ylabel("Account ($)", color=text_color, fontsize=9)
    ax.tick_params(colors=text_color, labelsize=8)
    ax.grid(True, alpha=0.1, color="#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")



def _plot_stats_table(ax, result, text_color, green, red, gold):
    """Render key statistics as a formatted table."""
    ax.axis("off")
    ax.set_title("KEY METRICS", fontsize=11, fontweight="bold",
                 color=text_color, pad=10)

    stats = [
        ("Total Trades", f"{result.total_trades}"),
        ("Win Rate", f"{result.win_rate:.1f}%"),
        ("Profit Factor", f"{result.profit_factor:.2f}"),
        ("Total P&L", f"${result.total_pnl:.2f}"),
        ("Total Return", f"{result.total_return_pct:.2f}%"),
        ("Max Drawdown", f"{result.max_drawdown_pct:.2f}%"),
        ("Sharpe Ratio", f"{result.sharpe_ratio:.2f}"),
        ("Avg Winner", f"${result.avg_winner:.2f}"),
        ("Avg Loser", f"${result.avg_loser:.2f}"),
        ("Largest Win", f"${result.largest_win:.2f}"),
        ("Largest Loss", f"${result.largest_loss:.2f}"),
        ("Max Consec Wins", f"{result.max_consecutive_wins}"),
        ("Max Consec Losses", f"{result.max_consecutive_losses}"),
        ("Avg Duration", f"{result.avg_trade_duration_min:.0f} min"),
    ]

    y_start = 0.95
    for i, (label, value) in enumerate(stats):
        y = y_start - i * 0.065
        # Color code P&L values
        val_color = text_color
        if "$" in value:
            val_color = green if not value.startswith("$-") else red
        if "%" in value and "Return" in label:
            val_color = green if not value.startswith("-") else red

        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=9,
                color="#aaa", va="top", fontfamily="monospace")
        ax.text(0.75, y, value, transform=ax.transAxes, fontsize=9,
                color=val_color, va="top", fontweight="bold",
                fontfamily="monospace", ha="right")



def _plot_pnl_distribution(ax, result, green, red, text_color):
    """Histogram of trade P&L."""
    if not result.trades:
        ax.text(0.5, 0.5, "No Trades", ha="center", color=text_color)
        return

    pnls = [t.pnl_dollars for t in result.trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    bins = 20
    if winners:
        ax.hist(winners, bins=bins, color=green, alpha=0.7, label=f"Wins ({len(winners)})")
    if losers:
        ax.hist(losers, bins=bins, color=red, alpha=0.7, label=f"Losses ({len(losers)})")

    ax.axvline(x=0, color="#fff", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("P&L DISTRIBUTION", fontsize=11, fontweight="bold",
                 color=text_color, pad=10)
    ax.set_xlabel("P&L ($)", color=text_color, fontsize=9)
    ax.set_ylabel("Frequency", color=text_color, fontsize=9)
    ax.tick_params(colors=text_color, labelsize=8)
    ax.legend(fontsize=8, facecolor="#16213e", edgecolor="#555", labelcolor=text_color)
    ax.grid(True, alpha=0.1, color="#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")


def _plot_session_performance(ax, result, text_color, green, red, blue):
    """Bar chart of session performance."""
    sessions = result.session_stats
    if not sessions:
        ax.text(0.5, 0.5, "No Session Data", ha="center", color=text_color)
        return

    names = list(sessions.keys())
    pnls = [sessions[s]["total_pnl"] for s in names]
    win_rates = [sessions[s]["win_rate"] for s in names]
    trade_counts = [sessions[s]["trades"] for s in names]

    colors = [green if p >= 0 else red for p in pnls]
    bars = ax.bar(names, pnls, color=colors, alpha=0.8, edgecolor="#333")

    # Add win rate labels
    for i, (bar, wr, tc) in enumerate(zip(bars, win_rates, trade_counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{wr:.0f}%\n({tc}t)", ha="center", fontsize=8, color=text_color)

    ax.set_title("SESSION PERFORMANCE", fontsize=11, fontweight="bold",
                 color=text_color, pad=10)
    ax.set_ylabel("P&L ($)", color=text_color, fontsize=9)
    ax.tick_params(colors=text_color, labelsize=8)
    ax.axhline(y=0, color="#555", linewidth=0.8)
    ax.grid(True, alpha=0.1, color="#555", axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")



def _plot_win_rate_donut(ax, result, green, red, text_color, gold):
    """Donut chart showing win/loss ratio."""
    wins = result.winning_trades
    losses = result.losing_trades

    if wins + losses == 0:
        ax.text(0.5, 0.5, "No Trades", ha="center", color=text_color)
        return

    sizes = [wins, losses]
    colors_list = [green, red]
    labels = [f"Wins\n{wins}", f"Losses\n{losses}"]

    wedges, texts = ax.pie(sizes, colors=colors_list, startangle=90,
                            wedgeprops=dict(width=0.4, edgecolor="#1a1a2e"))

    # Center text
    ax.text(0, 0, f"{result.win_rate:.1f}%", ha="center", va="center",
            fontsize=20, fontweight="bold", color=gold)
    ax.text(0, -0.15, "WIN RATE", ha="center", va="center",
            fontsize=8, color=text_color)

    ax.set_title("WIN / LOSS RATIO", fontsize=11, fontweight="bold",
                 color=text_color, pad=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=green, label=f"Wins: {wins}"),
        mpatches.Patch(facecolor=red, label=f"Losses: {losses}"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", fontsize=8,
              facecolor="#16213e", edgecolor="#555", labelcolor=text_color)


def _plot_price_with_trades(ax, result, price_data, green, red, text_color, gold):
    """Price chart with buy/sell markers."""
    if price_data is None or len(price_data) == 0:
        ax.text(0.5, 0.5, "No Price Data", ha="center", color=text_color)
        return

    # Sample price data for performance
    step = max(1, len(price_data) // 800)
    sampled = price_data.iloc[::step]

    ax.plot(sampled.index, sampled["close"], color=gold, linewidth=0.8, alpha=0.9)

    # Plot trade entries
    for trade in result.trades:
        if trade.entry_time is None:
            continue
        # Find closest index
        try:
            idx = price_data.index[price_data["timestamp"] >= trade.entry_time][0]
        except (IndexError, KeyError):
            continue

        if trade.direction == "BUY":
            marker_color = green
            marker = "^"
        else:
            marker_color = red
            marker = "v"

        ax.scatter(idx, trade.entry_price, color=marker_color,
                   marker=marker, s=40, zorder=5, alpha=0.8)

    ax.set_title("PRICE ACTION + TRADE ENTRIES", fontsize=11,
                 fontweight="bold", color=text_color, pad=10)
    ax.set_ylabel("XAUUSD Price ($)", color=text_color, fontsize=9)
    ax.set_xlabel("Bar Index", color=text_color, fontsize=9)
    ax.tick_params(colors=text_color, labelsize=8)
    ax.grid(True, alpha=0.1, color="#555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
