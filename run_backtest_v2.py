"""
XAUUSD Signal Bot V2 — Backtest
Win Rate Maximized | 1:1 RR | Reversals at Range Boundaries
"""
import sys
sys.path.insert(0, "/projects/sandbox/claude")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime

from xauusd_bot.backtest.realistic_data import generate_realistic_xauusd
from xauusd_bot.bot_v2 import XAUUSDBotV2

print("=" * 60)
print("XAUUSD SIGNAL BOT V2 — WIN RATE MAXIMIZED BACKTEST")
print("=" * 60)
print("Strategy: Reversals at Range Boundaries")
print("Risk:Reward: 1:1 Fixed (Set & Forget)")
print("Target Win Rate: 70%+")
print("=" * 60)

# Generate data
print("\n[1/3] Generating 90 days of XAUUSD M15 data (realistic regime model)...")
price_data = generate_realistic_xauusd(
    start_date="2025-10-01",
    days=90,
    base_price=4100.0,
    seed=42,
)
print(f"       {len(price_data)} bars | ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")

# Run backtest
print("\n[2/3] Running backtest...")
bot = XAUUSDBotV2(account_balance=10000.0)
bot.initialize()

signals_generated = 0

for idx, row in price_data.iterrows():
    ts = row["timestamp"]
    result = bot.on_bar(
        timestamp=ts,
        open_price=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
        spread_pips=15.0,
    )
    if result:
        signals_generated += 1

# Get stats
stats = bot.get_stats()

print(f"\n{'━' * 60}")
print(f"  BACKTEST RESULTS — 90 Days")
print(f"{'━' * 60}")
print(f"  Total Trades:      {stats['total_trades']}")
print(f"  Wins:              {stats['wins']}")
print(f"  Losses:            {stats['losses']}")
print(f"  Win Rate:          {stats['win_rate']}")
print(f"  Profit Factor:     {stats['profit_factor']}")
print(f"  Total P&L:         {stats['total_pnl']}")
print(f"  Total Return:      {stats['total_return']}")
print(f"  Max Drawdown:      {stats['max_drawdown']}")
print(f"  Avg Winner:        {stats['avg_winner']}")
print(f"  Avg Loser:         {stats['avg_loser']}")
print(f"  Final Equity:      {stats['equity']}")
print(f"{'━' * 60}")

# Session breakdown
print(f"\n  TRADE LOG (last 10):")
for t in bot.trade_log[-10:]:
    print(f"    {t['entry_time'].strftime('%m/%d %H:%M')} | "
          f"{t['direction']:4s} | "
          f"entry={t['entry_price']:.2f} | "
          f"exit={t['exit_price']:.2f} | "
          f"P&L={t['pnl_dollars']:+.2f} | "
          f"{t['result']} ({t['reason']})")

# ============================================================
# GENERATE DASHBOARD
# ============================================================
print("\n[3/3] Generating dashboard...")

fig = plt.figure(figsize=(20, 14), facecolor="#0d1117")
fig.suptitle("XAUUSD SIGNAL BOT V2 — WIN RATE MAXIMIZED\n"
             "Strategy: Range Reversals | RR: 1:1 Fixed | Set & Forget",
             fontsize=16, fontweight="bold", color="#e6edf3", y=0.98)

gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
              left=0.05, right=0.95, top=0.92, bottom=0.05)

# Colors
bg = "#0d1117"
panel = "#161b22"
text = "#e6edf3"
green = "#3fb950"
red = "#f85149"
gold = "#d29922"
blue = "#58a6ff"

# ---- Panel 1: Equity Curve ----
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(panel)

if bot.equity_history:
    timestamps = [e[0] for e in bot.equity_history]
    equities = [e[1] for e in bot.equity_history]
    step = max(1, len(timestamps) // 500)
    ts_s = timestamps[::step]
    eq_s = equities[::step]

    ax1.plot(ts_s, eq_s, color=gold, linewidth=1.5)
    ax1.fill_between(ts_s, 10000, eq_s,
                     where=[e >= 10000 for e in eq_s],
                     alpha=0.15, color=green)
    ax1.fill_between(ts_s, 10000, eq_s,
                     where=[e < 10000 for e in eq_s],
                     alpha=0.15, color=red)
    ax1.axhline(y=10000, color="#555", linestyle="--", linewidth=0.8, alpha=0.5)

ax1.set_title("EQUITY CURVE", fontsize=12, fontweight="bold", color=text, pad=10)
ax1.set_ylabel("Account ($)", color=text, fontsize=9)
ax1.tick_params(colors=text, labelsize=8)
ax1.grid(True, alpha=0.1, color="#555")
for spine in ax1.spines.values():
    spine.set_color("#333")

# ---- Panel 2: Key Stats ----
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(panel)
ax2.axis("off")
ax2.set_title("KEY METRICS", fontsize=12, fontweight="bold", color=text, pad=10)

stat_lines = [
    ("Total Trades", str(bot.total_trades)),
    ("Win Rate", stats["win_rate"]),
    ("Profit Factor", stats["profit_factor"]),
    ("Total P&L", stats["total_pnl"]),
    ("Total Return", stats["total_return"]),
    ("Max Drawdown", stats["max_drawdown"]),
    ("Avg Winner", stats["avg_winner"]),
    ("Avg Loser", stats["avg_loser"]),
    ("Risk:Reward", "1:1 Fixed"),
    ("Strategy", "Range Reversal"),
    ("Position Size", "2% Risk"),
    ("Trade Mgmt", "Set & Forget"),
]

for i, (label, value) in enumerate(stat_lines):
    y = 0.92 - i * 0.075
    val_color = text
    if "$" in str(value) and "-" in str(value):
        val_color = red
    elif "$" in str(value):
        val_color = green

    ax2.text(0.05, y, label, transform=ax2.transAxes, fontsize=9,
             color="#8b949e", va="top", fontfamily="monospace")
    ax2.text(0.95, y, str(value), transform=ax2.transAxes, fontsize=9,
             color=val_color, va="top", fontweight="bold",
             fontfamily="monospace", ha="right")

# ---- Panel 3: Win/Loss Distribution ----
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(panel)

if bot.trade_log:
    pnls = [t["pnl_dollars"] for t in bot.trade_log]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    if winners:
        ax3.hist(winners, bins=15, color=green, alpha=0.7, label=f"Wins ({len(winners)})")
    if losers:
        ax3.hist(losers, bins=15, color=red, alpha=0.7, label=f"Losses ({len(losers)})")
    ax3.axvline(x=0, color="#fff", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=8, facecolor=panel, edgecolor="#555", labelcolor=text)

ax3.set_title("P&L DISTRIBUTION", fontsize=12, fontweight="bold", color=text, pad=10)
ax3.set_xlabel("P&L ($)", color=text, fontsize=9)
ax3.tick_params(colors=text, labelsize=8)
ax3.grid(True, alpha=0.1, color="#555")
for spine in ax3.spines.values():
    spine.set_color("#333")

# ---- Panel 4: Win Rate Donut ----
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(panel)

if bot.total_trades > 0:
    sizes = [bot.wins, bot.losses]
    colors_pie = [green, red]
    wedges, _ = ax4.pie(sizes, colors=colors_pie, startangle=90,
                         wedgeprops=dict(width=0.4, edgecolor=bg))
    wr = bot.wins / bot.total_trades * 100
    ax4.text(0, 0.05, f"{wr:.1f}%", ha="center", va="center",
             fontsize=24, fontweight="bold", color=gold)
    ax4.text(0, -0.15, "WIN RATE", ha="center", va="center",
             fontsize=9, color=text)

ax4.set_title("WIN / LOSS", fontsize=12, fontweight="bold", color=text, pad=10)

# ---- Panel 5: Trade Results Sequence ----
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(panel)

if bot.trade_log:
    results = [1 if t["result"] == "WIN" else -1 for t in bot.trade_log]
    colors_seq = [green if r == 1 else red for r in results]
    ax5.bar(range(len(results)), results, color=colors_seq, width=0.8, alpha=0.8)
    ax5.axhline(y=0, color="#555", linewidth=0.5)
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_yticks([-1, 1])
    ax5.set_yticklabels(["LOSS", "WIN"])

ax5.set_title("TRADE SEQUENCE", fontsize=12, fontweight="bold", color=text, pad=10)
ax5.set_xlabel("Trade #", color=text, fontsize=9)
ax5.tick_params(colors=text, labelsize=8)
for spine in ax5.spines.values():
    spine.set_color("#333")

# ---- Panel 6: Price + Signals ----
ax6 = fig.add_subplot(gs[2, :])
ax6.set_facecolor(panel)

step = max(1, len(price_data) // 800)
sampled = price_data.iloc[::step]
ax6.plot(sampled.index, sampled["close"], color=gold, linewidth=0.7, alpha=0.8)

# Plot trade entries
for trade in bot.trade_log:
    if trade["entry_time"]:
        try:
            idx = price_data.index[price_data["timestamp"] >= trade["entry_time"]][0]
            color = green if trade["result"] == "WIN" else red
            marker = "^" if trade["direction"] == "BUY" else "v"
            ax6.scatter(idx, trade["entry_price"], color=color,
                       marker=marker, s=50, zorder=5, alpha=0.8,
                       edgecolors="white", linewidths=0.3)
        except (IndexError, KeyError):
            pass

ax6.set_title("XAUUSD PRICE + TRADE SIGNALS (▲=BUY ▼=SELL | Green=WIN Red=LOSS)",
              fontsize=12, fontweight="bold", color=text, pad=10)
ax6.set_ylabel("Price ($)", color=text, fontsize=9)
ax6.set_xlabel("Bar Index (M15)", color=text, fontsize=9)
ax6.tick_params(colors=text, labelsize=8)
ax6.grid(True, alpha=0.1, color="#555")
for spine in ax6.spines.values():
    spine.set_color("#333")

# Save
output_path = "/projects/sandbox/claude/backtest_v2_dashboard.png"
plt.savefig(output_path, dpi=150, facecolor=bg, bbox_inches="tight", pad_inches=0.2)
plt.close()

print(f"\n       Dashboard saved: {output_path}")
print("\n" + "=" * 60)
print("DONE — V2 Win Rate Maximized Strategy")
print("=" * 60)
