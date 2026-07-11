"""
XAUUSD Signal Bot V3 - Adaptive Engine Backtest
Full Year: Jan 2025 -> Jan 2026
6 Adaptive Systems | HUD Dashboard Output

Optimal settings:
- 6 trades/day max
- NY session added
- 30min loss cooldown / 10min win cooldown
- Adaptive R:R (0.75-1.5x)
- Kelly Criterion lite sizing (1%-3%)
- Early exit on probability shift
- Sentiment-aware (simulated for backtest)
"""
import sys
sys.path.insert(0, "/projects/sandbox/claude")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

from xauusd_bot.backtest.realistic_data import generate_realistic_xauusd
from xauusd_bot.bot_v3 import XAUUSDBotV3

print("=" * 70)
print("  XAUUSD SIGNAL BOT V3 - ADAPTIVE ENGINE BACKTEST")
print("  Period: Jan 1, 2025 -> Jan 1, 2026 (365 days)")
print("  6 Adaptive Systems Active")
print("=" * 70)

# ============================================================
# GENERATE DATA
# ============================================================
print("\n[1/4] Generating 365 days of XAUUSD M15 data...")
price_data = generate_realistic_xauusd(
    start_date="2025-01-01",
    days=365,
    base_price=4100.0,
    seed=55,
)
print(f"       {len(price_data)} bars generated")
print(f"       Range: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")

# ============================================================
# RUN BACKTEST (optimized: convert to lists before iterating)
# ============================================================
print("\n[2/4] Running V3 adaptive backtest...")
bot = XAUUSDBotV3(account_balance=10000.0)
bot.initialize()

# Convert DataFrame to lists for performance (avoid .iterrows() overhead)
timestamps = price_data["timestamp"].tolist()
opens = price_data["open"].tolist()
highs_list = price_data["high"].tolist()
lows_list = price_data["low"].tolist()
closes = price_data["close"].tolist()

total_bars = len(timestamps)
progress_step = total_bars // 10

for i in range(total_bars):
    bot.on_bar(
        timestamp=timestamps[i],
        open_price=opens[i],
        high=highs_list[i],
        low=lows_list[i],
        close=closes[i],
        spread_pips=15.0,
    )
    if (i + 1) % progress_step == 0:
        pct = (i + 1) / total_bars * 100
        print(f"       {pct:.0f}% complete ({bot.total_trades} trades so far)")

# ============================================================
# COMPUTE STATS
# ============================================================
print("\n[3/4] Computing statistics...")

stats = bot.get_stats()
total_trades = bot.total_trades
win_rate = bot.wins / total_trades * 100 if total_trades > 0 else 0
total_pnl = bot.total_pnl
final_equity = bot.equity

# Profit factor
winners = [t["pnl_dollars"] for t in bot.trade_log if t["pnl_dollars"] > 0]
losers_list_pnl = [t["pnl_dollars"] for t in bot.trade_log if t["pnl_dollars"] < 0]
gross_profit = sum(winners) if winners else 0
gross_loss = abs(sum(losers_list_pnl)) if losers_list_pnl else 1
pf = gross_profit / gross_loss if gross_loss > 0 else 0

# Max drawdown
equities = [e[1] for e in bot.equity_history]
peak = equities[0]
max_dd = 0
for eq in equities:
    if eq > peak:
        peak = eq
    dd = (peak - eq) / peak
    max_dd = max(max_dd, dd)

# Sharpe
daily_eq = equities[::96]
daily_rets = [(daily_eq[i] - daily_eq[i-1]) / daily_eq[i-1]
              for i in range(1, len(daily_eq)) if daily_eq[i-1] != 0]
sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
          if daily_rets and np.std(daily_rets) > 0 else 0)

# Monthly breakdown
monthly_stats = {}
daily_pnl = {}
for t in bot.trade_log:
    if t.get("exit_time"):
        month_key = t["exit_time"].strftime("%b %Y")
        if month_key not in monthly_stats:
            monthly_stats[month_key] = {"trades": 0, "wins": 0, "pnl": 0}
        monthly_stats[month_key]["trades"] += 1
        monthly_stats[month_key]["pnl"] += t["pnl_dollars"]
        if t["result"] == "WIN":
            monthly_stats[month_key]["wins"] += 1
        day = t["exit_time"].strftime("%Y-%m-%d")
        daily_pnl[day] = daily_pnl.get(day, 0) + t["pnl_dollars"]

# Top wins
sorted_trades = sorted(bot.trade_log, key=lambda t: t["pnl_dollars"], reverse=True)
top_wins = sorted_trades[:5]

# Condition breakdown
condition_stats = {"GOOD": {"trades": 0, "wins": 0},
                   "OKAY": {"trades": 0, "wins": 0},
                   "CHOPPY": {"trades": 0, "wins": 0}}
for t in bot.trade_log:
    cond = t.get("condition", "OKAY")
    if cond in condition_stats:
        condition_stats[cond]["trades"] += 1
        if t["result"] == "WIN":
            condition_stats[cond]["wins"] += 1

# Avg win/loss
avg_win = np.mean(winners) if winners else 0
avg_loss = np.mean(losers_list_pnl) if losers_list_pnl else 0

# Trading days
trading_days = len(set(t["exit_time"].strftime("%Y-%m-%d")
                       for t in bot.trade_log if t.get("exit_time")))
trades_per_day = total_trades / trading_days if trading_days > 0 else 0

print(f"\n       Results:")
print(f"       Trades: {total_trades} | WR: {win_rate:.1f}% | PF: {pf:.2f}")
print(f"       P&L: ${total_pnl:.2f} | Equity: ${final_equity:.2f}")
print(f"       Max DD: {max_dd*100:.1f}% | Sharpe: {sharpe:.2f}")
print(f"       Early Exits (profit): {bot.early_exits_profit}")
print(f"       Early Exits (loss): {bot.early_exits_loss}")
print(f"       Sentiment Avoids: {bot.sentiment_avoids}")
print(f"       Trades/Day: {trades_per_day:.1f}")

# ============================================================
# GENERATE HUD DASHBOARD
# ============================================================
print("\n[4/4] Rendering HUD dashboard...")

fig = plt.figure(figsize=(24, 16), facecolor="#0a0a0a")

# Master grid
gs = GridSpec(12, 12, figure=fig, hspace=0.5, wspace=0.4,
              left=0.02, right=0.98, top=0.96, bottom=0.02)

# Colors (terminal HUD style)
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
# TOP BAR - Ticker tape
# ================================================================
ax_top = fig.add_subplot(gs[0, :])
ax_top.set_facecolor("#111111")
ax_top.axis("off")

top_text = (
    f"  XAUUSD V3 ADAPTIVE   |   "
    f"TRADES: {total_trades}   |   "
    f"WIN RATE: {win_rate:.0f}%   |   "
    f"PF: {pf:.2f}   |   "
    f"SHARPE: {sharpe:.2f}   |   "
    f"6 ADAPTIVE SYSTEMS   |   "
    f"RISK: KELLY LITE   |   "
    f"JAN 2025 -> JAN 2026   |   "
    f"NET P&L: ${total_pnl:,.0f}"
)
ax_top.text(0.5, 0.5, top_text, transform=ax_top.transAxes,
            fontsize=8, color=CYAN, fontfamily="monospace",
            ha="center", va="center")

for spine in ax_top.spines.values():
    spine.set_color(BORDER)
    spine.set_linewidth(0.5)

# ================================================================
# LEFT PANEL - P&L Stats (rows 1-5, cols 0-3)
# ================================================================
ax_left = fig.add_subplot(gs[1:6, :3])
ax_left.set_facecolor(PANEL)
ax_left.axis("off")
for spine in ax_left.spines.values():
    spine.set_color(BORDER)
    spine.set_linewidth(0.5)

pnl_color = GREEN if total_pnl >= 0 else RED
ax_left.text(0.05, 0.95, "ALL-TIME P&L . V3 ADAPTIVE", transform=ax_left.transAxes,
             fontsize=7, color=DIM, fontfamily="monospace", va="top")
ax_left.text(0.05, 0.82, f"${final_equity:,.0f}", transform=ax_left.transAxes,
             fontsize=28, color=WHITE, fontfamily="monospace",
             va="top", fontweight="bold")
ax_left.text(0.75, 0.85, f"+${total_pnl:,.0f}", transform=ax_left.transAxes,
             fontsize=12, color=GREEN if total_pnl >= 0 else RED,
             fontfamily="monospace", va="top")

y = 0.62
ax_left.text(0.05, y, "365 DAYS | 6 ADAPTIVE SYSTEMS", transform=ax_left.transAxes,
             fontsize=7, color=DIM, fontfamily="monospace")
ax_left.text(0.7, y, f"+{(final_equity-10000)/10000*100:.1f}%", transform=ax_left.transAxes,
             fontsize=9, color=GREEN if total_pnl >= 0 else RED, fontfamily="monospace")

y -= 0.12
stats_row = [
    ("TRADES", f"{total_trades}"),
    ("WIN RATE", f"{win_rate:.0f}%"),
    ("AVG R:R", stats.get("avg_adaptive_rr", "1.0")),
]
for i, (label, val) in enumerate(stats_row):
    x = 0.05 + i * 0.33
    ax_left.text(x, y, label, transform=ax_left.transAxes,
                 fontsize=7, color=DIM, fontfamily="monospace")
    ax_left.text(x, y - 0.08, str(val), transform=ax_left.transAxes,
                 fontsize=14, color=WHITE, fontfamily="monospace", fontweight="bold")

y -= 0.22
ax_left.text(0.05, y, "PERFORMANCE METRICS", transform=ax_left.transAxes,
             fontsize=7, color=CYAN, fontfamily="monospace")
y -= 0.08

perf_items = [
    ("MAX DD", f"{max_dd*100:.1f}%"),
    ("SHARPE", f"{sharpe:.2f}"),
    ("PF", f"{pf:.2f}"),
]
for i, (label, val) in enumerate(perf_items):
    x = 0.05 + i * 0.33
    ax_left.text(x, y, label, transform=ax_left.transAxes,
                 fontsize=7, color=DIM, fontfamily="monospace")
    ax_left.text(x, y - 0.07, val, transform=ax_left.transAxes,
                 fontsize=11, color=WHITE, fontfamily="monospace")

# Top wins
y -= 0.18
ax_left.text(0.05, y, "TOP 5 . BIGGEST WINS", transform=ax_left.transAxes,
             fontsize=7, color=GOLD, fontfamily="monospace")
for i, trade in enumerate(top_wins[:5]):
    y -= 0.065
    date_str = trade["entry_time"].strftime("%b %d") if trade.get("entry_time") else "N/A"
    ax_left.text(0.05, y, f"0{i+1}", transform=ax_left.transAxes,
                 fontsize=7, color=DIM, fontfamily="monospace")
    ax_left.text(0.12, y, f"{date_str}  +${trade['pnl_dollars']:,.0f}  RR={trade.get('adaptive_rr', 1.0):.1f}",
                 transform=ax_left.transAxes, fontsize=7, color=GREEN,
                 fontfamily="monospace")

# ================================================================
# CENTER - Equity Curve (rows 1-5, cols 3-9)
# ================================================================
ax_eq = fig.add_subplot(gs[1:6, 3:9])
ax_eq.set_facecolor(PANEL)

# Sample equity for plotting
equity_curve = bot.equity_history
step = max(1, len(equity_curve) // 800)
ts_plot = [equity_curve[i][0] for i in range(0, len(equity_curve), step)]
eq_plot = [equity_curve[i][1] for i in range(0, len(equity_curve), step)]

ax_eq.plot(ts_plot, eq_plot, color=GREEN, linewidth=1.2, alpha=0.9)
ax_eq.fill_between(ts_plot, 10000, eq_plot,
                   where=[e >= 10000 for e in eq_plot],
                   alpha=0.08, color=GREEN)
ax_eq.fill_between(ts_plot, 10000, eq_plot,
                   where=[e < 10000 for e in eq_plot],
                   alpha=0.08, color=RED)
ax_eq.axhline(y=10000, color=DIM, linewidth=0.5, linestyle="--", alpha=0.3)

ax_eq.set_title("EQUITY CURVE | V3 ADAPTIVE", fontsize=8, color=DIM,
                fontfamily="monospace", loc="left", pad=5)
ax_eq.text(0.02, 0.02, f"LIVE  ${final_equity:,.0f}", transform=ax_eq.transAxes,
           fontsize=10, color=GREEN if total_pnl >= 0 else RED, fontfamily="monospace",
           bbox=dict(boxstyle="round,pad=0.3", facecolor="#0a0a0a",
                     edgecolor=GREEN, linewidth=0.5))

ax_eq.tick_params(colors=DIM, labelsize=7)
ax_eq.grid(True, alpha=0.05, color=DIM)
for spine in ax_eq.spines.values():
    spine.set_color(BORDER)
    spine.set_linewidth(0.5)
ax_eq.set_ylabel("$", color=DIM, fontsize=8, fontfamily="monospace")

# ================================================================
# RIGHT PANEL - Adaptive Systems Status (rows 1-5, cols 9-12)
# ================================================================
ax_right = fig.add_subplot(gs[1:6, 9:])
ax_right.set_facecolor(PANEL)
ax_right.axis("off")
for spine in ax_right.spines.values():
    spine.set_color(BORDER)
    spine.set_linewidth(0.5)

ax_right.text(0.05, 0.95, "6 ADAPTIVE SYSTEMS", transform=ax_right.transAxes,
              fontsize=7, color=CYAN, fontfamily="monospace")

systems = [
    ("1. ADAPTIVE R:R", f"AVG {stats.get('avg_adaptive_rr', '1.0')}", GREEN),
    ("2. KELLY SIZE", f"AVG {stats.get('avg_position_size', '2.0%')}", GREEN),
    ("3. TIGHT SL", "ACTIVE (0.3-0.7x)", GREEN),
    ("4. EARLY EXIT", f"P:{bot.early_exits_profit} L:{bot.early_exits_loss}", GREEN),
    ("5. SENTIMENT", f"AVOIDS: {bot.sentiment_avoids}", GOLD),
    ("6. INFLUENCER", "TRACKING", GOLD),
]

for i, (name, status, color) in enumerate(systems):
    y = 0.85 - i * 0.09
    ax_right.text(0.05, y, name, transform=ax_right.transAxes,
                  fontsize=7, color=DIM, fontfamily="monospace")
    ax_right.text(0.95, y, status, transform=ax_right.transAxes,
                  fontsize=7, color=color, fontfamily="monospace", ha="right")

# Condition breakdown
y -= 0.12
ax_right.text(0.05, y, "CONDITION BREAKDOWN", transform=ax_right.transAxes,
              fontsize=7, color=GOLD, fontfamily="monospace")
for cond_name, cond_data in condition_stats.items():
    y -= 0.07
    wr = cond_data["wins"] / cond_data["trades"] * 100 if cond_data["trades"] > 0 else 0
    wr_color = GREEN if wr >= 60 else RED if wr < 45 else WHITE
    ax_right.text(0.05, y, f"{cond_name:<8} {cond_data['trades']:>3}T  {wr:.0f}% WR",
                  transform=ax_right.transAxes, fontsize=7, color=wr_color,
                  fontfamily="monospace")

# Strategy profile
y -= 0.12
ax_right.text(0.05, y, "STRATEGY PROFILE", transform=ax_right.transAxes,
              fontsize=7, color=CYAN, fontfamily="monospace")
profile = [
    ("TYPE", "RANGE REVERSAL"),
    ("TIMEFRAME", "M15"),
    ("MAX DAILY", "6 TRADES"),
    ("SESSIONS", "ASIA+LON+NY"),
    ("COOLDOWN W", "10 MIN"),
    ("COOLDOWN L", "30 MIN"),
]
for label, val in profile:
    y -= 0.06
    ax_right.text(0.05, y, label, transform=ax_right.transAxes,
                  fontsize=6, color=DIM, fontfamily="monospace")
    ax_right.text(0.95, y, val, transform=ax_right.transAxes,
                  fontsize=6, color=WHITE, fontfamily="monospace", ha="right")

# ================================================================
# BOTTOM LEFT - Adaptive R:R Distribution (rows 6-9, cols 0-4)
# ================================================================
ax_rr = fig.add_subplot(gs[6:10, :4])
ax_rr.set_facecolor(PANEL)

rr_values = [t.get("adaptive_rr", 1.0) for t in bot.trade_log]
if rr_values:
    rr_wins = [t.get("adaptive_rr", 1.0) for t in bot.trade_log if t["result"] == "WIN"]
    rr_losses = [t.get("adaptive_rr", 1.0) for t in bot.trade_log if t["result"] == "LOSS"]
    bins = np.linspace(0.5, 1.8, 20)
    if rr_wins:
        ax_rr.hist(rr_wins, bins=bins, color=GREEN, alpha=0.6, label=f"Wins ({len(rr_wins)})")
    if rr_losses:
        ax_rr.hist(rr_losses, bins=bins, color=RED, alpha=0.6, label=f"Losses ({len(rr_losses)})")
    ax_rr.axvline(x=1.0, color=GOLD, linewidth=1, linestyle="--", alpha=0.7)
    ax_rr.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)

ax_rr.set_title("ADAPTIVE R:R DISTRIBUTION", fontsize=7, color=DIM,
                fontfamily="monospace", loc="left", pad=3)
ax_rr.tick_params(colors=DIM, labelsize=6)
ax_rr.grid(True, alpha=0.05, color=DIM, axis="y")
for spine in ax_rr.spines.values():
    spine.set_color(BORDER)
    spine.set_linewidth(0.5)

# ================================================================
# BOTTOM CENTER - P&L Ticks (rows 6-9, cols 4-8)
# ================================================================
ax_ticks = fig.add_subplot(gs[6:10, 4:8])
ax_ticks.set_facecolor(PANEL)

if bot.trade_log:
    pnls = [t["pnl_dollars"] for t in bot.trade_log]
    colors_bar = [GREEN if p > 0 else RED for p in pnls]
    ax_ticks.bar(range(len(pnls)), pnls, color=colors_bar, width=0.8, alpha=0.7)
    ax_ticks.axhline(y=0, color=DIM, linewidth=0.5)

ax_ticks.set_title("P&L TICKS . ALL TRADES", fontsize=7, color=DIM,
                   fontfamily="monospace", loc="left", pad=3)
ax_ticks.tick_params(colors=DIM, labelsize=6)
ax_ticks.set_xlabel("Trade #", fontsize=7, color=DIM, fontfamily="monospace")
ax_ticks.grid(True, alpha=0.05, color=DIM, axis="y")
for spine in ax_ticks.spines.values():
    spine.set_color(BORDER)
    spine.set_linewidth(0.5)

# ================================================================
# BOTTOM RIGHT - Monthly Performance (rows 6-9, cols 8-12)
# ================================================================
ax_monthly = fig.add_subplot(gs[6:10, 8:])
ax_monthly.set_facecolor(PANEL)
ax_monthly.axis("off")
for spine in ax_monthly.spines.values():
    spine.set_color(BORDER)
    spine.set_linewidth(0.5)

ax_monthly.text(0.05, 0.95, "MONTHLY BREAKDOWN", transform=ax_monthly.transAxes,
                fontsize=7, color=CYAN, fontfamily="monospace")

y = 0.85
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
# VERY BOTTOM - Summary Bar (rows 10-11)
# ================================================================
ax_bottom = fig.add_subplot(gs[10:, :])
ax_bottom.set_facecolor("#080808")
ax_bottom.axis("off")
for spine in ax_bottom.spines.values():
    spine.set_color(BORDER)
    spine.set_linewidth(0.5)

# V3 improvement summary
ax_bottom.text(0.02, 0.7, "V3 ADAPTIVE SYSTEMS", transform=ax_bottom.transAxes,
               fontsize=7, color=GOLD, fontfamily="monospace")
ax_bottom.text(0.02, 0.25,
               f"SCANNED: {total_bars}  ->  SIGNALS: {total_trades}  |  "
               f"AVG RR: {stats.get('avg_adaptive_rr', '1.0')}  |  "
               f"KELLY SIZE: {stats.get('avg_position_size', '2.0%')}  |  "
               f"EARLY EXITS: {bot.early_exits_profit + bot.early_exits_loss}",
               transform=ax_bottom.transAxes, fontsize=8, color=WHITE,
               fontfamily="monospace")

# Daily P&L meter
ax_bottom.text(0.45, 0.7, "DAILY P&L METER", transform=ax_bottom.transAxes,
               fontsize=7, color=GOLD, fontfamily="monospace")
avg_daily_pnl = np.mean(list(daily_pnl.values())) if daily_pnl else 0
best_day = max(daily_pnl.values()) if daily_pnl else 0
worst_day = min(daily_pnl.values()) if daily_pnl else 0
ax_bottom.text(0.45, 0.25,
               f"AVG: ${avg_daily_pnl:,.0f}/DAY  |  "
               f"BEST: ${best_day:,.0f}  |  WORST: ${worst_day:,.0f}",
               transform=ax_bottom.transAxes, fontsize=8,
               color=GREEN if avg_daily_pnl >= 0 else RED,
               fontfamily="monospace")

# Bot status
ax_bottom.text(0.82, 0.7, "BOT STATUS", transform=ax_bottom.transAxes,
               fontsize=7, color=GOLD, fontfamily="monospace")
ax_bottom.text(0.82, 0.25, f"V3 ADAPTIVE  |  {trades_per_day:.1f} T/DAY",
               transform=ax_bottom.transAxes, fontsize=8, color=GREEN,
               fontfamily="monospace")

# ================================================================
# SAVE
# ================================================================
output_path = "/projects/sandbox/claude/backtest_v3_dashboard.png"
plt.savefig(output_path, dpi=150, facecolor=BG,
            bbox_inches="tight", pad_inches=0.1)
plt.close()

print(f"\n       Dashboard saved: {output_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'=' * 70}")
print(f"  V3 ADAPTIVE ENGINE - FINAL RESULTS")
print(f"{'=' * 70}")
print(f"  Total Trades:        {total_trades}")
print(f"  Win Rate:            {win_rate:.1f}%")
print(f"  Profit Factor:       {pf:.2f}")
print(f"  Total P&L:           ${total_pnl:,.2f}")
print(f"  Total Return:        {(final_equity-10000)/10000*100:.1f}%")
print(f"  Max Drawdown:        {max_dd*100:.1f}%")
print(f"  Sharpe Ratio:        {sharpe:.2f}")
print(f"  Avg Winner:          ${avg_win:.2f}")
print(f"  Avg Loser:           ${avg_loss:.2f}")
print(f"  Trades/Day:          {trades_per_day:.1f}")
print(f"  Avg Adaptive R:R:    {stats.get('avg_adaptive_rr', 'N/A')}")
print(f"  Avg Position Size:   {stats.get('avg_position_size', 'N/A')}")
print(f"  Early Exits (profit): {bot.early_exits_profit}")
print(f"  Early Exits (loss):   {bot.early_exits_loss}")
print(f"  Sentiment Avoids:    {bot.sentiment_avoids}")
print(f"{'=' * 70}")
print(f"  Dashboard: {output_path}")
print(f"{'=' * 70}")
