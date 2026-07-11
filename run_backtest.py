"""
Run XAUUSD Backtest and Generate Dashboard
"""
import sys
sys.path.insert(0, "/projects/sandbox/claude")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from xauusd_bot.backtest.data_generator import generate_xauusd_data, generate_dxy_data
from xauusd_bot.backtest.dashboard import render_dashboard
from xauusd_bot.bot import XAUUSDBot
from xauusd_bot.backtest.engine import BacktestResult, TradeRecord

print("=" * 60)
print("XAUUSD SIGNAL BOT — BACKTEST ENGINE")
print("=" * 60)

# Generate 90 days of synthetic XAUUSD M15 data
print("\n[1/4] Generating synthetic XAUUSD M15 data (90 days)...")
price_data = generate_xauusd_data(
    start_date="2025-10-01",
    days=90,
    base_price=4100.0,
    seed=42,
)
print(f"       Generated {len(price_data)} bars")
print(f"       Price range: ${price_data['low'].min():.2f} - ${price_data['high'].max():.2f}")

# Generate correlated DXY data
print("\n[2/4] Generating correlated DXY data...")
dxy_data = generate_dxy_data(price_data, seed=123)
print(f"       DXY range: {dxy_data.min():.2f} - {dxy_data.max():.2f}")

# Run backtest
print("\n[3/4] Running backtest...")
starting_balance = 10000.0

# Collect trades via log parsing
all_trades = []
trade_opens = []
trade_closes = []

def on_log(entry):
    if entry.level == "TRADE":
        if "OPENED" in entry.message:
            trade_opens.append({
                "time": entry.timestamp,
                "message": entry.message,
                "data": entry.data.copy(),
            })
        elif "CLOSED" in entry.message:
            trade_closes.append({
                "time": entry.timestamp,
                "message": entry.message,
                "data": entry.data.copy(),
            })

bot = XAUUSDBot(account_balance=starting_balance, on_log=on_log)
bot.initialize()

equity_curve = []

for idx, row in price_data.iterrows():
    timestamp = row["timestamp"]

    # Get DXY slice
    dxy_slice = None
    if idx >= 25:
        dxy_slice = dxy_data[max(0, idx - 25):idx + 1]

    # Feed bar
    status = bot.on_bar(
        timestamp=timestamp,
        high=row["high"],
        low=row["low"],
        close=row["close"],
        open_price=row["open"],
        spread_pips=15.0,
        dxy_closes=dxy_slice,
    )

    # Track equity
    eq = bot.risk_manager.equity if bot.risk_manager else starting_balance
    equity_curve.append((timestamp, eq))

# Pair up opens and closes into TradeRecords
trades = []
for i in range(min(len(trade_opens), len(trade_closes))):
    op = trade_opens[i]
    cl = trade_closes[i]
    
    # Parse direction from OPENED message
    direction = "BUY" if "BUY" in op["message"] else "SELL"
    
    trade = TradeRecord(
        entry_time=op["time"],
        exit_time=cl["time"],
        direction=direction,
        strategy=op["data"].get("strategy", "TREND"),
        entry_price=cl["data"].get("entry", 0),
        exit_price=cl["data"].get("exit", 0),
        stop_loss=op["data"].get("sl", 0),
        take_profit=op["data"].get("tp", 0),
        lot_size=op["data"].get("lot_size", 0.01),
        pnl_pips=cl["data"].get("pnl_pips", 0),
        pnl_dollars=cl["data"].get("pnl_dollars", 0),
        exit_reason=cl["data"].get("reason", ""),
        score=op["data"].get("rr_ratio", 0),
    )
    trades.append(trade)

print(f"       Trade opens captured: {len(trade_opens)}")
print(f"       Trade closes captured: {len(trade_closes)}")
print(f"       Complete trades: {len(trades)}")

# Build result object
result = BacktestResult()
result.starting_balance = starting_balance
result.equity_curve = equity_curve
result.trades = trades
result.total_trades = len(trades)

if trades:
    pnls = [t.pnl_dollars for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    result.winning_trades = len(winners)
    result.losing_trades = len(losers)
    result.win_rate = len(winners) / len(trades) * 100 if trades else 0
    result.total_pnl = sum(pnls)
    result.avg_trade_pnl = np.mean(pnls) if pnls else 0
    result.avg_winner = np.mean(winners) if winners else 0
    result.avg_loser = np.mean(losers) if losers else 0
    result.largest_win = max(winners) if winners else 0
    result.largest_loss = min(losers) if losers else 0

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 1
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Final equity
    equities = [e[1] for e in equity_curve]
    result.final_balance = equities[-1] if equities else starting_balance
    result.total_return_pct = ((result.final_balance - starting_balance) /
                                starting_balance * 100)

    # Max drawdown
    peak = equities[0]
    max_dd = 0
    dd_curve = []
    for eq_val in equities:
        if eq_val > peak:
            peak = eq_val
        dd = (peak - eq_val) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        dd_curve.append(dd)
    result.max_drawdown_pct = max_dd * 100
    result.drawdown_curve = dd_curve

    # Sharpe
    if len(equities) > 96:
        daily_eq = equities[::96]
        daily_rets = [(daily_eq[i] - daily_eq[i-1]) / daily_eq[i-1]
                      for i in range(1, len(daily_eq)) if daily_eq[i-1] != 0]
        if daily_rets:
            mean_r = np.mean(daily_rets)
            std_r = np.std(daily_rets)
            result.sharpe_ratio = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0

    # Consecutive
    cw, cl, max_cw, max_cl = 0, 0, 0, 0
    for t in trades:
        if t.pnl_dollars > 0:
            cw += 1; cl = 0; max_cw = max(max_cw, cw)
        else:
            cl += 1; cw = 0; max_cl = max(max_cl, cl)
    result.max_consecutive_wins = max_cw
    result.max_consecutive_losses = max_cl

    # Duration
    durations = []
    for t in trades:
        if t.entry_time and t.exit_time:
            dur = (t.exit_time - t.entry_time).total_seconds() / 60
            t.duration_minutes = int(dur)
            durations.append(dur)
    result.avg_trade_duration_min = np.mean(durations) if durations else 0

    # Session stats
    session_map = {"asian": [], "london": [], "overlap": [], "new_york": []}
    for t in trades:
        if t.entry_time:
            h = t.entry_time.hour
            if 0 <= h < 7:
                session_map["asian"].append(t.pnl_dollars)
            elif 7 <= h < 12:
                session_map["london"].append(t.pnl_dollars)
            elif 12 <= h < 16:
                session_map["overlap"].append(t.pnl_dollars)
            elif 16 <= h < 21:
                session_map["new_york"].append(t.pnl_dollars)

    for session, pnl_list in session_map.items():
        if pnl_list:
            w = sum(1 for p in pnl_list if p > 0)
            result.session_stats[session] = {
                "trades": len(pnl_list),
                "wins": w,
                "win_rate": w / len(pnl_list) * 100,
                "total_pnl": sum(pnl_list),
            }

    result.test_period_days = 90
else:
    equities = [e[1] for e in equity_curve]
    result.final_balance = equities[-1] if equities else starting_balance
    result.test_period_days = 90

# Print results
print(f"\n{'─' * 60}")
print(f"  BACKTEST COMPLETE — {result.test_period_days} Days")
print(f"{'─' * 60}")
print(f"  Total Trades:      {result.total_trades}")
print(f"  Win Rate:          {result.win_rate:.1f}%")
print(f"  Profit Factor:     {result.profit_factor:.2f}")
print(f"  Total P&L:         ${result.total_pnl:.2f}")
print(f"  Total Return:      {result.total_return_pct:.2f}%")
print(f"  Max Drawdown:      {result.max_drawdown_pct:.2f}%")
print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
print(f"  Avg Winner:        ${result.avg_winner:.2f}")
print(f"  Avg Loser:         ${result.avg_loser:.2f}")
print(f"  Max Consec Wins:   {result.max_consecutive_wins}")
print(f"  Max Consec Losses: {result.max_consecutive_losses}")
print(f"  Starting Balance:  ${result.starting_balance:.2f}")
print(f"  Final Balance:     ${result.final_balance:.2f}")
print(f"{'─' * 60}")

if result.session_stats:
    print(f"\n  SESSION BREAKDOWN:")
    for s, stats in result.session_stats.items():
        print(f"    {s:12s}: {stats['trades']} trades | "
              f"WR {stats['win_rate']:.0f}% | P&L ${stats['total_pnl']:.2f}")

# Generate dashboard
print("\n[4/4] Generating dashboard image...")
output_path = "/projects/sandbox/claude/backtest_dashboard.png"
render_dashboard(result, price_data, output_path)
print(f"       Dashboard saved to: {output_path}")
print("\nDone!")
