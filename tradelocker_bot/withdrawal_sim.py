"""
Withdrawal simulation: Mar 5, 2026 -> Jul 13, 2026 (~18.5 weeks).
Starting equity $5,000. Withdraw minimum $100 as soon as available:
  - First withdrawal: after 14 days
  - Then: every 7 days thereafter
Uses the FLOW circuit-breaker config (base $45, ignition exempt at $25 in DD,
daily stop $120, throttle normal signals to 35% at DD >= $200).
30/40/30 allocation with trend tilt.
"""
import numpy as np
import pandas as pd
import tsai_optimize as T
import phase4_usage as P4
import phase7_candleEV as P7


def flow_ignition(c):
    s = np.zeros(c["n"], dtype=int)
    s += (c["vol_ratio"] > 1.2).astype(int)
    s += (c["trend_strength"] < 1.0).astype(int)
    if "rsi" in c:
        s += ((c["rsi"] >= 42) & (c["rsi"] <= 58)).astype(int)
    return s


def simulate_with_withdrawals(df, start_date, end_date):
    c = T.build_candidates(df)
    pc = P4.precompute(c)
    pc["dt_sorted"] = c["dt"][pc["order"]]
    dts = pd.to_datetime(pc["dt_sorted"])

    # Filter to the target window
    mask = (dts >= pd.Timestamp(start_date)) & (dts <= pd.Timestamp(end_date))
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print("No signals in window!")
        return

    score = flow_ignition(c)[pc["order"]]
    ign = score >= 2
    r = P7.per_signal_r(pc, 0.30, 0.40, 0.19)

    # Sequential simulation with withdrawals
    sld = pc["sl_dist"]
    ent = pc["entries"]
    day_code = pc["day_code"]

    equity = 5000.0
    peak = 5000.0
    total_withdrawn = 0.0
    withdrawals = []
    daily_pnls = {}  # date -> pnl
    cur_day = -1
    dpnl = 0.0
    blocked = False

    # Withdrawal schedule
    first_wd_date = pd.Timestamp(start_date) + pd.Timedelta(days=14)
    last_wd_date = None
    wd_interval = None  # 14 for first, 7 after

    signal_count = 0
    wins = 0
    losses = 0

    for k in idx:
        dt = dts[k]
        d = day_code[k]

        if d != cur_day:
            # New day — check for withdrawal eligibility
            cur_day = d
            dpnl = 0.0
            blocked = False

            # Withdrawal logic
            if last_wd_date is None:
                # First withdrawal: after 14 days
                if dt >= first_wd_date:
                    profit = equity - 5000.0
                    if profit >= 100:
                        wd = profit  # withdraw all available profit
                        equity -= wd
                        total_withdrawn += wd
                        withdrawals.append({"date": dt.strftime("%Y-%m-%d"),
                                          "amount": round(wd, 2),
                                          "equity_after": round(equity, 2),
                                          "total_withdrawn": round(total_withdrawn, 2)})
                        last_wd_date = dt
                        peak = equity  # reset peak after withdrawal
            else:
                # Subsequent: every 7 days
                if dt >= last_wd_date + pd.Timedelta(days=7):
                    profit = equity - 5000.0
                    if profit >= 100:
                        wd = profit
                        equity -= wd
                        total_withdrawn += wd
                        withdrawals.append({"date": dt.strftime("%Y-%m-%d"),
                                          "amount": round(wd, 2),
                                          "equity_after": round(equity, 2),
                                          "total_withdrawn": round(total_withdrawn, 2)})
                        last_wd_date = dt
                        peak = equity

        if blocked:
            continue

        # Circuit-breaker with Flow
        dd = peak - equity
        if dd >= 200.0:
            risk = 25.0 if ign[k] else 45.0 * 0.35
        else:
            risk = 45.0

        lpl = 100.0 * sld[k]
        lots = min(0.12, risk / lpl)
        lots = max(0.01, np.floor(lots * 100) / 100)
        if lots * 100.0 * ent[k] / 10.0 > equity * 0.95:
            lots = max(0.01, np.floor((equity * 0.95 * 10.0 / (100.0 * ent[k])) * 100) / 100)

        pnl = r[k] * lots * lpl
        equity += pnl
        dpnl += pnl
        signal_count += 1
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

        if equity > peak:
            peak = equity
        if dpnl <= -120.0:
            blocked = True

        # Track daily P&L
        day_str = dt.strftime("%Y-%m-%d")
        daily_pnls[day_str] = daily_pnls.get(day_str, 0.0) + pnl

    # Final unrealized profit (available but not yet withdrawn)
    remaining_profit = equity - 5000.0

    print(f"{'=' * 70}")
    print(f"WITHDRAWAL SIMULATION: {start_date} -> {end_date}")
    print(f"Starting equity: $5,000 | Flow circuit-breaker config")
    print(f"{'=' * 70}")
    print(f"\nSignals taken: {signal_count} | Wins: {wins} | Losses: {losses} | "
          f"WR: {wins/max(1,wins+losses)*100:.1f}%")
    print(f"Final equity: ${equity:.2f} (profit still in account: ${remaining_profit:.2f})")
    print(f"\n{'─' * 70}")
    print(f"WITHDRAWALS:")
    print(f"{'─' * 70}")
    if withdrawals:
        print(f"{'#':>3} {'Date':>12} {'Amount':>10} {'Equity After':>14} {'Total Withdrawn':>17}")
        for i, w in enumerate(withdrawals, 1):
            print(f"{i:>3} {w['date']:>12} ${w['amount']:>8.2f} ${w['equity_after']:>12.2f} "
                  f"${w['total_withdrawn']:>15.2f}")
    else:
        print("  No withdrawals made (profit never reached $100 at a withdrawal window)")
    print(f"{'─' * 70}")
    print(f"\nTOTAL WITHDRAWN: ${total_withdrawn:.2f}")
    print(f"STILL IN ACCOUNT: ${remaining_profit:.2f}")
    print(f"TOTAL EARNED: ${total_withdrawn + remaining_profit:.2f}")
    print(f"WITHDRAWAL COUNT: {len(withdrawals)}")
    print(f"{'=' * 70}")

    # Weekly breakdown
    daily_df = pd.Series(daily_pnls)
    daily_df.index = pd.to_datetime(daily_df.index)
    weekly = daily_df.resample("W").sum()
    print(f"\nWEEKLY P&L BREAKDOWN (before withdrawals):")
    for dt, val in weekly.items():
        if val != 0:
            print(f"  {dt.strftime('%Y-%m-%d')}: ${val:+.2f}")


if __name__ == "__main__":
    df = T.load_data()
    simulate_with_withdrawals(df, "2026-03-05", "2026-07-13")
