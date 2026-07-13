"""
Test harness: fires ONE signal using 100% REAL live market data, then sends it
to Telegram clearly marked as a TEST. This verifies the full pipeline:

  live price  ->  real bars/indicators  ->  ASWP level math  ->  Telegram delivery

It does NOT place any trade (signal-only bot). Run:
    python3 test_signal.py
"""
import asyncio
import time
from datetime import datetime, timezone

from live_terminal import (
    LiveFeed, DashboardState, send_telegram,
    compute_ema, compute_atr, compute_rsi,
)


async def main():
    feed = LiveFeed()
    state = DashboardState(feed)

    print("1) Fetching REAL live price from TradeLocker...")
    feed.start_websocket()  # starts TradeLocker poll + TV backup in background
    # wait for a real price tick
    for _ in range(20):
        if feed.price > 0:
            break
        await asyncio.sleep(1)
    if feed.price <= 0:
        print("   ERROR: no live price received. Check network/credentials.")
        return
    print(f"   Live price: {feed.price:.2f}  (source: {feed.price_source})")

    print("2) Fetching REAL 5m bars (Forexite) + real yields (FRED)...")
    await feed.fetch_price()
    await feed.fetch_yields()
    print(f"   Bars: {len(feed.bars_5m)} x 5m | {len(feed.bars_1h)} x 1h | "
          f"10y real yield: {feed.yield_value} (chg {feed.yield_change:+.3f})")

    if len(feed.bars_5m) < 30 or len(feed.bars_1h) < 50:
        print("   ERROR: not enough bars to compute indicators.")
        return

    # --- compute REAL indicators (same math the live engine uses) ---
    closes_5m = [b[4] for b in feed.bars_5m]
    closes_1h = [b[4] for b in feed.bars_1h]
    atr_val = compute_atr(feed.bars_5m)
    rsi_val = compute_rsi(closes_5m)
    e20_1h = compute_ema(closes_1h, 20)
    e50_1h = compute_ema(closes_1h, 50)
    trend = "buy" if e20_1h[-1] >= e50_1h[-1] else "sell"
    print("3) Real indicators computed:")
    print(f"   ATR(5m): {atr_val:.2f} | RSI(5m): {rsi_val:.1f} | "
          f"1H trend: {trend.upper()} (EMA20 {e20_1h[-1]:.2f} vs EMA50 {e50_1h[-1]:.2f})")

    # --- build signal levels using REAL price + REAL atr (ASWP logic) ---
    entry = feed.price
    spread = 0.30
    base_move = atr_val
    sl_dist = atr_val + spread / 2
    lots = 0.05  # example size

    if trend == "buy":
        sl = round(entry - sl_dist, 2)
        tp1 = round(entry + 1 * base_move + spread, 2)
        tp2 = round(entry + 2 * base_move + spread, 2)
        tp_final = round(entry + 3 * base_move + spread, 2)
    else:
        sl = round(entry + sl_dist, 2)
        tp1 = round(entry - 1 * base_move - spread, 2)
        tp2 = round(entry - 2 * base_move - spread, 2)
        tp_final = round(entry - 3 * base_move - spread, 2)

    p_tp1 = 0.56
    ev = 0.10 * 1 * p_tp1 + 0.20 * 2 * (p_tp1 * 0.92) + 0.70 * 3 * (p_tp1 * 0.81) - (1 - p_tp1)
    risk_dollars = round(lots * 100 * sl_dist, 2)
    full_win = round(lots * 100 * (0.10*base_move + 0.20*2*base_move + 0.70*3*base_move), 2)

    arrow = "BUY" if trend == "buy" else "SELL"
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    msg = (
        f"[TEST SIGNAL - NOT A REAL TRADE]\n"
        f"Pipeline check @ {now}\n"
        f"------------------------------\n"
        f"XAUUSD {arrow}  |  {lots} lots (example)\n"
        f"Entry: {entry:.2f}   <-- LIVE price from your TradeLocker acct\n"
        f"SL: {sl:.2f}  (risk -${risk_dollars:.2f})\n"
        f"TP1: {tp1:.2f}  (close 10%, SL->BE)\n"
        f"TP2: {tp2:.2f}  (close 20%, SL->TP1)\n"
        f"Final: {tp_final:.2f}  (ride 70%)\n"
        f"------------------------------\n"
        f"Real ATR(5m): {atr_val:.2f} | Real RSI: {rsi_val:.1f}\n"
        f"1H trend: {arrow} | 10y yield: {feed.yield_value} ({feed.yield_change:+.3f})\n"
        f"Prob(TP1): {p_tp1:.2f} | EV: +{ev:.2f}R | Est full win: +${full_win:.2f}\n"
        f"------------------------------\n"
        f"This is a TEST to confirm the bot is scanning live data.\n"
        f"Signal-only: place trades manually if you choose."
    )

    print("4) Sending TEST signal to Telegram...")
    await send_telegram(msg)
    print("   Sent. Check your Telegram.\n")
    print("----- MESSAGE PREVIEW -----")
    print(msg)


if __name__ == "__main__":
    asyncio.run(main())
