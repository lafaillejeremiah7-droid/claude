"""
GOLD VORTEX v5 — Live Signal Bot
==================================
Fetches real-time NAS100 (Nasdaq 100) 5-minute data, runs the strategy logic,
serves a futuristic dashboard at http://localhost:5000, and sends
signals to Telegram.

Instrument: NAS100 (switched from XAUUSD — Nasdaq trends cleaner with this
momentum strategy: 57% WR vs 29% on gold in recent data).

Usage:
  pip install fastapi uvicorn httpx pandas numpy
  python live_bot.py

Then open: http://localhost:5000
"""
import asyncio, json, time, os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np, pandas as pd
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse
import uvicorn

# Strategy imports (same logic as backtest)
from strategy import (
    ema, atr_calc, rsi_calc,
    SL_MULT, TP_RATIO, COOLDOWN, MAX_PER_DAY, MAX_HOLD,
    RISK_PCT, SESSION_START, SESSION_END, BAR_MINUTES, CONTRACT
)

# ===================== CONFIG =====================

TELEGRAM_BOT_TOKEN = "8926622863:AAF0QHHYAyEVQZiYV35b5vyeKxDC_ouMnmQ"
TELEGRAM_CHAT_ID = "7040023207"
PORT = 5000
SCAN_INTERVAL = 300  # 5 minutes in seconds
STARTING_BALANCE = 5000.0

# ===================== STATE =====================

class BotState:
    def __init__(self):
        self.equity = STARTING_BALANCE
        self.peak_equity = STARTING_BALANCE
        self.signals_history = []       # all signals sent
        self.open_trade = None          # currently open trade (or None)
        self.daily_pnl = {}
        self.trades_today = {}
        self.last_signal_time = None
        self.bars_5m = pd.DataFrame()   # rolling 5m bar history
        self.price = 0.0
        self.running = False
        self.last_scan = None
        self.load_state()

    def save_state(self):
        data = {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "signals_history": self.signals_history[-100:],
            "open_trade": self.open_trade,
            "daily_pnl": self.daily_pnl,
            "trades_today": self.trades_today,
        }
        Path("bot_state.json").write_text(json.dumps(data, default=str, indent=2))

    def load_state(self):
        if Path("bot_state.json").exists():
            try:
                data = json.loads(Path("bot_state.json").read_text())
                self.equity = data.get("equity", STARTING_BALANCE)
                self.peak_equity = data.get("peak_equity", STARTING_BALANCE)
                self.signals_history = data.get("signals_history", [])
                self.open_trade = data.get("open_trade", None)
                self.daily_pnl = data.get("daily_pnl", {})
                self.trades_today = data.get("trades_today", {})
                print(f"State loaded: equity=${self.equity:.2f}, {len(self.signals_history)} signals")
            except:
                pass


state = BotState()

# ===================== PRICE FEED =====================

async def fetch_5m_bars():
    """Fetch recent NAS100 15-minute bars from Yahoo Finance."""
    async with httpx.AsyncClient(timeout=30) as client:
        # Yahoo Finance: NQ=F (Nasdaq 100 Futures) 15m data (last 60 days)
        url = "https://query1.finance.yahoo.com/v8/finance/chart/NQ=F"
        params = {"interval": "15m", "range": "60d"}
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            r = await client.get(url, params=params, headers=headers)
            if r.status_code != 200:
                print(f"Yahoo API error: {r.status_code}")
                return None
            data = r.json()
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]

            df = pd.DataFrame({
                "dt": pd.to_datetime(timestamps, unit="s", utc=True),
                "Open": quotes["open"],
                "High": quotes["high"],
                "Low": quotes["low"],
                "Close": quotes["close"],
            }).dropna()
            df = df.set_index("dt").sort_index()
            return df
        except Exception as e:
            print(f"Price fetch error: {e}")
            return None


# ===================== SIGNAL LOGIC =====================

def compute_features(bars):
    """Compute all indicators on the 5m bar DataFrame."""
    if len(bars) < 60:
        return None
    d = bars.copy()
    h, l, c = d['High'], d['Low'], d['Close']
    d['atr'] = atr_calc(h, l, c, 14)
    d['ema5'] = ema(c, 5)
    d['ema9'] = ema(c, 9)
    d['ema21'] = ema(c, 21)
    d['ema50'] = ema(c, 50)
    d['rsi'] = rsi_calc(c, 14)
    d['macd'] = ema(c, 12) - ema(c, 26)
    d['macd_sig'] = ema(d['macd'], 9)
    d['macd_hist'] = d['macd'] - d['macd_sig']
    d['bb_mid'] = c.rolling(20).mean()
    d['bb_std'] = c.rolling(20).std()
    d['bb_upper'] = d['bb_mid'] + 2 * d['bb_std']
    d['bb_lower'] = d['bb_mid'] - 2 * d['bb_std']
    # 1h trend
    h1 = bars.resample('1h').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    if len(h1) >= 50:
        h1['ema20'] = ema(h1['Close'], 20)
        h1['ema50'] = ema(h1['Close'], 50)
        h1['trend'] = np.where(h1['ema20'] > h1['ema50'], 1,
                               np.where(h1['ema20'] < h1['ema50'], -1, 0))
        d['trend_1h'] = h1['trend'].reindex(d.index, method='ffill').fillna(0)
    else:
        d['trend_1h'] = 0
    return d.dropna(subset=['atr', 'ema5', 'ema21', 'rsi'])


def check_signal(feat):
    """Check the latest bar for a signal. Returns signal dict or None."""
    if len(feat) < 4:
        return None
    row = feat.iloc[-1]
    prev = feat.iloc[-2]
    hour = row.name.hour if hasattr(row.name, 'hour') else 12

    if hour < SESSION_START or hour > SESSION_END:
        return None
    if row['atr'] < 0.3:
        return None

    direction = None
    confidence = 50
    reason = ""

    # TYPE 1: FAST EMA CROSS
    cross_up = prev['ema5'] <= prev['ema21'] and row['ema5'] > row['ema21']
    cross_dn = prev['ema5'] >= prev['ema21'] and row['ema5'] < row['ema21']
    if cross_up and row['trend_1h'] >= 0 and row['rsi'] > 45:
        direction = 'BUY'; confidence = 70
        reason = "EMA5 crossed above EMA21 (momentum shift bullish), RSI confirms above 45, 1H trend aligned UP"
    elif cross_dn and row['trend_1h'] <= 0 and row['rsi'] < 55:
        direction = 'SELL'; confidence = 70
        reason = "EMA5 crossed below EMA21 (momentum shift bearish), RSI confirms below 55, 1H trend aligned DOWN"

    # TYPE 2: RSI MOMENTUM SHIFT
    if direction is None:
        if row['rsi'] > 55 and prev['rsi'] < 50 and row['trend_1h'] >= 0 and row['Close'] > row['ema21']:
            direction = 'BUY'; confidence = 65
            reason = "RSI punched through 50 from below (momentum ignition), price above EMA21, 1H uptrend"
        elif row['rsi'] < 45 and prev['rsi'] > 50 and row['trend_1h'] <= 0 and row['Close'] < row['ema21']:
            direction = 'SELL'; confidence = 65
            reason = "RSI dropped through 50 from above (momentum collapse), price below EMA21, 1H downtrend"

    # TYPE 3: BOLLINGER BOUNCE
    if direction is None:
        if 'bb_lower' in row.index and not pd.isna(row['bb_lower']):
            if prev['Close'] <= prev['bb_lower'] and row['Close'] > row['bb_lower'] and row['trend_1h'] >= 0:
                direction = 'BUY'; confidence = 60
                reason = "Price bounced off lower Bollinger Band (oversold snap-back), 1H trend bullish"
            elif prev['Close'] >= prev['bb_upper'] and row['Close'] < row['bb_upper'] and row['trend_1h'] <= 0:
                direction = 'SELL'; confidence = 60
                reason = "Price rejected from upper Bollinger Band (overbought reversal), 1H trend bearish"

    # TYPE 4: MACD FLIP
    if direction is None:
        if prev['macd_hist'] < 0 and row['macd_hist'] > 0 and row['ema9'] > row['ema21'] and row['trend_1h'] >= 0:
            direction = 'BUY'; confidence = 60
            reason = "MACD histogram flipped positive (buying pressure resuming), EMA stack bullish, 1H uptrend"
        elif prev['macd_hist'] > 0 and row['macd_hist'] < 0 and row['ema9'] < row['ema21'] and row['trend_1h'] <= 0:
            direction = 'SELL'; confidence = 60
            reason = "MACD histogram flipped negative (selling pressure resuming), EMA stack bearish, 1H downtrend"

    if direction is None:
        return None

    # Confidence bonus
    if direction == 'BUY' and row['Close'] > row['ema50']:
        confidence += 10; reason += " | Price above EMA50 (strong structure)"
    elif direction == 'SELL' and row['Close'] < row['ema50']:
        confidence += 10; reason += " | Price below EMA50 (strong structure)"

    return {
        "time": str(row.name)[:16] + " UTC",
        "direction": direction,
        "confidence": min(100, confidence),
        "close": float(row['Close']),
        "atr": float(row['atr']),
        "reason": reason,
    }


# ===================== TRADE MANAGEMENT =====================

def build_trade_signal(sig):
    """Compute full trade parameters from a raw signal."""
    entry_price = sig['close']
    atr_val = sig['atr']
    sl_dist = SL_MULT * atr_val
    tp_dist = TP_RATIO * sl_dist

    # Adaptive risk with DD caps
    dd_pct = (state.peak_equity - state.equity) / state.peak_equity if state.peak_equity > 0 else 0
    if dd_pct >= 0.085:
        return None  # halt near 10% cap
    elif dd_pct >= 0.06:
        eff_risk = 0.012
    elif dd_pct >= 0.035:
        eff_risk = 0.018
    else:
        eff_risk = RISK_PCT

    risk_dollars = state.equity * eff_risk

    if sig['direction'] == 'BUY':
        sl = entry_price - sl_dist
        tp = entry_price + tp_dist
    else:
        sl = entry_price + sl_dist
        tp = entry_price - tp_dist

    lots = max(0.01, round(risk_dollars / (sl_dist * 20.0), 2))
    actual_risk = lots * sl_dist * 20.0

    return {
        "time": sig['time'],
        "entry_ts": datetime.now(timezone.utc).isoformat(),  # for timeout tracking
        "direction": sig['direction'],
        "entry": round(entry_price, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "sl_dist": round(sl_dist, 2),
        "tp_dist": round(tp_dist, 2),
        "rr": f"1:{TP_RATIO}",
        "lots": lots,
        "risk": round(actual_risk, 2),
        "riskPct": round(eff_risk * 100, 1),
        "confidence": sig['confidence'],
        "atr": round(atr_val, 2),
        "equity": round(state.equity, 2),
        "reason": sig['reason'],
        "result": "PENDING",
        "pnl": None,
        "rMultiple": None,
        "duration": None,
    }


# ===================== TELEGRAM (send + receive) =====================

LAST_UPDATE_ID = 0

async def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            await client.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
        except:
            pass


async def check_telegram_messages():
    """Poll for incoming Telegram messages and respond."""
    global LAST_UPDATE_ID
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            params = {"offset": LAST_UPDATE_ID + 1, "timeout": 5}
            r = await client.get(url, params=params)
            if r.status_code != 200:
                return
            data = r.json()
            if not data.get("ok"):
                return
            for update in data.get("result", []):
                LAST_UPDATE_ID = update["update_id"]
                msg = update.get("message", {})
                text = msg.get("text", "").strip().lower()
                chat_id = msg.get("chat", {}).get("id")
                if str(chat_id) != TELEGRAM_CHAT_ID:
                    continue

                # Respond to commands
                if text in ["/status", "status", "stats"]:
                    wins = [s for s in state.signals_history if s.get('pnl') and s['pnl'] > 0]
                    losses = [s for s in state.signals_history if s.get('pnl') and s['pnl'] <= 0]
                    completed = wins + losses
                    wr = len(wins)/len(completed)*100 if completed else 0
                    reply = (
                        f"VORTEX STATUS\n"
                        f"Equity: ${state.equity:.2f}\n"
                        f"P&L: ${state.equity - STARTING_BALANCE:+.2f}\n"
                        f"Trades: {len(completed)} (W{len(wins)}/L{len(losses)})\n"
                        f"Win Rate: {wr:.1f}%\n"
                        f"Open: {'YES - ' + state.open_trade['direction'] + ' @ $' + str(state.open_trade['entry']) if state.open_trade else 'No'}\n"
                        f"Last scan: {state.last_scan or 'not yet'}\n"
                        f"Price: ${state.price:.2f}"
                    )
                    await send_telegram(reply)

                elif text in ["/config", "config", "settings"]:
                    reply = (
                        f"VORTEX CONFIG\n"
                        f"Instrument: NAS100\n"
                        f"Timeframe: {TIMEFRAME}\n"
                        f"Session: {SESSION_START}:00-{SESSION_END}:00 UTC\n"
                        f"SL: {SL_MULT}x ATR | TP: {TP_RATIO}:1\n"
                        f"Risk: {RISK_PCT*100}% per trade\n"
                        f"Cooldown: {COOLDOWN} bars ({COOLDOWN*BAR_MINUTES}min)\n"
                        f"Max: {MAX_PER_DAY}/day | Hold: {MAX_HOLD} bars ({MAX_HOLD*BAR_MINUTES/60:.1f}h)"
                    )
                    await send_telegram(reply)

                elif text in ["/trade", "trade", "open trade", "current"]:
                    if state.open_trade:
                        t = state.open_trade
                        reply = (
                            f"OPEN TRADE\n"
                            f"{t['direction']} @ ${t['entry']:.2f}\n"
                            f"SL: ${t['sl']:.2f} | TP: ${t['tp']:.2f}\n"
                            f"Size: {t['lots']} lots | Risk: ${t['risk']:.2f}\n"
                            f"Current price: ${state.price:.2f}"
                        )
                    else:
                        reply = "No open trade. Waiting for next signal."
                    await send_telegram(reply)

                elif text in ["/help", "help", "commands"]:
                    reply = (
                        "VORTEX COMMANDS:\n"
                        "/status - Account stats\n"
                        "/config - Current settings\n"
                        "/trade - Open trade info\n"
                        "/help - This menu"
                    )
                    await send_telegram(reply)

                elif text:
                    await send_telegram(f"Unknown command. Type /help for options.")

        except Exception as e:
            pass


async def send_signal_telegram(trade):
    msg = (
        f"{'BUY' if trade['direction']=='BUY' else 'SELL'} NAS100\n"
        f"Entry: ${trade['entry']:.2f}\n"
        f"SL: ${trade['sl']:.2f} ({trade['sl_dist']:.2f})\n"
        f"TP: ${trade['tp']:.2f} ({trade['tp_dist']:.2f})\n"
        f"R:R: {trade['rr']}\n"
        f"Size: {trade['lots']} lots\n"
        f"Risk: ${trade['risk']:.2f} ({trade['riskPct']}%)\n"
        f"Confidence: {trade['confidence']}/100\n"
        f"---\n"
        f"WHY: {trade['reason']}"
    )
    await send_telegram(msg)


# ===================== MAIN SCAN LOOP =====================

async def scan_loop():
    """Main loop: every 5 min, fetch bars, compute features, check for signal.
    Enforces ALL strategy rules identically to the backtest:
      - Session filter (07:00-20:00 UTC)
      - Cooldown between trades (30 min)
      - Max 2 trades per day
      - Daily DD halt (5% of equity)
      - Total DD halt (10% from peak)
      - Adaptive risk scaling
      - Trade timeout (MAX_HOLD * 5 min = 5 hours)
    """
    state.running = True
    print(f"Scan loop started. Checking every {SCAN_INTERVAL}s during session {SESSION_START}:00-{SESSION_END}:00 UTC")
    print(f"Strategy: SL={SL_MULT}x ATR | TP={TP_RATIO}:1 | Max {MAX_PER_DAY}/day | Timeout {MAX_HOLD*BAR_MINUTES/60:.1f}h")
    print(f"Telegram: listening for commands (/status, /config, /trade, /help)")

    while state.running:
        try:
            # Check for Telegram messages (respond to user commands)
            await check_telegram_messages()

            now = datetime.now(timezone.utc)
            state.last_scan = now.strftime("%Y-%m-%d %H:%M UTC")
            day_key = str(now.date())

            # Reset daily counters at the start of each new day
            if day_key not in state.trades_today:
                state.trades_today = {day_key: 0}
                state.daily_pnl = {day_key: 0.0}

            # --- CHECK OPEN TRADE FIRST (timeout + SL/TP) ---
            if state.open_trade is not None:
                bars = await fetch_5m_bars()
                if bars is not None and len(bars) > 0:
                    state.price = float(bars['Close'].iloc[-1])

                t = state.open_trade
                hit = None
                pnl = 0.0
                r_mult = 0.0

                # 1) Check timeout (MAX_HOLD bars = 5 hours)
                try:
                    entry_dt = datetime.fromisoformat(t['entry_ts'])
                    elapsed_min = (now - entry_dt).total_seconds() / 60
                    if elapsed_min >= MAX_HOLD * BAR_MINUTES:
                        # Timeout: close at current price
                        if t['direction'] == 'BUY':
                            pnl = (state.price - t['entry']) * t['lots'] * 20.0
                        else:
                            pnl = (t['entry'] - state.price) * t['lots'] * 20.0
                        r_mult = pnl / t['risk'] if t['risk'] > 0 else 0
                        hit = 'TIMEOUT'
                except:
                    pass

                # 2) Check SL/TP hit (only if not already timed out)
                if hit is None and state.price > 0:
                    if t['direction'] == 'BUY':
                        if state.price <= t['sl']:
                            hit = 'SL'
                            pnl = -(t['lots'] * t['sl_dist'] * 20.0)
                            r_mult = -1.0
                        elif state.price >= t['tp']:
                            hit = 'TP'
                            pnl = t['lots'] * t['tp_dist'] * 20.0
                            r_mult = TP_RATIO
                    else:
                        if state.price >= t['sl']:
                            hit = 'SL'
                            pnl = -(t['lots'] * t['sl_dist'] * 20.0)
                            r_mult = -1.0
                        elif state.price <= t['tp']:
                            hit = 'TP'
                            pnl = t['lots'] * t['tp_dist'] * 20.0
                            r_mult = TP_RATIO

                # 3) If trade closed, update state
                if hit:
                    state.equity += pnl
                    state.peak_equity = max(state.peak_equity, state.equity)
                    state.daily_pnl[day_key] = state.daily_pnl.get(day_key, 0.0) + pnl
                    try:
                        entry_dt = datetime.fromisoformat(t['entry_ts'])
                        dur_h = (now - entry_dt).total_seconds() / 3600
                    except:
                        dur_h = 0
                    t['result'] = hit
                    t['pnl'] = round(pnl, 2)
                    t['rMultiple'] = round(r_mult, 1)
                    t['duration'] = f"{dur_h:.1f}h"
                    state.open_trade = None
                    state.save_state()

                    result_msg = (f"{'WIN' if pnl>0 else 'LOSS'} | {hit} | "
                                  f"PnL ${pnl:+.2f} ({r_mult:+.1f}R) | "
                                  f"Duration: {dur_h:.1f}h | Equity: ${state.equity:.2f}")
                    print(f"  >> {result_msg}")
                    await send_telegram(result_msg)

            # --- LOOK FOR NEW SIGNAL (only if no open trade) ---
            elif SESSION_START <= now.hour <= SESSION_END:
                # Daily DD check
                today_loss = state.daily_pnl.get(day_key, 0.0)
                if today_loss <= -(0.05 * state.equity):
                    pass  # halted for the day
                # Total DD check
                elif (state.peak_equity - state.equity) / state.peak_equity >= 0.085:
                    pass  # halted near 10% cap
                # Max trades/day check
                elif state.trades_today.get(day_key, 0) >= MAX_PER_DAY:
                    pass  # already traded max today
                else:
                    bars = await fetch_5m_bars()
                    if bars is not None and len(bars) > 60:
                        state.bars_5m = bars
                        state.price = float(bars['Close'].iloc[-1])

                        feat = compute_features(bars)
                        if feat is not None:
                            sig = check_signal(feat)
                            if sig is not None:
                                # Cooldown check (COOLDOWN * 5 min = 30 min between signals)
                                cooldown_ok = True
                                if state.last_signal_time:
                                    try:
                                        last_ts = state.last_signal_time.replace(" UTC", "")
                                        last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M")
                                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                                        gap_min = (now - last_dt).total_seconds() / 60
                                        if gap_min < COOLDOWN * BAR_MINUTES:
                                            cooldown_ok = False
                                    except:
                                        pass  # if parsing fails, allow

                                if cooldown_ok:
                                    trade = build_trade_signal(sig)
                                    if trade is not None:
                                        state.open_trade = trade
                                        state.signals_history.append(trade)
                                        state.last_signal_time = sig['time']
                                        state.trades_today[day_key] = state.trades_today.get(day_key, 0) + 1
                                        state.save_state()
                                        print(f"\n  NEW SIGNAL: {trade['direction']} @ ${trade['entry']:.2f} "
                                              f"| SL ${trade['sl']:.2f} | TP ${trade['tp']:.2f} "
                                              f"| {trade['lots']} lots | Conf {trade['confidence']}")
                                        print(f"  WHY: {trade['reason']}")
                                        await send_signal_telegram(trade)

        except Exception as e:
            print(f"Scan error: {e}")

        await asyncio.sleep(SCAN_INTERVAL)


# ===================== FASTAPI SERVER =====================

app = FastAPI(title="VORTEX v5 — NAS100")


@app.on_event("startup")
async def startup():
    asyncio.create_task(scan_loop())
    print(f"\n  VORTEX v5 — NAS100 Live Signal Bot")
    print(f"  Dashboard: http://localhost:{PORT}")
    print(f"  Instrument: NAS100 (Nasdaq 100 Futures)")
    print(f"  Scanning every 5 min during {SESSION_START}:00-{SESSION_END}:00 UTC")
    print(f"  Press Ctrl+C to stop\n")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Dashboard not found</h1>")


@app.get("/api/state")
async def api_state():
    wins = [s for s in state.signals_history if s.get('pnl') and s['pnl'] > 0]
    losses = [s for s in state.signals_history if s.get('pnl') and s['pnl'] <= 0]
    completed = wins + losses
    return JSONResponse({
        "equity": state.equity,
        "peak_equity": state.peak_equity,
        "net_pnl": state.equity - STARTING_BALANCE,
        "price": state.price,
        "total_trades": len(completed),
        "win_rate": len(wins) / len(completed) * 100 if completed else 0,
        "profit_factor": sum(s['pnl'] for s in wins) / abs(sum(s['pnl'] for s in losses)) if losses and sum(s['pnl'] for s in losses) != 0 else 0,
        "drawdown_pct": (state.peak_equity - state.equity) / state.peak_equity * 100 if state.peak_equity > 0 else 0,
        "open_trade": state.open_trade,
        "last_scan": state.last_scan,
        "running": state.running,
    })


@app.get("/api/signals")
async def api_signals():
    return JSONResponse(state.signals_history[-50:])


@app.get("/api/health")
async def api_health():
    return JSONResponse({"status": "ok", "price": state.price, "last_scan": state.last_scan})


# ===================== ENTRY POINT =====================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
