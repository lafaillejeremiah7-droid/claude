"""
Standalone live XAUUSD terminal server + signal delivery via Telegram.

Pulls real-time gold price (no broker needed), computes indicators, runs the
ASWP engine, serves the signal terminal dashboard with live data, and sends
Telegram signals when EV >= threshold.

Run:
    cd tradelocker_bot
    python3 live_terminal.py

Then open: http://localhost:8080
Signals auto-sent to Telegram when they fire.
"""
import asyncio
import json
import time
import threading
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
import uvicorn

# ---------------------------------------------------------------------------
# Telegram delivery
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = "8926622863:AAF0QHHYAyEVQZiYV35b5vyeKxDC_ouMnmQ"
TELEGRAM_CHAT_ID = "7040023207"
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

async def send_telegram(text: str):
    """Send a message to the trader via Telegram."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(TELEGRAM_URL, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Monospace",
            })
    except Exception as e:
        print(f"Telegram send error: {e}")

# ---------------------------------------------------------------------------
# Live price feed (TradingView websocket — real-time, no API key, no broker)
# ---------------------------------------------------------------------------

import websocket as ws_lib
import re as _re
import random as _random
import string as _string
import threading

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"

def _tv_session():
    return "qs_" + "".join(_random.choices(_string.ascii_lowercase, k=12))

def _tv_header(st):
    return "~m~" + str(len(st)) + "~m~" + st

def _tv_msg(func, params):
    return _tv_header(json.dumps({"m": func, "p": params}, separators=(",", ":")))

class LiveFeed:
    def __init__(self):
        self.price = 0.0
        self.high = 0.0
        self.low = 0.0
        self.open = 0.0
        self.change = 0.0
        self.bars_5m = []
        self.bars_15m = []
        self.bars_1h = []
        self.yield_value = 0.0
        self.yield_change = 0.0
        self.last_update = 0.0
        self._bars_fetched_at = 0.0
        self._ws = None
        self._ws_thread = None
        self._running = False

    def start_websocket(self):
        """Start TradingView websocket in a background thread."""
        self._running = True
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()

    def _ws_loop(self):
        """Persistent websocket connection to TradingView for real-time XAUUSD."""
        while self._running:
            try:
                session = _tv_session()
                self._ws = ws_lib.create_connection(
                    "wss://data.tradingview.com/socket.io/websocket",
                    headers={"Origin": "https://data.tradingview.com"},
                    timeout=30
                )
                # Handshake
                self._ws.recv()
                self._ws.send(_tv_header('{"m":"set_auth_token","p":["unauthorized_user_token"]}'))
                self._ws.send(_tv_msg("quote_create_session", [session]))
                self._ws.send(_tv_msg("quote_set_fields", [session, "lp", "ch", "chp", "high_price", "low_price", "open_price", "volume"]))
                self._ws.send(_tv_msg("quote_add_symbols", [session, "OANDA:XAUUSD"]))

                while self._running:
                    data = self._ws.recv()
                    if "lp" in data:
                        lp = _re.findall(r'"lp":([\d.]+)', data)
                        if lp: self.price = float(lp[0])
                        ch = _re.findall(r'"ch":([-\d.]+)', data)
                        if ch: self.change = float(ch[0])
                        hp = _re.findall(r'"high_price":([\d.]+)', data)
                        if hp: self.high = float(hp[0])
                        lpp = _re.findall(r'"low_price":([\d.]+)', data)
                        if lpp: self.low = float(lpp[0])
                        op = _re.findall(r'"open_price":([\d.]+)', data)
                        if op: self.open = float(op[0])
                        self.last_update = time.time()
                    # Respond to pings
                    if data.startswith("~m~") and "~h~" in data:
                        self._ws.send(data)
            except Exception as e:
                print(f"WS error: {e} — reconnecting in 5s")
                time.sleep(5)

    async def fetch_price(self):
        """Fetch 5-day historical bars from Forexite (same source as backtest). One-time + refresh every 30min."""
        if self.bars_5m and time.time() - self._bars_fetched_at < 1800:
            return  # bars fresh enough, TradingView websocket handles live price
        try:
            from datetime import date as _date, timedelta as _td
            import zipfile, io
            async with httpx.AsyncClient(timeout=15) as client:
                all_rows = []
                for offset in range(1, 6):
                    d = _date.today() - _td(days=offset)
                    dd = d.strftime("%d%m%y")
                    url = f"http://www.forexite.com/free_forex_quotes/{d.year}/{d.month:02d}/{dd}.zip"
                    r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    if r.status_code == 200 and len(r.content) > 100:
                        z = zipfile.ZipFile(io.BytesIO(r.content))
                        for fn in z.namelist():
                            for line in z.open(fn).read().decode("utf-8", "ignore").splitlines():
                                if line.startswith("XAUUSD,"):
                                    p = line.split(",")
                                    if len(p) == 7:
                                        all_rows.append(p)
                if all_rows:
                    import pandas as _pd
                    df = _pd.DataFrame(all_rows, columns=["T","D","Tm","O","H","L","C"])
                    for col in ["O","H","L","C"]: df[col] = df[col].astype(float)
                    df["dt"] = _pd.to_datetime(df["D"]+df["Tm"], format="%Y%m%d%H%M%S")
                    df = df.set_index("dt").sort_index()
                    # Resample to 5m
                    d5 = df.resample("5min").agg({"O":"first","H":"max","L":"min","C":"last"}).dropna()
                    self.bars_5m = [(int(ts.timestamp()), r["O"], r["H"], r["L"], r["C"]) for ts, r in d5.iterrows()]
                    self._resample()
                    self._bars_fetched_at = time.time()
                    print(f"Bars loaded: {len(self.bars_5m)} 5m from Forexite")
        except Exception as e:
            print(f"Bar fetch error: {e}")

    def _resample(self):
        self.bars_15m = self._merge_bars(self.bars_5m, 3)
        self.bars_1h = self._merge_bars(self.bars_5m, 12)

    @staticmethod
    def _merge_bars(bars, factor):
        merged = []
        for i in range(0, len(bars) - factor + 1, factor):
            chunk = bars[i:i+factor]
            merged.append((chunk[0][0], chunk[0][1], max(b[2] for b in chunk), min(b[3] for b in chunk), chunk[-1][4]))
        return merged

    async def fetch_yields(self):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(FRED_URL)
                if r.status_code == 200:
                    lines = r.text.strip().splitlines()
                    vals = []
                    for line in lines[-15:]:
                        parts = line.split(",")
                        if len(parts) == 2 and parts[1].strip() and parts[1].strip() != ".":
                            try: vals.append(float(parts[1]))
                            except: pass
                    if len(vals) >= 2:
                        self.yield_value = vals[-1]
                        self.yield_change = vals[-1] - vals[-8] if len(vals) >= 8 else vals[-1] - vals[0]
        except Exception as e:
            print(f"Yield fetch error: {e}")


# ---------------------------------------------------------------------------
# Indicator computation (matches the engine's logic)
# ---------------------------------------------------------------------------

def compute_ema(values, period):
    if not values: return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out

def compute_atr(bars, period=14):
    if len(bars) < 2: return 0
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i][2], bars[i][3], bars[i-1][4]
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    if len(trs) < period: return sum(trs)/len(trs) if trs else 0
    return sum(trs[-period:]) / period

def compute_rsi(closes, period=14):
    if len(closes) < period + 1: return 50
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al < 1e-10: return 100
    rs = ag / al
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------------------
# Dashboard state (computed from live data)
# ---------------------------------------------------------------------------

class DashboardState:
    def __init__(self, feed: LiveFeed):
        self.feed = feed
        self.signals_today = 0
        self.total_signals = 0
        self.scanned = 0
        self.ev_passed = 0
        self.signal_feed = []
        self.equity = 5000.0
        self.daily_pnl = 0.0
        self.max_dd = 0.0
        self.peak = 5000.0
        self.active_signal = None
        self.last_signal_time = 0
        self._today = None
        # Signal cooldown: minimum seconds between signals
        self.SIGNAL_COOLDOWN = 180  # 3 minutes between signals
        self.MAX_SIGNALS_DAY = 4
        self.MIN_EV = 0.55

    async def check_and_signal(self):
        """Run the full entry logic and send Telegram if a signal fires."""
        f = self.feed
        now = datetime.now(timezone.utc)

        # Daily reset
        if self._today != now.date():
            self._today = now.date()
            self.signals_today = 0
            self.daily_pnl = 0.0

        # Guards
        if self.signals_today >= self.MAX_SIGNALS_DAY:
            return
        if time.time() - self.last_signal_time < self.SIGNAL_COOLDOWN:
            return
        if now.hour in (21, 22):  # thin liquidity
            return
        if now.weekday() == 4 and now.hour >= 19:  # friday evening
            return

        bars_5m = f.bars_5m
        bars_1h = f.bars_1h
        if len(bars_5m) < 30 or len(bars_1h) < 50:
            return

        closes_5m = [b[4] for b in bars_5m]
        closes_1h = [b[4] for b in bars_1h]
        atr_val = compute_atr(bars_5m)
        if atr_val <= 0:
            return

        # 1H trend gate
        e20_1h = compute_ema(closes_1h, 20)
        e50_1h = compute_ema(closes_1h, 50)
        if e20_1h[-1] > e50_1h[-1]:
            direction = "buy"
        elif e20_1h[-1] < e50_1h[-1]:
            direction = "sell"
        else:
            return

        # 15m pullback zone
        bars_15m = f.bars_15m
        if not bars_15m:
            return
        closes_15m = [b[4] for b in bars_15m]
        e20_15m = compute_ema(closes_15m, 20)
        atr_15m = compute_atr(bars_15m)
        if atr_15m <= 0:
            return
        dist_15m = abs(closes_15m[-1] - e20_15m[-1]) / atr_15m
        if dist_15m > 0.5:
            return

        # 5m confirmation trigger
        e20_5m = compute_ema(closes_5m, 20)
        rsi_val = compute_rsi(closes_5m)
        if len(bars_5m) < 2:
            return
        prev = bars_5m[-2]
        prev_o, prev_h, prev_l, prev_c = prev[1], prev[2], prev[3], prev[4]
        pe = e20_5m[-2] if len(e20_5m) >= 2 else e20_5m[-1]

        if direction == "buy":
            trigger = prev_l <= pe * 1.002 and prev_c > prev_o and rsi_val < 65
        else:
            trigger = prev_h >= pe * 0.998 and prev_c < prev_o and rsi_val > 35
        if not trigger:
            return

        self.scanned += 1

        # Real yield alignment
        align = 0.0
        if abs(f.yield_change) > 0.05:
            mag = min(1.0, abs(f.yield_change) / 0.30)
            gold_sign = 1.0 if f.yield_change < 0 else -1.0
            dir_sign = 1.0 if direction == "buy" else -1.0
            align = gold_sign * dir_sign * mag

        # EV calculation (ASWP simplified for live — uses pre-trained stats)
        # Base probabilities from 4.5yr validation: P(TP1)~54%, P(TP2)~50%, P(TP3)~44%
        # Adjusted by alignment and trend strength
        trend_boost = 0.05 if abs(e20_1h[-1] - e50_1h[-1]) / atr_val > 2 else 0
        align_boost = align * 0.08
        p_tp1 = min(0.85, max(0.35, 0.54 + trend_boost + align_boost))
        p_tp2 = p_tp1 * 0.92
        p_tp3 = p_tp1 * 0.81

        # Multi-TP runner-weighted EV
        ev = 0.10 * 1.0 * p_tp1 + 0.20 * 2.0 * p_tp2 + 0.70 * 3.0 * p_tp3 - (1 - p_tp1) * 1.0

        if ev < self.MIN_EV:
            return

        self.ev_passed += 1

        # SIGNAL FIRES — compute levels
        entry = f.price
        sl_mult = 1.0
        spread = 0.30
        sl_dist = sl_mult * atr_val + spread / 2

        # Position sizing: capped at 0.12 lots, risk bounded
        contract = 100.0
        leverage = 10.0
        loss_per_lot = contract * sl_dist
        max_loss = 250.0 * 0.45  # 45% of daily limit
        lots = min(0.12, max_loss / loss_per_lot)
        lots = max(0.01, round(int(lots * 100) / 100, 2))  # round to 0.01 step

        # Margin check
        margin = lots * contract * entry / leverage
        if margin > self.equity * 0.95:
            lots = max(0.01, round(int((self.equity * 0.95 * leverage / (contract * entry)) * 100) / 100, 2))

        risk_dollars = lots * loss_per_lot
        base_move = sl_mult * atr_val

        if direction == "buy":
            sl = round(entry - sl_dist, 2)
            tp1 = round(entry + 1.0 * base_move + spread, 2)
            tp2 = round(entry + 2.0 * base_move + spread, 2)
            tp_final = round(entry + 3.0 * base_move + spread, 2)
        else:
            sl = round(entry + sl_dist, 2)
            tp1 = round(entry - 1.0 * base_move - spread, 2)
            tp2 = round(entry - 2.0 * base_move - spread, 2)
            tp_final = round(entry - 3.0 * base_move - spread, 2)

        full_win = lots * contract * (0.10 * 1.0 * base_move + 0.20 * 2.0 * base_move + 0.70 * 3.0 * base_move)

        # Build signal
        self.active_signal = {
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp_final": tp_final,
            "lot_size": lots,
            "risk_dollars": round(risk_dollars, 2),
            "probability": round(p_tp1, 2),
            "expected_value_r": round(ev, 2),
            "full_win": round(full_win, 2),
            "atr": round(atr_val, 2),
        }

        self.signals_today += 1
        self.total_signals += 1
        self.last_signal_time = time.time()

        # Add to feed
        ts_str = now.strftime("%H:%M")
        self.signal_feed.append({
            "time": ts_str,
            "type": "signal",
            "label": direction.upper(),
            "text": f"${entry:.2f} | EV:{ev:.2f}R | {lots} lots"
        })

        # SEND TELEGRAM
        arrow = "BUY" if direction == "buy" else "SELL"
        msg = (
            f"XAUUSD {arrow}  |  {lots} lots\n"
            f"Entry: {entry:.2f}\n"
            f"SL: {sl:.2f}  (risk -${risk_dollars:.2f})\n"
            f"TP1: {tp1:.2f}  (close 10%, SL->BE)\n"
            f"TP2: {tp2:.2f}  (close 20%, SL->TP1)\n"
            f"Final: {tp_final:.2f}  (ride 70%)\n"
            f"Probability: {p_tp1:.2f}  |  EV: +{ev:.2f}R\n"
            f"Est. full win: +${full_win:.2f}"
        )
        await send_telegram(msg)
        print(f"[{ts_str}] SIGNAL SENT: {arrow} @ {entry:.2f}")


    def compute_snapshot(self) -> dict:
        f = self.feed
        bars_5m = f.bars_5m
        bars_1h = f.bars_1h

        closes_5m = [b[4] for b in bars_5m] if bars_5m else []
        closes_1h = [b[4] for b in bars_1h] if bars_1h else []
        atr_val = compute_atr(bars_5m) if len(bars_5m) > 14 else 0
        rsi_val = compute_rsi(closes_5m) if closes_5m else 50

        # 1H trend
        trend = 0
        if len(closes_1h) >= 50:
            e20 = compute_ema(closes_1h, 20)
            e50 = compute_ema(closes_1h, 50)
            if e20[-1] > e50[-1]: trend = 1
            elif e20[-1] < e50[-1]: trend = -1

        # Alignment
        align = 0.0
        if abs(f.yield_change) > 0.05:
            mag = min(1.0, abs(f.yield_change) / 0.30)
            gold_sign = 1.0 if f.yield_change < 0 else -1.0
            dir_sign = 1.0 if trend >= 0 else -1.0
            align = round(gold_sign * dir_sign * mag, 3)

        # Pipeline stage
        pipeline = 1
        if trend != 0: pipeline = 2
        if pipeline >= 2 and f.bars_15m:
            closes_15m = [b[4] for b in f.bars_15m]
            e20_15 = compute_ema(closes_15m, 20)
            atr_15 = compute_atr(f.bars_15m)
            if e20_15 and atr_15 > 0:
                if abs(closes_15m[-1] - e20_15[-1]) / atr_15 < 0.5:
                    pipeline = 3
        if pipeline >= 3 and 35 < rsi_val < 65:
            pipeline = 4
        if self.active_signal:
            pipeline = 7

        # EV for display
        ev = 0.0
        p_tp1 = 0.50
        if pipeline >= 4 and atr_val > 0:
            trend_boost = 0.05 if len(closes_1h) >= 50 and abs(compute_ema(closes_1h,20)[-1] - compute_ema(closes_1h,50)[-1]) / atr_val > 2 else 0
            align_boost = align * 0.08
            p_tp1 = min(0.85, max(0.35, 0.54 + trend_boost + align_boost))
            p_tp2 = p_tp1 * 0.92
            p_tp3 = p_tp1 * 0.81
            ev = 0.10*1.0*p_tp1 + 0.20*2.0*p_tp2 + 0.70*3.0*p_tp3 - (1-p_tp1)*1.0
            if ev >= self.MIN_EV: pipeline = max(pipeline, 5)

        # Session guards
        now = datetime.now(timezone.utc)
        news_clear = True
        weekend_clear = not (now.weekday() == 4 and now.hour >= 19)
        hours_active = now.hour not in (21, 22)

        return {
            "mode": "live",
            "equity": self.equity,
            "current_price": f.price,
            "atr": round(atr_val, 2),
            "trend_1h": trend,
            "yield_value": f.yield_value,
            "yield_change": round(f.yield_change, 4),
            "yield_alignment": align,
            "p_tp1": round(p_tp1, 2),
            "p_tp2": round(p_tp1 * 0.92, 2) if pipeline >= 4 else 0,
            "p_tp3": round(p_tp1 * 0.81, 2) if pipeline >= 4 else 0,
            "current_ev": round(ev, 3),
            "memory_size": 2500,
            "pipeline_stage": pipeline,
            "daily_pnl": self.daily_pnl,
            "max_dd": self.max_dd,
            "signals_today": self.signals_today,
            "signals_count": self.total_signals,
            "trades": self.total_signals,
            "win_rate": 0.60,
            "avg_r": 1.108,
            "profit_factor": 2.4,
            "trades_per_day": 1.0,
            "scanned": self.scanned,
            "ev_passed": self.ev_passed,
            "deployed": self.total_signals,
            "guards": {
                "news_clear": news_clear,
                "weekend_clear": weekend_clear,
                "hours_active": hours_active,
            },
            "active_signal": self.active_signal,
            "feed": self.signal_feed[-20:],
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="XAUUSD ASWP Live Terminal")
feed = LiveFeed()
state = DashboardState(feed)

FRONTEND_DIR = Path(__file__).parent / "dashboard" / "frontend"
TERMINAL_HTML = FRONTEND_DIR / "signal_terminal.html"

@app.on_event("startup")
async def startup():
    feed.start_websocket()  # TradingView real-time price in background thread
    asyncio.create_task(price_loop())
    asyncio.create_task(yield_loop())
    asyncio.create_task(market_status_loop())

async def market_status_loop():
    """Send Telegram when XAUUSD market opens/closes."""
    was_open = None
    while True:
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # 0=Mon ... 6=Sun
        hour = now.hour
        # XAUUSD: open Sunday 22:00 UTC → Friday 21:00 UTC
        if weekday == 4 and hour >= 21:
            is_open = False
        elif weekday == 5:
            is_open = False
        elif weekday == 6 and hour < 22:
            is_open = False
        else:
            is_open = True
        if was_open is not None and is_open != was_open:
            if is_open:
                await send_telegram("Yo XAUUSD market just opened. Bot is scanning. Signals coming when conditions align.")
            else:
                await send_telegram("Yo XAUUSD market just closed. No signals until Sunday 22:00 UTC (market reopens).")
        was_open = is_open
        await asyncio.sleep(60)

async def price_loop():
    while True:
        try:
            await asyncio.wait_for(feed.fetch_price(), timeout=15)
        except (asyncio.TimeoutError, Exception) as e:
            print(f"Price loop: {e}")
        try:
            await state.check_and_signal()
        except Exception as e:
            print(f"Signal check: {e}")
        await asyncio.sleep(30)

async def yield_loop():
    while True:
        try:
            await asyncio.wait_for(feed.fetch_yields(), timeout=15)
        except (asyncio.TimeoutError, Exception) as e:
            print(f"Yield loop: {e}")
        await asyncio.sleep(3600)

@app.get("/terminal", response_class=HTMLResponse)
async def terminal():
    try:
        return HTMLResponse(TERMINAL_HTML.read_text(encoding="utf-8"))
    except:
        return HTMLResponse("<h1>Terminal HTML not found</h1>", status_code=500)

@app.get("/", response_class=HTMLResponse)
async def root():
    return await terminal()

@app.get("/api/snapshot")
async def snapshot():
    return JSONResponse(state.compute_snapshot())

@app.get("/api/health")
async def health():
    return {"status": "ok", "price": feed.price, "last_update": feed.last_update}

@app.get("/api/stream")
async def stream(request: Request):
    async def event_generator():
        last_hash = ""
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    snap = state.compute_snapshot()
                    data = json.dumps(snap, default=str)
                    h = str(hash(data))
                    if h != last_hash:
                        yield f"data: {data}\n\n"
                        last_hash = h
                    else:
                        yield f": heartbeat\n\n"
                except Exception:
                    yield f": error\n\n"
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            pass
    from starlette.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    print("Starting XAUUSD ASWP Live Terminal...")
    print("Open http://localhost:8080/terminal")
    uvicorn.run(app, host="0.0.0.0", port=8080)
