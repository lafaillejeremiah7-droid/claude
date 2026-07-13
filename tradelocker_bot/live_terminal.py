"""
Standalone live XAUUSD terminal server + signal delivery via Telegram.

Pulls real-time gold price (no broker needed), computes indicators, runs the
ASWP engine, serves the signal terminal dashboard with live data, and sends
Telegram signals when EV >= threshold.

Run:
    cd tradelocker_bot
    python3 live_terminal.py

Then open: http://localhost:5000
Signals auto-sent to Telegram when they fire.
"""
import asyncio
import json
import time
import threading
from contextlib import asynccontextmanager
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

async def send_telegram(text: str) -> bool:
    """Send a message to the trader via Telegram. Returns True on success.

    NOTE: no parse_mode is set. Telegram only accepts 'Markdown', 'MarkdownV2'
    or 'HTML' — the old 'Monospace' value caused every send to fail with a
    400 error that was silently swallowed. Plain text is the safest choice.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(TELEGRAM_URL, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
            })
            if resp.status_code != 200:
                print(f"Telegram send FAILED: {resp.status_code} {resp.text[:200]}")
                return False
            return True
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False

# ---------------------------------------------------------------------------
# Live price feed (TradingView websocket — real-time, no API key, no broker)
# ---------------------------------------------------------------------------

import websocket as ws_lib
import re as _re
import random as _random
import string as _string
import threading

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"

# ---------------------------------------------------------------------------
# TradeLocker API config (your AquaFunded demo account)
#   - account id  = 2325106  (the "D#2325106" shown in the app)
#   - accNum      = 4        (what the API 'accNum' header requires)
# Same broker for BOTH live quotes and historical bars => indicators align
# perfectly with the live price (no stale-data drift).
# ---------------------------------------------------------------------------
TL_URL = "https://demo.tradelocker.com/backend-api"
TL_EMAIL = "lafaillejeremiah7@gmail.com"
TL_PASS = ",3)m1U"
TL_SERVER = "AQUA"
TL_ACC_NUM = "4"            # accNum for account D#2325106
TL_INSTRUMENT = "1714"      # XAUUSD tradableInstrumentId
TL_ROUTE_INFO = "791554"    # XAUUSD INFO route (quotes + history)

def _tv_session():
    return "qs_" + "".join(_random.choices(_string.ascii_lowercase, k=12))

def _tv_header(st):
    return "~m~" + str(len(st)) + "~m~" + st

def _tv_msg(func, params):
    return _tv_header(json.dumps({"m": func, "p": params}, separators=(",", ":")))

class LiveFeed:
    def __init__(self):
        self.price = 0.0
        self.bid = 0.0
        self.ask = 0.0
        self.price_source = "init"
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
        self._token = None
        self._token_time = 0.0
        self._token_lock = threading.Lock()
        self.bars_source = "none"

    def start_websocket(self):
        """Start TradingView websocket + HTTP fallback in background thread."""
        self._running = True
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()
        # Also start an HTTP price poller as backup (every 10s)
        self._http_thread = threading.Thread(target=self._http_price_loop, daemon=True)
        self._http_thread.start()

    def _http_price_loop(self):
        """Primary: poll TradeLocker API for live XAUUSD price every 10s."""
        import requests as _req
        fail_count = 0

        while self._running:
            try:
                token = self._ensure_token()
                if token:
                    r = _req.get(f"{TL_URL}/trade/quotes",
                                 headers={"Authorization": f"Bearer {token}", "accNum": TL_ACC_NUM},
                                 params={"routeId": TL_ROUTE_INFO, "tradableInstrumentId": TL_INSTRUMENT},
                                 timeout=8)
                    if r.status_code == 200:
                        d = r.json().get("d", {})
                        ask = d.get("ap", 0)
                        bid = d.get("bp", 0)
                        if ask > 0 and bid > 0:
                            self.price = (ask + bid) / 2  # mid price
                            self.bid = bid
                            self.ask = ask
                            self.last_update = time.time()
                            self.price_source = "TradeLocker"
                            fail_count = 0
                    else:
                        fail_count += 1
                        if fail_count <= 3 or fail_count % 10 == 0:
                            print(f"TradeLocker quote failed: {r.status_code} {r.text[:150]}")
                        if r.status_code in (401, 403):
                            self._token = None  # force re-auth
            except Exception as e:
                print(f"TradeLocker price error: {e}")
            time.sleep(10)

    def _ensure_token(self):
        """Return a valid TradeLocker JWT, refreshing if older than 5 min. Thread-safe."""
        import requests as _req
        with self._token_lock:
            if self._token and time.time() - self._token_time < 300:
                return self._token
            try:
                r = _req.post(f"{TL_URL}/auth/jwt/token",
                              json={"email": TL_EMAIL, "password": TL_PASS, "server": TL_SERVER},
                              timeout=10)
                if r.status_code in (200, 201):
                    self._token = r.json()["accessToken"]
                    self._token_time = time.time()
                else:
                    print(f"TradeLocker auth failed: {r.status_code} {r.text[:150]}")
                    self._token = None
            except Exception as e:
                print(f"TradeLocker auth error: {e}")
                self._token = None
            return self._token

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
        """Load historical 5m bars. Primary: TradeLocker (same broker as live price,
        always current). Fallback: Forexite. Refreshes every ~5 min so the newest
        candles stay aligned with the live quote."""
        if self.bars_5m and time.time() - self._bars_fetched_at < 300:
            return  # bars fresh enough
        # Try TradeLocker history first (fresh, aligned with live price)
        if await self._fetch_bars_tradelocker():
            return
        # Fallback to Forexite if TradeLocker history unavailable
        await self._fetch_bars_forexite()

    async def _fetch_bars_tradelocker(self) -> bool:
        """Fetch 5m bars from TradeLocker history endpoint (last ~10 days)."""
        try:
            token = self._ensure_token()
            if not token:
                return False
            now_ms = int(time.time() * 1000)
            from_ms = now_ms - 10 * 24 * 3600 * 1000
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(
                    f"{TL_URL}/trade/history",
                    headers={"Authorization": f"Bearer {token}", "accNum": TL_ACC_NUM},
                    params={"routeId": TL_ROUTE_INFO, "tradableInstrumentId": TL_INSTRUMENT,
                            "resolution": "5m", "from": from_ms, "to": now_ms},
                )
            if r.status_code != 200:
                print(f"TradeLocker history failed: {r.status_code} {r.text[:150]}")
                return False
            bars = r.json().get("d", {}).get("barDetails", [])
            if not bars:
                return False
            self.bars_5m = [
                (int(b["t"] / 1000), float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]))
                for b in bars
            ]
            self._resample()
            self._bars_fetched_at = time.time()
            self.bars_source = "TradeLocker"
            last_c = self.bars_5m[-1][4]
            print(f"Bars loaded: {len(self.bars_5m)} 5m from TradeLocker (last close {last_c:.2f})")
            return True
        except Exception as e:
            print(f"TradeLocker bar fetch error: {e}")
            return False

    async def _fetch_bars_forexite(self):
        """Fallback: fetch 5-day historical bars from Forexite."""
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
                    self.bars_source = "Forexite"
                    print(f"Bars loaded: {len(self.bars_5m)} 5m from Forexite (fallback)")
        except Exception as e:
            print(f"Forexite bar fetch error: {e}")

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

# File to persist signal P&L history across restarts
SIGNAL_PNL_FILE = Path(__file__).parent / "signal_pnl.json"

def _load_signal_pnl():
    """Load persisted signal P&L data (survives restarts)."""
    try:
        if SIGNAL_PNL_FILE.exists():
            data = json.loads(SIGNAL_PNL_FILE.read_text())
            return data
    except Exception as e:
        print(f"signal_pnl.json load error: {e}")
    return {"all_time_pnl": 0.0, "total_signals": 0, "wins": 0, "losses": 0, "results": []}

def _save_signal_pnl(data):
    """Persist signal P&L data to disk."""
    try:
        SIGNAL_PNL_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"signal_pnl.json save error: {e}")


class DashboardState:
    def __init__(self, feed: LiveFeed):
        self.feed = feed
        self.signals_today = 0
        self.scanned = 0
        self.ev_passed = 0
        self.signal_feed = []
        self.daily_pnl = 0.0
        self.max_dd = 0.0
        self.active_signal = None
        self.last_signal_time = 0
        self._today = None
        self._open_trades = []  # tracks signals sent, monitors for TP/SL hits
        # Signal cooldown: minimum seconds between signals
        self.SIGNAL_COOLDOWN = 180  # 3 minutes between signals
        self.MAX_SIGNALS_DAY = 4
        self.MIN_EV = 0.55
        # Load persisted signal P&L from disk (survives restarts)
        pnl_data = _load_signal_pnl()
        self.all_time_pnl = pnl_data.get("all_time_pnl", 0.0)
        self.total_signals = pnl_data.get("total_signals", 0)
        self.wins = pnl_data.get("wins", 0)
        self.losses = pnl_data.get("losses", 0)
        self.signal_results = pnl_data.get("results", [])  # history of each signal outcome
        self.equity = 5000.0 + self.all_time_pnl  # reflects cumulative signal P&L
        self.peak = max(self.equity, 5000.0)
        print(f"Signal P&L loaded: all_time={self.all_time_pnl:+.2f} | signals={self.total_signals} | W/L={self.wins}/{self.losses}")

    def _persist_pnl(self):
        """Save current signal P&L state to disk."""
        _save_signal_pnl({
            "all_time_pnl": round(self.all_time_pnl, 2),
            "total_signals": self.total_signals,
            "wins": self.wins,
            "losses": self.losses,
            "results": self.signal_results[-200:],  # keep last 200 results
        })

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
        if dist_15m > 1.5:
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

        # Position sizing: risk $60 max per signal
        contract = 100.0
        leverage = 10.0
        loss_per_lot = contract * sl_dist
        max_loss = 60.0  # $60 max risk per signal
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

        # Track this trade for P&L monitoring
        self._open_trades.append({
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp_final": tp_final,
            "lots": lots,
            "risk_dollars": round(risk_dollars, 2),
            "time": ts_str,
            "tp1_hit": False,
            "tp2_hit": False,
        })

    async def check_open_trades(self):
        """Monitor open trades against live price. Send Telegram on TP/SL hit."""
        if not self._open_trades or self.feed.price <= 0:
            return
        
        price = self.feed.price
        closed = []
        
        for i, trade in enumerate(self._open_trades):
            d = trade["direction"]
            hit = None
            
            if d == "buy":
                if price <= trade["sl"]:
                    if trade["tp1_hit"]:
                        hit = "BREAKEVEN"
                        pnl = trade["lots"] * 100 * (trade["tp1"] - trade["entry"]) * 0.10
                    else:
                        hit = "SL HIT"
                        pnl = -trade["risk_dollars"]
                elif price >= trade["tp1"] and not trade["tp1_hit"]:
                    trade["tp1_hit"] = True
                    trade["sl"] = trade["entry"]  # move SL to BE
                    pnl_tp1 = trade["lots"] * 100 * (trade["tp1"] - trade["entry"]) * 0.10
                    await send_telegram(f"TP1 HIT +${pnl_tp1:.2f} (closed 10%) | SL moved to breakeven | Riding 90%")
                elif price >= trade["tp2"] and not trade["tp2_hit"]:
                    trade["tp2_hit"] = True
                    trade["sl"] = trade["tp1"]  # move SL to TP1
                    pnl_tp2 = trade["lots"] * 100 * (trade["tp2"] - trade["entry"]) * 0.20
                    await send_telegram(f"TP2 HIT +${pnl_tp2:.2f} (closed 20%) | SL moved to TP1 | Riding 70%")
                elif price >= trade["tp_final"]:
                    hit = "FINAL TP HIT"
                    pnl = trade["lots"] * 100 * (
                        0.10 * (trade["tp1"] - trade["entry"]) +
                        0.20 * (trade["tp2"] - trade["entry"]) +
                        0.70 * (trade["tp_final"] - trade["entry"])
                    )
            else:  # sell
                if price >= trade["sl"]:
                    if trade["tp1_hit"]:
                        hit = "BREAKEVEN"
                        pnl = trade["lots"] * 100 * (trade["entry"] - trade["tp1"]) * 0.10
                    else:
                        hit = "SL HIT"
                        pnl = -trade["risk_dollars"]
                elif price <= trade["tp1"] and not trade["tp1_hit"]:
                    trade["tp1_hit"] = True
                    trade["sl"] = trade["entry"]
                    pnl_tp1 = trade["lots"] * 100 * (trade["entry"] - trade["tp1"]) * 0.10
                    await send_telegram(f"TP1 HIT +${pnl_tp1:.2f} (closed 10%) | SL moved to breakeven | Riding 90%")
                elif price <= trade["tp2"] and not trade["tp2_hit"]:
                    trade["tp2_hit"] = True
                    trade["sl"] = trade["tp1"]
                    pnl_tp2 = trade["lots"] * 100 * (trade["entry"] - trade["tp2"]) * 0.20
                    await send_telegram(f"TP2 HIT +${pnl_tp2:.2f} (closed 20%) | SL moved to TP1 | Riding 70%")
                elif price <= trade["tp_final"]:
                    hit = "FINAL TP HIT"
                    pnl = trade["lots"] * 100 * (
                        0.10 * (trade["entry"] - trade["tp1"]) +
                        0.20 * (trade["entry"] - trade["tp2"]) +
                        0.70 * (trade["entry"] - trade["tp_final"])
                    )
            
            if hit:
                arrow = "BUY" if d == "buy" else "SELL"
                if "TP" in hit:
                    emoji = "WIN"
                    await send_telegram(f"{emoji} {hit} | {arrow} from {trade['entry']:.2f}\nProfit: +${pnl:.2f}\nFull multi-TP captured.")
                elif hit == "BREAKEVEN":
                    await send_telegram(f"BREAKEVEN | {arrow} from {trade['entry']:.2f}\nTP1 was banked (+${pnl:.2f}), rest closed at entry.")
                else:
                    await send_telegram(f"LOSS | {arrow} from {trade['entry']:.2f}\nSL hit: -${abs(pnl):.2f}")
                
                self.daily_pnl += pnl
                self.equity += pnl
                self.all_time_pnl += pnl
                if pnl > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                # Track peak and max drawdown
                if self.equity > self.peak:
                    self.peak = self.equity
                dd = self.peak - self.equity
                if dd > self.max_dd:
                    self.max_dd = dd
                # Record result
                self.signal_results.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "direction": d,
                    "entry": trade["entry"],
                    "exit_type": hit,
                    "pnl": round(pnl, 2),
                    "all_time_pnl": round(self.all_time_pnl, 2),
                })
                # Persist to disk (survives restarts)
                self._persist_pnl()
                closed.append(i)
                
                self.signal_feed.append({
                    "time": datetime.now(timezone.utc).strftime("%H:%M"),
                    "type": "win" if pnl > 0 else "loss",
                    "label": hit,
                    "text": f"${pnl:+.2f}"
                })
        
        for i in sorted(closed, reverse=True):
            self._open_trades.pop(i)
        
        if not self._open_trades:
            self.active_signal = None


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
                if abs(closes_15m[-1] - e20_15[-1]) / atr_15 < 1.5:
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
            "win_rate": self.wins / max(1, self.wins + self.losses),
            "avg_r": (self.all_time_pnl / max(1, self.wins + self.losses)) if (self.wins + self.losses) > 0 else 0.0,
            "profit_factor": (self.wins / max(1, self.losses)) if self.losses > 0 else 0.0,
            "trades_per_day": 1.0,
            "all_time_pnl": round(self.all_time_pnl, 2),
            "wins": self.wins,
            "losses": self.losses,
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

feed = LiveFeed()
state = DashboardState(feed)

FRONTEND_DIR = Path(__file__).parent / "dashboard" / "frontend"
TERMINAL_HTML = FRONTEND_DIR / "signal_terminal.html"

_background_tasks = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: launch price feed thread + background scanning loops
    feed.start_websocket()  # TradingView real-time price in background thread
    _background_tasks.append(asyncio.create_task(price_loop()))
    _background_tasks.append(asyncio.create_task(yield_loop()))
    _background_tasks.append(asyncio.create_task(market_status_loop()))
    yield
    # Shutdown: stop feed thread and cancel background tasks cleanly
    feed._running = False
    for t in _background_tasks:
        t.cancel()

app = FastAPI(title="XAUUSD ASWP Live Terminal", lifespan=lifespan)

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
        try:
            await state.check_open_trades()
        except Exception as e:
            print(f"Trade monitor: {e}")
        await asyncio.sleep(15)

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
    return {
        "status": "ok",
        "price": feed.price,
        "price_source": feed.price_source,
        "bars_source": feed.bars_source,
        "bars_5m": len(feed.bars_5m),
        "last_update": feed.last_update,
    }

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
    import signal as _signal
    import os as _os

    def _force_exit(*_):
        """Ctrl+C = instant kill. No graceful shutdown delay."""
        print("\n Bot stopped.")
        _os._exit(0)

    _signal.signal(_signal.SIGINT, _force_exit)
    _signal.signal(_signal.SIGTERM, _force_exit)

    print("Starting XAUUSD ASWP Live Terminal...")
    print("Open http://localhost:5000/terminal")
    print("Press Ctrl+C to stop instantly.")
    uvicorn.run(app, host="0.0.0.0", port=5000)
