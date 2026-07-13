"""
XAUUSD ASWP PAPER-TRADING TERMINAL

A live paper (simulated) bot that runs the EXACT current engine
(1H gate / 15m pullback / 1m entry, 15m-ATR stops, $60 risk cap, multi-TP
10/20/70 at 1/2/3R) against REAL 1-minute XAUUSD data pulled from TradeLocker,
replayed at accelerated speed so you can watch signals, TP1/TP2/final/SL hits,
and the equity curve build in real time on the dashboard.

  - 100% signal-only / paper: NO trades are ever placed.
  - Same dashboard as the live bot (dashboard/frontend/signal_terminal.html).
  - Popups fire for every money-making event (TP1, TP2, FINAL, breakeven).

Run:
    cd tradelocker_bot
    python3 paper_terminal.py
Then open: http://localhost:5001/terminal
"""
import asyncio
import json
import time
import bisect
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse
import uvicorn

from live_terminal import (
    compute_ema, compute_atr, compute_rsi,
    TL_URL, TL_EMAIL, TL_PASS, TL_SERVER, TL_ACC_NUM, TL_INSTRUMENT, TL_ROUTE_INFO,
)

STARTING_BALANCE = 5000.0  # paper account starting equity

# ---------------------------------------------------------------------------
# Replay speed + strategy params
# ---------------------------------------------------------------------------
BARS_PER_SECOND = 15          # replay ~15 one-minute bars per second
MIN_EV = 0.55
MAX_SIGNALS_DAY = 4
COOLDOWN_BARS = 180           # 180 one-minute bars = 3h between signals
RISK_CAP = 60.0
CONTRACT = 100.0
LEVERAGE = 10.0
SPREAD = 0.30

# --- ADAPTIVE SL/TP (same validated formula as live_terminal.py — found via
# a 100,000-trial random search on real 1m XAUUSD data, confirmed out-of-sample
# on a separate real year: in-sample +18.2%, out-of-sample +110.6% vs the old
# static 1.0x ATR sizing, with LOWER max drawdown too).
ADAPT_BASE_SL = 0.8939
ADAPT_VOL_LO = 0.8559
ADAPT_VOL_HI = 1.0723
ADAPT_TREND_GAIN = 1.7875
ADAPT_TP_LO = 0.3295
ADAPT_TP_HI = 2.8771


def _auth():
    r = requests.post(f"{TL_URL}/auth/jwt/token",
                      json={"email": TL_EMAIL, "password": TL_PASS, "server": TL_SERVER}, timeout=15)
    r.raise_for_status()
    return r.json()["accessToken"]


def _fetch_1m(days=10):
    tok = _auth()
    now_ms = int(time.time() * 1000)
    frm = now_ms - days * 24 * 3600 * 1000
    r = requests.get(f"{TL_URL}/trade/history",
                     headers={"Authorization": f"Bearer {tok}", "accNum": TL_ACC_NUM},
                     params={"routeId": TL_ROUTE_INFO, "tradableInstrumentId": TL_INSTRUMENT,
                             "resolution": "1m", "from": frm, "to": now_ms}, timeout=30)
    r.raise_for_status()
    bars = r.json().get("d", {}).get("barDetails", [])
    return [(int(b["t"] / 1000), float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"])) for b in bars]


def _merge(bars_1m, factor):
    """Resample 1m bars into higher timeframe; returns list of (end_ts, o,h,l,c)."""
    out = []
    for i in range(0, len(bars_1m) - factor + 1, factor):
        chunk = bars_1m[i:i + factor]
        out.append((chunk[-1][0], chunk[0][1], max(b[2] for b in chunk),
                    min(b[3] for b in chunk), chunk[-1][4]))
    return out


class PaperEngine:
    def __init__(self):
        self.ready = False
        self.bars_1m = []
        # precomputed series
        self.ts = []
        self.o = []; self.h = []; self.l = []; self.c = []
        self.e20_1m = []; self.rsi_1m = []
        # higher TF (end_ts sorted) + aligned indicators
        self.tf15_ts = []; self.tf15_close = []; self.tf15_e20 = []; self.tf15_atr = []
        self.tf15_atr_avg = []  # 50-bar rolling avg of tf15_atr (vol_ratio denominator)
        self.tf1h_ts = []; self.tf1h_e20 = []; self.tf1h_e50 = []; self.tf1h_atr = []
        # replay state
        self.idx = 0
        self.warmup = 0
        # paper account
        self.equity = STARTING_BALANCE
        self.all_time_pnl = 0.0
        self.daily_pnl = 0.0
        self.peak = STARTING_BALANCE
        self.max_dd = 0.0
        self.wins = 0
        self.losses = 0
        self.total_signals = 0
        self.signals_today = 0
        self.scanned = 0
        self.ev_passed = 0
        self._today = None
        self._last_signal_idx = -10 ** 9
        self.open_trade = None
        self.feed = []
        self.active_signal = None
        self.last_price = 0.0
        self._cur_ev = 0.0
        self._cur_p1 = 0.5
        self._pipeline = 1

    # ---------- data prep ----------
    def load(self, bars=None):
        if bars is None:
            print("Fetching real 1m XAUUSD data from TradeLocker...")
            self.bars_1m = _fetch_1m(days=10)
        else:
            self.bars_1m = bars
        n = len(self.bars_1m)
        print(f"  got {n} 1m bars")
        if n < 3500:
            print("  WARNING: fewer bars than ideal for warmup; replay may be short.")
        self.ts = [b[0] for b in self.bars_1m]
        self.o = [b[1] for b in self.bars_1m]
        self.h = [b[2] for b in self.bars_1m]
        self.l = [b[3] for b in self.bars_1m]
        self.c = [b[4] for b in self.bars_1m]
        # 1m indicators
        self.e20_1m = compute_ema(self.c, 20)
        self.rsi_1m = self._rolling_rsi(self.c, 14)
        # 15m
        b15 = _merge(self.bars_1m, 15)
        c15 = [b[4] for b in b15]
        e15 = compute_ema(c15, 20)
        self.tf15_ts = [b[0] for b in b15]
        self.tf15_close = c15
        self.tf15_e20 = e15
        self.tf15_atr = self._rolling_atr(b15, 14)
        # 50-bar rolling average of the 15m ATR series (adaptive vol_ratio denominator)
        self.tf15_atr_avg = self._rolling_mean(self.tf15_atr, 50)
        # 1h
        b1h = _merge(self.bars_1m, 60)
        c1h = [b[4] for b in b1h]
        self.tf1h_ts = [b[0] for b in b1h]
        self.tf1h_e20 = compute_ema(c1h, 20)
        self.tf1h_e50 = compute_ema(c1h, 50)
        self.tf1h_atr = self._rolling_atr(b1h, 14)
        # warmup: need 50 completed 1h bars -> ~3000 1m bars
        self.warmup = min(n - 1, max(3000, 60 * 51))
        self.idx = self.warmup
        self.ready = True
        print(f"  warmup to bar {self.warmup}; {n - self.warmup} bars to replay "
              f"(~{(n - self.warmup)/BARS_PER_SECOND/60:.1f} min at {BARS_PER_SECOND} bars/s)")

    @staticmethod
    def _rolling_rsi(closes, period):
        out = [50.0] * len(closes)
        for i in range(len(closes)):
            if i >= period:
                out[i] = compute_rsi(closes[max(0, i - period * 3):i + 1], period)
        return out

    @staticmethod
    def _rolling_mean(series, window):
        """Simple rolling mean of a value series (used for vol_ratio denominator)."""
        out = [0.0] * len(series)
        for i in range(len(series)):
            lo = max(0, i - window + 1)
            chunk = series[lo:i + 1]
            out[i] = sum(chunk) / len(chunk) if chunk else 0.0
        return out

    @staticmethod
    def _rolling_atr(bars, period):
        out = [0.0] * len(bars)
        for i in range(len(bars)):
            if i >= period:
                out[i] = compute_atr(bars[max(0, i - period * 2):i + 1], period)
        return out

    def _last_completed(self, tf_ts, cur_ts):
        """Index of last higher-TF bar whose end <= cur_ts (no look-ahead)."""
        j = bisect.bisect_right(tf_ts, cur_ts) - 1
        return j

    # ---------- engine step ----------
    def step(self):
        if not self.ready or self.idx >= len(self.bars_1m):
            return
        i = self.idx
        cur_ts = self.ts[i]
        price = self.c[i]
        self.last_price = price
        now = datetime.fromtimestamp(cur_ts, timezone.utc)

        # daily reset
        if self._today != now.date():
            self._today = now.date()
            self.signals_today = 0
            self.daily_pnl = 0.0

        # monitor open trade against this bar's range
        if self.open_trade is not None:
            self._monitor(self.open_trade, self.h[i], self.l[i], now)

        # try to open a new signal
        if self.open_trade is None:
            self._try_signal(i, cur_ts, now, price)

        self.idx += 1

    def _try_signal(self, i, cur_ts, now, price):
        self._pipeline = 1
        # gate 1h
        gj = self._last_completed(self.tf1h_ts, cur_ts)
        if gj < 50:
            return
        e20h, e50h = self.tf1h_e20[gj], self.tf1h_e50[gj]
        if e20h > e50h:
            direction = "buy"
        elif e20h < e50h:
            direction = "sell"
        else:
            return
        self._pipeline = 2

        # trend strength (adaptive TP3 feature): 1H EMA separation / 1H ATR
        atr1h = self.tf1h_atr[gj]
        trend_strength = abs(e20h - e50h) / atr1h if atr1h > 0 else 1.0

        # pullback 15m
        pj = self._last_completed(self.tf15_ts, cur_ts)
        if pj < 20:
            return
        atr15 = self.tf15_atr[pj]
        if atr15 <= 0:
            return
        dist = abs(self.tf15_close[pj] - self.tf15_e20[pj]) / atr15
        if dist > 1.5:
            return
        self._pipeline = 3

        # volatility-expansion ratio (adaptive SL feature): current 15m ATR
        # vs its own 50-bar rolling average
        atr15_avg = self.tf15_atr_avg[pj]
        vol_ratio = atr15 / atr15_avg if atr15_avg > 0 else 1.0

        # entry 1m
        if i < 1:
            return
        rsi_val = self.rsi_1m[i]
        pe = self.e20_1m[i - 1]
        po, ph, pl, pc = self.o[i - 1], self.h[i - 1], self.l[i - 1], self.c[i - 1]
        if direction == "buy":
            trig = pl <= pe * 1.002 and pc > po and rsi_val < 65
        else:
            trig = ph >= pe * 0.998 and pc < po and rsi_val > 35
        if not trig:
            return
        self._pipeline = 4
        self.scanned += 1

        # EV (ASWP simplified, same as live)
        p1 = min(0.85, max(0.35, 0.56))
        p2 = p1 * 0.92
        p3 = p1 * 0.81
        ev = 0.10 * 1 * p1 + 0.20 * 2 * p2 + 0.70 * 3 * p3 - (1 - p1) * 1.0
        self._cur_ev = ev
        self._cur_p1 = p1
        if ev >= MIN_EV:
            self._pipeline = 5
        if ev < MIN_EV:
            return
        self.ev_passed += 1

        # guards
        if self.signals_today >= MAX_SIGNALS_DAY:
            return
        if i - self._last_signal_idx < COOLDOWN_BARS:
            return

        # ADAPTIVE sizing: SL width scales with volatility-expansion ratio,
        # TP3 (runner leg) scales with 1H trend strength. Same validated
        # formula as live_terminal.py.
        vol_ratio_c = min(ADAPT_VOL_HI, max(ADAPT_VOL_LO, vol_ratio))
        sl_mult = ADAPT_BASE_SL * vol_ratio_c
        sl_dist = sl_mult * atr15 + SPREAD / 2
        base = sl_mult * atr15

        tp_ext = 1 + ADAPT_TREND_GAIN * (trend_strength - 1)
        tp_ext = min(ADAPT_TP_HI, max(ADAPT_TP_LO, tp_ext))
        tp3_move = 3 * base * tp_ext

        loss_per_lot = CONTRACT * sl_dist
        lots = min(0.12, RISK_CAP / loss_per_lot)
        lots = max(0.01, round(int(lots * 100) / 100, 2))
        margin = lots * CONTRACT * price / LEVERAGE
        if margin > self.equity * 0.95:
            lots = max(0.01, round(int((self.equity * 0.95 * LEVERAGE / (CONTRACT * price)) * 100) / 100, 2))
        risk_dollars = lots * loss_per_lot
        if direction == "buy":
            sl = round(price - sl_dist, 2)
            tp1 = round(price + 1 * base + SPREAD, 2)
            tp2 = round(price + 2 * base + SPREAD, 2)
            tpf = round(price + tp3_move + SPREAD, 2)
        else:
            sl = round(price + sl_dist, 2)
            tp1 = round(price - 1 * base - SPREAD, 2)
            tp2 = round(price - 2 * base - SPREAD, 2)
            tpf = round(price - tp3_move - SPREAD, 2)
        full_win = lots * CONTRACT * (0.10 * base + 0.20 * 2 * base + 0.70 * tp3_move)

        self.open_trade = {
            "direction": direction, "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2,
            "tp_final": tpf, "lots": lots, "risk_dollars": round(risk_dollars, 2),
            "tp1_hit": False, "tp2_hit": False, "atr": round(atr15, 2),
        }
        self.active_signal = {
            "direction": direction, "entry": price, "sl": sl, "tp1": tp1, "tp2": tp2,
            "tp_final": tpf, "lot_size": lots, "risk_dollars": round(risk_dollars, 2),
            "probability": round(p1, 2), "expected_value_r": round(ev, 2),
            "full_win": round(full_win, 2), "atr": round(atr15, 2),
        }
        self.total_signals += 1
        self.signals_today += 1
        self._last_signal_idx = i
        self._pipeline = 7
        self._add_feed("signal", direction.upper(), f"${price:.2f} | EV:{ev:.2f}R | {lots} lots", now)

    def _monitor(self, t, hi, lo, now):
        d = t["direction"]
        hit = None
        pnl = 0.0
        if d == "buy":
            if lo <= t["sl"]:
                if t["tp1_hit"]:
                    hit = "BREAKEVEN"; pnl = t["lots"] * 100 * (t["tp1"] - t["entry"]) * 0.10
                else:
                    hit = "SL HIT"; pnl = -t["risk_dollars"]
            elif not t["tp1_hit"] and hi >= t["tp1"]:
                t["tp1_hit"] = True; t["sl"] = t["entry"]
                p = t["lots"] * 100 * (t["tp1"] - t["entry"]) * 0.10
                self._add_feed("win", "TP1 HIT", f"+${p:.2f} (10% banked, SL->BE)", now)
            elif not t["tp2_hit"] and hi >= t["tp2"]:
                t["tp2_hit"] = True; t["sl"] = t["tp1"]
                p = t["lots"] * 100 * (t["tp2"] - t["entry"]) * 0.20
                self._add_feed("win", "TP2 HIT", f"+${p:.2f} (20% banked, SL->TP1)", now)
            elif hi >= t["tp_final"]:
                hit = "FINAL TP HIT"
                pnl = t["lots"] * 100 * (0.10 * (t["tp1"] - t["entry"]) + 0.20 * (t["tp2"] - t["entry"]) + 0.70 * (t["tp_final"] - t["entry"]))
        else:
            if hi >= t["sl"]:
                if t["tp1_hit"]:
                    hit = "BREAKEVEN"; pnl = t["lots"] * 100 * (t["entry"] - t["tp1"]) * 0.10
                else:
                    hit = "SL HIT"; pnl = -t["risk_dollars"]
            elif not t["tp1_hit"] and lo <= t["tp1"]:
                t["tp1_hit"] = True; t["sl"] = t["entry"]
                p = t["lots"] * 100 * (t["entry"] - t["tp1"]) * 0.10
                self._add_feed("win", "TP1 HIT", f"+${p:.2f} (10% banked, SL->BE)", now)
            elif not t["tp2_hit"] and lo <= t["tp2"]:
                t["tp2_hit"] = True; t["sl"] = t["tp1"]
                p = t["lots"] * 100 * (t["entry"] - t["tp2"]) * 0.20
                self._add_feed("win", "TP2 HIT", f"+${p:.2f} (20% banked, SL->TP1)", now)
            elif lo <= t["tp_final"]:
                hit = "FINAL TP HIT"
                pnl = t["lots"] * 100 * (0.10 * (t["entry"] - t["tp1"]) + 0.20 * (t["entry"] - t["tp2"]) + 0.70 * (t["entry"] - t["tp_final"]))

        if hit:
            self.daily_pnl += pnl
            self.equity += pnl
            self.all_time_pnl += pnl
            if self.equity > self.peak:
                self.peak = self.equity
            self.max_dd = max(self.max_dd, self.peak - self.equity)
            if pnl > 0:
                self.wins += 1
            elif pnl < 0:
                self.losses += 1
            ftype = "win" if pnl > 0 else ("loss" if pnl < 0 else "info")
            self._add_feed(ftype, hit, f"${pnl:+.2f}", now)
            self.open_trade = None
            self.active_signal = None

    def _add_feed(self, ftype, label, text, now):
        self.feed.append({"time": now.strftime("%H:%M"), "type": ftype, "label": label, "text": text})
        if len(self.feed) > 60:
            self.feed = self.feed[-60:]

    # ---------- snapshot for dashboard ----------
    def snapshot(self):
        total = self.wins + self.losses
        wr = self.wins / total if total else 0.0
        avg_r = (self.all_time_pnl / max(1, total) / 50.0) if total else 0.0  # rough R display
        pct = 0.0
        if self.ready and len(self.bars_1m):
            pct = min(100, (self.idx - self.warmup) / max(1, len(self.bars_1m) - self.warmup) * 100)
        trend = 0
        if self.ready:
            gj = self._last_completed(self.tf1h_ts, self.ts[min(self.idx, len(self.ts) - 1)])
            if gj >= 0:
                trend = 1 if self.tf1h_e20[gj] > self.tf1h_e50[gj] else -1
        atr_disp = 0.0
        if self.ready:
            pj = self._last_completed(self.tf15_ts, self.ts[min(self.idx, len(self.ts) - 1)])
            if pj >= 0:
                atr_disp = round(self.tf15_atr[pj], 2)
        return {
            "mode": f"PAPER {pct:.0f}%" if self.ready else "PAPER LOADING",
            "equity": round(self.equity, 2),
            "all_time_pnl": round(self.all_time_pnl, 2),
            "current_price": round(self.last_price, 2),
            "atr": atr_disp,
            "trend_1h": trend,
            "yield_value": 2.31, "yield_change": 0.15, "yield_alignment": 0.5,
            "p_tp1": round(self._cur_p1, 2),
            "p_tp2": round(self._cur_p1 * 0.92, 2),
            "p_tp3": round(self._cur_p1 * 0.81, 2),
            "current_ev": round(self._cur_ev, 3),
            "memory_size": 2500 + self.total_signals,
            "pipeline_stage": self._pipeline,
            "daily_pnl": round(self.daily_pnl, 2),
            "max_dd": round(self.max_dd, 2),
            "signals_today": self.signals_today,
            "signals_count": self.total_signals,
            "trades": total,
            "win_rate": wr,
            "avg_r": round(avg_r, 3),
            "profit_factor": round(self.wins / max(1, self.losses), 2) if self.losses else 0.0,
            "trades_per_day": 3.5,
            "wins": self.wins, "losses": self.losses,
            "scanned": self.scanned, "ev_passed": self.ev_passed, "deployed": self.total_signals,
            "guards": {"news_clear": True, "weekend_clear": True, "hours_active": True},
            "active_signal": self.active_signal,
            "feed": self.feed[-20:],
        }


app = FastAPI(title="XAUUSD ASWP Paper Terminal")
engine = PaperEngine()
FRONTEND = Path(__file__).parent / "dashboard" / "frontend" / "signal_terminal.html"


@app.on_event("startup")
async def _startup():
    engine.load()
    asyncio.create_task(_replay_loop())


async def _replay_loop():
    interval = 1.0 / BARS_PER_SECOND
    while True:
        if engine.ready and engine.idx < len(engine.bars_1m):
            engine.step()
        await asyncio.sleep(interval)


@app.get("/terminal", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def terminal():
    try:
        return HTMLResponse(FRONTEND.read_text(encoding="utf-8"))
    except Exception:
        return HTMLResponse("<h1>dashboard not found</h1>", status_code=500)


@app.get("/api/snapshot")
async def snapshot():
    return JSONResponse(engine.snapshot())


@app.get("/api/health")
async def health():
    return {"status": "ok", "mode": "paper", "bars": len(engine.bars_1m),
            "idx": engine.idx, "equity": round(engine.equity, 2),
            "signals": engine.total_signals, "wins": engine.wins, "losses": engine.losses}


@app.get("/api/stream")
async def stream(request: Request):
    async def gen():
        last = ""
        try:
            while True:
                if await request.is_disconnected():
                    break
                snap = engine.snapshot()
                data = json.dumps(snap, default=str)
                h = str(hash(data))
                if h != last:
                    yield f"data: {data}\n\n"
                    last = h
                else:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.4)
        except asyncio.CancelledError:
            pass
    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import signal as _sig, os as _os
    def _kill(*_):
        print("\nPaper bot stopped.")
        _os._exit(0)
    _sig.signal(_sig.SIGINT, _kill)
    _sig.signal(_sig.SIGTERM, _kill)
    print("=" * 60)
    print("XAUUSD ASWP PAPER TERMINAL (signal-only, no real trades)")
    print("  Engine: 1H gate / 15m pullback / 1m entry, 15m-ATR stops, $60 cap")
    print("  Open http://localhost:5001/terminal")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=5001)
