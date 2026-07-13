"""
Timeframe Optimizer: Tests every valid combination of gate/pullback/entry
timeframes against real TradeLocker XAUUSD data to find the optimal setup.

Combinations tested (gate > pullback > entry):
  Gate:     4h, 1h, 30m, 15m
  Pullback: 1h, 30m, 15m, 5m
  Entry:    15m, 5m, 1m

Valid combos: gate_tf > pullback_tf > entry_tf (in minutes)
"""
import requests
import time
import json
from datetime import datetime

# === DATA FETCH ===
TL_URL = "https://demo.tradelocker.com/backend-api"


def get_token():
    r = requests.post(f"{TL_URL}/auth/jwt/token",
                      json={"email": "lafaillejeremiah7@gmail.com", "password": ",3)m1U", "server": "AQUA"},
                      timeout=15)
    return r.json()["accessToken"]


def fetch_bars(token, resolution, days=10):
    now_ms = int(time.time() * 1000)
    from_ms = now_ms - days * 24 * 3600 * 1000
    h = {"Authorization": f"Bearer {token}", "accNum": "4"}
    r = requests.get(f"{TL_URL}/trade/history", headers=h,
                     params={"routeId": 791554, "tradableInstrumentId": 1714,
                             "resolution": resolution, "from": from_ms, "to": now_ms},
                     timeout=20)
    if r.status_code == 200:
        return r.json().get("d", {}).get("barDetails", [])
    return []


def resample(bars_5m, factor):
    """Resample 5m bars to higher timeframe by merging N bars."""
    merged = []
    for i in range(0, len(bars_5m) - factor + 1, factor):
        chunk = bars_5m[i:i + factor]
        merged.append({
            "t": chunk[0]["t"],
            "o": chunk[0]["o"],
            "h": max(b["h"] for b in chunk),
            "l": min(b["l"] for b in chunk),
            "c": chunk[-1]["c"],
        })
    return merged


# === INDICATORS ===
def compute_ema(values, period):
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def compute_atr(bars, period=14):
    if len(bars) < 2:
        return 0
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["h"], bars[i]["l"], bars[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return sum(trs) / len(trs) if trs else 0
    return sum(trs[-period:]) / period


def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al < 1e-10:
        return 100
    rs = ag / al
    return 100 - 100 / (1 + rs)


# === STRATEGY SIMULATION ===
def simulate_combo(gate_bars, pullback_bars, entry_bars, entry_tf_minutes):
    """Simulate the ASWP strategy with given timeframe bars.
    Returns stats dict."""
    if len(gate_bars) < 50 or len(pullback_bars) < 20 or len(entry_bars) < 30:
        return None

    signals = []
    cooldown_bars = max(1, 180 // (entry_tf_minutes * 60)) if entry_tf_minutes < 3 else 3

    last_signal_idx = -999

    # Walk through entry bars and check conditions at each bar
    for i in range(30, len(entry_bars)):
        if i - last_signal_idx < cooldown_bars:
            continue

        # Find corresponding gate and pullback bars (by timestamp)
        entry_time = entry_bars[i]["t"]

        # Get latest gate bars up to this time
        gate_closes = [b["c"] for b in gate_bars if b["t"] <= entry_time]
        if len(gate_closes) < 50:
            continue

        # Get latest pullback bars up to this time
        pb_bars_valid = [b for b in pullback_bars if b["t"] <= entry_time]
        if len(pb_bars_valid) < 20:
            continue
        pb_closes = [b["c"] for b in pb_bars_valid]

        # Get entry bar closes up to this point
        entry_closes = [b["c"] for b in entry_bars[:i + 1]]

        # 1. GATE: EMA20 vs EMA50 on gate timeframe
        e20_gate = compute_ema(gate_closes, 20)
        e50_gate = compute_ema(gate_closes, 50)
        if e20_gate[-1] > e50_gate[-1]:
            direction = "buy"
        elif e20_gate[-1] < e50_gate[-1]:
            direction = "sell"
        else:
            continue

        # 2. PULLBACK: price within 1.5x ATR of EMA20
        e20_pb = compute_ema(pb_closes, 20)
        atr_pb = compute_atr(pb_bars_valid[-20:])
        if atr_pb <= 0:
            continue
        dist_pb = abs(pb_closes[-1] - e20_pb[-1]) / atr_pb
        if dist_pb > 1.5:
            continue

        # 3. ENTRY: candle touch + close in direction + RSI
        e20_entry = compute_ema(entry_closes, 20)
        rsi_val = compute_rsi(entry_closes)
        if len(entry_bars) < i or i < 1:
            continue
        prev = entry_bars[i - 1]
        pe = e20_entry[-2] if len(e20_entry) >= 2 else e20_entry[-1]

        if direction == "buy":
            trigger = prev["l"] <= pe * 1.002 and prev["c"] > prev["o"] and rsi_val < 65
        else:
            trigger = prev["h"] >= pe * 0.998 and prev["c"] < prev["o"] and rsi_val > 35

        if not trigger:
            continue

        # SIGNAL FIRED - simulate outcome using future bars
        entry_price = entry_bars[i]["c"]
        atr_val = compute_atr(entry_bars[max(0, i - 14):i + 1])
        if atr_val <= 0:
            continue

        sl_dist = atr_val + 0.15
        tp1_dist = atr_val + 0.30
        tp2_dist = 2 * atr_val + 0.30
        tp_final_dist = 3 * atr_val + 0.30

        # Walk forward to find outcome
        outcome = None
        for j in range(i + 1, min(i + 100, len(entry_bars))):
            fb = entry_bars[j]
            if direction == "buy":
                if fb["l"] <= entry_price - sl_dist:
                    outcome = "sl"
                    break
                if fb["h"] >= entry_price + tp_final_dist:
                    outcome = "tp_final"
                    break
                if fb["h"] >= entry_price + tp1_dist and outcome is None:
                    # Check if it later hits final or SL
                    pass
            else:
                if fb["h"] >= entry_price + sl_dist:
                    outcome = "sl"
                    break
                if fb["l"] <= entry_price - tp_final_dist:
                    outcome = "tp_final"
                    break

        # Simplified: check TP1/TP2/Final/SL
        if outcome is None:
            # Check partial TPs
            for j in range(i + 1, min(i + 100, len(entry_bars))):
                fb = entry_bars[j]
                if direction == "buy":
                    if fb["l"] <= entry_price - sl_dist:
                        outcome = "sl"
                        break
                    if fb["h"] >= entry_price + tp2_dist:
                        outcome = "tp2"
                        break
                    if fb["h"] >= entry_price + tp1_dist:
                        outcome = "tp1"
                        break
                else:
                    if fb["h"] >= entry_price + sl_dist:
                        outcome = "sl"
                        break
                    if fb["l"] <= entry_price - tp2_dist:
                        outcome = "tp2"
                        break
                    if fb["l"] <= entry_price - tp1_dist:
                        outcome = "tp1"
                        break

        if outcome is None:
            outcome = "timeout"  # didn't hit anything in 100 bars

        # Calculate P&L (simplified multi-TP)
        if outcome == "tp_final":
            pnl_r = 0.10 * 1.0 + 0.20 * 2.0 + 0.70 * 3.0  # 2.6R
        elif outcome == "tp2":
            pnl_r = 0.10 * 1.0 + 0.20 * 2.0 + 0.70 * 0  # 0.5R (partial)
        elif outcome == "tp1":
            pnl_r = 0.10 * 1.0 + 0.90 * 0  # 0.1R (only TP1 banked)
        elif outcome == "sl":
            pnl_r = -1.0
        else:
            pnl_r = 0.0  # timeout

        signals.append({
            "direction": direction,
            "entry": entry_price,
            "outcome": outcome,
            "pnl_r": pnl_r,
        })
        last_signal_idx = i

    if not signals:
        return None

    wins = sum(1 for s in signals if s["pnl_r"] > 0)
    losses = sum(1 for s in signals if s["pnl_r"] < 0)
    total = len(signals)
    total_r = sum(s["pnl_r"] for s in signals)
    win_rate = wins / total if total > 0 else 0
    avg_r = total_r / total if total > 0 else 0
    tp_final_count = sum(1 for s in signals if s["outcome"] == "tp_final")

    return {
        "signals": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate * 100, 1),
        "total_r": round(total_r, 2),
        "avg_r": round(avg_r, 3),
        "tp_final_pct": round(tp_final_count / total * 100, 1) if total > 0 else 0,
    }


# === MAIN ===
if __name__ == "__main__":
    print("=" * 70)
    print("XAUUSD TIMEFRAME OPTIMIZER")
    print("Testing all valid gate > pullback > entry combinations")
    print("=" * 70)
    print()

    token = get_token()

    # Fetch 5m bars (base data, 10 days)
    print("Fetching 5m bars from TradeLocker (10 days)...")
    bars_5m_raw = fetch_bars(token, "5m", days=10)
    print(f"  Got {len(bars_5m_raw)} 5m bars")

    # Fetch 1m bars (3 days for fine entry)
    time.sleep(1)
    print("Fetching 1m bars from TradeLocker (3 days)...")
    bars_1m_raw = fetch_bars(token, "1m", days=3)
    print(f"  Got {len(bars_1m_raw)} 1m bars")

    # Build all timeframes from 5m by resampling
    tf_bars = {
        "1m": bars_1m_raw,
        "5m": bars_5m_raw,
        "15m": resample(bars_5m_raw, 3),
        "30m": resample(bars_5m_raw, 6),
        "1h": resample(bars_5m_raw, 12),
        "4h": resample(bars_5m_raw, 48),
    }

    for name, bars in tf_bars.items():
        print(f"  {name}: {len(bars)} bars")

    print()

    # All valid combinations: gate > pullback > entry (in minutes)
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
    gate_options = ["4h", "1h", "30m", "15m"]
    pullback_options = ["1h", "30m", "15m", "5m"]
    entry_options = ["15m", "5m", "1m"]

    results = []

    for gate_tf in gate_options:
        for pb_tf in pullback_options:
            for entry_tf in entry_options:
                # Validate: gate > pullback > entry
                if tf_minutes[gate_tf] <= tf_minutes[pb_tf]:
                    continue
                if tf_minutes[pb_tf] <= tf_minutes[entry_tf]:
                    continue

                gate_bars = tf_bars[gate_tf]
                pb_bars = tf_bars[pb_tf]
                entry_bars = tf_bars[entry_tf]

                stats = simulate_combo(gate_bars, pb_bars, entry_bars, tf_minutes[entry_tf])
                if stats is None:
                    continue

                combo_name = f"{gate_tf}/{pb_tf}/{entry_tf}"
                stats["combo"] = combo_name
                stats["gate"] = gate_tf
                stats["pullback"] = pb_tf
                stats["entry"] = entry_tf
                results.append(stats)

    # Sort by total R (best overall P&L)
    results.sort(key=lambda x: x["total_r"], reverse=True)

    print(f"{'COMBO':<16} {'SIG':>4} {'W':>3} {'L':>3} {'WR%':>6} {'TOT R':>7} {'AVG R':>7} {'TP3%':>5}")
    print("-" * 70)
    for r in results:
        print(f"{r['combo']:<16} {r['signals']:>4} {r['wins']:>3} {r['losses']:>3} "
              f"{r['win_rate']:>5.1f}% {r['total_r']:>+7.2f} {r['avg_r']:>+7.3f} {r['tp_final_pct']:>4.1f}%")

    print()
    print("=" * 70)
    if results:
        best = results[0]
        print(f"OPTIMAL COMBINATION: {best['combo']}")
        print(f"  Gate:     {best['gate']} (trend direction)")
        print(f"  Pullback: {best['pullback']} (pullback zone)")
        print(f"  Entry:    {best['entry']} (trigger)")
        print(f"  Signals:  {best['signals']} | Win Rate: {best['win_rate']}% | Total R: {best['total_r']:+.2f} | Avg R: {best['avg_r']:+.3f}")
    print("=" * 70)
