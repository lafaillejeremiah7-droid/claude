# Full Codebase Audit: TradeLocker Self-Adaptive Trading Bot

A rigorous, file-by-file audit of the entire trading bot and dashboard. This bot manages real capital ($10,000 paper / $5,000 funded), so every bug identified below could translate to dollars lost or incorrect trading behavior.

**Watch for:** Live-mode position close uses stale spot price instead of actual fill (P0, confirmed). Paper equity accumulates only today's PnL and resets to starting equity on day rollover, losing all prior gains (P0, confirmed). The `_build_features` function references `pd` as a global but `pd` is only defined inside `main()` — live mode will crash on the first trade attempt when run via import rather than `__main__` (P0, confirmed). Adaptive engine records features for live trades but never sets `result`/`pnl_r` before calling `record_trade`, feeding garbage into the learning system (P0, confirmed).

---

## High-level view

The position close handler in live mode (`_handle_position_closed` in `trade_manager.py`) estimates exit price from a spot quote taken *after* the position already closed on the exchange — the actual fill could be materially different, especially during volatile BTC moves. Every closed trade's P&L, R-multiple, and win/loss classification fed to the risk manager and adaptive engine is based on this unreliable estimate.

Paper equity is defined as `starting_equity + daily_stats.realized_pnl`, but `daily_stats` resets every UTC midnight. After the first day boundary, accumulated paper gains vanish and equity snaps back to the starting value. All subsequent sizing, drawdown calculations, and dashboard equity displays are wrong.

The adaptive engine's live-mode learning loop records a `TradeFeatures` object on close, but the code path that sets `features.result` and `features.pnl_r` is commented-out / incomplete — the features go into the learning history with empty `result=""` and `pnl_r=0.0`, corrupting all subsequent optimization cycles.

The `pd` (pandas) global is declared inside `main()` via `global pd; import pandas as pd`. Any code path that reaches `_build_features` without going through `main()` (e.g., unit tests importing `TradingBot`, or a future refactor) will raise `NameError: name 'pd' is not defined`.

The API client's `get_account_balance` has a hardcoded fallback (`$10,109.58`) that silently activates when all endpoints fail — the bot continues trading with a stale, possibly fictional equity figure without any circuit-breaker.

---

<details>
<summary>Issues (22)</summary>

1. **Stale exit price on live close** — `TradeManager._handle_position_closed` fetches a spot quote after the position already closed; actual fill price may differ significantly. Use the broker's order history or position close response for the real fill.
2. **Paper equity resets at midnight** — `current_equity` is `starting_equity + daily_stats.realized_pnl`; daily_stats resets every UTC day. Accumulate a separate running total that persists across days.
3. **Adaptive learning receives empty result/pnl_r in live mode** — The live close path in `_manage_positions_and_learn` never assigns `features.result` or `features.pnl_r`. Populate them from the trade manager's journal or the API.
4. **`pd` global only defined inside `main()`** — Any import of `TradingBot` outside `__main__` will crash at `_build_features` where `pd.isna()` is called. Move the import to module level.
5. **Hardcoded fallback equity in API client** — `get_account_balance` returns `$10,109.58` when all endpoints fail. The bot then sizes trades off a fictional number. Add a fail-stop or at minimum log at ERROR level and skip the scan cycle.
6. **`confidence_to_risk_pct` uses module-level `CONFIDENCE_MAX=10.0` not `gate+span`** — When `gate > 10`, `span <= 0` returns `max_pct` directly (3%), which means a misconfigured adaptive gate silently uses max risk. Add a bounds check or clamp the gate.
7. **Position sizing returns `min_lot` on zero SL distance** — `calculate_position_size` returns `(min_lot, risk_amount)` when `sl_distance <= 0`, which could place a real order with unbounded loss per lot. Reject the trade instead.
8. **Weekly stats reset uses *current* equity at first check** — `_update_weekly_stats` sets `starting_equity = current_equity` on rollover, but if the bot was offline over a weekend and equity changed, the weekly drawdown baseline is wrong. Persist the start-of-week equity.
9. **`DailyStats` loaded with unknown fields will crash** — `DailyStats(**daily_data)` will raise `TypeError` if the JSON on disk has extra keys (e.g., from a newer version). Use a filtered constructor or `dataclasses` `__post_init__`.
10. **Paper `open_from_signal` calls `can_trade` twice** — `open_from_signal` checks `can_trade`, then calls `open_position` which checks it again. The second check could trip between the two calls if another paper trade opened concurrently (unlikely but wasteful).
11. **`find_swing_highs/lows` is O(n × lookback)** — With 200-bar DataFrames and lookback=10, this is fine, but the window slicing creates new Series objects each iteration (800 allocations per call). Use vectorized rolling max/min.
12. **`asdict` import missing in `main.py` reset-adaptive path** — `from dataclasses import asdict` is not imported in `main.py`; the `--reset-adaptive` branch will crash with `NameError`.
13. **Breakeven move in paper mode doesn't account for spread** — `PaperTradeManager._check_breakeven` sets SL to `entry_price` exactly. The live `TradeManager` adds half-spread. Paper results are slightly optimistic vs. live.
14. **No upper bound on `_history_cache` size** — If instruments change over time (or the bot is run for weeks), stale cache keys are never evicted. Add a max-size or periodic purge.
15. **`scan_for_entry` modifies DataFrame in place when indicators missing** — `add_all_indicators` returns a copy, but the local `df_5m` in `scan_for_entry` rebinds to the copy while the caller keeps the original. If indicators are already present this is fine; if not, the two diverge silently.
16. **`get_latest_price` doesn't use retry logic** — It calls `self._throttle()` manually and uses `self.session.get` directly (no `_request_with_retry`), so a single 429/5xx kills the price fetch with no retry.
17. **Authentication calls bypass rate-limit/retry** — `authenticate()` and `refresh_access_token()` use `self.session.post` directly, not `_request_with_retry`. A transient 429 on auth will fail the entire scan cycle.
18. **`check_session_status` for BTC doesn't special-case 24/7 trading** — BTC trades 24/7 but the session filter restricts it to London+NY hours, reducing uptime unnecessarily. If intentional, document why.
19. **Journal file writes are not atomic** — Both live and paper journal appends open with `"a"` mode and write one line. A crash mid-write could leave a partial JSON line that `parse_jsonl` silently skips, losing a trade record.
20. **`create_order` doesn't use `_request_with_retry`** — The most critical API call (placing real orders) uses raw `self.session.post` with no retry on transient failures. A 429 at order submission = missed trade.
21. **Dashboard SSE poller runs `store.refresh()` in a thread but `build_snapshot` reads files non-atomically** — A partial write to `daily_stats.json` (by the bot writing mid-cycle) could give the dashboard a truncated JSON, which `_read_json` catches and returns `None`. The dashboard degrades gracefully, but equity/stats flicker to "unavailable" momentarily.
22. **`_blend_params` zero-value parameters are permanently stuck** — `max_shift = abs(current) * MAX_SHIFT_PCT` with `current == 0` yields `max_shift = 0`, so the adaptive engine can never adjust a parameter that starts at zero (e.g., `min_ema20_slope`). Use `max(abs(current), epsilon) * MAX_SHIFT_PCT`.

</details>

<details>
<summary>Details</summary>

## P0 — Critical (will cause money loss, crashes, or data corruption)

### 1. Live close uses spot quote, not actual fill price

**File:** `modules/trade_manager.py`, `_handle_position_closed` (line ~170)

The moment a position disappears from the API's open-positions list, `_handle_position_closed` fetches `get_latest_price` and uses the current bid/ask as the "exit price." But the position was already closed — potentially seconds or minutes ago, at a completely different price. For BTC, a 0.3% move in 60 seconds is routine; on a $200 risk trade that's $60 of phantom P&L error.

The P&L calculated here flows into `risk_manager.record_trade_closed` (which controls the daily drawdown lock and consecutive-loss counter), and into the adaptive engine's learning history. A single badly-estimated loss could trip the 4% daily drawdown lock prematurely, halting trading for the day. Conversely, a phantom win could prevent the consecutive-loss lock from triggering when it should.

**Fix:** Query `get_orders_history()` for the actual fill price/time of the closed position. The API returns order history with fill details. Fall back to the spot estimate only if the history lookup fails.

---

### 2. Paper equity resets to starting equity every midnight

**File:** `modules/paper_trading.py`, `current_equity` property (line ~70)

```python
@property
def current_equity(self) -> float:
    """Paper equity = starting equity + realized paper PnL."""
    return self.starting_equity + self.risk_manager.daily_stats.realized_pnl
```

`daily_stats.realized_pnl` resets to `0.0` when `RiskManager.can_trade` detects a new day (via `_reset_daily_stats`). On the second day of paper trading, equity snaps back to `$10,000` regardless of Day 1's gains/losses. Position sizing, drawdown headroom, and the dashboard equity display are all wrong from that point forward.

**Impact:** After a profitable Day 1 ($+300), Day 2 sizes trades off $10,000 instead of $10,300. After a losing Day 1 ($-200), Day 2 sizes trades off $10,000 instead of $9,800 — taking too much risk.

**Fix:** Maintain a separate `cumulative_realized_pnl` field in the paper stats file that persists across day boundaries. Use `starting_equity + cumulative_realized_pnl` for sizing.

---

### 3. Adaptive engine records empty result/pnl_r for live trades

**File:** `main.py`, `_manage_positions_and_learn` (line ~215)

In the live-mode branch:
```python
for pos_id in closed_ids:
    if pos_id in self.pending_features:
        features = self.pending_features.pop(pos_id)
        # The trade_manager already logged the PnL - we need to update features
        # ... (no actual assignment of features.result or features.pnl_r)
        self.adaptive.record_trade(features)
```

The code has comments acknowledging the need to set `features.result` and `features.pnl_r`, but never actually does it. The features are recorded into `trade_features.jsonl` with `result=""` and `pnl_r=0.0`. The optimization cycle then counts these as neither wins nor losses (empty result is not 'win' or 'loss'), so `winners` and `losers` lists exclude them. After 20 live trades, the optimization cycle fires with potentially zero usable data.

**Fix:** After `_handle_position_closed` returns, extract the PnL and win/loss from the journal entry it just wrote (or from the trade manager's state), and set `features.result` and `features.pnl_r` before calling `record_trade`.

---

### 4. `pd` (pandas) global only exists inside `main()`

**File:** `main.py`, line ~390

```python
def main():
    global pd
    import pandas as pd
```

But `_build_features` (line ~275) uses `pd.isna(slope_val)`. If the bot is instantiated by any code path other than calling `main()` (tests, a wrapper script, a process manager that imports `TradingBot` directly), the first trade attempt crashes with `NameError`.

**Fix:** Move `import pandas as pd` to the module level (alongside the other imports).

---

### 5. Hardcoded fallback equity bypasses all safety checks

**File:** `modules/api_client.py`, `get_account_balance` (line ~350)

```python
fallback_equity = float(os.getenv("ACCOUNT_BALANCE", "10109.58"))
logger.warning(f"Using fallback balance: ${fallback_equity:.2f}")
return {"balance": fallback_equity, "equity": fallback_equity, ...}
```

When all four API endpoint patterns fail (network down, API maintenance, auth expired between `ensure_authenticated` and the balance call), the bot silently uses a static number. It then passes `can_trade` (the equity is non-zero), calculates a position size based on potentially stale equity, and places a real trade.

If the real equity has dropped due to an external manual trade or drawdown, the bot over-sizes. If equity has grown, it under-sizes — less dangerous but still wrong.

**Fix:** Return `None` on complete failure (matching the function's `Optional[dict]` return type). The scan cycle already handles `balance is None` by returning early.

---

### 6. Position sizing on zero SL distance places an order with min_lot

**File:** `modules/risk_management.py`, `calculate_position_size` (line ~120)

```python
if sl_distance <= 0:
    logger.error("Stop loss distance is zero or negative")
    return min_lot, risk_amount
```

This returns `(0.01, risk_amount)` — the caller (`create_trade_setup`) proceeds to build a valid `TradeSetup` with `position_size=0.01` and the full risk_amount. The trade gets placed with no effective stop loss protection (since SL == entry or is on the wrong side). On BTC, 0.01 lot with no real SL distance means the actual risk is undefined.

**Fix:** Return `(0, 0)` and let `create_trade_setup` detect `position_size == 0` as invalid, or raise/return a signal that produces an invalid setup.

---

## P1 — High (incorrect behavior that could mislead or miss trades)

### 7. `asdict` not imported in `main.py`

**File:** `main.py`, `--reset-adaptive` branch (line ~395)

```python
from modules.adaptive_engine import AdaptiveParams, ADAPTIVE_CONFIG_FILE
import json
params = AdaptiveParams()
with open(ADAPTIVE_CONFIG_FILE, 'w') as f:
    json.dump(asdict(params), f, indent=2)
```

`asdict` is never imported. Running `python main.py --reset-adaptive` crashes with `NameError: name 'asdict' is not defined`.

**Fix:** Add `from dataclasses import asdict` at the top of the file or in the branch.

---

### 8. `get_latest_price` has no retry logic

**File:** `modules/api_client.py`, `get_latest_price` (line ~305)

This method uses `self.session.get` directly instead of `_request_with_retry`. A single HTTP 429 (common on the demo server) returns an exception, caught as `None`. In the live trade manager, this means `_check_breakeven` silently skips — the bot misses the breakeven move window. If the 429 persists across multiple cycles, the position never moves to breakeven and takes a full-stop loss that should have been eliminated.

**Fix:** Replace `self.session.get` with `self._request_with_retry("GET", ...)`.

---

### 9. `create_order` has no retry logic

**File:** `modules/api_client.py`, `create_order` (line ~380)

The single most important API call — placing the actual trade — uses `self.session.post` with no retry. A transient 429 or 5xx at the moment of execution means the signal is lost. The bot logs "TRADE EXECUTION FAILED" and moves on; the next signal might not come for hours.

**Fix:** Use `_request_with_retry` for order submission. Be careful with idempotency — a retried market order could double-fill. Consider adding a deduplication check (query open positions after a failed order attempt before retrying).

---

### 10. Weekly drawdown baseline set from first equity seen after rollover

**File:** `modules/risk_management.py`, `_update_weekly_stats` (line ~315)

```python
if self.weekly_stats.week_start != week_start:
    self.weekly_stats = WeeklyStats(
        week_start=week_start,
        starting_equity=current_equity,
    )
```

If the bot is offline from Friday close to Monday morning and the account moved (e.g., a swap charge, or another system traded), the "starting equity" is whatever the API reports on the first Monday scan — not the actual start-of-week value. The weekly drawdown calculation is anchored to the wrong baseline.

**Impact:** Could allow trading past the real 4% weekly limit or lock prematurely.

---

### 11. `DailyStats` / `WeeklyStats` deserialization crashes on extra keys

**File:** `modules/risk_management.py`, `_load_stats` (line ~340)

```python
self.daily_stats = DailyStats(**daily_data)
```

If a future version adds a field to the JSON (or if the file is manually edited with an extra key), this raises `TypeError: __init__() got an unexpected keyword argument`. The bot falls back to empty stats, losing the daily trade count and potentially allowing extra trades past the limit.

**Fix:** Filter `daily_data` to only the fields `DailyStats` accepts, or wrap in try/except with partial recovery.

---

### 12. Authentication bypass in rate-limit scenario

**File:** `modules/api_client.py`, `authenticate` and `refresh_access_token` (lines ~155, ~175)

Both methods use `self.session.post(url, json=payload)` directly — no throttle, no retry. If the server is rate-limiting (429), the auth call fails, `ensure_authenticated` returns False, and the entire scan cycle is skipped. If this persists (e.g., backoff needed), the bot sits idle indefinitely, re-attempting auth every 60s without any backoff.

---

## P2 — Medium (edge cases, degraded behavior)

### 13. Paper breakeven doesn't account for spread

**File:** `modules/paper_trading.py`, `_check_breakeven` (line ~165)

```python
breakeven_price = position.entry_price
position.stop_loss = breakeven_price
```

The live `TradeManager._check_breakeven` adds `spread * 0.5` to the breakeven price to account for the cost of closing at the bid/ask. Paper mode sets SL exactly at entry, making paper results slightly more optimistic than live. Over hundreds of trades, this could overstate win rate by 1-2%.

---

### 14. Unbounded `_history_cache` growth

**File:** `modules/api_client.py`, `_history_cache` dict

Cache entries are added on every successful price-history fetch but never evicted (only expired entries are not *used*). If instruments change or the bot runs for weeks, memory grows unboundedly. Not a crisis (each entry is a DataFrame copy ~50KB), but in a long-running daemon this accumulates.

**Fix:** Purge entries older than 2× their TTL on each access, or cap the cache at N entries with LRU eviction.

---

### 15. Swing high/low detection is O(n × lookback) with many small allocations

**File:** `modules/indicators.py`, `find_swing_highs`/`find_swing_lows` (lines ~110-140)

Each iteration slices `highs.iloc[i - lookback:i]` and `highs.iloc[i + 1:i + lookback + 1]`, creating new Series objects. With 200 bars and lookback=10, that's ~380 intermediate Series per call, 6 calls per instrument per cycle (swing_highs + swing_lows on 5m, plus `get_recent_swing_high/low` in entry signals and risk management). Total: ~2000+ transient allocations per scan cycle.

Not a correctness issue, but adds ~50-100ms of GC pressure per cycle that's avoidable with a rolling-max approach.

---

### 16. Non-atomic journal writes risk partial lines

**File:** `modules/trade_manager.py` and `modules/paper_trading.py`, `_journal_entry`

```python
with open(journal_file, "a") as f:
    f.write(json.dumps(entry) + "\n")
```

A crash (SIGKILL, OOM) between `open` and the completed `write` can leave a truncated JSON line. `parse_jsonl` in the dashboard skips it gracefully, but the trade record is lost — the daily report underreports trades, and the adaptive engine has a gap in its history.

**Fix:** Write to a temp file and `os.rename` (atomic on POSIX), or use `fcntl.flock` for append safety.

---

### 17. `scan_for_entry` can shadow caller's DataFrame

**File:** `modules/entry_signals.py`, `scan_for_entry` (line ~275)

```python
if "ema_20" not in df_5m.columns:
    df_5m = add_all_indicators(df_5m, "5m")
```

`add_all_indicators` does `df = df.copy()` internally and returns the copy. The local `df_5m` rebinds to this copy, but all subsequent checks (`check_pullback_to_value`, etc.) use the *new* local copy. The caller's original DataFrame remains unmodified. If the caller (main.py `_scan_instrument`) later uses `df_5m` expecting indicators to be there, they won't be — but in practice the caller adds indicators *before* calling `scan_for_entry`, so this is only a latent trap, not an active bug.

---

### 18. `_confidence_recent` in `store.py` can include stale log entries

**File:** `dashboard/backend/store.py`, `_confidence_recent`

The function merges journal confidence entries with log-parsed entries but doesn't deduplicate. A trade that appears in both the journal (via `entry_reasons`) and the log (via the APPROVED block) shows up twice in the dashboard's confidence feed.

---

## P3 — Low (style, minor optimizations, cleanup)

### 19. BTC session restriction is conservative

**File:** `modules/session_filter.py`

BTC/USD trades 24/7 on crypto exchanges, but the session filter restricts it to London/NY hours (07:00-21:00 UTC). This is a design choice (concentrating on high-liquidity hours), but it eliminates overnight BTC opportunities. If intentional, adding a comment explaining the reasoning would prevent future "fix" attempts.

---

### 20. `AVOID_HOURS` constant in `main.py` is never used

**File:** `main.py`, line ~80

```python
AVOID_HOURS = [15, 16, 17]  # UTC hours with high loss rate (London close)
```

This module-level constant is shadowed by the adaptive engine's `params.avoid_hours` and the `AVOID_HOURS` env override logic. The constant is dead code.

---

### 21. `_scan_cycle` creates `datetime.now(UTC)` but `check_session_status` creates its own

The scan cycle passes no timestamp to `can_trade_now()`, which internally calls `get_current_utc()` (a new `datetime.now(UTC)` call). Between the cycle's `last_scan_time` and the session check, up to 1 second could elapse (due to API calls). Not a bug, but passing the already-computed time would make the system more deterministic and testable.

---

### 22. `_blend_params` clips to `MAX_SHIFT_PCT` of current value — zero values never change

**File:** `modules/adaptive_engine.py`, `_blend_params` (line ~350)

```python
max_shift = abs(current) * MAX_SHIFT_PCT
blended = np.clip(blended, current - max_shift, current + max_shift)
```

If `current == 0` (e.g., `min_ema20_slope` starts at 0.0), `max_shift = 0` and the parameter can never be adjusted upward. The optimization cycle proposes a new value, but the blend clamps it back to 0. The parameter is permanently stuck.

**Fix:** Use `max(abs(current), epsilon) * MAX_SHIFT_PCT` as the shift bound.

</details>

---

<details>
<summary>File map</summary>

| File | Role |
|------|------|
| `main.py` | Orchestrator: scan loop, instrument pipeline, feature building, trade execution, adaptive learning wiring |
| `config.py` | Environment parsing, all tunables, rate-limit/cache config |
| `modules/api_client.py` | TradeLocker REST client: auth, token refresh, rate limiting, retry, price history, order management |
| `modules/risk_management.py` | Position sizing, SL/TP calculation, confidence scaling, drawdown locks, daily/weekly stats persistence |
| `modules/trade_manager.py` | Live position lifecycle: open, breakeven, close detection, journal writing |
| `modules/paper_trading.py` | Paper position lifecycle: simulated open/close against real prices, paper stats |
| `modules/adaptive_engine.py` | Self-optimization: feature recording, confidence scoring, parameter tuning |
| `modules/session_filter.py` | Session windows, news blocks, weekend detection |
| `modules/trend_analysis.py` | 4H/30M trend detection via EMA alignment |
| `modules/indicators.py` | RSI, ATR, EMA, volume avg, swing highs/lows, candlestick patterns |
| `modules/entry_signals.py` | 5M entry signal detection: pullback, RSI, sweep, structure break, candle, volume |
| `modules/reporting.py` | Daily/weekly/monthly report generation, improvement suggestions |
| `dashboard/backend/app.py` | FastAPI endpoints, SSE stream, snapshot serving |
| `dashboard/backend/store.py` | Snapshot assembly, content hashing, thread-safe store |
| `dashboard/backend/readers.py` | File reading, JSONL/log parsing, mode-aware path resolution |
| `dashboard/backend/guard.py` | Read-only enforcement for dashboard file access |
| `dashboard/backend/security.py` | Credential handling, secret redaction |
| `dashboard/backend/derivations/*.py` | Pure functions: equity, positions, streaks, confidence, feed, countdown, freshness |

</details>
