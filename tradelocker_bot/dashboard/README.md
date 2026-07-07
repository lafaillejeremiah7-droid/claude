# TradeLocker Trading Dashboard (read-only)

A **strictly read-only** web dashboard for the TradeLocker trading bot. It reads the
bot's on-disk state files (and, optionally, the live TradeLocker read API), derives
account/PnL/win-rate/streak/confidence/position/equity-curve metrics, and serves them
to a self-contained browser page over HTTP + Server-Sent Events (SSE).

It **never** writes to bot files and **never** sends trading API calls — every file
open goes through `ReadOnlyGuard.open_readonly`, and secrets are stripped from every
response with `redact_secrets`.

## Architecture

```
bot files (logs/, journal/, logs/reports/)          optional TradeLocker GET API
        │  (read-only, via ReadOnlyGuard)                    │ (best-effort)
        ▼                                                    ▼
  readers.py  ──►  store.py (build_snapshot + SnapshotStore)  ──►  app.py (FastAPI)
   (parsers)        (pure derivations from derivations/*)          /  /api/*
                                                                     │  SSE
                                                                     ▼
                                                       frontend/index.html (browser)
```

- `backend/readers.py` — MODE-aware, tolerant file readers (missing/malformed → empty).
- `backend/store.py` — assembles the full `DashboardSnapshot` from readers + the pure
  `backend/derivations/*` functions; equity falls back to `daily_stats` when the API is off.
- `backend/app.py` — FastAPI app + background file poller (≤2s) + SSE.
- `frontend/index.html` — self-contained page (inline CSS/JS, hand-rolled canvas chart).

## Run

Install deps (from the repo root):

```bash
pip install fastapi uvicorn httpx      # runtime
pip install -r tradelocker_bot/dashboard/requirements.txt   # test deps (pytest, hypothesis)
```

Run from the **repository root** (the folder that contains `tradelocker_bot/`):

### Paper mode (pair with `python main.py --dry`)

```bash
DASHBOARD_MODE=paper uvicorn tradelocker_bot.dashboard.backend.app:app --port 8080
```

### Live mode

```bash
DASHBOARD_MODE=live uvicorn tradelocker_bot.dashboard.backend.app:app --port 8080
```

Then open <http://localhost:8080>.

You can also run it directly: `python -m tradelocker_bot.dashboard.backend.app`
(honours the `PORT` env var, default 8080).

## Endpoints

| Method | Path            | Purpose                                             |
| ------ | --------------- | --------------------------------------------------- |
| GET    | `/`             | Serves the dashboard page                           |
| GET    | `/api/snapshot` | Current `DashboardSnapshot` JSON (no secrets)       |
| GET    | `/api/stream`   | SSE stream: snapshot on change + heartbeat (~10s)   |
| GET    | `/api/health`   | `{status, mode, uptime_s}` (no secrets)             |
| GET    | `/api/reports`  | Latest daily / weekly / monthly report payloads     |

Any request whose path/query looks like it is trying to read a credential/token
(e.g. `?field=TL_PASSWORD`) is rejected with HTTP 400 and no secret in the body.

## Environment variables

| Variable             | Default                       | Meaning                                                            |
| -------------------- | ----------------------------- | ------------------------------------------------------------------ |
| `DASHBOARD_MODE`     | `live`                        | `live` reads primary files; `paper` reads the `paper_*` shadow files. |
| `BOT_DIR`            | bundled `tradelocker_bot/`    | Directory that contains the bot's `logs/`, `journal/`, `.env`.     |
| `API_READER_ENABLED` | `false`                       | Enable the optional live TradeLocker GET reader (else file-only).  |
| `INSTRUMENTS`        | `BTCUSD,XAUUSD`               | Monitored instruments (capped at two).                             |
| `PORT`               | `8080`                        | Port when launched via `python -m ...`.                            |

The dashboard reuses the bot's `.env` (via `BOT_DIR`) only to read credentials for the
**optional** live-price reader. With `API_READER_ENABLED` off (the default) the dashboard
runs entirely from files with **no network** — equity falls back to
`daily_stats.current_equity` and positions use best-effort last-known prices parsed from
the bot log.

## Files read (per mode)

| Data              | live                          | paper                                |
| ----------------- | ----------------------------- | ------------------------------------ |
| Daily stats       | `logs/daily_stats.json`       | `logs/paper_daily_stats.json`        |
| Active positions  | `logs/active_positions.json`  | `logs/paper_active_positions.json`   |
| Journal           | `journal/journal_*.jsonl`     | `journal/paper_journal_*.jsonl`      |
| Trade features    | `logs/trade_features.jsonl`   | `logs/paper_trade_features.jsonl`    |
| Adaptive config   | `logs/adaptive_config.json`   | `logs/adaptive_config.json`          |
| Bot log           | `logs/bot_YYYY-MM-DD.log`     | `logs/bot_YYYY-MM-DD.log`            |
| Reports           | `logs/reports/{daily,weekly,monthly}_*.json`, `logs/reports/history.jsonl` | (same) |

Missing files are treated as empty — the dashboard shows "waiting for data / bot offline"
states instead of crashing.

## Tests

```bash
python -m pytest tradelocker_bot/dashboard/tests -q
```

Covers the pure derivations (25 property-based tests), example/edge cases, the MODE-aware
readers, the snapshot builder, the FastAPI endpoints, and a read-only guarantee test that
asserts a full poll cycle leaves every bot file's bytes and mtime unchanged.
