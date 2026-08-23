# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Early-stage, pre-alpha. There is no `requirements.txt`/`pyproject.toml`, no test suite, no lint config,
and no README. `.venv` (Python 3.14) currently has only `python-dotenv` installed — `alpaca-py` and
`pandas`, which several modules import, are **not installed**, so those modules will fail to run as-is.
Do not assume build/lint/test commands exist; if you add tooling, add the manifest/config for it too.

## What this is

A market-data ingestion pipeline that pulls raw trades from Alpaca, normalizes them into a canonical
`Trade` record, and aggregates them into information-driven bars (tick bars, volume bars, dollar bars —
the sampling schemes from López de Prado's *Advances in Financial Machine Learning*) instead of
time-based OHLC bars.

## Architecture

- `src/trade.py` — `Trade`, the canonical, frozen dataclass every data source must convert into.
  `dollar_value` (price × volume) is derived, not stored.
- `src/ingestion/bar/bar.py` — `Bar`, the immutable output of a completed bar (OHLC + volume +
  dollar_volume + n_ticks + start/end timestamps).
- `src/ingestion/bar/base.py` — `BarBuilder`, an abstract, **stateful/incremental** accumulator: feed
  `Trade`s one at a time via `update()`, which returns a completed `Bar` (and resets internal state)
  once `_is_complete()` says the threshold is crossed, else `None`. Designed to work identically for
  historical replay and live streaming — don't build a batch/vectorized alternative path for this.
  Concrete builders only need to implement `_is_complete()`:
  - `src/ingestion/bar/volume_builder.py` — `VolumeBarBuilder` (closes on cumulative volume).
  - `src/ingestion/bar/tick_builder.py` — `TickBarBuilder` (closes on tick count).
  - `src/ingestion/bar/dollar_builder.py` — currently **empty**; the dollar-bar builder is unimplemented
    despite the module existing.
  - `src/ingestion/bar/__init__.py` is empty (no re-exports), so import concrete classes via their full
    module path, e.g. `from src.ingestion.bar.volume_builder import VolumeBarBuilder`.
- `src/ingestion.py` — the canonical `AlpacaTradeSource`: wraps `StockHistoricalDataClient`, exposes
  `iter_trades()` as a generator that pages through Alpaca's API lazily and yields `Trade` objects.
  Reads credentials from `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`.
- `src/ingestion/date_ranges.py` — `DateRange` value object with constructors for common windows
  (`single_day`, `last_n_days`, `year_to_date`, `month`). `last_n_days` is calendar days, not trading
  days — Alpaca just returns nothing for closed days.

## Known inconsistencies to be aware of

- **Two unrelated `AlpacaTradeSource` classes exist.** `src/ingestion.py` is the current, clean one
  (generator-based, paginated, importable safely). `src/sources/alpacha_ingestion.py` (note the
  "alpacha" typo) is an older/exploratory script version that reads different env var names
  (`APCA_API_KEY`/`APCA_SECRET_KEY`) and **executes a live API call and prints results at module import
  time**, plus has a bug (`trades.df.shape()` — `shape` is an attribute, not a method). Don't import
  this module for its side effects; treat it as scratch code pending cleanup/deletion.
- `src/ingestion/tick_builder.py` is a separate, incomplete stub (`TickBarBuilder` with only a
  constructor calling `self.reset()`, which isn't defined) that duplicates/conflicts with the real,
  working `src/ingestion/bar/tick_builder.py`. Prefer the `bar/` package version.
- `src/ingestion/__init__.py` and `src/sources/__init__.py` are empty.

## Credentials

Alpaca credentials are read from environment variables via `python-dotenv` (`load_dotenv()` + `.env`).
The canonical path (`src/ingestion.py`) expects `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`. `.env` is
gitignored going forward, but git history on `main` contains prior commits titled "creds" and a later
"Delete .env" — treat any credentials from before that point as compromised/rotated, not as secret.
