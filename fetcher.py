"""
Smart data fetcher.

Priority order:
  1. Neon DB  — if data is present and up-to-date for the requested range
  2. yfinance — for any missing date range; result is persisted to DB

Graceful fallback: if the DB is unreachable, fetches directly from yfinance
so the app still works without a database connection.
"""
import datetime
import warnings

import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# Lazy DB init — only runs once per Streamlit server process
@st.cache_resource
def _init_db():
    try:
        from db import init_db
        init_db()
        return True
    except Exception as e:
        st.warning(f"⚠️ DB init failed — running in offline mode: {e}")
        return False


def _yfinance_fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df


def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Return a OHLCV DataFrame indexed by Date for the given ticker and range.

    Serves from Neon DB when possible. Fetches from yfinance only for
    data that isn't already stored, then persists it for future calls.
    Falls back silently to yfinance-only if DB is unavailable.
    """
    db_ok = _init_db()

    today = datetime.date.today()
    start_dt = datetime.date.fromisoformat(start)
    end_dt   = min(datetime.date.fromisoformat(end), today)

    if not db_ok:
        return _yfinance_fetch(ticker, start, str(end_dt))

    try:
        from db import get_sync_info, upsert_ohlcv, query_ohlcv

        sync = get_sync_info(ticker)

        if sync:
            last_synced, db_from, db_to = sync
            synced_today = (last_synced.date() == today)
            covers_range = (db_from <= start_dt) and (db_to >= end_dt)

            if synced_today and covers_range:
                # DB is fresh and covers the full requested range — no API call
                return query_ohlcv(ticker, start, str(end_dt))

            # DB has older data; only fetch the tail we're missing
            if db_to is not None and db_from <= start_dt:
                fetch_start = str(db_to + datetime.timedelta(days=1))
            else:
                fetch_start = start
        else:
            fetch_start = start

        # Fetch only the missing window from yfinance
        raw = _yfinance_fetch(ticker, fetch_start, str(today + datetime.timedelta(days=1)))
        if not raw.empty:
            upsert_ohlcv(ticker, raw)

        return query_ohlcv(ticker, start, str(end_dt))

    except Exception as e:
        # DB unreachable mid-session — fall back to live yfinance
        st.warning(f"⚠️ DB unavailable, using live data: {e}")
        return _yfinance_fetch(ticker, start, str(end_dt))
