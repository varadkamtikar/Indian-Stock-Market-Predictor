"""
Daily DB refresh script.
Fetches latest OHLCV from yfinance for every stock in both exchanges
and upserts into Neon DB. Runs via GitHub Actions after market close.

Usage:
    DATABASE_URL=<neon-url> python scripts/refresh_db.py
"""
import os
import sys
import time
import datetime

import pandas as pd
import psycopg2
import psycopg2.extras
import yfinance as yf

# Allow importing stocks.py from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stocks import NSE_STOCKS, BSE_STOCKS

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    sys.exit("ERROR: DATABASE_URL environment variable is not set.")

FETCH_FROM_DEFAULT = "2015-01-01"   # initial history window
RATE_LIMIT_SLEEP   = 0.4            # seconds between yfinance requests


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(DATABASE_URL)


def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                ticker  TEXT    NOT NULL,
                date    DATE    NOT NULL,
                open    FLOAT8,
                high    FLOAT8,
                low     FLOAT8,
                close   FLOAT8,
                volume  BIGINT,
                PRIMARY KEY (ticker, date)
            );
            CREATE TABLE IF NOT EXISTS sync_log (
                ticker       TEXT        PRIMARY KEY,
                last_synced  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                date_from    DATE,
                date_to      DATE
            );
        """)
    conn.commit()


def get_date_to(conn, ticker: str):
    with conn.cursor() as cur:
        cur.execute("SELECT date_to FROM sync_log WHERE ticker = %s", (ticker,))
        row = cur.fetchone()
    return row[0] if row else None


def upsert(conn, ticker: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    rows = []
    for row in df.itertuples():
        rows.append((
            ticker,
            row.Index.date() if hasattr(row.Index, "date") else row.Index,
            float(row.Open)   if pd.notna(row.Open)   else None,
            float(row.High)   if pd.notna(row.High)   else None,
            float(row.Low)    if pd.notna(row.Low)     else None,
            float(row.Close)  if pd.notna(row.Close)   else None,
            int(row.Volume)   if pd.notna(row.Volume)  else None,
        ))

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO ohlcv (ticker, date, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (ticker, date) DO UPDATE SET
                open   = EXCLUDED.open,
                high   = EXCLUDED.high,
                low    = EXCLUDED.low,
                close  = EXCLUDED.close,
                volume = EXCLUDED.volume
            """,
            rows,
        )
        dates = [r[1] for r in rows]
        cur.execute(
            """
            INSERT INTO sync_log (ticker, last_synced, date_from, date_to)
            VALUES (%s, NOW(), %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                last_synced = NOW(),
                date_from   = LEAST(sync_log.date_from,   EXCLUDED.date_from),
                date_to     = GREATEST(sync_log.date_to, EXCLUDED.date_to)
            """,
            (ticker, min(dates), max(dates)),
        )
    conn.commit()
    return len(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def refresh_all():
    # All unique tickers across both exchanges
    all_tickers = sorted(set(NSE_STOCKS.values()) | set(BSE_STOCKS.values()))
    today       = datetime.date.today()
    yesterday   = today - datetime.timedelta(days=1)

    print(f"[{today}] Refreshing {len(all_tickers)} tickers\n")

    conn = get_conn()
    ensure_tables(conn)

    ok, skipped, failed = 0, 0, []

    for i, ticker in enumerate(all_tickers, 1):
        prefix = f"[{i:>3}/{len(all_tickers)}] {ticker:<22}"
        try:
            date_to = get_date_to(conn, ticker)

            if date_to and date_to >= yesterday:
                print(f"{prefix} up-to-date (last: {date_to})")
                skipped += 1
                continue

            fetch_start = (
                str(date_to + datetime.timedelta(days=1))
                if date_to else FETCH_FROM_DEFAULT
            )

            raw = yf.download(
                ticker,
                start=fetch_start,
                end=str(today + datetime.timedelta(days=1)),
                progress=False,
                auto_adjust=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            n = upsert(conn, ticker, raw)
            print(f"{prefix} +{n:>5} rows  (from {fetch_start})")
            ok += 1

        except Exception as exc:
            print(f"{prefix} FAILED — {exc}")
            failed.append(ticker)

        time.sleep(RATE_LIMIT_SLEEP)

    conn.close()

    print(f"\n{'─'*50}")
    print(f"Done: {ok} updated · {skipped} skipped · {len(failed)} failed")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    refresh_all()
