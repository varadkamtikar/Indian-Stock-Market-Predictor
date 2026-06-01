"""
Neon DB (PostgreSQL) operations.
Tables are created automatically on first run.
"""
import os
import datetime
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
import pandas as pd
import streamlit as st


# ── Connection ────────────────────────────────────────────────────────────────

def _dsn() -> str:
    try:
        return st.secrets["DATABASE_URL"]
    except Exception:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL not found. "
                "Add it to .streamlit/secrets.toml or as an environment variable."
            )
        return url


@contextmanager
def _conn():
    conn = psycopg2.connect(_dsn())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with _conn() as conn:
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


# ── Write ─────────────────────────────────────────────────────────────────────

def upsert_ohlcv(ticker: str, df: pd.DataFrame) -> None:
    if df.empty:
        return

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

    with _conn() as conn:
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


# ── Read ──────────────────────────────────────────────────────────────────────

def query_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT date, open, high, low, close, volume
                FROM ohlcv
                WHERE ticker = %s AND date BETWEEN %s AND %s
                ORDER BY date
                """,
                (ticker, start, end),
            )
            rows = cur.fetchall()

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def get_sync_info(ticker: str):
    """Returns (last_synced, date_from, date_to) or None."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_synced, date_from, date_to FROM sync_log WHERE ticker = %s",
                (ticker,),
            )
            return cur.fetchone()
