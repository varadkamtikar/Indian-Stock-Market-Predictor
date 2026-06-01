"""
News fetching and VADER sentiment scoring.
Uses yfinance .news endpoint (Yahoo Finance) — no API key required.

yfinance 1.x changed the response shape: data is now nested under
a 'content' key instead of being flat. Both formats are handled.
"""
import datetime
import html as html_lib
import streamlit as st
import yfinance as yf


def _parse(raw: dict) -> dict:
    """Normalise old (flat) and new (nested content) yfinance news formats."""
    if "content" in raw:
        c = raw["content"]
        title = c.get("title", "")
        link = (c.get("canonicalUrl") or {}).get("url", "#") or "#"
        publisher = (c.get("provider") or {}).get("displayName", "Unknown")
        pub = c.get("pubDate", "")
        try:
            dt = datetime.datetime.fromisoformat(pub.replace("Z", "+00:00"))
            time_str = dt.strftime("%d %b %Y, %H:%M")
        except Exception:
            time_str = pub[:10] if pub else "—"
    else:
        title = raw.get("title", "")
        link = raw.get("link", "#") or "#"
        publisher = raw.get("publisher", "Unknown")
        ppt = raw.get("providerPublishTime", 0)
        try:
            time_str = datetime.datetime.fromtimestamp(ppt).strftime("%d %b %Y, %H:%M")
        except Exception:
            time_str = "—"

    return {
        "title": html_lib.escape(title),   # prevent HTML injection in cards
        "link": link,
        "publisher": html_lib.escape(publisher),
        "time": time_str,
    }


@st.cache_data(ttl=1800)
def get_stock_news(ticker: str) -> list:
    try:
        raw = yf.Ticker(ticker).news or []
        return [_parse(a) for a in raw[:15]]
    except Exception:
        return []


@st.cache_resource
def _sia():
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def sentiment_score(text: str) -> float:
    try:
        return _sia().polarity_scores(text)["compound"]
    except Exception:
        return 0.0


def sentiment_label(score: float) -> tuple:
    if score >= 0.05:
        return "Positive", "#00d084"
    elif score <= -0.05:
        return "Negative", "#ff4b4b"
    return "Neutral", "#f59e0b"
