"""
News fetching and VADER sentiment scoring.
Uses yfinance .news endpoint (Yahoo Finance) — no API key required.
"""
import datetime
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=1800)
def get_stock_news(ticker: str) -> list:
    try:
        articles = yf.Ticker(ticker).news or []
        return articles[:15]
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
