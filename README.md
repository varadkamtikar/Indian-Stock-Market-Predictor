# Indian Stock Market Predictor

A production-grade stock price prediction dashboard for **NSE** and **BSE** listed equities, powered by a stacked LSTM neural network and served through an interactive Streamlit web application.

**Live app →** [indian-stock-market-predictor-varadkamtikar.streamlit.app](https://indian-stock-market-predictor-varadkamtikar.streamlit.app)

---

## Features

| Feature | Details |
|---|---|
| **Exchange toggle** | Switch between NSE (NIFTY 50) and BSE (SENSEX) |
| **Stock universe** | 116 NSE + 120 BSE verified stocks across 12 sectors |
| **Candlestick chart** | OHLCV with MA 20 / 50 / 200 overlays and volume bars |
| **Technical indicators** | RSI (14), MACD, Bollinger Bands — all interactive Plotly charts |
| **LSTM prediction** | 3-layer stacked LSTM with Dropout trained on 80% holdout |
| **Forecast table** | Configurable 7–90 day price projection with % change |
| **Model metrics** | MAE, RMSE, MAPE on the 20% test set |
| **Neon DB cache** | OHLCV data persisted in PostgreSQL — fetched once, served forever |
| **Daily refresh** | GitHub Actions cron job updates all stocks after market close |

---

## Architecture

```
                        ┌─────────────────┐
                        │  Streamlit UI   │  app.py
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │    fetcher.py   │  smart fetch layer
                        └────┬───────┬───┘
                  DB hit?    │       │  miss / stale
               ┌─────────────┘       └────────────────┐
      ┌────────▼────────┐                   ┌─────────▼────────┐
      │    Neon DB      │  ◄── upsert ───── │    yfinance      │
      │  (PostgreSQL)   │                   │  (Yahoo Finance) │
      └─────────────────┘                   └──────────────────┘
               ▲
               │  daily cron (GitHub Actions)
      ┌────────┴────────┐
      │  refresh_db.py  │  scripts/refresh_db.py
      └─────────────────┘
```

---

## Model Architecture

```
Input → LSTM(128) → Dropout(0.2)
      → LSTM(64)  → Dropout(0.2)
      → LSTM(32)  → Dropout(0.2)
      → Dense(16, ReLU)
      → Dense(1)           ← predicted close price
```

- **Lookback window:** 60 trading days
- **Train / Test split:** 80 % / 20 %
- **Optimiser:** Adam · **Loss:** MSE
- **Early stopping:** patience = 5 epochs

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit · Plotly |
| ML | TensorFlow / Keras (LSTM) |
| Data source | yfinance (Yahoo Finance) |
| Database | Neon (serverless PostgreSQL) |
| CI / Scheduler | GitHub Actions |
| Language | Python 3.12 |

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/varadkamtikar/Indian-Stock-Market-Predictor.git
cd Indian-Stock-Market-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Neon DB
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and paste your Neon DATABASE_URL

# 4. Run
streamlit run app.py
```

---

## GitHub Actions — Daily DB Refresh

The workflow at `.github/workflows/refresh_db.yml` runs automatically at **4:15 PM IST every weekday** (45 minutes after NSE/BSE close).

To enable it, add your Neon connection string as a GitHub secret:

> **Repo → Settings → Secrets and variables → Actions → New repository secret**
> Name: `DATABASE_URL`
> Value: `postgresql://user:password@host/dbname?sslmode=require`

You can also trigger it manually from the **Actions** tab → **Daily DB Refresh** → **Run workflow**.

---

## Disclaimer

This project is built for **educational purposes** and as a portfolio demonstration.  
It does **not** constitute financial advice. Past performance is not indicative of future results.

---

*Originally developed as a BEng Computer Science & Engineering Final Year Project (2021). Rebuilt and extended in 2026 with a modern stack.*
