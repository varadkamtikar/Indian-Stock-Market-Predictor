import math
import warnings
import datetime
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fetcher import get_stock_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Stock Market Predictor | LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
    border: 1px solid #2d3548;
    border-radius: 12px;
    padding: 18px 12px;
    text-align: center;
}
.metric-card .label {
    color: #8b95a1;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-card .value {
    color: #e8ecf0;
    font-size: 1.45rem;
    font-weight: 700;
    line-height: 1.2;
}
.metric-card .sublabel { color: #8b95a1; font-size: 0.72rem; margin-top: 5px; }
.pos { color: #00d084; font-size: 0.82rem; font-weight: 600; margin-top: 4px; }
.neg { color: #ff4b4b; font-size: 0.82rem; font-weight: 600; margin-top: 4px; }
.neu { color: #f59e0b; font-size: 0.82rem; font-weight: 600; margin-top: 4px; }

.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8ecf0;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #2d3548;
}

.info-banner {
    background: linear-gradient(135deg, #1a2744, #1e2f55);
    border: 1px solid #2d4080;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #93c5fd;
    font-size: 0.88rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #161b2e;
    border-radius: 10px;
    padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 7px;
    color: #8b95a1;
    font-weight: 500;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: #2d3a5c !important;
    color: #60a5fa !important;
}

[data-testid="stSidebar"] {
    background: #131722;
    border-right: 1px solid #1e2535;
}

footer { visibility: hidden; }

.exchange-badge-nse {
    display: inline-block;
    background: linear-gradient(135deg, #1a3a6e, #1e4d9e);
    border: 1px solid #2563eb;
    color: #93c5fd;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 5px;
    margin-right: 6px;
    vertical-align: middle;
}
.exchange-badge-bse {
    display: inline-block;
    background: linear-gradient(135deg, #6e1a1a, #9e2e1e);
    border: 1px solid #dc2626;
    color: #fca5a5;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 5px;
    margin-right: 6px;
    vertical-align: middle;
}
.index-label {
    color: #8b95a1;
    font-size: 0.85rem;
    vertical-align: middle;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Stock universes (sourced from stocks.py, all tickers verified) ────────────
from stocks import NSE_STOCKS, BSE_STOCKS

EXCHANGE_META = {
    "NSE": {
        "label": "NSE",
        "index": "NIFTY 50",
        "badge_class": "exchange-badge-nse",
        "stocks": NSE_STOCKS,
    },
    "BSE": {
        "label": "BSE",
        "index": "SENSEX",
        "badge_class": "exchange-badge-bse",
        "stocks": BSE_STOCKS,
    },
}

LOOKBACK = 60
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    font=dict(family="Inter", color="#c9d1d9"),
    margin=dict(l=0, r=0, t=45, b=0),
)
GRID = dict(showgrid=True, gridcolor="#1e2535", zeroline=False)


# ── Helpers ───────────────────────────────────────────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series):
    e12 = series.ewm(span=12, adjust=False).mean()
    e26 = series.ewm(span=26, adjust=False).mean()
    m = e12 - e26
    sig = m.ewm(span=9, adjust=False).mean()
    return m, sig, m - sig


def bollinger(series: pd.Series, period: int = 20):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + 2 * std, sma, sma - 2 * std


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = rsi(df["Close"])
    df["MACD"], df["Signal"], df["MACD_Hist"] = macd(df["Close"])
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = bollinger(df["Close"])
    return df


def prices_hash(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


# ── LSTM pipeline ─────────────────────────────────────────────────────────────
def build_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    return X.reshape(-1, lookback, 1), y


@st.cache_resource
def train_model(cache_key: str, prices_bytes: bytes, epochs: int):
    prices = np.frombuffer(prices_bytes, dtype=np.float64).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    train_size = int(len(scaled) * 0.80)
    X_train, y_train = build_sequences(scaled[:train_size], LOOKBACK)
    X_test, y_test = build_sequences(scaled[train_size - LOOKBACK :], LOOKBACK)

    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=(LOOKBACK, 1)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0,
    )

    pred_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    return model, scaler, predictions, actual, train_size


def forecast_future(model, scaler, last_prices: np.ndarray, days: int) -> np.ndarray:
    seq = last_prices.copy().astype(np.float64)
    preds = []
    for _ in range(days):
        scaled_seq = scaler.transform(seq.reshape(-1, 1)).reshape(1, LOOKBACK, 1)
        p = model.predict(scaled_seq, verbose=0)[0, 0]
        price = scaler.inverse_transform([[p]])[0, 0]
        preds.append(price)
        seq = np.append(seq[1:], price)
    return np.array(preds)


# ── Chart helpers ─────────────────────────────────────────────────────────────
def apply_theme(fig, height: int, title: str = ""):
    fig.update_layout(height=height, title=title, **PLOTLY_BASE)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Indian Stock Predictor")
    st.markdown("*LSTM Neural Network · Indian Markets*")
    st.divider()

    exchange = st.radio(
        "Exchange",
        ["NSE", "BSE"],
        horizontal=True,
        help="NSE uses NIFTY 50 stocks · BSE uses SENSEX stocks",
    )
    meta = EXCHANGE_META[exchange]
    stock_dict = meta["stocks"]

    # Clear stale predictions when the user switches exchange or stock
    prev_exchange = st.session_state.get("_exchange")
    prev_stock = st.session_state.get("_stock")

    selected_name = st.selectbox("Select Stock", list(stock_dict.keys()))
    ticker = stock_dict[selected_name]

    if prev_exchange != exchange or prev_stock != selected_name:
        st.session_state.pop("results", None)
        st.session_state["_exchange"] = exchange
        st.session_state["_stock"] = selected_name

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input(
            "From", datetime.date(2015, 1, 1), min_value=datetime.date(2010, 1, 1)
        )
    with c2:
        end_date = st.date_input("To", datetime.date.today())

    st.divider()
    st.markdown("### ⚙️ Model Settings")
    epochs = st.slider("Training Epochs", 10, 60, 25, 5)
    future_days = st.slider("Forecast Horizon (days)", 7, 90, 30, 7)

    st.divider()
    train_btn = st.button("🚀  Train & Predict", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        """
    <div style='font-size:0.76rem;color:#8b95a1;line-height:1.6'>
    <b>How it works</b><br>
    Downloads historical OHLCV data from Yahoo Finance, trains a stacked LSTM
    network (3 layers + dropout) on 80 % of the data, evaluates on the held-out
    20 %, and projects prices forward.<br><br>
    <b>⚠ Disclaimer:</b> Educational purposes only. Not financial advice.
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
display_name = selected_name.rsplit("(", 1)[0].strip()
st.markdown(f"# {display_name}")
st.markdown(
    f'<span class="{meta["badge_class"]}">{meta["label"]}</span>'
    f'<span class="index-label"><b>{ticker}</b> &nbsp;·&nbsp; {meta["index"]} &nbsp;·&nbsp; India</span>',
    unsafe_allow_html=True,
)
st.markdown("")

# ── Fetch & enrich data ───────────────────────────────────────────────────────
with st.spinner("Fetching market data…"):
    df = get_stock_data(ticker, str(start_date), str(end_date))

if df.empty:
    st.error("No data returned. Check the ticker or widen the date range.")
    st.stop()

df = add_indicators(df)
latest = df.iloc[-1]
prev = df.iloc[-2]
change = float(latest["Close"]) - float(prev["Close"])
pct = change / float(prev["Close"]) * 100
arrow = "▲" if change >= 0 else "▼"
d_cls = "pos" if change >= 0 else "neg"
high52 = float(df["High"].tail(252).max())
low52 = float(df["Low"].tail(252).min())
rsi_val = float(latest["RSI"])
rsi_signal = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
rsi_cls = "neg" if rsi_val > 70 else ("pos" if rsi_val < 30 else "neu")

# ── Metrics row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value, delta, cls in [
    (c1, "Current Price", f"₹{float(latest['Close']):,.2f}", f"{arrow} ₹{abs(change):.2f} ({pct:+.2f}%)", d_cls),
    (c2, "52-Week High", f"₹{high52:,.2f}", "", ""),
    (c3, "52-Week Low", f"₹{low52:,.2f}", "", ""),
    (c4, "Volume", f"{int(latest['Volume']):,}", "", ""),
    (c5, "RSI (14)", f"{rsi_val:.1f}", rsi_signal, rsi_cls),
]:
    with col:
        delta_html = f'<div class="{cls}">{delta}</div>' if delta else ""
        st.markdown(
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>{delta_html}</div>',
            unsafe_allow_html=True,
        )

st.markdown("")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📊  Price & Volume", "📉  Technical Indicators", "🤖  LSTM Prediction"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · Price & Volume
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.04,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="#00d084",
            decreasing_line_color="#ff4b4b",
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    for col_name, color, lname in [
        ("MA20", "#60a5fa", "MA 20"),
        ("MA50", "#f59e0b", "MA 50"),
        ("MA200", "#a78bfa", "MA 200"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col_name],
                line=dict(color=color, width=1.5),
                name=lname,
            ),
            row=1,
            col=1,
        )

    bar_colors = [
        "#00d084" if float(c) >= float(o) else "#ff4b4b"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            marker_color=bar_colors,
            opacity=0.75,
            name="Volume",
        ),
        row=2,
        col=1,
    )

    fig = apply_theme(fig, 620)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw data expander
    with st.expander("📋  Raw OHLCV Data"):
        display_df = df[["Open", "High", "Low", "Close", "Volume"]].tail(60).copy()
        display_df.index = display_df.index.strftime("%d %b %Y")
        st.dataframe(display_df.style.format("{:,.2f}", subset=["Open","High","Low","Close"]), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · Technical Indicators
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    # RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#60a5fa", width=2), name="RSI")
    )
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00d084", annotation_text="Oversold (30)")
    fig_rsi.add_hrect(y0=30, y1=70, fillcolor="#2d3548", opacity=0.25, layer="below")
    fig_rsi = apply_theme(fig_rsi, 260, "RSI — Relative Strength Index (14)")
    fig_rsi.update_layout(yaxis=dict(range=[0, 100]), showlegend=False)
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    fig_macd = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05
    )
    fig_macd.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#60a5fa", width=2), name="MACD"),
        row=1, col=1,
    )
    fig_macd.add_trace(
        go.Scatter(x=df.index, y=df["Signal"], line=dict(color="#f59e0b", width=2), name="Signal"),
        row=1, col=1,
    )
    hist_clr = ["#00d084" if v >= 0 else "#ff4b4b" for v in df["MACD_Hist"]]
    fig_macd.add_trace(
        go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=hist_clr, name="Histogram"),
        row=2, col=1,
    )
    fig_macd = apply_theme(fig_macd, 360, "MACD — Moving Average Convergence Divergence")
    fig_macd.update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    st.plotly_chart(fig_macd, use_container_width=True)

    # Bollinger Bands
    fig_bb = go.Figure()
    fig_bb.add_trace(
        go.Scatter(
            x=df.index, y=df["BB_Upper"],
            line=dict(color="rgba(96,165,250,0.45)", width=1), name="Upper Band",
        )
    )
    fig_bb.add_trace(
        go.Scatter(
            x=df.index, y=df["BB_Lower"],
            fill="tonexty", fillcolor="rgba(96,165,250,0.08)",
            line=dict(color="rgba(96,165,250,0.45)", width=1), name="Lower Band",
        )
    )
    fig_bb.add_trace(
        go.Scatter(
            x=df.index, y=df["BB_Mid"],
            line=dict(color="#60a5fa", width=1.5, dash="dash"), name="Middle (SMA 20)",
        )
    )
    fig_bb.add_trace(
        go.Scatter(x=df.index, y=df["Close"], line=dict(color="#e8ecf0", width=1.5), name="Close")
    )
    fig_bb = apply_theme(fig_bb, 360, "Bollinger Bands (20-day, 2σ)")
    fig_bb.update_layout(legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    st.plotly_chart(fig_bb, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · LSTM Prediction
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if not train_btn and "results" not in st.session_state:
        st.markdown(
            '<div class="info-banner">👈  Click <b>Train & Predict</b> in the sidebar to train the LSTM model '
            "and generate price forecasts.</div>",
            unsafe_allow_html=True,
        )

    else:
        prices = df["Close"].values.astype(np.float64)
        cache_key = f"{ticker}|{start_date}|{end_date}|{epochs}|{prices_hash(prices)}"

        if train_btn:
            with st.spinner("Training LSTM model… this may take 1–2 minutes."):
                model, scaler, preds, actual, train_size = train_model(
                    cache_key, prices.tobytes(), epochs
                )
            future_seq = prices[-LOOKBACK:]
            with st.spinner("Generating future forecast…"):
                future_preds = forecast_future(model, scaler, future_seq, future_days)
            future_dates = pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1), periods=future_days, freq="B"
            )
            st.session_state["results"] = dict(
                preds=preds, actual=actual, train_size=train_size,
                future_preds=future_preds, future_dates=future_dates,
                df_index=df.index, df_close=prices,
            )

        r = st.session_state["results"]
        preds = r["preds"]
        actual = r["actual"]
        train_size = r["train_size"]
        future_preds = r["future_preds"]
        future_dates = r["future_dates"]
        df_close = r["df_close"]
        df_index = r["df_index"]

        # Performance metrics
        mae = mean_absolute_error(actual, preds)
        rmse = math.sqrt(mean_squared_error(actual, preds))
        mape = float(np.mean(np.abs((actual - preds) / actual)) * 100)
        next_day = float(future_preds[0])
        today_price = float(df_close[-1])
        next_day_chg = (next_day - today_price) / today_price * 100

        mc1, mc2, mc3, mc4 = st.columns(4)
        for col, label, val, sub, cls in [
            (mc1, "MAE", f"₹{mae:,.2f}", "Mean Absolute Error", ""),
            (mc2, "RMSE", f"₹{rmse:,.2f}", "Root Mean Square Error", ""),
            (mc3, "MAPE", f"{mape:.2f}%", "Mean Abs. % Error", ""),
            (
                mc4,
                "Next Day Forecast",
                f"₹{next_day:,.2f}",
                f"{'+' if next_day_chg >= 0 else ''}{next_day_chg:.2f}% vs today",
                "pos" if next_day_chg >= 0 else "neg",
            ),
        ]:
            with col:
                sub_html = f'<div class="{cls}">{sub}</div>' if cls else f'<div class="sublabel">{sub}</div>'
                st.markdown(
                    f'<div class="metric-card"><div class="label">{label}</div>'
                    f'<div class="value">{val}</div>{sub_html}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Main prediction chart
        test_index = df_index[train_size:]
        fig_p = go.Figure()
        fig_p.add_trace(
            go.Scatter(
                x=df_index[:train_size], y=df_close[:train_size],
                line=dict(color="#4a5568", width=1.5), name="Training Data", opacity=0.7,
            )
        )
        fig_p.add_trace(
            go.Scatter(
                x=test_index, y=actual.flatten(),
                line=dict(color="#e8ecf0", width=2), name="Actual Price",
            )
        )
        fig_p.add_trace(
            go.Scatter(
                x=test_index, y=preds.flatten(),
                line=dict(color="#60a5fa", width=2, dash="dot"), name="LSTM Prediction",
            )
        )
        fig_p.add_trace(
            go.Scatter(
                x=future_dates, y=future_preds,
                line=dict(color="#f59e0b", width=2.5), name=f"{future_days}-Day Forecast",
            )
        )
        fig_p.add_vrect(
            x0=df_index[-1],
            x1=future_dates[-1],
            fillcolor="rgba(245,158,11,0.05)",
            layer="below",
            line_width=0,
        )
        fig_p = apply_theme(fig_p, 520, f"LSTM Prediction vs Actual · {exchange} · {display_name}")
        fig_p.update_layout(
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
            yaxis_title="Price (₹)",
            xaxis_title="Date",
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Forecast table + model info side-by-side
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown(f'<div class="section-title">📅 {future_days}-Day Price Forecast</div>', unsafe_allow_html=True)
            fdf = pd.DataFrame(
                {
                    "Date": future_dates.strftime("%d %b %Y"),
                    "Predicted Price": future_preds.round(2),
                    "Change from Today": ((future_preds - today_price) / today_price * 100).round(2),
                }
            )

            def color_change(val):
                if isinstance(val, float):
                    return "color: #00d084" if val > 0 else "color: #ff4b4b"
                return ""

            st.dataframe(
                fdf.style.format(
                    {"Predicted Price": "₹{:,.2f}", "Change from Today": "{:+.2f}%"}
                ).map(color_change, subset=["Change from Today"]),
                use_container_width=True,
                hide_index=True,
            )

        with col_right:
            st.markdown('<div class="section-title">🏗️  Model Architecture</div>', unsafe_allow_html=True)
            arch_rows = [
                ("Layer", "Details"),
                ("Input", f"Sequence: {LOOKBACK} days"),
                ("LSTM 1", "128 units, return_seq=True"),
                ("Dropout 1", "rate = 0.20"),
                ("LSTM 2", "64 units, return_seq=True"),
                ("Dropout 2", "rate = 0.20"),
                ("LSTM 3", "32 units"),
                ("Dropout 3", "rate = 0.20"),
                ("Dense 1", "16 units, ReLU"),
                ("Output", "1 unit (price)"),
            ]
            arch_df = pd.DataFrame(arch_rows[1:], columns=arch_rows[0])
            st.dataframe(arch_df, use_container_width=True, hide_index=True)

            st.markdown(
                f"""
            <div style='background:#1e2130;border:1px solid #2d3548;border-radius:8px;padding:12px;margin-top:10px;font-size:0.82rem;color:#c9d1d9;line-height:1.7'>
            <b>Training config</b><br>
            Optimizer: Adam &nbsp;|&nbsp; Loss: MSE<br>
            Epochs: {epochs} (early stop, patience=5)<br>
            Batch size: 32 &nbsp;|&nbsp; Val split: 10 %<br>
            Train / Test: 80 % / 20 %<br>
            Lookback window: {LOOKBACK} days
            </div>
            """,
                unsafe_allow_html=True,
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
<div style='text-align:center;color:#8b95a1;font-size:0.78rem;padding:8px 0'>
    Indian Stock Market Predictor &nbsp;·&nbsp; NSE &amp; BSE &nbsp;·&nbsp; LSTM Neural Network<br>
    ⚠️ For educational purposes only — not financial advice.
</div>
""",
    unsafe_allow_html=True,
)
