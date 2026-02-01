import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import feedparser
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(page_title="AI Stock Insight Pro", layout="wide")

# --- 🎨 CSS สำหรับจัดหัวข้อให้อยู่ตรงกลาง ---
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 45px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 10px;
    }
    .centered-subtitle {
        text-align: center;
        font-size: 18px;
        color: #BBBBBB;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 📰 Super Stable News Function ---
def get_super_stable_news(ticker, num_news):
    analyzer = SentimentIntensityAnalyzer()
    news_data = []
    try:
        raw_news = yf.Ticker(ticker).news
        if raw_news:
            for n in raw_news[:num_news]:
                title = n.get('title', '')
                if title:
                    s = analyzer.polarity_scores(title)['compound']
                    news_data.append({"title": title, "score": s, "link": n.get('link', '#')})
    except: pass

    if len(news_data) < 3:
        try:
            rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:num_news]:
                title = entry.title
                s = analyzer.polarity_scores(title)['compound']
                if not any(d['title'] == title for d in news_data):
                    news_data.append({"title": title, "score": s, "link": entry.link})
        except: pass

    for n in news_data:
        n['status'] = "🟢 Positive" if n['score'] > 0.05 else "🔴 Negative" if n['score'] < -0.05 else "⚪ Neutral"
    
    avg_score = np.mean([n['score'] for n in news_data]) if news_data else 0.0
    return news_data, avg_score

# --- Header Section (แบบอยู่ตรงกลาง) ---
st.markdown('<div class="centered-title">🔮 AI Stock Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="centered-subtitle">Hybrid Deep Learning Forecasting & Sentiment Analysis</div>', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("⚙️ Configuration")
ticker_input = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
train_years = st.sidebar.slider("Historical Period (Years)", 1, 10, 5)
epochs_num = st.sidebar.slider("AI Training Epochs", 10, 100, 30)

# Time Estimation
est_seconds = 5 + (epochs_num * 0.45 * train_years)
st.sidebar.warning(f"⏱️ Estimated Time: ~{int(est_seconds)}s")

if st.sidebar.button("Execute Full Analysis"):
    start_time = time.time()
    
    with st.spinner(f'AI is training for {epochs_num} epochs...'):
        # 1. Fetch News & Data
        news_list, avg_sentiment = get_super_stable_news(ticker_input, 10)
        df = yf.download(ticker_input, start=datetime.now()-timedelta(days=train_years*365))

        if df.empty:
            st.error("Ticker symbol not found.")
        else:
            # Data Preparation
            data = df[['Close']].copy()
            data['Sentiment'] = avg_sentiment
            data['Target'] = data['Close'].shift(-1)
            data.dropna(inplace=True)

            scaler_x, scaler_y = RobustScaler(), RobustScaler()
            x_scaled = scaler_x.fit_transform(data[['Close', 'Sentiment']])
            y_scaled = scaler_y.fit_transform(data[['Target']])

            lookback = 30
            X, y = [], []
            for i in range(lookback, len(x_scaled)):
                X.append(x_scaled[i-lookback:i, :])
                y.append(y_scaled[i, 0])
            
            X, y = np.array(X), np.array(y)
            split = int(len(X) * 0.8)

            # 2. AI Model
            model = Sequential([
                Input(shape=(lookback, 2)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X[:split], y[:split], epochs=epochs_num, batch_size=16, verbose=0)

            # 3. Predict & Results
            preds_real = scaler_y.inverse_transform(model.predict(X[split:]))
            y_real = scaler_y.inverse_transform(y[split:].reshape(-1, 1))
            tomorrow = scaler_y.inverse_transform(model.predict(x_scaled[-lookback:].reshape(1, lookback, 2))).item()
            curr_price = float(data['Close'].iloc[-1])
            actual_time = time.time() - start_time

            # 4. Display Metrics
            st.success(f"Analysis Complete in {actual_time:.1f}s")
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"${curr_price:.2f}")
            m2.metric("AI Forecast", f"${tomorrow:.2f}", f"{tomorrow-curr_price:.2f}")
            m3.metric("Market Sentiment", f"{avg_sentiment:.2f}")

            # 5. Chart
            st.subheader(f"📊 Market Trend Analysis: {ticker_input}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-len(y_real):], y=y_real.flatten(), name="Actual", line=dict(color="#00ff88")))
            fig.add_trace(go.Scatter(x=df.index[-len(y_real):], y=preds_real.flatten(), name="AI Prediction", line=dict(color="#ff3366", dash='dot')))
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

            # 6. News Feed
            st.divider()
            st.subheader("📰 Relevant News Headlines")
            if news_list:
                for n in news_list:
                    with st.expander(f"{n['status']} | {n['title']}"):
                        st.write(f"Sentiment Impact: {n['score']}")
                        st.write(f"[Source Article]({n['link']})")
            else:
                st.warning("News sources temporarily unavailable. Using neutral sentiment for AI model.")
else:
    st.info("👈 Adjust your configuration in the sidebar and click 'Execute Full Analysis' to start.")
