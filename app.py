import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import feedparser # ต้องเพิ่มใน requirements.txt
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(page_title="AI Stock Insight Pro", layout="wide")

# --- 📰 Super Stable News Function ---
def get_super_stable_news(ticker, num_news):
    analyzer = SentimentIntensityAnalyzer()
    news_data = []
    
    # Layer 1: Try yfinance built-in
    try:
        raw_news = yf.Ticker(ticker).news
        if raw_news:
            for n in raw_news[:num_news]:
                title = n.get('title', '')
                if title:
                    s = analyzer.polarity_scores(title)['compound']
                    news_data.append({"title": title, "score": s, "link": n.get('link', '#')})
    except: pass

    # Layer 2: Try Google News RSS (เสถียรมาก ไม่ค่อยโดนบล็อก)
    if len(news_data) < 3:
        try:
            rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+news&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:num_news]:
                title = entry.title
                s = analyzer.polarity_scores(title)['compound']
                if not any(d['title'] == title for d in news_data): # ป้องกันข่าวซ้ำ
                    news_data.append({"title": title, "score": s, "link": entry.link})
        except: pass

    # ประมวลผล Status และค่าเฉลี่ย
    for n in news_data:
        n['status'] = "🟢 Positive" if n['score'] > 0.05 else "🔴 Negative" if n['score'] < -0.05 else "⚪ Neutral"
    
    avg_score = np.mean([n['score'] for n in news_data]) if news_data else 0.0
    return news_data, avg_score

# --- Sidebar ---
st.sidebar.header("🛡️ AI Controls")
ticker_input = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
train_years = st.sidebar.slider("Data Range (Years)", 1, 10, 5)
epochs_num = st.sidebar.slider("Training Epochs", 10, 100, 30)

# Time Estimation
est_time = 5 + (epochs_num * 0.4 * train_years)
st.sidebar.warning(f"⏱️ Estimated: ~{int(est_time)}s")

if st.sidebar.button("Execute Analysis"):
    start_t = time.time()
    
    with st.spinner('Fetching news and training AI model...'):
        # 1. Fetch News (New Stable Method)
        news_list, avg_s = get_super_stable_news(ticker_input, 10)
        
        # 2. Fetch Stock Data
        df = yf.download(ticker_input, start=datetime.now()-timedelta(days=train_years*365))

        if df.empty:
            st.error("Ticker not found.")
        else:
            # Data Prep
            data = df[['Close']].copy()
            data['Sentiment'] = avg_s
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
            
            # 3. Build Model
            model = Sequential([
                Input(shape=(lookback, 2)),
                LSTM(50),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X[:split], y[:split], epochs=epochs_num, batch_size=16, verbose=0)

            # 4. Predict
            preds_real = scaler_y.inverse_transform(model.predict(X[split:]))
            y_real = scaler_y.inverse_transform(y[split:].reshape(-1, 1))
            tomorrow = scaler_y.inverse_transform(model.predict(x_scaled[-lookback:].reshape(1, lookback, 2))).item()
            
            # 5. Dashboard UI
            st.success(f"Processing Complete in {time.time()-start_t:.1f}s")
            
            c1, c2, c3 = st.columns(3)
            curr_p = float(data['Close'].iloc[-1])
            c1.metric("Current Price", f"${curr_p:.2f}")
            c2.metric("AI Prediction", f"${tomorrow:.2f}", f"{tomorrow-curr_p:.2f}")
            c3.metric("Sentiment Score", f"{avg_s:.2f}")

            # Chart (The "One Chart" Layout)
            st.subheader(f"📊 Accuracy Check: {ticker_input}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-len(y_real):], y=y_real.flatten(), name="Actual", line=dict(color="#00ff88")))
            fig.add_trace(go.Scatter(x=df.index[-len(y_real):], y=preds_real.flatten(), name="AI Prediction", line=dict(color="#ff3366", dash='dot')))
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)

            # News Feed (Force Display)
            st.subheader("📰 Latest Market News")
            if news_list:
                for n in news_list:
                    with st.expander(f"{n['status']} | {n['title']}"):
                        st.write(f"Sentiment Score: {n['score']}")
                        st.write(f"[Source Link]({n['link']})")
            else:
                st.warning("All news sources are currently unavailable. AI used neutral sentiment.")
