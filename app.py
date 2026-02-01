import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# --- Page Setup ---
st.set_page_config(page_title="AI Stock Insight", layout="wide")

# --- News Engine (Improved for AAPL) ---
def get_news_and_sentiment(ticker):
    analyzer = SentimentIntensityAnalyzer()
    news_data = []
    try:
        t_obj = yf.Ticker(ticker)
        raw_news = t_obj.news
        if not raw_news: return [], 0.0
        
        scores = []
        for n in raw_news[:15]:  # ดึงมา 15 ข่าวเพื่อความแม่นยำ
            title = n.get('title', '')
            if title:
                s = analyzer.polarity_scores(title)['compound']
                scores.append(s)
                status = "🟢 Positive" if s > 0.05 else "🔴 Negative" if s < -0.05 else "⚪ Neutral"
                news_data.append({"title": title, "score": s, "status": status, "link": n.get('link', '#')})
        
        return news_data, np.mean(scores) if scores else 0.0
    except:
        return [], 0.0

# --- UI Sidebar ---
st.sidebar.header("🛡️ AI Controls")
ticker_input = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
train_years = st.sidebar.slider("Data Range (Years)", 1, 10, 5)
epochs_num = st.sidebar.slider("Training Epochs", 10, 100, 30)

# Time Estimation
est_time = 5 + (epochs_num * 0.3 * train_years)
st.sidebar.info(f"⏱️ Estimated: ~{int(est_time)}s")

if st.sidebar.button("Run Analysis"):
    start_t = time.time()
    
    with st.spinner('Reading market news and training AI...'):
        # 1. Fetch News
        news_list, avg_s = get_news_and_sentiment(ticker_input)
        
        # 2. Fetch Data
        end_d = datetime.now()
        start_d = end_d - timedelta(days=train_years*365)
        df = yf.download(ticker_input, start=start_d, end=end_d)

        if df.empty:
            st.error("Ticker not found.")
        else:
            # Prepare Data
            data = df[['Close']].copy()
            data['Sentiment'] = avg_s
            data['Target'] = data['Close'].shift(-1)
            data.dropna(inplace=True)

            scaler_x, scaler_y = RobustScaler(), RobustScaler()
            x_scaled = scaler_x.fit_transform(data[['Close', 'Sentiment']])
            y_scaled = scaler_y.fit_transform(data[['Target']])

            # Windows
            lookback = 30 # ลด Lookback ลงเพื่อให้เข้ากับ Epochs น้อยๆ
            X, y = [], []
            for i in range(lookback, len(x_scaled)):
                X.append(x_scaled[i-lookback:i, :])
                y.append(y_scaled[i, 0])
            
            X, y = np.array(X), np.array(y)
            split = int(len(X) * 0.8)
            X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

            # 3. Simple & Robust Model
            model = Sequential([
                Input(shape=(lookback, 2)),
                LSTM(50, activation='tanh'), # ใช้โครงสร้างที่เล็กลงเพื่อให้เรียนรู้ไวขึ้น
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=epochs_num, batch_size=16, verbose=0)

            # 4. Results
            preds = model.predict(X_test)
            preds_real = scaler_y.inverse_transform(preds)
            y_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            
            tomorrow = scaler_y.inverse_transform(model.predict(x_scaled[-lookback:].reshape(1, lookback, 2))).item()
            curr_p = float(data['Close'].iloc[-1])
            actual_t = time.time() - start_t

            # 5. Dashboard
            st.success(f"Complete in {actual_t:.1f}s")
            c1, c2, c3 = st.columns(3)
            c1.metric("Current", f"${curr_p:.2f}")
            c2.metric("AI Prediction", f"${tomorrow:.2f}", f"{tomorrow-curr_p:.2f}")
            c3.metric("Market Sentiment", f"{avg_s:.2f}")

            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[-len(y_real):], y=y_real.flatten(), name="Actual", line=dict(color="#00ff88")))
            fig.add_trace(go.Scatter(x=df.index[-len(y_real):], y=preds_real.flatten(), name="AI Prediction", line=dict(color="#ff3366", dash='dot')))
            fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

            # News Feed
            st.subheader("📰 Latest News Feed")
            if news_list:
                for n in news_list:
                    with st.expander(f"{n['status']} | {n['title']}"):
                        st.write(f"Sentiment Score: {n['score']}")
                        st.write(f"[Source]({n['link']})")
            else:
                st.warning("Could not fetch news. Using neutral sentiment.")
