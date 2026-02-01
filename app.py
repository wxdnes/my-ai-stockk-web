import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="AI Stock Insight Pro", layout="wide")

# --- Sentiment Function ---
def get_detailed_news(ticker, num_news):
    analyzer = SentimentIntensityAnalyzer()
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    headers = {'User-Agent': 'Mozilla/5.0'}
    news_list = []
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3') if len(h.text) > 10]
        for text in headlines[:num_news]:
            score = analyzer.polarity_scores(text)['compound']
            status = "🟢 Positive" if score > 0.05 else "🔴 Negative" if score < -0.05 else "⚪ Neutral"
            news_list.append({"Headline": text, "Score": score, "Status": status})
        return news_list
    except: return []

# --- Header ---
st.title("🔮 AI Stock Intelligence Dashboard")

# --- Sidebar & Estimation Logic ---
st.sidebar.header("⚙️ Model Configuration")
stock_symbol = st.sidebar.text_input("Stock Ticker Symbol", "AAPL")
num_news_to_show = st.sidebar.slider("News Articles to Analyze", 5, 20, 10)
train_years = st.sidebar.slider("Historical Data Period (Years)", 1, 10, 5)
epochs_num = st.sidebar.slider("AI Training Epochs", 10, 100, 30)

# --- ส่วนคำนวณเวลาล่วงหน้า (Estimation) ---
# ค่าเฉลี่ยโดยประมาณ: ข้อมูล 1 ปี ใช้เวลา ~0.5 วิ/Epoch, ดึงข้อมูล+ข่าว ~3-5 วิ
base_setup_time = 5.0 
time_per_epoch_per_year = 0.4 
estimated_seconds = base_setup_time + (epochs_num * train_years * time_per_epoch_per_year)

st.sidebar.warning(f"⏱️ **Estimated Time:** ~{int(estimated_seconds)} seconds")
st.sidebar.caption("_*Estimation based on server average speed._")

if st.sidebar.button("Execute Full Analysis"):
    start_time = time.time()
    
    with st.spinner(f'AI is training for {epochs_num} epochs. Please wait...'):
        # 1. Data Fetching
        latest_news = get_detailed_news(stock_symbol, num_news_to_show)
        avg_sentiment = np.mean([n['Score'] for n in latest_news]) if latest_news else 0.0
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=train_years*365)
        df_stock = yf.download(stock_symbol, start=start_date, end=end_date)
        df_market = yf.download("^GSPC", start=start_date, end=end_date)

        if df_stock.empty:
            st.error("Ticker symbol not found.")
        else:
            # Data Preparation
            if isinstance(df_stock.columns, pd.MultiIndex):
                data = df_stock['Close'][stock_symbol].to_frame(name='Close')
                data['Volume'] = df_stock['Volume'][stock_symbol]
                market_close = df_market['Close']['^GSPC']
            else:
                data = df_stock[['Close', 'Volume']].copy()
                market_close = df_market['Close']

            data['Market'] = market_close
            data['Sentiment'] = avg_sentiment
            data.fillna(method='ffill', inplace=True)

            # Features
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            data['Target'] = data['Close'].shift(-1)
            data.dropna(inplace=True)

            features = ['Close', 'Volume', 'Market', 'SMA20', 'RSI', 'Sentiment']
            scaler_X, scaler_y = RobustScaler(), RobustScaler()
            X_scaled = scaler_X.fit_transform(data[features])
            y_scaled = scaler_y.fit_transform(data[['Target']])

            lookback = 60
            X_windows, y_windows = [], []
            for i in range(lookback, len(X_scaled)):
                X_windows.append(X_scaled[i-lookback:i, :])
                y_windows.append(y_scaled[i, 0])

            X_windows, y_windows = np.array(X_windows), np.array(y_windows)
            split = int(len(X_windows) * 0.8)
            X_train, X_test = X_windows[:split], X_windows[split:]
            y_train, y_test = y_windows[:split], y_windows[split:]

            # 2. AI Training
            model = Sequential([
                Input(shape=(lookback, len(features))),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                GRU(32),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=epochs_num, batch_size=32, verbose=0)

            # 3. Results
            test_predictions = model.predict(X_test)
            test_predictions_real = scaler_y.inverse_transform(test_predictions)
            y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            tomorrow_pred = scaler_y.inverse_transform(model.predict(X_scaled[-lookback:].reshape(1, lookback, len(features)))).item()
            
            proc_time = time.time() - start_time

            # 4. Display
            st.success(f"Analysis Complete! Actual time: {proc_time:.2f}s")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"{float(data['Close'].iloc[-1]):.2f}")
            m2.metric("AI Forecast", f"{tomorrow_pred:.2f}", f"{tomorrow_pred - float(data['Close'].iloc[-1]):.2f}")
            m3.metric("Avg Sentiment", f"{avg_sentiment:.2f}")

            # Charts
            st.subheader("📈 Backtesting & Market Trends")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=data.index[split+lookback:], y=y_test_real.flatten(), name="Actual"))
            fig1.add_trace(go.Scatter(x=data.index[split+lookback:], y=test_predictions_real.flatten(), name="AI Prediction", line=dict(dash='dot')))
            fig1.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig1, use_container_width=True)

            # News
            st.divider()
            st.subheader("📰 Sentiment Analysis")
            for n in latest_news:
                with st.expander(f"{n['Status']} | {n['Headline']}"):
                    st.write(f"Score: {n['Score']}")
                    st.progress((n['Score'] + 1) / 2)
else:
    st.info("👈 Adjust Epochs to see the estimated processing time.")
