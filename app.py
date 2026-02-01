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
st.set_page_config(page_title="AI Stock Predictor Pro", layout="wide")

# --- Enhanced News & Sentiment Function ---
def get_detailed_news(ticker, num_news):
    analyzer = SentimentIntensityAnalyzer()
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    headers = {'User-Agent': 'Mozilla/5.0'}
    news_list = []
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        # ค้นหาหัวข้อข่าวจากโครงสร้าง Yahoo Finance
        headlines = [h.text for h in soup.find_all('h3') if len(h.text) > 10]
        
        for text in headlines[:num_news]:
            score = analyzer.polarity_scores(text)['compound']
            status = "🟢 Positive" if score > 0.05 else "🔴 Negative" if score < -0.05 else "⚪ Neutral"
            news_list.append({"Headline": text, "Score": score, "Status": status})
            
        return news_list
    except:
        return []

# --- Header Section ---
st.title("🔮 AI Stock Analysis & News Dashboard")
st.markdown("Deep Learning Prediction with **Dynamic News Feed Analysis**")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Settings")
stock_symbol = st.sidebar.text_input("Stock Ticker Symbol", "AAPL")
num_news_to_show = st.sidebar.slider("Number of News to Analyze", 5, 20, 10) # เลือกจำนวนข่าวได้
train_years = st.sidebar.slider("Historical Data (Years)", 1, 10, 5)
epochs_num = st.sidebar.slider("Training Epochs", 10, 100, 30)

if st.sidebar.button("Run Analysis & Prediction"):
    start_time = time.time()
    
    with st.spinner('Fetching market data and analyzing latest news...'):
        # 1. Fetch News First
        latest_news = get_detailed_news(stock_symbol, num_news_to_show)
        avg_sentiment = np.mean([n['Score'] for n in latest_news]) if latest_news else 0.0
        
        # 2. Market Data Acquisition
        end_date = datetime.now()
        start_date = end_date - timedelta(days=train_years*365)
        df_stock = yf.download(stock_symbol, start=start_date, end=end_date)
        df_market = yf.download("^GSPC", start=start_date, end=end_date)

        if df_stock.empty:
            st.error("Ticker symbol not found.")
        else:
            # Data Handling for Multi-index
            if isinstance(df_stock.columns, pd.MultiIndex):
                data = df_stock['Close'][stock_symbol].to_frame(name='Close')
                data['Volume'] = df_stock['Volume'][stock_symbol]
                market_close = df_market['Close']['^GSPC']
            else:
                data = df_stock[['Close', 'Volume']].copy()
                market_close = df_market['Close']

            data['Market'] = market_close
            data['Sentiment'] = avg_sentiment # ใช้ค่าเฉลี่ยจากข่าวที่ดึงมา
            data.fillna(method='ffill', inplace=True)

            # Technical Indicators
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            data['Target'] = data['Close'].shift(-1)
            data.dropna(inplace=True)

            # 3. Model Training (LSTM + GRU)
            features = ['Close', 'Volume', 'Market', 'SMA20', 'RSI', 'Sentiment']
            scaler_X, scaler_y = RobustScaler(), RobustScaler()
            X_scaled = scaler_X.fit_transform(data[features])
            y_scaled = scaler_y.fit_transform(data[['Target']])

            lookback = 60
            X_windows, y_windows = [], []
            for i in range(lookback, len(X_scaled)):
                X_windows.append(X_scaled[i-lookback:i, :])
                y_windows.append(y_scaled[i, 0])

            X_train = np.array(X_windows)
            y_train = np.array(y_windows)

            model = Sequential([
                Input(shape=(lookback, len(features))),
                LSTM(64, return_sequences=True),
                GRU(32),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=epochs_num, batch_size=32, verbose=0)

            # 4. Results calculation
            last_window = X_scaled[-lookback:].reshape(1, lookback, len(features))
            tomorrow_pred = scaler_y.inverse_transform(model.predict(last_window)).item()
            current_price = float(data['Close'].iloc[-1])
            processing_time = time.time() - start_time

            # 5. Display Dashboard
            st.success(f"Analysis Finished in {processing_time:.2f}s")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"{current_price:.2f}")
            c2.metric("AI Prediction", f"{tomorrow_pred:.2f}", f"{tomorrow_pred - current_price:.2f}")
            c3.metric("Avg Sentiment", f"{avg_sentiment:.2f}")

            # Main Chart
            st.subheader(f"📈 Market Trend: {stock_symbol}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index[-100:], y=data['Close'].tail(100), name="Actual Price"))
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # --- News Feed Section ---
            st.subheader(f"📰 Latest {len(latest_news)} News Headlines")
            for n in latest_news:
                with st.expander(f"{n['Status']} | {n['Headline']}"):
                    st.write(f"**Sentiment Score:** {n['Score']}")
                    st.progress((n['Score'] + 1) / 2) # แสดงเป็นแถบพลัง

else:
    st.info("👈 Set the number of news and click 'Run Analysis'.")
