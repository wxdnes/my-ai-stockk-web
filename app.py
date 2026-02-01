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

# --- Enhanced News & Sentiment Function ---
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
    except:
        return []

# --- Header Section ---
st.title("🔮 AI Stock Intelligence Dashboard")
st.markdown("A professional-grade tool combining **Deep Learning Forecasting** with **Real-time News Sentiment Analysis**.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Model Configuration")
stock_symbol = st.sidebar.text_input("Stock Ticker Symbol", "AAPL")
num_news_to_show = st.sidebar.slider("News Articles to Analyze", 5, 20, 10)
train_years = st.sidebar.slider("Historical Data Period (Years)", 1, 10, 5)
epochs_num = st.sidebar.slider("AI Training Epochs", 10, 100, 30)

if st.sidebar.button("Execute Full Analysis"):
    start_time = time.time()
    
    with st.spinner('Synchronizing with market data and training neural networks...'):
        # 1. Fetch News & Data
        latest_news = get_detailed_news(stock_symbol, num_news_to_show)
        avg_sentiment = np.mean([n['Score'] for n in latest_news]) if latest_news else 0.0
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=train_years*365)
        df_stock = yf.download(stock_symbol, start=start_date, end=end_date)
        df_market = yf.download("^GSPC", start=start_date, end=end_date)

        if df_stock.empty:
            st.error("Ticker symbol not found. Please verify and try again.")
        else:
            # Data Handling
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

            # Indicators
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            data['Target'] = data['Close'].shift(-1)
            data.dropna(inplace=True)

            # 2. Data Preprocessing
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

            # 3. AI Model Construction
            model = Sequential([
                Input(shape=(lookback, len(features))),
                LSTM(100, return_sequences=True),
                Dropout(0.2),
                GRU(50),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=epochs_num, batch_size=32, verbose=0)

            # 4. Predictions & Metrics
            test_predictions = model.predict(X_test)
            test_predictions_real = scaler_y.inverse_transform(test_predictions)
            y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            
            last_window = X_scaled[-lookback:].reshape(1, lookback, len(features))
            tomorrow_pred = scaler_y.inverse_transform(model.predict(last_window)).item()
            current_price = float(data['Close'].iloc[-1])
            proc_time = time.time() - start_time

            # 5. UI Rendering
            st.success(f"Analysis Complete! Processed in {proc_time:.2f} seconds.")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"{current_price:.2f}")
            m2.metric("AI Forecast (Tomorrow)", f"{tomorrow_pred:.2f}", f"{tomorrow_pred - current_price:.2f}")
            m3.metric("Avg Sentiment", f"{avg_sentiment:.2f}")

            # --- CHART 1: ACCURACY ---
            st.subheader("📈 1. Backtesting: Actual vs AI Prediction")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=data.index[split+lookback:], y=y_test_real.flatten(), name="Actual", line=dict(color='#00CC96')))
            fig1.add_trace(go.Scatter(x=data.index[split+lookback:], y=test_predictions_real.flatten(), name="AI Model", line=dict(color='#EF553B', dash='dot')))
            fig1.update_layout(hovermode="x unified", template="plotly_dark", height=400)
            st.plotly_chart(fig1, use_container_width=True)

            # --- CHART 2: RECENT TREND ---
            st.subheader("📊 2. Recent Market Sentiment Trend")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index[-100:], y=data['Close'].tail(100), fill='tozeroy', name="Recent Price", line=dict(color='#1f77b4')))
            fig2.update_layout(hovermode="x unified", template="plotly_dark", height=300)
            st.plotly_chart(fig2, use_container_width=True)

            # --- NEWS SECTION ---
            st.divider()
            st.subheader(f"📰 Top {len(latest_news)} Market Headlines for {stock_symbol}")
            for n in latest_news:
                with st.expander(f"{n['Status']} | {n['Headline']}"):
                    st.write(f"**Score:** {n['Score']}")
                    st.progress((n['Score'] + 1) / 2)
else:
    st.info("👈 Enter ticker and configure settings in the sidebar, then click 'Execute Full Analysis'.")
