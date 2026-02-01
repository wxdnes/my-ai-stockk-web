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
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="AI Stock Predictor Pro", layout="wide")

# --- Optimized News & Sentiment Function (Stable Version) ---
def get_stable_news(ticker, num_news):
    analyzer = SentimentIntensityAnalyzer()
    news_list = []
    try:
        # ใช้ yfinance built-in news เพื่อความเสถียรสูงสุด
        ticker_obj = yf.Ticker(ticker)
        raw_news = ticker_obj.news
        
        for n in raw_news[:num_news]:
            title = n.get('title', '')
            if title:
                score = analyzer.polarity_scores(title)['compound']
                status = "🟢 Positive" if score > 0.05 else "🔴 Negative" if score < -0.05 else "⚪ Neutral"
                news_list.append({
                    "Headline": title, 
                    "Score": score, 
                    "Status": status,
                    "Link": n.get('link', '#')
                })
        return news_list
    except:
        return []

# --- Header ---
st.title("🔮 AI Stock Intelligence Dashboard")
st.markdown("Forecasting market trends using Hybrid Deep Learning & Stable News Sentiment.")

# --- Sidebar & Estimation Logic ---
st.sidebar.header("⚙️ Configuration")
stock_symbol = st.sidebar.text_input("Stock Ticker Symbol", "AAPL")
num_news_to_show = st.sidebar.slider("News Articles to Analyze", 5, 20, 10)
train_years = st.sidebar.slider("Historical Data Period (Years)", 1, 10, 5)
epochs_num = st.sidebar.slider("AI Training Epochs", 10, 100, 30)

# Time Estimation Formula
base_setup_time = 4.0 
time_per_unit = 0.45 
est_seconds = base_setup_time + (epochs_num * train_years * time_per_unit)

st.sidebar.warning(f"⏱️ **Estimated Time:** ~{int(est_seconds)} seconds")
st.sidebar.caption("Processing time depends on server load and data volume.")

if st.sidebar.button("Execute Full Analysis"):
    start_time = time.time()
    
    with st.spinner(f'AI is learning from {epochs_num} training cycles...'):
        # 1. Fetching Stable News
        latest_news = get_stable_news(stock_symbol, num_news_to_show)
        avg_sentiment = np.mean([n['Score'] for n in latest_news]) if latest_news else 0.0
        
        # 2. Market Data Acquisition
        end_date = datetime.now()
        start_date = end_date - timedelta(days=train_years*365)
        df_stock = yf.download(stock_symbol, start=start_date, end=end_date)
        df_market = yf.download("^GSPC", start=start_date, end=end_date)

        if df_stock.empty:
            st.error("Ticker symbol not found. Please use valid symbols like AAPL or PTT.BK")
        else:
            # Multi-index Handling
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

            # Feature Engineering
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            data['Target'] = data['Close'].shift(-1)
            data.dropna(inplace=True)

            # Preprocessing
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
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=epochs_num, batch_size=32, verbose=0)

            # 4. Final Calculations
            test_preds_real = scaler_y.inverse_transform(model.predict(X_test))
            y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            
            tomorrow_pred = scaler_y.inverse_transform(model.predict(X_scaled[-lookback:].reshape(1, lookback, len(features)))).item()
            actual_time = time.time() - start_time

            # 5. Dashboard UI
            st.success(f"Analysis Finished! (Actual: {actual_time:.2f}s | Predicted: {int(est_seconds)}s)")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"{float(data['Close'].iloc[-1]):.2f}")
            m2.metric("AI Forecast (Next Day)", f"{tomorrow_pred:.2f}", f"{tomorrow_pred - float(data['Close'].iloc[-1]):.2f}")
            m3.metric("Sentiment Score", f"{avg_sentiment:.2f}")

            # --- Chart ---
            st.subheader(f"📊 Accuracy Validation: {stock_symbol}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index[split+lookback:], y=y_test_real.flatten(), name="Actual Price", line=dict(color='#00CC96')))
            fig.add_trace(go.Scatter(x=data.index[split+lookback:], y=test_preds_real.flatten(), name="AI Model", line=dict(color='#EF553B', dash='dot')))
            fig.update_layout(hovermode="x unified", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # --- Improved News Section ---
            st.divider()
            st.subheader(f"📰 Latest News Highlights for {stock_symbol}")
            if latest_news:
                for n in latest_news:
                    with st.expander(f"{n['Status']} | {n['Headline']}"):
                        st.write(f"**Sentiment Impact Score:** {n['Score']}")
                        st.progress((n['Score'] + 1) / 2)
                        st.write(f"[Read Full Article]({n['Link']})")
            else:
                st.warning("No recent news found for this ticker. Sentiment analysis used a neutral score.")

else:
    st.info("👈 Set your preferences and click 'Execute Full Analysis' to begin.")
