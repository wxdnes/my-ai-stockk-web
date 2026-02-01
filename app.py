import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# --- ?????????????????? ---
st.set_page_config(page_title="AI Stock Predictor", layout="wide")

# --- ????????????????????? ---
def get_sentiment(ticker):
    analyzer = SentimentIntensityAnalyzer()
    url = f'https://finance.yahoo.com/quote/{ticker}/news'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3')[:10]]
        scores = [analyzer.polarity_scores(text)['compound'] for text in headlines]
        return np.mean(scores) if scores else 0.0
    except:
        return 0.0

# --- ???? UI ??????? ---
st.title("?? AI Stock Analysis Dashboard")
st.markdown("??????????????????????? **Deep Learning (Hybrid LSTM+GRU)** ?????????????? Sentiment ???????")

# --- Sidebar ????????????? ---
st.sidebar.header("?? ??????????")
stock_symbol = st.sidebar.text_input("???????? (Ticker Symbol)", "AAPL")
train_years = st.sidebar.slider("???????????????????????", 1, 10, 5)
epochs_num = st.sidebar.slider("????????????????? (Epochs)", 10, 100, 30)

if st.sidebar.button("????????????????????????"):
    with st.spinner('?????????????????????????? AI...'):
        # 1. ?????????
        end_date = datetime.now()
        start_date = end_date - timedelta(days=train_years*365)
        
        df_stock = yf.download(stock_symbol, start=start_date, end=end_date)
        df_market = yf.download("^GSPC", start=start_date, end=end_date)

        if df_stock.empty:
            st.error("??????????????? ???????????????? Ticker ????????")
        else:
            # ???????????? Multi-index
            if isinstance(df_stock.columns, pd.MultiIndex):
                data = df_stock['Close'][stock_symbol].to_frame(name='Close')
                data['Volume'] = df_stock['Volume'][stock_symbol]
                market_close = df_market['Close']['^GSPC']
            else:
                data = df_stock[['Close', 'Volume']].copy()
                market_close = df_market['Close']

            data['Market'] = market_close
            data['Sentiment'] = get_sentiment(stock_symbol)
            data.fillna(method='ffill', inplace=True)

            # ????? Indicators
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            data['Target'] = data['Close'].shift(-1)
            data.dropna(inplace=True)

            # 2. ?????? Data ?????? AI
            features = ['Close', 'Volume', 'Market', 'SMA20', 'RSI', 'Sentiment']
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()

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

            # 3. ??????????? Model
            model = Sequential([
                Input(shape=(lookback, len(features))),
                LSTM(100, return_sequences=True),
                Dropout(0.2),
                GRU(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=epochs_num, batch_size=32, verbose=0)

            # 4. ???????
            test_predictions = model.predict(X_test)
            test_predictions_real = scaler_y.inverse_transform(test_predictions)
            y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            
            last_window = X_scaled[-lookback:].reshape(1, lookback, len(features))
            tomorrow_pred = scaler_y.inverse_transform(model.predict(last_window)).item()
            current_price = float(data['Close'].iloc[-1])

            # 5. ???????? Web
            st.success("?????????????????????!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("?????????????", f"{current_price:.2f}")
            col2.metric("AI ???????????????", f"{tomorrow_pred:.2f}", f"{tomorrow_pred - current_price:.2f}")
            col3.metric("?????????? (Sentiment)", f"{data['Sentiment'].iloc[-1]:.2f}")

            # ???? Interactive
            st.subheader(f"?? ???????????? vs ??????? ({stock_symbol})")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index[split+lookback:], y=y_test_real.flatten(), name="Actual Price", line=dict(color='royalblue')))
            fig.add_trace(go.Scatter(x=data.index[split+lookback:], y=test_predictions_real.flatten(), name="AI Prediction", line=dict(color='firebrick', dash='dot')))
            fig.update_layout(hovermode="x unified", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("?? ????????????????????? '??????????????' ????????????????????????")