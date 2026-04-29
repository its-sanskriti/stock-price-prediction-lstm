import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from datetime import date

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("Stock Price Predictor using LSTM")
st.markdown("Predicts stock prices using a trained LSTM model. Built with TCS data — works best with Indian stocks.")

# Load model and scaler
@st.cache_resource
def load_assets():
    model = load_model('tcs_lstm_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", value="TCS.NS")
start_date = st.sidebar.date_input("Start Date", value=date(2010, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

if st.sidebar.button("Predict"):
    with st.spinner("Fetching data and running prediction..."):
        try:
            # Fetch data
            df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                st.error("No data found. Please check the ticker symbol.")
                st.stop()

            close = df['Close'].values.reshape(-1, 1)

            if len(close) < 200:
                st.error("Not enough data. Please choose a longer date range.")
                st.stop()

            # Split
            split = int(len(close) * 0.70)
            train_close = close[:split]
            test_close = close[split:]

            # Scale — fit only on train
            train_scaled = scaler.fit_transform(train_close)
            test_scaled = scaler.transform(test_close)

            # Create test sequences with last 100 days of train as context
            test_inputs = np.concatenate([train_scaled[-100:], test_scaled])
            X_test, y_test = [], []
            for i in range(100, len(test_inputs)):
                X_test.append(test_inputs[i-100:i])
                y_test.append(test_inputs[i, 0])
            X_test = np.array(X_test)
            y_test = np.array(y_test).reshape(-1, 1)

            # Predict
            y_pred = model.predict(X_test)

            # Inverse transform
            y_pred_actual = scaler.inverse_transform(y_pred)
            y_test_actual = scaler.inverse_transform(y_test)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            r2 = r2_score(y_test_actual, y_pred_actual)
            mae_pct = (mae / np.mean(y_test_actual)) * 100

            # Display metrics
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"₹{rmse:.2f}")
            col2.metric("MAE", f"₹{mae:.2f} ({mae_pct:.1f}%)")
            col3.metric("R² Score", f"{r2:.4f}")

            # Plot
            st.subheader("Actual vs Predicted Price")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(y_test_actual, label='Actual Price', color='royalblue', linewidth=1.5)
            ax.plot(y_pred_actual, label='Predicted Price', color='orange', linewidth=1.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('Price (INR)')
            ax.set_title(f'{ticker} — Actual vs Predicted')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Raw data
            with st.expander("View raw data"):
                st.dataframe(df.tail(50))

        except Exception as e:
            st.error(f"Something went wrong: {e}")
