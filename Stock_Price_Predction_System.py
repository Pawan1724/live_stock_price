import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to fetch stock data
def get_live_stock_data(ticker, period="1mo", interval="1h"):  # Fetch hourly data
    data = yf.download(ticker, period=period, interval=interval)
    #print(data)
    return data

# Function to preprocess data
def preprocess_data(data, n_days=8):
    for i in range(1, n_days + 1):
        data[f'Close_{i}'] = data['Close'].shift(i)
    data = data.dropna()
    if len(data) < 10:
        raise ValueError("Not enough data after preprocessing for training/testing.")
    X = data[[f'Close_{i}' for i in range(1, n_days + 1)]]
    y = data['Close']
    return X, y
# Function to train the models and make predictions
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Linear Regression Model
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)

    # Random Forest Model
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)

    return model_lr, model_rf, y_test, y_pred_lr, y_pred_rf, mse_lr, mse_rf

# Function to predict the next 24 hours
def predict_next_24_hours(model_lr, model_rf, data, n_days=3):
    last_n_days = data[['Close']].tail(n_days).values.flatten().reshape(1, -1)
    if last_n_days.shape[1] != n_days:
        return None

    predictions_lr = []
    predictions_rf = []

    for _ in range(24):
        pred_lr = model_lr.predict(last_n_days)[0]
        pred_rf = model_rf.predict(last_n_days)[0]

        predictions_lr.append(pred_lr)
        predictions_rf.append(pred_rf)

        last_n_days = np.roll(last_n_days, -1)
        last_n_days[0, -1] = pred_lr  # Update with latest prediction

    return predictions_lr, predictions_rf

# Streamlit UI
st.title("📈 Advanced Stock Price Prediction App")

# ✅ Input moved to the main page
ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, GOOGL):", "AAPL")

if st.button("Predict"):
    try:
        live_data = get_live_stock_data(ticker)
        X, y = preprocess_data(live_data)
        model_lr, model_rf, y_test, y_pred_lr, y_pred_rf, mse_lr, mse_rf = train_models(X, y)
        predictions_lr, predictions_rf = predict_next_24_hours(model_lr, model_rf, live_data)

        st.subheader("🔹 Model Performance")
        st.write(f"Linear Regression MSE: {mse_lr}")
        st.write(f"Random Forest MSE: {mse_rf}")

        st.subheader("📊 Stock Price Prediction Graph")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(y_test.index, y_test, label="Actual Prices", color='blue')
        ax.plot(y_test.index, y_pred_lr, label="Predicted (Linear Regression)", color='red')
        ax.plot(y_test.index, y_pred_rf, label="Predicted (Random Forest)", color='green')
        ax.set_title(f'{ticker} Stock Price Prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

        st.subheader("⏳ Next 24 Hours Prediction")
        hours = [f"{i+1}h" for i in range(24)]
        df_predictions = pd.DataFrame({
            "Hour": hours,
            "Linear Regression": predictions_lr,
            "Random Forest": predictions_rf
        })
        st.dataframe(df_predictions)

        st.subheader("📉 Visualizing 24-Hour Predictions")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(hours, predictions_lr, marker='o', linestyle='-', label="Linear Regression", color='red')
        ax.plot(hours, predictions_rf, marker='o', linestyle='-', label="Random Forest", color='green')
        ax.set_title(f'Next 24 Hours Stock Price Prediction for {ticker}')
        ax.set_xlabel('Hours Ahead')
        ax.set_ylabel('Predicted Price')
        ax.legend()
        st.pyplot(fig)

    except ValueError as e:
        st.error(f"Error: {e}")
