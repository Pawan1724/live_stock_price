# 📈 Advanced Stock Price Prediction App

A **Streamlit** web application that predicts stock prices using **Linear Regression** and **Random Forest Regressor** based on hourly stock data from Yahoo Finance.

---

## 🔧 Features

- 📥 Fetches live hourly stock data with `yfinance`
- 🤖 Uses **Linear Regression** and **Random Forest Regressor** from `scikit-learn`
- 📊 Plots **actual vs predicted** prices on historical data
- ⏳ Predicts the **next 24 hours** of stock prices
- 📉 Displays **Mean Squared Error** for model performance
- 🖼️ Visualizes predictions with `matplotlib`
- 💡 Simple and interactive web interface using `streamlit`

---

## 🚀 How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-price-predictor.git
cd stock-price-predictor
```

### 2. Install Dependencies

Ensure you have Python 3.8 or later installed. Then install required libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Replace `app.py` with your actual filename if different.

---

## 📝 Requirements

Here are the dependencies required to run the project. Save these in a file named `requirements.txt`:

```
streamlit
yfinance
pandas
numpy
matplotlib
scikit-learn
```

To install everything:

```bash
pip install -r requirements.txt
```

---

## 📌 How It Works

### Live Data Fetching:
- Downloads 1 month of hourly stock data from Yahoo Finance.
- Default stock ticker: AAPL.

### Data Preprocessing:
- Creates lag features like Close_1, Close_2, ..., Close_n from the closing price.
- Drops missing values and prepares features (X) and target (y) for training.

### Model Training:
- Splits data into train/test (80/20 split).
- Trains:
  - `LinearRegression()`
  - `RandomForestRegressor(n_estimators=100)`
- Computes Mean Squared Error (MSE) for both models.

### Future Prediction (Next 24 Hours):
- Uses the latest 3 closing prices to predict next 24 hours (one at a time in rolling fashion).
- Displays predictions for both models in a table and line graph.

### Visualization:
- Shows historical predictions (actual vs predicted).
- Displays 24-hour forecast in a clean chart and dataframe.

---

## 📂 Project Structure

```
.
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 📋 Example Usage

1. Launch the Streamlit app.
2. Enter a stock ticker (e.g., AAPL, GOOGL, MSFT, TSLA).
3. Click the **Predict** button.
4. View:
   - Historical actual vs predicted stock prices
   - Mean Squared Error (MSE) of both models
   - 24-hour forecast (table and graph)

---


## 👨‍💻 Author

**Salikanti Pawan Kumar**  

---

## 🪪 License

This project is licensed under the MIT License.
