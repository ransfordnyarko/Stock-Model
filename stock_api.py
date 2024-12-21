import datetime as dt
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from datetime import date

ticker_map = {
    'GCF': 'GC=F', 
    'AAPL': 'AAPL',
    'RTX': 'RTX',
    'CRWD': 'CRWD',
    'GSPC': '^GSPC'
}
model_name = 'CRWD'
ticker = ticker_map[model_name]
today = date.today().strftime('%Y-%m-%d')
print(today)

model = joblib.load(f'{model_name}_Stock_Model.joblib')
df = yf.download(ticker, start='2020-1-1', end='2024-12-1')
recent_df = yf.download(ticker, start='2024-12-02', end=today)
actual_future_prices = recent_df['Adj Close'].tolist()

num_future_days = len(actual_future_prices)
predictions = []
latest_window = df['Adj Close'].iloc[-2:].values

current_window = latest_window.copy()

for i in range(num_future_days):
    # Normalize the current window using the first price
    first_price = current_window[0]
    normalized_window = [(price - first_price) / first_price for price in current_window]
    normalized_window = np.array(normalized_window).reshape(1, 2, 1)

    # Predict the next day
    next_price_normalized = model.predict(normalized_window)
    next_price = next_price_normalized[0][0] * first_price + first_price

    # Store the prediction
    predictions.append(next_price)

    # Instead of appending the predicted price, append the actual price for comparison
    current_window = np.append(current_window[1:], actual_future_prices[i])

first_price = current_window[0]
normalized_window = [(price - first_price) / first_price for price in current_window]
normalized_window = np.array(normalized_window).reshape(1, 2, 1)

# Predict the additional day
extra_day_normalized = model.predict(normalized_window)
extra_day_price = extra_day_normalized[0][0] * first_price + first_price

# Add the extra prediction to the list
predictions.append(extra_day_price)

# Display the predicted and actual prices
print("Predicted vs Actual Prices:")
for i, (pred, actual) in enumerate(zip(predictions, actual_future_prices)):
    print(f"Day {i+1}: Predicted: {pred:.2f}, Actual: {actual:.2f}")

print(f"Day {num_future_days + 1}: Predicted: {predictions[-1]:.2f} (no actual data)")