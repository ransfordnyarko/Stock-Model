from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
# import datetime as dt
from datetime import datetime, date, timedelta
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Ticker map
ticker_map = {
    'GCF': 'GC=F',
    'AAPL': 'AAPL',
    'RTX': 'RTX',
    'CRWD': 'CRWD',
    'GSPC': '^GSPC'
}

@app.route('/predict', methods=['GET'])
def predict():
    try:
        model_name = request.args.get('model_name')

        if model_name not in ticker_map:
            return jsonify({'error': 'Invalid model_name. Available options: GCF, AAPL, RTX, CRWD, GSPC'}), 400

        # Load the model
        model = joblib.load(f'{model_name}_Stock_Model.joblib')

        # Get the ticker symbol
        ticker = ticker_map[model_name]

        # Today's date
        today = date.today().strftime('%Y-%m-%d')

        # Download historical and recent data
        df = yf.download(ticker, start='2020-01-01', end='2024-12-01')
        recent_df = yf.download(ticker, start='2024-12-02', end=today)
        actual_future_prices = recent_df['Adj Close'].tolist()
        actual_future_dates = recent_df.index.strftime('%b %d %Y').tolist()

        num_future_days = len(actual_future_prices)
        predictions = []
        prediction_dates = actual_future_dates[:]

        # Use the last 2 prices as the initial window
        latest_window = df['Adj Close'].iloc[-2:].values
        current_window = latest_window.copy()

        # Generate predictions
        for i in range(num_future_days):
            # Normalize the current window using the first price
            first_price = current_window[0]
            normalized_window = [(price - first_price) / first_price for price in current_window]
            normalized_window = np.array(normalized_window).reshape(1, 2, 1)

            # Predict the next day
            next_price_normalized = model.predict(normalized_window)
            next_price = next_price_normalized[0][0] * first_price + first_price

            # Store the prediction
            predictions.append(round(next_price, 2))

            # Update the current window with the actual price
            current_window = np.append(current_window[1:], actual_future_prices[i])

        # Predict one additional day
        first_price = current_window[0]
        normalized_window = [(price - first_price) / first_price for price in current_window]
        normalized_window = np.array(normalized_window).reshape(1, 2, 1)

        extra_day_normalized = model.predict(normalized_window)
        extra_day_price = round(extra_day_normalized[0][0] * first_price + first_price, 2)

        
        extra_day_date = date.today().strftime('%a %d %b %Y')

        # Append extra day prediction and date
        # predictions.append(extra_day_price)
        # prediction_dates.append(extra_day_date)

        # Prepare the response with the latest 4 actuals, 4 predictions, and the extra day'

        print(prediction_dates[-4: ])
        response = {
            'model_name': model_name,
            'dates': prediction_dates[-4:],  # Latest 4 dates + extra day
            'actual_prices': [round(price, 2) for price in actual_future_prices[-4:]],
            'predictions': predictions[-4:],  # Latest 4 predictions + extra day
            'extra_day_prediction': {
                'date': extra_day_date,
                'price': extra_day_price
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
