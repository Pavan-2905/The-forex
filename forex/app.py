from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Load scalers (we need to save them during training; for simplicity, refit here)
scalers = {}
for pair in ['usdinr', 'usdjpy', 'usdcny', 'usdkor']:
    data = pd.read_csv(f'data/{pair}.csv', index_col=0, skiprows=2)
    # Get the first column (Close prices)
    close_data = data.iloc[:, 0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_data.values.reshape(-1, 1))
    scalers[pair] = scaler

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    timestamp = None
    if request.method == 'POST':
        pair = request.form['pair']
        model = load_model(f'models/{pair}_lstm.h5')
        
        # Fetch latest 30 days + predict next
        ticker = {'usdinr': 'INR=X', 'usdjpy': 'JPY=X', 'usdcny': 'CNY=X', 'usdkor': 'KRW=X'}[pair]
        data = yf.download(ticker, period='31d', interval='1d')['Close']
        data_scaled = scalers[pair].transform(data.values.reshape(-1, 1))
        X_input = data_scaled[-30:].reshape(1, 30, 1)  # Last 30 days
        
        pred_scaled = model.predict(X_input)
        prediction = scalers[pair].inverse_transform(pred_scaled)[0][0]
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('index_clean.html', prediction=prediction, timestamp=timestamp)

if __name__ == '__main__':
    app.run(debug=True)
