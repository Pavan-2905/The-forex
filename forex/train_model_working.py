import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import os

# Function to prepare data for LSTM
def prepare_data(data, look_back=30):  # Look back 30 days
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:(i + look_back), 0])
        y.append(data_scaled[i + look_back, 0])
    return np.array(X), np.array(y), scaler

# Train model for each pair
pairs = ['usdinr', 'usdjpy', 'usdcny', 'usdkor']
for pair in pairs:
    # Load data - skip header rows and use only Close prices
    data = pd.read_csv(f'data/{pair}.csv', index_col=0, skiprows=2)
    # Get the second column (Close prices)
    close_data = data.iloc[:, 0]
    
    # Prepare data
    X, y, scaler = prepare_data(close_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM needs 3D input
    
    # Split into train/test (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build LSTM model (simple: 1 LSTM layer, 50 units)
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], 1)))  # Fixed: removed extra parenthesis
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train (10 epochs for simplicity; increase for better results)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    # Save model
    model.save(f'models/{pair}_lstm.h5')
    
    # Predict and evaluate
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Unscale
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
    mae = mean_absolute_error(y_test_unscaled, predictions)
    print(f'{pair.upper()}: RMSE={rmse:.4f}, MAE={mae:.4f}')
    
    # Plot actual vs predicted (save as image for web app)
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_unscaled, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(f'{pair.upper()} Actual vs Predicted')
    plt.legend()
    plt.savefig(f'static/{pair}_plot.png')
    plt.close()
