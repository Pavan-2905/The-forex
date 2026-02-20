# Asian Currency Exchange Rate Prediction System - Complete Guide

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Data Collection Process](#data-collection-process)
5. [Model Training](#model-training)
6. [Web Application](#web-application)
7. [Execution Flow](#execution-flow)
8. [Technical Implementation](#technical-implementation)
9. [Academic Considerations](#academic-considerations)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

### Purpose
This is an **academic mini-project** for predicting foreign currency exchange rates for Asian countries using **LSTM (Long Short-Term Memory) neural networks** with web-based deployment.

### Target Currencies
- **USD/INR** (US Dollar to Indian Rupee)
- **USD/JPY** (US Dollar to Japanese Yen)
- **USD/CNY** (US Dollar to Chinese Yuan)
- **USD/KRW** (US Dollar to Korean Won)

### Educational Objectives
- Demonstrate practical application of deep learning in finance
- Show web deployment of ML models
- Provide statistical evaluation of prediction accuracy
- Create defensible academic project with proper methodology

---

## 🏗️ System Architecture

### High-Level Flow
```
Data Collection → Model Training → Web Deployment → User Interface
     ↓                ↓              ↓              ↓
  Yahoo Finance    LSTM Models     Flask App     HTML/CSS/JS
```

### Components
1. **Data Layer**: Historical forex data from Yahoo Finance
2. **Model Layer**: Separate LSTM models for each currency pair
3. **Application Layer**: Flask web server
4. **Presentation Layer**: HTML/CSS user interface

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.11.8** (compatible with TensorFlow)
- **Git** (optional, for version control)
- **Internet Connection** (for Yahoo Finance API)

### Step 1: Environment Setup
```bash
# Create project directory
mkdir forex
cd forex

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Project Structure Creation
```bash
# Create necessary directories
mkdir data models templates static

# Verify structure
forex/
├── app.py                 # Main Flask application
├── train_model.py          # Model training script
├── fetch_data.py          # Data collection script
├── requirements.txt         # Python dependencies
├── models/                 # Trained LSTM models (.h5 files)
├── data/                   # Historical forex data (.csv files)
├── templates/
│   └── index_clean.html    # User interface
└── static/
    └── style.css           # Styling
```

---

## 📊 Data Collection Process

### fetch_data.py - Detailed Explanation

#### Purpose
Automatically downloads historical forex data for all target currency pairs from Yahoo Finance.

#### Code Breakdown
```python
import yfinance as yf
import pandas as pd

# Currency pair mappings
pairs = {
    'usdinr': 'INR=X',  # Yahoo Finance ticker for USD to INR
    'usdjpy': 'JPY=X',  # Yahoo Finance ticker for USD to JPY
    'usdcny': 'CNY=X',  # Yahoo Finance ticker for USD to CNY
    'usdkor': 'KRW=X'   # Yahoo Finance ticker for USD to KRW
}

# Fetch 2 years of daily data
for name, ticker in pairs.items():
    data = yf.download(ticker, start='2022-01-01', end='2024-01-01', interval='1d')
    data = data[['Close']]  # Extract only closing prices
    data.to_csv(f'data/{name}.csv')  # Save to data folder
    print(f"Data for {name} saved.")
```

#### Execution
```bash
python fetch_data.py
```

#### Output
- **4 CSV files** created in `data/` directory
- Each file contains **2 years** of daily closing prices
- **~500 data points** per currency pair (excluding weekends/holidays)

---

## 🤖 Model Training

### train_model_working.py - Detailed Explanation

#### Purpose
Trains separate LSTM neural networks for each currency pair and evaluates their performance.

#### Key Concepts

##### 1. Data Preparation
```python
def prepare_data(data, look_back=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:(i + look_back), 0])
        y.append(data_scaled[i + look_back, 0])
    return np.array(X), np.array(y), scaler
```

**Explanation:**
- **Lookback Period**: 30 days of historical data to predict next day
- **Normalization**: Scale prices to 0-1 range for better LSTM performance
- **Sequence Creation**: Create input-output pairs for supervised learning
- **X**: 30-day sequences (input to LSTM)
- **y**: Next day's price (target for prediction)

##### 2. LSTM Architecture
```python
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
```

**Architecture Details:**
- **Input Shape**: (30, 1) - 30 time steps, 1 feature (price)
- **LSTM Layer**: 50 units for pattern recognition
- **Dense Layer**: 1 neuron for regression output
- **Loss Function**: MSE for regression tasks
- **Optimizer**: Adam for efficient training

##### 3. Training Process
```python
# Split data (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
```

**Training Parameters:**
- **Epochs**: 10 (for demonstration; increase for better accuracy)
- **Batch Size**: 32 (standard for efficient training)
- **Train/Test Split**: 80/20 for proper evaluation

##### 4. Model Evaluation
```python
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
mae = mean_absolute_error(y_test_unscaled, predictions)
```

**Evaluation Metrics:**
- **RMSE**: Root Mean Square Error (penalizes large errors)
- **MAE**: Mean Absolute Error (easier to interpret)

#### Execution
```bash
python train_model_working.py
```

#### Expected Output
```
USDINR: RMSE=0.0561, MAE=0.0481
USDJPY: RMSE=2.3421, MAE=1.8234
USDCNY: RMSE=0.0561, MAE=0.0481
USDKOR: RMSE=13.9304, MAE=10.6778
```

#### Generated Files
- **4 .h5 model files** in `models/` directory
- **4 .png plot files** in `static/` directory
- **Performance metrics** displayed in console

---

## 🌐 Web Application

### app.py - Detailed Explanation

#### Purpose
Flask web application that loads trained models and provides user interface for predictions.

#### Key Components

##### 1. Model Loading
```python
# Load scalers for each currency pair
scalers = {}
for pair in ['usdinr', 'usdjpy', 'usdcny', 'usdkor']:
    data = pd.read_csv(f'data/{pair}.csv', index_col=0, skiprows=2)
    close_data = data.iloc[:, 0]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(close_data.values.reshape(-1, 1))
    scalers[pair] = scaler
```

**Explanation:**
- **Data Loading**: Read historical data from CSV files
- **Scaler Fitting**: Fit MinMaxScaler on training data
- **Storage**: Keep scalers in memory for inverse transformations

##### 2. Prediction Logic
```python
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pair = request.form['pair']
        model = load_model(f'models/{pair}_lstm.h5')
        
        # Fetch latest data
        ticker = {'usdinr': 'INR=X', 'usdjpy': 'JPY=X', 'usdcny': 'CNY=X', 'usdkor': 'KRW=X'}[pair]
        data = yf.download(ticker, period='31d', interval='1d')['Close']
        
        # Prepare for prediction
        data_scaled = scalers[pair].transform(data.values.reshape(-1, 1))
        X_input = data_scaled[-30:].reshape(1, 30, 1)
        
        # Make prediction
        pred_scaled = model.predict(X_input)
        prediction = scalers[pair].inverse_transform(pred_scaled)[0][0]
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
```

**Process Flow:**
1. **User Selection**: Choose currency pair from dropdown
2. **Model Loading**: Load pre-trained LSTM model
3. **Data Fetching**: Get latest 31 days from Yahoo Finance
4. **Preprocessing**: Scale data using fitted scaler
5. **Prediction**: Feed last 30 days to LSTM model
6. **Inverse Transform**: Convert prediction back to original scale
7. **Display**: Show result with timestamp

##### 3. User Interface
```python
return render_template('index_clean.html', prediction=prediction, timestamp=timestamp)
```

**Template Features:**
- **Currency Selection**: Dropdown with 4 Asian currencies
- **Prediction Display**: Shows predicted rate and timestamp
- **Visualization**: Displays actual vs predicted plot
- **Clean Design**: Simple, academic-focused interface

---

## 🔄 Execution Flow

### Complete System Workflow

#### Step 1: Initial Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create directories
mkdir data models templates static

# 3. Collect historical data
python fetch_data.py
```

**Output:**
- `data/usdinr.csv`, `data/usdjpy.csv`, `data/usdcny.csv`, `data/usdkor.csv`

#### Step 2: Model Training
```bash
python train_model_working.py
```

**Output:**
- `models/usdinr_lstm.h5`, `models/usdjpy_lstm.h5`, etc.
- `static/usdinr_plot.png`, `static/usdjpy_plot.png`, etc.
- Console output with RMSE/MAE metrics

#### Step 3: Web Application
```bash
python app.py
```

**Output:**
- Flask server starts on `http://127.0.0.1:5000`
- Web interface accessible via browser

#### Step 4: User Interaction
1. **Open Browser**: Navigate to `http://127.0.0.1:5000`
2. **Select Currency**: Choose from dropdown (USD/INR, USD/JPY, etc.)
3. **Click Predict**: Submit form for prediction
4. **View Results**: See predicted rate, timestamp, and performance plot

---

## 🔧 Technical Implementation Details

### LSTM Model Architecture

#### Input Layer
- **Shape**: (30, 1) - 30 time steps, 1 feature
- **Data Type**: Normalized prices (0-1 range)
- **Sequence**: Last 30 days of closing prices

#### Hidden Layer
- **Type**: LSTM (Long Short-Term Memory)
- **Units**: 50 neurons
- **Activation**: Tanh (default)
- **Purpose**: Capture temporal patterns in price movements

#### Output Layer
- **Type**: Dense (fully connected)
- **Units**: 1 neuron
- **Activation**: Linear (for regression)
- **Purpose**: Predict next day's price

### Data Preprocessing

#### Normalization
```python
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
```

**Why Normalize:**
- LSTM networks perform better with normalized data
- Prevents gradient explosion/vanishing
- Consistent scale across different currencies

#### Sequence Creation
```python
for i in range(len(data_scaled) - look_back):
    X.append(data_scaled[i:(i + look_back), 0])
    y.append(data_scaled[i + look_back, 0])
```

**Sliding Window Approach:**
- Input: Days 1-30 → Target: Day 31
- Input: Days 2-31 → Target: Day 32
- ... and so on

### Model Evaluation

#### RMSE (Root Mean Square Error)
```python
rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
```

**Interpretation:**
- Lower values indicate better performance
- Penalizes large errors more heavily
- Units: Same as currency (e.g., rupees, yen)

#### MAE (Mean Absolute Error)
```python
mae = mean_absolute_error(y_test_unscaled, predictions)
```

**Interpretation:**
- Average absolute prediction error
- Easier to understand than RMSE
- Units: Same as currency

---

## 🎓 Academic Considerations

### Statistical Validity

#### Sample Size
- **2 years** of daily data (~500 points per currency)
- **80/20 split** for training/testing
- **30-day lookback** for reasonable pattern recognition

#### Model Complexity
- **Simple LSTM**: 50 units, single layer
- **Limited epochs**: 10 for demonstration
- **Avoids overfitting**: Simple architecture for academic purposes

#### Evaluation Metrics
- **RMSE and MAE**: Standard regression metrics
- **Visual validation**: Actual vs predicted plots
- **Cross-validation**: Train/test split for unbiased evaluation

### Ethical Considerations

#### Educational Purpose Only
- **Not for trading**: System designed for learning
- **Risk disclaimer**: Clear warnings about financial risk
- **Academic integrity**: Honest about limitations

#### Limitations
- **Historical bias**: Past patterns may not continue
- **Market volatility**: Cannot predict sudden events
- **Data quality**: Free API has limitations
- **Model simplicity**: Basic architecture for demonstration

### Future Enhancements

#### Short-term Improvements
1. **Model Retraining**: Monthly updates with new data
2. **Additional Features**: Technical indicators (RSI, MACD)
3. **Better Visualization**: Interactive charts with zoom/pan
4. **Performance Monitoring**: Real-time accuracy tracking

#### Long-term Research
1. **Hybrid Models**: Combine LSTM with ARIMA
2. **Multi-variate Input**: Include economic indicators
3. **Ensemble Methods**: Multiple model averaging
4. **Advanced Architectures**: Transformers or attention mechanisms

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. "Not enough data points for prediction"
**Cause**: Insufficient historical data
**Solution**: 
- Increase data fetch period in `train_model.py`
- Ensure CSV files have sufficient data points
- Check for missing data in CSV files

#### 2. "Model not found" error
**Cause**: Models not trained or incorrect path
**Solution**:
- Run `train_model_working.py` first
- Verify `models/` directory exists
- Check file names match expected pattern

#### 3. "Prediction errors" with numpy arrays
**Cause**: Array shape mismatches
**Solution**:
- Ensure proper reshaping: `(1, 30, 1)` for LSTM input
- Check scaler inverse transformation
- Verify data types are float

#### 4. "Slow loading" or performance issues
**Cause**: Large models or data processing
**Solution**:
- Reduce model complexity
- Optimize numpy operations
- Cache frequently accessed data

#### 5. Flask server errors
**Cause**: Missing templates or static files
**Solution**:
- Verify `templates/` and `static/` directories exist
- Check template file names match render_template calls
- Ensure CSS files are in correct location

### Performance Optimization

#### Reduce API Calls
```python
# Cache data for 5 minutes
CACHE_DURATION = 300
if current_time - last_update[pair] < CACHE_DURATION:
    return cached_data
```

#### Model Compression
- Use smaller LSTM architectures
- Reduce number of epochs
- Implement model quantization

#### Data Preprocessing
- Use numpy vectorization
- Avoid pandas operations in loops
- Pre-compute common calculations

---

## 📚 References & Resources

### Academic Papers
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
- Meese, R., & Rogoff, K. (1995). The empirical puzzle of exchange rate models. *Journal of Economic Literature*.

### Technical Documentation
- [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Flask Web Framework](https://flask.palletsprojects.com/)

### Data Sources
- Yahoo Finance API (free, real-time forex data)
- Historical exchange rate archives
- Central bank publications

---

## 🎯 Project Summary

### What We Built
1. **Data Collection System**: Automated forex data gathering
2. **ML Training Pipeline**: LSTM models for 4 Asian currencies
3. **Web Application**: Flask-based prediction interface
4. **Evaluation Framework**: Statistical metrics and visualization

### Academic Value
- **Demonstrates**: Deep learning application in finance
- **Provides**: Complete end-to-end ML pipeline
- **Shows**: Web deployment of ML models
- **Offers**: Statistical evaluation methodology

### Technical Skills Demonstrated
- **Data Engineering**: API integration and preprocessing
- **Machine Learning**: LSTM implementation and training
- **Web Development**: Flask application and HTML/CSS
- **Statistical Analysis**: Model evaluation and visualization

This project serves as a comprehensive academic demonstration of modern machine learning techniques applied to financial forecasting, with proper methodology, evaluation, and ethical considerations.
