import yfinance as yf
import pandas as pd

# List of currency pairs (tickers for Yahoo Finance)
pairs = {
    'usdinr': 'INR=X',  # USD to INR
    'usdjpy': 'JPY=X',  # USD to JPY
    'usdcny': 'CNY=X',  # USD to CNY
    'usdkor': 'KRW=X'   # USD to KRW (note: KRW for Korea)
}

# Fetch 2 years of daily data (adjust as needed for your project)
for name, ticker in pairs.items():
    data = yf.download(ticker, start='2022-01-01', end='2024-01-01', interval='1d')
    data = data[['Close']]  # We only need closing prices
    data.to_csv(f'data/{name}.csv')  # Save to data folder
    print(f"Data for {name} saved.")
