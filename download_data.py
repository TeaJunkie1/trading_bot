# download_data.py
import yfinance as yf
import pandas as pd

# Download historical data for selected stocks
symbols = ["AAPL", "MSFT", "AMZN", "GOOGL"]
data = {}

for symbol in symbols:
    stock_data = yf.download(symbol, start="2015-01-01", end="2024-01-01")
    data[symbol] = stock_data

# Combine data into a single DataFrame
combined_data = pd.concat(data, axis=1)
combined_data.columns = [f"{symbol}_{col}" for symbol in symbols for col in stock_data.columns]
combined_data.to_csv("multi_stock_data.csv")  # Save data locally for the bot to access
