import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date=None, end_date=None):
    """
    Fetches historical stock data using yfinance.
    """
    period = "max" if not start_date else None

    print(f"Fetching data for {ticker}...")
    if start_date and end_date:
        df = yf.download(ticker, start=start_date, end=end_date)
    else:
        df = yf.download(ticker, period="5y") # Default to 5y if no dates

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    return df
