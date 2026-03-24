# Stock Price Prediction using LSTM

Deep learning project to predict stock prices using Long Short-Term Memory (LSTM) networks.

## Features
- Fetches real-time historical data using `yfinance`.
- Preprocesses and scales data for LSTM ingestion.
- Stacked LSTM architecture with Dropout for regularization.
- Visualizes training loss and prediction vs actual prices.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run training and prediction:
   ```bash
   python main.py --ticker MSFT --epochs 25
   ```
