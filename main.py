import argparse
import numpy as np
from src.data_loader import fetch_stock_data
from src.preprocessing import DataPreprocessor
from src.model import build_lstm_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualization import plot_history, plot_predictions

def main():
    parser = argparse.ArgumentParser(description="Stock Price Prediction using LSTM")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock Ticker Symbol")
    parser.add_argument("--epochs", type=int, default=20, help="Training Epochs")
    args = parser.parse_args()

    df = fetch_stock_data(args.ticker)
    data = df['Close']

    preprocessor = DataPreprocessor()
    scaled_data = preprocessor.scale_data(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    SEQ_LENGTH = 60
    X_train, y_train = preprocessor.create_sequences(train_data, SEQ_LENGTH)
    X_test, y_test = preprocessor.create_sequences(test_data, SEQ_LENGTH)

    model = build_lstm_model((X_train.shape[1], 1))

    print("Starting training...")
    history = train_model(model, X_train, y_train, None, None, epochs=args.epochs)
    plot_history(history)

    print("Evaluating model...")
    rmse, mae, preds, actuals = evaluate_model(model, X_test, y_test, preprocessor)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    plot_predictions(actuals, preds, args.ticker)
    print("Execution Finished.")

if __name__ == "__main__":
    main()
