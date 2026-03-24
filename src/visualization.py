import matplotlib.pyplot as plt
import os

def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()

def plot_predictions(y_test, predictions, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, color='blue', label='Actual Price')
    plt.plot(predictions, color='red', label='Predicted Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('prediction_plot.png')
    plt.show()
