import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def scale_data(self, data):
        """
        Scales the data (Close price) using MinMaxScaler.
        """
        if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
            data = data.values.reshape(-1, 1)

        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def create_sequences(self, dataset, seq_length=60):
        """
        Creates sequences for LSTM input.
        X: [dat[i], dat[i+1], ..., dat[i+seq-1]]
        y: dat[i+seq]
        """
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length, 0])
            y.append(dataset[i + seq_length, 0])
        return np.array(X), np.array(y)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
