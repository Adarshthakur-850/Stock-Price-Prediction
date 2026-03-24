import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def evaluate_model(model, X_test, y_test, preprocessor):
    """
    Evaluates the model and returns metrics.
    """
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    predictions = model.predict(X_test_reshaped)

    predictions_actual = preprocessor.inverse_transform(predictions)
    y_test_reshaped = y_test.reshape(-1, 1)
    y_test_actual = preprocessor.inverse_transform(y_test_reshaped)

    rmse = math.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    mae = mean_absolute_error(y_test_actual, predictions_actual)

    return rmse, mae, predictions_actual, y_test_actual
