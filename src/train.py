from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Trains the LSTM model with callbacks.
    """
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    if X_val is not None:
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val) if X_val is not None else None,
        callbacks=callbacks,
        verbose=1
    )
    return history
