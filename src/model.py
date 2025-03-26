import tensorflow as tf

def load_model(model_path):
    """Loads a TensorFlow Keras model from the given path."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# If you need to retrain the model, add a retrain_model function here
def retrain_model(model, X_train, y_train, epochs=10, batch_size=32):
    """Retrains the model with new data.

    Args:
        model: The model to retrain.
        X_train: NumPy array of training images.
        y_train: NumPy array of training labels.
        epochs: Number of retraining epochs.
        batch_size: Batch size for retraining.
    """
    try:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        print("Model retrained successfully.")
        return model
    except Exception as e:
        print(f"Error retraining model: {e}")
        return None
