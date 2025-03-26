import tensorflow as tf
import numpy as np

def load_model(model_path):
    """Loads a TensorFlow Keras model from the given path."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def make_predictions(model, preprocessed_images):
    """Makes predictions using the loaded model.

    Args:
        model: The loaded TensorFlow Keras model.
        preprocessed_images: A NumPy array of preprocessed images.

    Returns:
        A NumPy array of predictions (class labels).
    """
    try:
        predictions = model.predict(preprocessed_images)
        predicted_labels = np.argmax(predictions, axis=1) #get the highest probability
        return predicted_labels
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None