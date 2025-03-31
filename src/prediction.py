import boto3
import tensorflow as tf
import os
import numpy as np
import preprocessing  

# Create the S3 client
s3 = boto3.client('s3', region_name='eu-north-1')

def load_model_from_s3(bucket_name, s3_file_path, local_file_path):
    """Loads a model from Amazon S3."""
    try:
        s3.download_file(bucket_name, s3_file_path, local_file_path)
        model = tf.keras.models.load_model(local_file_path)
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        return None

def make_predictions(model, preprocessed_images):
    """Makes predictions using the loaded model."""
    try:
        predictions = model.predict(preprocessed_images)
        predicted_labels = np.argmax(predictions, axis=1)
        return predicted_labels
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

bucket_name = "theosummative"
s3_model_file = "models/second_model.keras"
local_model_path = "/tmp/model.keras"  # inside the docker container

loaded_model = load_model_from_s3(bucket_name, s3_model_file, local_model_path)

# Receive image path from user (or API)
# Replace this with actual method of getting the image path.
s3_image_path = "data/test/images/20190207_172525_jpg.rf.f8dbadf227b82ad4b5caf737ed904a06.jpg" # This is a placeholder.

# Download test image from S3
local_image_path = "/tmp/test_image.jpg"
s3.download_file(bucket_name, s3_image_path, local_image_path)

# Preprocess the image using my preprocessing.py functions
preprocessed_image = preprocessing.preprocess_image(local_image_path)
preprocessed_images = np.expand_dims(preprocessed_image, axis=0) 

# Make predictions
predictions = make_predictions(loaded_model, preprocessed_images)
print(predictions)