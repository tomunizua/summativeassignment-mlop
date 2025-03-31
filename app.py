from flask import Flask, request, jsonify
import boto3
import tensorflow as tf
import numpy as np
import preprocessing
import os

app = Flask(__name__)

# Create the S3 client 
s3 = boto3.client('s3', region_name='eu-north-1')

# Load model from S3 (same as in prediction.py)
def load_model_from_s3(bucket_name, s3_file_path, local_file_path):
    try:
        s3.download_file(bucket_name, s3_file_path, local_file_path)
        model = tf.keras.models.load_model(local_file_path)
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        return None

# Make predictions (same as in prediction.py)
def make_predictions(model, preprocessed_images):
    try:
        predictions = model.predict(preprocessed_images)
        predicted_labels = np.argmax(predictions, axis=1)
        return predicted_labels
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

bucket_name = "theosummative"
s3_model_file = "models/second_model.keras"
local_model_path = "/tmp/model.keras"
loaded_model = load_model_from_s3(bucket_name, s3_model_file, local_model_path)

# Add the label mapping
label_map = {
    0: "bxw",
    1: "healthy",
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        s3_image_path = data['image_path']

        local_image_path = "/tmp/user_image.jpg"
        s3.download_file(bucket_name, s3_image_path, local_image_path)

        preprocessed_image = preprocessing.preprocess_image(local_image_path)
        preprocessed_images = np.expand_dims(preprocessed_image, axis=0)

        predictions = make_predictions(loaded_model, preprocessed_images)
        predicted_label = int(predictions[0])
        predicted_class = label_map.get(predicted_label, "Unknown")

        return jsonify({'prediction': predicted_class}) # return class name

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')