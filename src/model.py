import tensorflow as tf
import boto3
import pandas as pd
import preprocessing  
import os
import pickle

# Create the S3 client
s3 = boto3.client('s3', region_name='eu-north-1')

def load_model_from_s3(bucket_name, s3_file_path, local_file_path):
    """Loads a TensorFlow Keras model from Amazon S3."""
    try:
        s3.download_file(bucket_name, s3_file_path, local_file_path)
        model = tf.keras.models.load_model(local_file_path)
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        return None

def save_model_to_s3(bucket_name, s3_file_path, local_file_path):
    """Saves a TensorFlow Keras model to Amazon S3."""
    try:
        s3.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f"Model saved to S3: {s3_file_path}")
    except Exception as e:
        print(f"Error saving model to S3: {e}")

def retrain_model_from_s3(bucket_name, s3_model_file, local_model_path, s3_test_csv, local_test_csv, s3_test_images, local_test_images):
    """Retrains the model using test data from S3."""
    try:
        loaded_model = load_model_from_s3(bucket_name, s3_model_file, local_model_path)

        # Download test CSV from S3
        s3.download_file(bucket_name, s3_test_csv, local_test_csv)
        test_csv = pd.read_csv(local_test_csv)

        # Download test images from S3 and preprocess
        image_paths = [os.path.join(local_test_images, filename) for filename in test_csv['filename']]
        for index, row in test_csv.iterrows():
            s3.download_file(bucket_name, f"{s3_test_images}{row['filename']}", f"{local_test_images}{row['filename']}")

        preprocessed_images, encoded_labels, label_encoder = preprocessing.preprocess_and_encode(local_test_csv, local_test_images)

        # Retrain the model
        retrained_model = loaded_model.fit(preprocessed_images, encoded_labels, epochs=10, batch_size=32)

        # Save the retrained model to S3
        retrained_model.model.save("/tmp/retrained_model.keras")
        save_model_to_s3(bucket_name, "models/retrained_model.keras", "/tmp/retrained_model.keras")

        # save label encoder to s3.
        with open('/tmp/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        save_model_to_s3(bucket_name, "models/label_encoder.pkl", "/tmp/label_encoder.pkl")

        return retrained_model

    except Exception as e:
        print(f"Error retraining model from S3: {e}")
        return None

def load_label_encoder_from_s3(bucket_name, s3_file_path, local_file_path):
    """Loads a LabelEncoder from Amazon S3."""
    try:
        s3.download_file(bucket_name, s3_file_path, local_file_path)
        with open(local_file_path, 'rb') as f:
            le = pickle.load(f)
        return le
    except Exception as e:
        print(f"Error loading LabelEncoder from S3: {e}")
        return None

def make_predictions(model, preprocessed_images):
    """Makes predictions using the loaded model."""
    try:
        predictions = model.predict(preprocessed_images)
        predicted_labels = tf.argmax(predictions, axis=1).numpy() #converts tensor to numpy array
        return predicted_labels
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None