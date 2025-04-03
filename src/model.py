import tensorflow as tf
import boto3
import pandas as pd
from . import preprocessing  
import os
import pickle
import tempfile
import sqlite3
from sklearn.metrics import classification_report

# Create the S3 client
s3 = boto3.client('s3', region_name='eu-north-1')

# Database Configuration
DATABASE_FILE = "my_base.db"

def load_model_from_s3(bucket_name, s3_file_path, local_file_path):
    """Loads a TensorFlow Keras model from Amazon S3."""
    try:
        s3.download_file(bucket_name, s3_file_path, local_file_path)
        print(f"Model downloaded to: {local_file_path}") 
        if os.path.exists(local_file_path): 
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

def insert_retrain_image_data(image_path, label, data_type='retrain'):
    """Inserts retrain image data and label into the database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cur = conn.cursor()

        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        cur.execute("INSERT INTO images (image_data, label, data_type) VALUES (?, ?, ?);", (sqlite3.Binary(image_data), label, data_type))

        conn.commit()
        conn.close()
        print(f"Retrain image '{image_path}' inserted successfully.")

    except sqlite3.Error as e:
        print(f"Error inserting retrain image: {e}")

def populate_retrain_database_from_csv(csv_path, images_dir):
    """Populates the database with retrain data using a CSV and images directory."""
    try:
        df = pd.read_csv(csv_path)

        for index, row in df.iterrows():
            image_filename = row['filename']
            label = row['class']
            image_path = os.path.join(images_dir, image_filename)

            if os.path.exists(image_path):
                insert_retrain_image_data(image_path, label)
            else:
                print(f"Warning: Retrain image '{image_path}' not found.")

    except Exception as e:
        print(f"Error populating retrain database from CSV: {e}")

def get_retrain_data_from_db(database_file):
    """Retrieves all retraining data from the database."""
    conn = sqlite3.connect(database_file)
    cur = conn.cursor()
    cur.execute("SELECT image_data, label FROM images WHERE data_type = 'retrain';")
    retrain_data = cur.fetchall()
    conn.close()
    return retrain_data

def retrain_model_from_db(database_file, bucket_name, s3_model_file, local_model_path):
    """Retrains the model using data from the database and returns metrics."""
    try:
        loaded_model = load_model_from_s3(bucket_name, s3_model_file, local_model_path)

        retrain_data = get_retrain_data_from_db(database_file)

        # Process retrain_data
        local_test_images = os.path.join(tempfile.gettempdir(), "retrain_images")
        os.makedirs(local_test_images, exist_ok=True)

        csv_data = []
        for index, row in enumerate(retrain_data):
            image_filename = f"image_{index}.jpg"
            image_path = os.path.join(local_test_images, image_filename)
            with open(image_path, "wb") as f:
                f.write(row[0]) #image data
            csv_data.append({"filename": image_filename, "class": row[1]}) #label

        local_test_csv = os.path.join(tempfile.gettempdir(), "retrain_data.csv")
        pd.DataFrame(csv_data).to_csv(local_test_csv, index=False)

        preprocessed_images, encoded_labels, label_encoder = preprocessing.preprocess_and_encode(local_test_csv, local_test_images)

        # Retrain the model
        retrained_model = loaded_model.fit(preprocessed_images, encoded_labels, epochs=10, batch_size=32)

        # Evaluate the model
        evaluation_metrics = loaded_model.evaluate(preprocessed_images, encoded_labels) 

        # Save the retrained model to S3
        retrained_model_local_path = os.path.join(tempfile.gettempdir(), "retrained_model.keras")
        retrained_model.model.save(retrained_model_local_path)
        save_model_to_s3(bucket_name, "models/retrained_model.keras", retrained_model_local_path)

        # save label encoder to s3.
        label_encoder_local_path = os.path.join(tempfile.gettempdir(), "label_encoder.pkl")
        with open(label_encoder_local_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        save_model_to_s3(bucket_name, "models/label_encoder.pkl", label_encoder_local_path)

        # Generate classification report
        predicted_labels = loaded_model.predict(preprocessed_images)
        predicted_labels = tf.argmax(predicted_labels, axis=1).numpy()
        decoded_labels = label_encoder.inverse_transform(encoded_labels)
        decoded_predicted_labels = label_encoder.inverse_transform(predicted_labels)
        report = classification_report(decoded_labels, decoded_predicted_labels)

        return {"loss": evaluation_metrics[0], "accuracy": evaluation_metrics[1], "report": report}

    except Exception as e:
        print(f"Error retraining model: {e}")
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