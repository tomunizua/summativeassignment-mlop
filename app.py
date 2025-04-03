import os
from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from src import preprocessing
from src import model
import tempfile
import boto3
from PIL import Image
import io
import zipfile
import sqlite3
from tabulate import tabulate 
import threading
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global boto3 client
try:
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_DEFAULT_REGION'),
    )
except Exception as e:
    print(f"Error initializing boto3 client: {e}")
    s3_client = None

# Database Configuration
DATABASE_FILE = "my_base.db"

from database import create_table
# # Create the 'images' table if it doesn't exist
# create_table()

# Model Loading (from S3)
bucket_name = "theosummative"
s3_model_file = "models/second_model.keras"
temp_dir = tempfile.gettempdir()
local_model_path = os.path.join(temp_dir, "model.keras")
loaded_model = model.load_model_from_s3(bucket_name, s3_model_file, local_model_path)
print(f"Model loaded successfully: {loaded_model is not None}")

# Define label mapping
label_map = {
    0: "bxw",
    1: "healthy",
}

def get_db_connection():
    """Establishes a database connection."""
    try:
        temp_dir = tempfile.gettempdir()
        local_db_path = os.path.join(temp_dir, "my_base.db")

        # Check if the database file exists locally
        if not os.path.exists(local_db_path):
            # Download from S3 if it doesn't exist
            print("Downloading database from S3...")
            s3.download_file(bucket_name, DATABASE_FILE, local_db_path)
            print("Database downloaded from S3.")

        conn = sqlite3.connect(local_db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def get_image_from_db(image_id):
    """Retrieves image data from the database."""
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT image_data FROM images WHERE ROWID = ?;", (image_id,))

        row = cur.fetchone()
        if row:
            image_data = row[0]
            return image_data
        else:
            return None  # Return None if no matching row is found

    except sqlite3.Error as e:
        print(f"Error retrieving image from database: {e}")
        return None

    finally:
        if conn:
            conn.close()

def upload_database_to_s3():
    """Uploads the database to S3."""
    try:
        local_db_path = os.path.join(tempfile.gettempdir(), DATABASE_FILE)
        s3.upload_file(local_db_path, bucket_name, DATABASE_FILE)
        print("Database uploaded to S3.")
    except Exception as e:
        print(f"Error uploading database to S3: {e}")
        time.sleep(5)  # Wait for 5 seconds before retrying
        upload_database_to_s3() #retry the upload.

@app.route('/image/<image_id>', methods=['GET'])
def get_image(image_id):
    """Serves image data from the database."""
    image_data = get_image_from_db(image_id)
    if image_data is None:
        return jsonify({'error': 'Image not found'}), 404

    try:
        # Create a temporary file to hold the image data
        temp_dir = tempfile.gettempdir()
        local_image_path = os.path.join(temp_dir, f"image_{image_id}.jpg")

        with open(local_image_path, "wb") as f:
            f.write(image_data)

        # Send the file as a response
        return send_file(local_image_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    try:
        if 'image' in request.files:
            # Image uploaded via file upload
            image_file = request.files['image']
            try:
                img = Image.open(io.BytesIO(image_file.read()))
                temp_dir = tempfile.gettempdir()
                local_image_path = os.path.join(temp_dir, "uploaded_image.jpg")
                img.save(local_image_path)

                preprocessed_image = preprocessing.preprocess_image(local_image_path)
                preprocessed_images = np.expand_dims(preprocessed_image, axis=0)

                predictions = model.make_predictions(loaded_model, preprocessed_images)
                predicted_label = int(predictions[0])
                predicted_class = label_map.get(predicted_label, "Unknown")

                return jsonify({'prediction': predicted_class})

            except Exception as e:
                return jsonify({'error': f'Error processing uploaded image: {str(e)}'}), 500

        else:
            return jsonify({'error': 'No image provided'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/predict_lib', methods=['POST'])
def predict_lib():
    try:
        data = request.get_json()
        image_id = data['image_id']

        image_data = get_image_from_db(image_id)
        if image_data is None:
            return jsonify({'error': 'Image not found'}), 404

        # Convert image_data to a format that preprocessing can handle
        temp_dir = tempfile.gettempdir()
        local_image_path = os.path.join(temp_dir, "predicted_image.jpg")

        try:
            with open(local_image_path, "wb") as f:
                f.write(image_data)
            print(f"Image saved to: {local_image_path}") 
            if not os.path.exists(local_image_path): 
                return jsonify({'error': 'Failed to save image data'}), 500

        except Exception as save_error:
            print(f"Error saving image data: {save_error}")
            return jsonify({'error': 'Error saving image data'}), 500

        preprocessed_image = preprocessing.preprocess_image(local_image_path)
        preprocessed_images = np.expand_dims(preprocessed_image, axis=0)

        predictions = model.make_predictions(loaded_model, preprocessed_images)
        predicted_label = int(predictions[0])
        predicted_class = label_map.get(predicted_label, "Unknown")

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

retraining_status = {}  # Dictionary to store retraining status

def retrain_and_monitor(retrain_id, zip_file_path):
    """Retrains the model and updates the retraining status."""
    retraining_status[retrain_id] = {"status": "retraining", "progress": 0, "message": "Retraining started"}
    try:
        # Extract images from the zip file
        temp_dir = os.path.join(tempfile.gettempdir(), f"retrain_images_{retrain_id}")
        os.makedirs(temp_dir, exist_ok=True)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Retrain the model using the extracted images
        metrics = model.retrain_model_from_folder(temp_dir, bucket_name, s3_model_file, local_model_path) #change to your function.

        global loaded_model
        loaded_model = model.load_model_from_s3(bucket_name, "models/retrained_model.keras", os.path.join(tempfile.gettempdir(), "retrained_model.keras"))

        retraining_status[retrain_id] = {"status": "completed", "progress": 100, "message": "Retraining completed", "metrics": metrics}

    except Exception as e:
        retraining_status[retrain_id] = {"status": "failed", "progress": 0, "message": f"Retraining failed: {e}"}

@app.route('/upload_retrain_data', methods=['POST'])
def upload_retrain_data():
    """Uploads a zip file with images and _annotations.csv and saves data to the database."""
    
    try:
        create_table() #Create table before processing the upload.
        if 'zip_file' not in request.files:
            return jsonify({'error': 'No zip file uploaded'}), 400

        zip_file = request.files['zip_file']
        if zip_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        temp_dir = tempfile.gettempdir()
        extracted_dir = os.path.join(temp_dir, 'retrain_data_extracted')
        os.makedirs(extracted_dir, exist_ok=True)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)

        images_dir = os.path.join(extracted_dir, 'images')
        csv_path = os.path.join(extracted_dir, '_annotations.csv')

        if os.path.exists(csv_path) and os.path.exists(images_dir):
            model.populate_retrain_database_from_csv(csv_path, images_dir)
            upload_database_to_s3()
            return jsonify({'message': 'Data uploaded to database successfully'}), 200
        else:
            return jsonify({'error': 'Missing images folder or _annotations.csv in zip file'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain_status/<retrain_id>', methods=['GET'])
def get_retrain_status(retrain_id):
    """Gets the status of a retraining process."""
    if retrain_id not in retraining_status:
        return jsonify({'error': 'Retraining process not found'}), 404
    return jsonify(retraining_status[retrain_id])
    
@app.route('/retrain', methods=['POST'])
def retrain():
    """Triggers retraining using data from the database and returns metrics."""
    try:
        metrics = model.retrain_model_from_db(DATABASE_FILE, bucket_name, s3_model_file, local_model_path)
        global loaded_model
        loaded_model = model.load_model_from_s3(bucket_name, "models/retrained_model.keras", os.path.join(tempfile.gettempdir(), "retrained_model.keras"))

        report = metrics['report']
        accuracy = metrics['accuracy']
        loss = metrics['loss']

        return jsonify({'message': 'Retraining completed', 'metrics': {
            'accuracy': accuracy,
            'loss': loss,
            'report': report
        }})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, host='0.0.0.0')