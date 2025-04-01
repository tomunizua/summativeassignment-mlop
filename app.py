import os
from flask import Flask, request, jsonify
import sqlite3
import tensorflow as tf
import numpy as np
from src import preprocessing
from src import model
import pandas as pd
import pickle
import tempfile

app = Flask(__name__)

# SQLite Database Configuration
DATABASE_FILE = "my_base.db"

# Model Loading (from S3)
bucket_name = "theosummative"
s3_model_file = "models/second_model.keras"
temp_dir = tempfile.gettempdir()
local_model_path = os.path.join(temp_dir, "model.keras")
loaded_model = model.load_model_from_s3(bucket_name, s3_model_file, local_model_path)

# Define label mapping
label_map = {
    0: "bxw",
    1: "healthy",
}

def get_db_connection():
    """Establishes a database connection."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def create_table():
    """Creates the 'images' table if it doesn't exist."""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            print("Creating table 'images'...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_data BLOB NOT NULL,
                    label TEXT NOT NULL,
                    data_type TEXT DEFAULT 'train'
                );
            """)
            conn.commit()
            conn.close()
            print("Table 'images' created or already exists.")
        except sqlite3.Error as e:
            print(f"Error creating table: {e}")
    else:
        print("Could not connect to database")
create_table()

def get_image_from_db(image_id):
    """Retrieves image data from the database."""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT image_data FROM images WHERE id = ?;", (image_id,))
            image_data = cur.fetchone()[0]
            conn.close()
            return image_data
        except sqlite3.Error as e:
            print(f"Error retrieving image from database: {e}")
            return None

@app.route('/predict', methods=['POST'])
def predict():
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

@app.route('/retrain', methods=['POST'])
def retrain():
    """Triggers retraining using data from the database and returns metrics."""
    try:
        metrics = model.retrain_model_from_db(DATABASE_FILE, bucket_name, s3_model_file, local_model_path)
        global loaded_model
        loaded_model = model.load_model_from_s3(bucket_name, "models/retrained_model.keras", os.path.join(tempfile.gettempdir(), "retrained_model.keras"))

        return jsonify({'message': 'Retraining completed', 'metrics': metrics})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, host='0.0.0.0')
