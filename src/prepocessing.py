import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image

def load_image(image_path, target_size=(128, 128)):
    """Loads and resizes an image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocesses a single image."""
    return load_image(image_path, target_size)

def preprocess_batch(image_paths, target_size=(128, 128)):
    """Preprocesses a batch of images."""
    images = []
    for path in image_paths:
        img_array = load_image(path, target_size)
        if img_array is not None:
            images.append(img_array)
    return np.array(images)

def load_data(csv_path, image_folder):
    """Loads data from a CSV file and returns image paths and labels."""
    try:
        df = pd.read_csv(csv_path)
        image_paths = [os.path.join(image_folder, filename) for filename in df['filename']]
        labels = df['class']  
        return image_paths, labels
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None, None

def encode_labels(labels):
    """Encodes labels using LabelEncoder."""
    try:
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        return le, encoded_labels
    except Exception as e:
        print(f"Error encoding labels: {e}")
        return None, None

def preprocess_and_encode(csv_path, image_folder):
    """Combines data loading, preprocessing, and label encoding."""
    image_paths, labels = load_data(csv_path, image_folder)
    if image_paths is None or labels is None:
        return None, None, None

    le, encoded_labels = encode_labels(labels)
    if le is None or encoded_labels is None:
        return None, None, None

    preprocessed_images = preprocess_batch(image_paths)
    return preprocessed_images, encoded_labels, le