import sqlite3
import os
import pandas as pd

# Database connection details 
DATABASE_FILE = "my_base.db"

def create_table():
    """Creates the 'images' table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE_FILE)
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
#create_table()

def insert_image_data(image_path, label):
    """Inserts image data and label into the database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cur = conn.cursor()

        # Read image data as bytes
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Insert data into the 'images' table
        cur.execute("INSERT INTO images (image_data, label) VALUES (?, ?);", (sqlite3.Binary(image_data), label))

        conn.commit()
        cur.close()
        conn.close()
        print(f"Image '{image_path}' inserted successfully.")

    except sqlite3.Error as e:
        print(f"Error inserting image: {e}")

def populate_database_from_csv(csv_path, images_dir):
    """Populates the database using a CSV and images directory."""
    try:
        df = pd.read_csv(csv_path)

        for index, row in df.iterrows():
            image_filename = row['filename']
            label = row['class']
            image_path = os.path.join(images_dir, image_filename)

            if os.path.exists(image_path):
                insert_image_data(image_path, label)
            else:
                print(f"Warning: Image '{image_path}' not found.")

    except Exception as e:
        print(f"Error populating database from CSV: {e}")

# # Example usage 
# train_csv_path = "dataset/train/_annotations.csv"
# train_images_dir = "dataset/train/images"

# test_csv_path = "dataset/test/_annotations.csv"
# test_images_dir = "dataset/test/images"

# valid_csv_path = "dataset/valid/_annotations.csv"
# valid_images_dir = "dataset/valid/images"

# populate_database_from_csv(train_csv_path, train_images_dir)
# populate_database_from_csv(test_csv_path, test_images_dir)
# populate_database_from_csv(valid_csv_path, valid_images_dir)

