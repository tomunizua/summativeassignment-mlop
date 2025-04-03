# BXW Model Deployment Project

## Project Description

This application provides a user-friendly interface for image prediction and model retraining. Users can upload images for prediction or select images from a library. Additionally, the application allows users to retrain the model by uploading a ZIP file containing new training data. Retraining metrics are displayed upon completion, providing insights into the new model's performance.

## Video Demo

[YouTube Demo Link](https://youtu.be/S50ycIdLc3w)

## Usage Instructions

### Frontend Usage

1.  **Access the Web Application:**
    * Open the following URL in your web browser: [https://curious-pegasus-f40b0f.netlify.app/] (or open `index.html` locally - API is not deployed so the web link only shows a non-functional UI).

2.  **Image Prediction:**
    * **Upload Image:** Click "Choose File" under the "Prediction" section to upload an image.
    * **Select from Library:** Enter an image number (1-1400) in the "Select from Library" field.
    * Click "Predict" to get the model's prediction.
    * The predicted result will be displayed below the image preview.

3.  **Model Retraining:**
    * Click "Choose File" under the "Retrain Model" section to upload a ZIP file containing new training images.
    * Click "Start Retrain" to initiate the retraining process.
    * Retraining metrics will be displayed after the process completes.

### Backend API Usage (Local Setup)

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/tomunizua/summativeassignment-mlop.git
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask Application:**

    ```bash
    python app.py
    ```

    * The API will be running at `http://127.0.0.1:5000` (or the port you configured).

5.  **API Endpoints:**

    * **`/predict_upload` (POST):**
        * Upload an image file for prediction.
        * Use `multipart/form-data` with the file field named `image`.
        * Response: JSON with the prediction result.
    * **`/predict_lib` (POST):**
        * Provide an image ID from the library for prediction.
        * Use `application/json` with the request body `{"image_id": image_id}`.
        * Response: JSON with the prediction result.
    * **`/upload_retrain_data` (POST):**
        * Upload a ZIP file containing retraining images.
        * Use `multipart/form-data` with the file field named `zip_file`.
        * Response: JSON with retrain id, used for monitoring.
    * **`/retrain` (POST):**
        * Starts the retrain process with the uploaded zip file.
        * Response: JSON with retrain id, used for monitoring.
    * **`/retrain_status/<retrain_id>` (GET):**
        * Checks the retrain status.
        * Response: JSON with status, progress, and metrics.

### Important Notes

* Ensure the backend API is running before using the frontend.
* This is a basic example and might need adjustments based on your specific requirements and environment.