# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/preprocessing.py .
COPY src/prediction.py .
COPY src/model.py .
COPY app.py .

# Set the environment variable for the model path
ENV MODEL_PATH /app/models/second_model.keras

# Command to run 
CMD ["python", "app.py"]