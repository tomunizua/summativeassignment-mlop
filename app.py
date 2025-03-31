from flask import Flask, jsonify
import model
import boto3

app = Flask(__name__)

bucket_name = "theosummative"
s3_model_file = "models/second_model.keras"
local_model_path = "/tmp/second_model.keras"
s3_test_csv = "data/test/_annotations.csv"
local_test_csv = "/tmp/test.csv"
s3_test_images = "data/test/images/"
local_test_images = "/tmp/test/images/"

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        retrained_model = model.retrain_model_from_s3(bucket_name, s3_model_file, local_model_path, s3_test_csv, local_test_csv, s3_test_images, local_test_images)
        if retrained_model:
            return jsonify({'message': 'Retraining successful'})
        else:
            return jsonify({'message': 'Retraining failed'})
    except Exception as e:
        return jsonify({'message': f'Retraining error: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')