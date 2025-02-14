import io
import os
import numpy as np
from PIL import Image
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import img_to_array

from src.ImageCaptionP import logger
from src.ImageCaptionP.config.configuration import ConfigurationManager
from src.ImageCaptionP.pipeline.prediction import ImageCaptionPredict

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration for file upload

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():

    logger.info('=============== Inside Prediction app.py =====================')

    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    logger.info("Image file loaded and moving to prediction")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # filepath
        filepath = os.path.join(image_prediction_config.root_dir, filename)
        # saving Image to desire path
        file.save(filepath)

        # Open the image file and preprocess it
        rawImage = Image.open(file.stream).convert("RGB")
        
        predict_text = pipeline.predict(rawImage,filename)

        return jsonify({'prediction': predict_text})

    return jsonify({'error': 'File not allowed'})

if __name__ == '__main__':
    # getting configurations for prediction
    config = ConfigurationManager()
    image_prediction_config = config.get_prediction_config()

    # Object of Prediction pipeline
    pipeline = ImageCaptionPredict(image_prediction_config)
    logger.info('=============== Prediction app.py started =====================')
    app.run(debug=True,host='0.0.0.0', port=8080)
