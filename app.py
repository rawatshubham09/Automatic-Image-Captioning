import io
import os
import bcrypt
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, url_for, redirect, session, send_from_directory, abort
from tensorflow.keras.preprocessing.image import img_to_array

from src.ImageCaptionP import logger
from src.ImageCaptionP.config.configuration import ConfigurationManager
from src.ImageCaptionP.pipeline.prediction import ImageCaptionPredict

load_dotenv()

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# getting configurations for prediction and frontend
config = ConfigurationManager()

image_prediction_config = config.get_prediction_config()
frontend_config = config.get_frountend_config()

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# important links
IMG_FOLDER = frontend_config.image_folder
ARTIFACTS_DIR = frontend_config.artifact_dir

app.config["UPLOAD_FOLDER"] = IMG_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("SQLALCHEMY_DATABASE_URI")
app.secret_key = os.getenv("SECRET_KEY")

# Initialize SQL Alchemy
db = SQLAlchemy(app)

# Creating User Database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

# Configuration for file upload

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@cross_origin()
def index():
    images = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))] # only get images.
    return render_template('index.html', images=images, enumerate=enumerate)

@app.route("/topredict")
@cross_origin()
def to_predict():
    if "name" not in session:
        return redirect(url_for("login"))
    return render_template('predict.html',name=session["name"])

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if "name" not in session:
        return redirect(url_for("login"))

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

@app.route('/register', methods=['GET', 'POST'])
@cross_origin()
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        print("{name}: {email}: {password}")

        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match")

        new_user = User(name=name, email=email, password=password)

        db.session.add(new_user)
        db.session.commit()

        # Explicitly modify and save the session
        session["name"] = new_user.name
        session["email"] = new_user.email
        session.modified = True  # Mark the session as modified

        return redirect(url_for("to_predict"))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
@cross_origin()
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            print("Password check successful") #add this line
            session["name"] = user.name
            session["email"] = user.email
            return redirect(url_for("to_predict"))
        else:
            print("Password check failed") #add this line
            return render_template('login.html', error="Invalid User")

    return render_template('login.html')

@app.route('/logout')
@cross_origin()
def logout():
    session.pop("name", None)
    session.pop("email", None)
    return redirect(url_for("login"))

@app.route('/dashboard')
@cross_origin()
def dashboard():
    if "name" in session:
        folders = [f for f in os.listdir(ARTIFACTS_DIR) if os.path.isdir(os.path.join(ARTIFACTS_DIR, f))]
        return render_template("dashboard.html", name=session["name"], folders=folders)
    return redirect(url_for("login"))

@app.route('/logs')
@cross_origin()
def display_logs():
    if "name" in session:
        log_path = frontend_config.log_file_path
        with open(log_path, 'r') as file:
            log_content = file.read()
        return render_template('logs.html', log_content=log_content)
    return redirect(url_for("login"))

def create_artifact_routes():
    if not os.path.exists(ARTIFACTS_DIR):
        return

    for folder_name in os.listdir(ARTIFACTS_DIR):
        folder_path = os.path.join(ARTIFACTS_DIR, folder_name)
        if os.path.isdir(folder_path):
            # Generate unique endpoint names
            download_endpoint = f'download_{folder_name}'
            list_endpoint = f'list_{folder_name}'

            @app.route(f'/{folder_name}/<filename>', endpoint=download_endpoint)
            @cross_origin()
            def artifact_download(filename, folder_name=folder_name):
                artifact_path = os.path.join(ARTIFACTS_DIR, folder_name, filename)
                if os.path.isfile(artifact_path):
                    return send_from_directory(os.path.join(ARTIFACTS_DIR, folder_name), filename, as_attachment=True)
                else:
                    abort(404)

            @app.route(f'/{folder_name}', endpoint=list_endpoint)
            @cross_origin()
            def artifact_list(folder_name=folder_name):
                artifact_folder_path = os.path.join(ARTIFACTS_DIR, folder_name)
                files = [f for f in os.listdir(artifact_folder_path) if os.path.isfile(os.path.join(artifact_folder_path, f))]
                return render_template("artifact_list.html", folder_name=folder_name, files=files)

create_artifact_routes()

@app.route('/train')
@cross_origin()
def train():
    if "name" in session:
        return render_template('train.html', )
    return redirect(url_for("login"))

@app.route('/download/<filename>')
@cross_origin()
def download(filename):
    return send_from_directory("artifacts", filename, as_attachment=True)

@app.route('/dashboard/default-train',methods=['POST'])
@cross_origin()
def defaultTrain():
    if "name" in session:
        if request.method == "POST":
            #os.system("python main.py")
            # os.system("dvc repro")
            print("train successfully")
            return redirect(url_for("display_logs"))
        return render_template('train.html', )
    return redirect(url_for("login"))

@app.route('/dashboard/custom-train', methods=['GET', 'POST'])
@cross_origin()
def customeTrain():
    if "name" in session:
        if request.method == 'POST':
            mongo_link = request.form.get('mongoLink')
            epochs = request.form.get('epochs')
            batch_size = request.form.get('batchSize')
            test_train_split = request.form.get('testTrainSplit')

            # Process the form data (e.g., run your training script)
            print("Custom Train Parameters:")
            print(f"MongoDB Link: {mongo_link}")
            print(f"Epochs: {epochs}")
            print(f"Batch Size: {batch_size}")
            print(f"Test-Train Split: {test_train_split}")

            # Add your training logic here using the received parameters.
            # Example: os.system(f"python train.py --mongo {mongo_link} --epochs {epochs} ...")

            return "Training started with custom parameters." # Or render a result page.

        return render_template("custome_train.html")
    return redirect(url_for("login"))

if __name__ == '__main__':

    # Object of Prediction pipeline
    pipeline = ImageCaptionPredict(image_prediction_config)
    logger.info('=============== Prediction app.py started =====================')
    app.run(debug=True,host='0.0.0.0', port=8080)
