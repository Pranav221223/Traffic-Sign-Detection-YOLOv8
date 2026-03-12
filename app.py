import os
import uuid # To generate unique filenames for predicted images
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np # Ultralytics plot returns a numpy array

# --- Flask Configuration ---
app = Flask(__name__)
# Define upload folder - where original images are temporarily saved
UPLOAD_FOLDER = 'uploads'
# Define folder within static to serve predicted images
# Make sure this folder exists or is created
PREDICTED_FOLDER_NAME = 'predictions'
PREDICTED_FOLDER_PATH = os.path.join('static', PREDICTED_FOLDER_NAME)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTED_FOLDER_PATH'] = PREDICTED_FOLDER_PATH
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max upload size: 16MB

# Ensure upload and predicted folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTED_FOLDER_PATH'], exist_ok=True)


# --- Model Loading ---
# Load the YOLOv8 model once when the application starts
# CHANGE THIS PATH to your actual best.pt file
TRAINED_MODEL_PATH = 'runs/train3/weights/best.pt' # <--- CHANGE THIS

try:
    print(f"Loading model from: {TRAINED_MODEL_PATH}")
    model = YOLO(TRAINED_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None # Set model to None if loading fails

# --- Routes ---

@app.route('/')
def index():
    """Renders the image upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    """Handles image upload, runs prediction, and displays results."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url) # Or render an error template

        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(request.url) # Or render an error template

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Use a unique filename to save the uploaded image temporarily
            unique_filename_input = str(uuid.uuid4()) + '_' + filename
            filepath_input = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename_input)
            file.save(filepath_input)

            if model:
                print(f"Running prediction on: {filepath_input}")
                try:
                    # Perform inference
                    # conf: confidence threshold, iou: IoU threshold for NMS
                    results = model.predict(filepath_input, conf=0.25, iou=0.7)

                    # Get the result for the first (and likely only) image
                    r = results[0]

                    # Get the image array with detections drawn
                    im_array = r.plot() # Returns numpy array (BGR format)

                    # Convert numpy array to a savable format (e.g., PNG)
                    # Use a unique filename for the output image
                    unique_filename_output = str(uuid.uuid4()) + '_predicted_' + filename.rsplit('.', 1)[0] + '.png'
                    filepath_output = os.path.join(app.config['PREDICTED_FOLDER_PATH'], unique_filename_output)

                    # Save the output image
                    # cv2.imwrite expects BGR, which r.plot() provides
                    cv2.imwrite(filepath_output, im_array)
                    print(f"Predicted image saved to: {filepath_output}")

                    # Generate the URL path relative to the static folder
                    predicted_image_url = url_for('static', filename=f'{PREDICTED_FOLDER_NAME}/{unique_filename_output}')

                    # Render the results template, passing the image URL
                    return render_template('results.html', predicted_image_url=predicted_image_url)

                except Exception as e:
                    print(f"Error during prediction: {e}")
                    # You might want to render an error template here
                    return render_template('results.html', error="Error processing image.")

            else:
                # Model failed to load on startup
                return render_template('results.html', error="Model not loaded. Please check logs.")

        else:
             # File type not allowed
             return render_template('results.html', error="Invalid file type. Allowed types are: png, jpg, jpeg, gif.")

    # If somehow a GET request comes to /predict
    return redirect(url_for('index'))

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Running the app ---
if __name__ == '__main__':
    # Use debug=True for development. Set to False for production.
    app.run(debug=True)