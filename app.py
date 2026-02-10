import os
import numpy as np
import pickle
import cv2
from flask import Flask, request, render_template, url_for
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model and label binarizer
print("[INFO] Loading model and label binarizer...")
try:
    model = pickle.load(open('cnn_model.pkl', 'rb'))
    lb = pickle.load(open('label_transform.pkl', 'rb'))
    print("[INFO] Model and label binarizer loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load model/labels: {e}")
    # Create dummy objects to allow app to start (will fail on predict)
    model = None
    lb = None

def preprocess_image(image_path):
    default_image_size = tuple((256, 256))
    image = cv2.imread(image_path)
    image = cv2.resize(image, default_image_size)
    image = img_to_array(image)
    # Normalize using standard 0-1 range (divide by 255.0)
    np_image_list = np.array([image], dtype=np.float32) / 255.0
    return np_image_list

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    top_predictions = []
    supported_diseases = list(lb.classes_) if lb else []
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part", supported_diseases=supported_diseases)
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file", supported_diseases=supported_diseases)
            
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            image_url = url_for('static', filename=f'uploads/{filename}')
            
            if model and lb:
                try:
                    processed_image = preprocess_image(file_path)
                    
                    # Predict
                    prediction_prob = model.predict(processed_image, verbose=0)
                    
                    # Get the class label
                    idx = np.argmax(prediction_prob)
                    label = lb.classes_[idx]
                    confidence = prediction_prob[0][idx] * 100
                    
                    # Only show prediction if confidence is above 30%
                    if confidence >= 30:
                        prediction = f"{label} ({confidence:.2f}%)"
                    else:
                        prediction = f"Low Confidence: {label} ({confidence:.2f}%) - Result may be inaccurate"
                    
                    # Get top 5 predictions
                    top_indices = prediction_prob[0].argsort()[-5:][::-1]
                    for i in top_indices:
                        top_predictions.append({
                            'label': lb.classes_[i],
                            'confidence': float(prediction_prob[0][i] * 100)
                        })
                        
                except Exception as e:
                    print(f"[ERROR] Prediction error: {str(e)}")
                    prediction = f"Error during prediction: {str(e)}"
            else:
                prediction = "Model not loaded properly. Please run: python PlantDiseaseDetection.py"

    return render_template('index.html', prediction=prediction, top_predictions=top_predictions, image_url=image_url, supported_diseases=supported_diseases)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
