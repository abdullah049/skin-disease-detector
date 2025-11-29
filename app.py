from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

# Load model and classes
model = tf.keras.models.load_model('skin_disease_model.h5')
with open('class_names.json') as f:
    class_names = json.load(f)

def clean_name(name):
    return name.replace('BA-', '').replace('FU-', '').replace('PA-', '').replace('VI-', '').replace('-', ' ').title()

def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)[0]
    confidence = float(np.max(predictions))
    idx = np.argmax(predictions)
    disease_code = class_names[idx]
    disease_name = clean_name(disease_code)
    return disease_name, confidence

@app.route('/')
def home():
    return '''
    <h1>Skin Disease Detector</h1>
    <p>Upload an image to /predict</p>
    <form method="POST" enctype="multipart/form-data" action="/predict">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Check Disease</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filepath = 'temp.jpg'
    file.save(filepath)
    
    try:
        disease, confidence = predict_image(filepath)
        os.remove(filepath)
        return jsonify({
            'disease': disease,
            'confidence': round(confidence * 100, 1),
            'message': f"{disease} detected ({confidence*100:.1f}% confidence)"
        })
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)