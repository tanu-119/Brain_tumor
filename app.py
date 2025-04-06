from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

app = Flask(__name__)

IMG_SIZE = (128, 128)
label_mapping = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "Pituitary Tumor",
    3: "No Tumor"
}

# Load the model with error handling
model_path = os.path.join("model", "brain_tumor_model.h5")
try:
    model = load_model(model_path, compile=False)
except Exception as e:
    print(f"Error loading model with standard method: {e}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully with tf.keras.models.load_model")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def get_mask_image(mask_array):
    mask = (mask_array.squeeze() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask).convert("L")
    buffered = io.BytesIO()
    mask_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('braintumor.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    try:
        img = preprocess_image(file)
        seg_pred, clf_pred = model.predict(img)
        tumor_type = label_mapping[np.argmax(clf_pred)]
        mask_b64 = get_mask_image(seg_pred[0])

        return jsonify({
            'tumor_type': tumor_type,
            'mask_image': f"data:image/png;base64,{mask_b64}",
            'confidence': float(np.max(clf_pred))  # Add confidence score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
