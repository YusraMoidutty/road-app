import base64
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global variable to lazy-load the model
model = None

def get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        # Path matching your repository file name: best_preprocessed.pt
        model_path = os.path.join(os.path.dirname(__file__), "..", "best_preprocessed.pt")
        model = YOLO(model_path)
    return model

@app.route('/api/detect', methods=['GET', 'POST'])
def detect():
    # Handle simple GET requests to prevent 500 errors
    if request.method == 'GET':
        return jsonify({"status": "Backend running"}), 200

    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    img_bytes = file.read()

    # Convert image bytes to OpenCV format
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Lazy-load model and run inference
    yolo_model = get_model()
    results = yolo_model.predict(source=img, conf=0.25)
    annotated_frame = results[0].plot()

    # Encode annotated image to Base64
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"image": f"data:image/jpeg;base64,{encoded_img}"})

if __name__ == '__main__':
    app.run()
