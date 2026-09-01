import base64
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Locate model in the root directory
model_path = os.path.join(os.path.dirname(__file__), "..", "best_preprocessed.pt")
model = YOLO(model_path)

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    img_bytes = file.read()

    # Convert image bytes to OpenCV format
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model.predict(source=img, conf=0.25)
    annotated_frame = results[0].plot()

    # Encode annotated image as base64 string
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"image": f"data:image/jpeg;base64,{encoded_img}"})

if __name__ == '__main__':
    app.run(debug=True)
