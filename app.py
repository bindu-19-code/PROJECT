import torch
import cv2
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

# Create upload folder if not exists
UPLOAD_FOLDER = "static/uploads/"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the pre-trained YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

def load_and_preprocess_image(image_path):
    """ Load and preprocess the image for YOLOv5 """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_resized = cv2.resize(img, (640, 640))  # Resize to fit YOLO input
    return img, img_resized

@app.route('/')
def index():
    """ Render the upload page """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """ Handle image upload and perform object detection """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Load image for YOLO
    img, img_resized = load_and_preprocess_image(file_path)

    # Run object detection
    results = model(file_path)

    # Process YOLO predictions
    detected_objects = process_predictions(results)

    return jsonify({
        "image_path": file_path,
        "objects": detected_objects
    })

def process_predictions(results):
    """ Process YOLO predictions and return detected objects """
    detections = results.pandas().xyxy[0]  # Convert to Pandas DataFrame
    objects = []
    
    for _, row in detections.iterrows():
        objects.append({
            "label": row["name"],
            "confidence": float(row["confidence"]),
            "bbox": [int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])]
        })
    
    return objects

if __name__ == '__main__':
    app.run(debug=True)  
