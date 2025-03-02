from flask import Flask, request, render_template, jsonify, send_file
import torch
import numpy as np
import cv2
from PIL import Image
import io
import os
import time
import base64
import uuid
import sys

app = Flask(__name__)

# Create directories for uploads and results
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Model settings
MODEL_PATH = "garbage.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Global variable for the model
model = None

def load_model():
    """Load the YOLOv5 model with path fix for Windows"""
    global model
    if model is None:
        try:
            # Fix for the PosixPath issue on Windows
            import pathlib
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            
            # Force reload to avoid cache issues
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
            model.conf = CONF_THRESHOLD
            model.iou = IOU_THRESHOLD
            
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            sys.exit(1)
    return model

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle image upload and object detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Generate unique filename
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    upload_path = os.path.join('static/uploads', filename)
    result_path = os.path.join('static/results', filename)
    
    # Save the uploaded file
    file.save(upload_path)
    
    # Get confidence threshold from form if provided
    conf_threshold = request.form.get('confidence', CONF_THRESHOLD)
    try:
        conf_threshold = float(conf_threshold)
    except ValueError:
        conf_threshold = CONF_THRESHOLD
    
    # Load model and run inference
    try:
        model = load_model()
        model.conf = conf_threshold
        
        # Run inference
        start_time = time.time()
        results = model(upload_path)
        inference_time = time.time() - start_time
        
        # Save results image
        results.render()  # adds bounding boxes to images
        result_img = Image.fromarray(results.ims[0])
        result_img.save(result_path)
        
        # Get detection details
        detections = results.pandas().xyxy[0]
        detection_list = []
        
        for _, det in detections.iterrows():
            detection_list.append({
                'class': det['name'],
                'confidence': float(det['confidence']),
                'bbox': [
                    float(det['xmin']), 
                    float(det['ymin']), 
                    float(det['xmax']), 
                    float(det['ymax'])
                ]
            })
        
        # Return JSON response
        return jsonify({
            'success': True,
            'upload_path': upload_path,
            'result_path': result_path,
            'detections': detection_list,
            'inference_time': f"{inference_time:.2f}s",
            'detection_count': len(detection_list)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Create HTML template directory
os.makedirs('templates', exist_ok=True)

# Write the HTML template for the web interface
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv5 Object Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: #f8f9fa;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .upload-container {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .result-container {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            display: none;
        }
        .image-preview {
            width: 100%;
            max-height: 400px;
            object-fit: contain;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-radius: 5px;
        }
        .detection-item {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .confidence-high {
            color: green;
            font-weight: bold;
        }
        .confidence-medium {
            color: orange;
            font-weight: bold;
        }
        .confidence-low {
            color: red;
            font-weight: bold;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-radius: 50%;
            border-top: 5px solid #3498db;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>YOLOv5 Object Detection</h1>
            <p class="lead">Upload an image to detect objects using your custom trained YOLOv5 model</p>
        </div>
        
        <div class="row">
            <div class="col-md-6 mx-auto">
                <div class="upload-container">
                    <h3>Upload Image</h3>
                    <form id="upload-form" enctype="multipart/form-data">
                        <div class="mb-3">
                            <label for="image" class="form-label">Select Image</label>
                            <input class="form-control" type="file" id="image" name="image" accept="image/*" onchange="previewImage()">
                        </div>
                        <div class="mb-3">
                            <label for="confidence" class="form-label">Confidence Threshold: <span id="conf-value">0.25</span></label>
                            <input type="range" class="form-range" min="0.1" max="1.0" step="0.05" value="0.25" id="confidence" name="confidence" onchange="updateConfValue()">
                        </div>
                        <div class="mb-3">
                            <img id="preview" class="image-preview d-none">
                        </div>
                        <button type="submit" class="btn btn-primary w-100">Detect Objects</button>
                    </form>
                    <div id="loading" class="text-center mt-3 d-none">
                        <div class="loader"></div>
                        <p class="mt-2">Processing image...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-10 mx-auto">
                <div id="result-container" class="result-container">
                    <h3>Detection Results</h3>
                    <div class="row">
                        <div class="col-md-6">
                            <img id="result-image" class="image-preview">
                            <div class="d-grid gap-2">
                                <a id="download-btn" href="#" class="btn btn-success" download>Download Result</a>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div id="detection-info">
                                <p><strong>Inference Time:</strong> <span id="inference-time"></span></p>
                                <h4>Detected Objects: <span id="detection-count">0</span></h4>
                                <div id="detections-list"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function previewImage() {
            const preview = document.getElementById('preview');
            const file = document.getElementById('image').files[0];
            const reader = new FileReader();
            
            reader.onloadend = function() {
                preview.src = reader.result;
                preview.classList.remove('d-none');
            }
            
            if (file) {
                reader.readAsDataURL(file);
            } else {
                preview.src = '';
                preview.classList.add('d-none');
            }
        }
        
        function updateConfValue() {
            const value = document.getElementById('confidence').value;
            document.getElementById('conf-value').textContent = value;
        }
        
        function getConfidenceClass(conf) {
            if (conf > 0.7) return 'confidence-high';
            if (conf > 0.5) return 'confidence-medium';
            return 'confidence-low';
        }
        
        document.getElementById('upload-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const loading = document.getElementById('loading');
            const resultContainer = document.getElementById('result-container');
            
            // Show loading indicator
            loading.classList.remove('d-none');
            resultContainer.style.display = 'none';
            
            fetch('/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                loading.classList.add('d-none');
                
                if (data.success) {
                    // Display results
                    document.getElementById('result-image').src = data.result_path + '?' + new Date().getTime();
                    document.getElementById('download-btn').href = data.result_path;
                    document.getElementById('download-btn').download = 'detection_result.jpg';
                    document.getElementById('inference-time').textContent = data.inference_time;
                    document.getElementById('detection-count').textContent = data.detection_count;
                    
                    // Clear previous detections
                    const detectionsList = document.getElementById('detections-list');
                    detectionsList.innerHTML = '';
                    
                    // Add new detections
                    data.detections.forEach((det, index) => {
                        const detItem = document.createElement('div');
                        detItem.className = 'detection-item';
                        const confClass = getConfidenceClass(det.confidence);
                        
                        detItem.innerHTML = `
                            <h5>Detection #${index+1}: ${det.class}</h5>
                            <p>Confidence: <span class="${confClass}">${(det.confidence * 100).toFixed(1)}%</span></p>
                            <p>Bounding Box: X: ${det.bbox[0].toFixed(1)} to ${det.bbox[2].toFixed(1)}, 
                               Y: ${det.bbox[1].toFixed(1)} to ${det.bbox[3].toFixed(1)}</p>
                        `;
                        detectionsList.appendChild(detItem);
                    });
                    
                    // Show result container
                    resultContainer.style.display = 'block';
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                loading.classList.add('d-none');
                alert('Error: ' + error);
            });
        });
    </script>
</body>
</html>
    ''')

if __name__ == '__main__':
    print("Initializing YOLOv5 model...")
    # Initialize model
    load_model()
    
    # Run the Flask application
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)