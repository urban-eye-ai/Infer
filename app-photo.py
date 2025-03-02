from flask import Flask, request, render_template, jsonify, send_file, Response
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
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Create directories for uploads and results
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Model settings
MODEL_PATH = "best.pt"
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

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Generate unique filename
    filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
    upload_path = os.path.join('static/uploads', filename)
    
    # Save the uploaded file
    file.save(upload_path)
    
    # Get confidence threshold from form if provided
    conf_threshold = request.form.get('confidence', CONF_THRESHOLD)
    try:
        conf_threshold = float(conf_threshold)
    except ValueError:
        conf_threshold = CONF_THRESHOLD
    
    # Process the video and generate output
    output_filename = f"output_{filename.split('.')[0]}.mp4"
    output_path = os.path.join('static/results', output_filename)
    
    try:
        # Process video in background to not block the response
        # Return the paths that will be used for video processing
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully',
            'video_id': filename.split('.')[0],
            'upload_path': upload_path,
            'output_path': output_path,
            'conf_threshold': conf_threshold
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/process_video/<video_id>', methods=['GET'])
def process_video(video_id):
    """Process the uploaded video with object detection"""
    upload_path = os.path.join('static/uploads', f"{video_id}{os.path.splitext(request.args.get('file_extension', '.mp4'))[0]}")
    output_path = os.path.join('static/results', f"output_{video_id}.mp4")
    conf_threshold = float(request.args.get('conf_threshold', CONF_THRESHOLD))
    
    try:
        # Load model
        model = load_model()
        model.conf = conf_threshold
        
        # Process video with YOLOv5
        process_status = process_video_with_yolo(upload_path, output_path, model)
        
        return jsonify({
            'success': process_status['success'],
            'message': process_status['message'],
            'output_path': output_path if process_status['success'] else None,
            'error': process_status.get('error')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def process_video_with_yolo(video_path, output_path, model):
    """Process video with YOLOv5 and save output video with detections"""
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'message': 'Error opening video file', 'error': 'Could not open video file'}
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1'
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        start_time = time.time()
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Increment frame counter
            frame_count += 1
            
            # Apply YOLOv5 detection
            results = model(frame)
            
            # Render detection results on the frame
            rendered_frame = results.render()[0]
            
            # Write frame to output video
            out.write(rendered_frame)
            
            # Print progress
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / (frame_count / total_frames)
                remaining_time = estimated_total - elapsed_time
                print(f"Progress: {progress:.1f}% | Frames: {frame_count}/{total_frames} | Time remaining: {remaining_time:.1f}s")
        
        # Release resources
        cap.release()
        out.release()
        
        process_time = time.time() - start_time
        
        return {
            'success': True, 
            'message': f'Video processed successfully in {process_time:.2f} seconds',
            'processed_frames': frame_count,
            'process_time': process_time
        }
    
    except Exception as e:
        return {'success': False, 'message': 'Error processing video', 'error': str(e)}

@app.route('/video_status/<video_id>', methods=['GET'])
def video_status(video_id):
    """Check the status of video processing"""
    output_path = os.path.join('static/results', f"output_{video_id}.mp4")
    
    if os.path.exists(output_path):
        # Get file size and creation time
        file_size = os.path.getsize(output_path)
        creation_time = os.path.getctime(output_path)
        time_since_creation = time.time() - creation_time
        
        return jsonify({
            'status': 'complete',
            'output_path': output_path,
            'file_size': file_size,
            'time_elapsed': f"{time_since_creation:.2f}s"
        })
    else:
        return jsonify({
            'status': 'processing'
        })

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
    <title>YOLOv5 Video Object Detection</title>
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
        .video-preview {
            width: 100%;
            max-height: 400px;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-radius: 5px;
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
        .progress-container {
            margin-top: 20px;
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
            <h1>YOLOv5 Video Object Detection</h1>
            <p class="lead">Upload a video to detect objects using your custom trained YOLOv5 model</p>
        </div>
        
        <div class="row">
            <div class="col-md-6 mx-auto">
                <div class="upload-container">
                    <h3>Upload Video</h3>
                    <form id="upload-form" enctype="multipart/form-data">
                        <div class="mb-3">
                            <label for="video" class="form-label">Select Video File</label>
                            <input class="form-control" type="file" id="video" name="video" accept="video/*" onchange="previewVideo()">
                        </div>
                        <div class="mb-3">
                            <label for="confidence" class="form-label">Confidence Threshold: <span id="conf-value">0.25</span></label>
                            <input type="range" class="form-range" min="0.1" max="1.0" step="0.05" value="0.25" id="confidence" name="confidence" onchange="updateConfValue()">
                        </div>
                        <div class="mb-3">
                            <video id="preview" class="video-preview d-none" controls></video>
                        </div>
                        <button type="submit" class="btn btn-primary w-100">Process Video</button>
                    </form>
                    <div id="loading" class="text-center mt-3 d-none">
                        <div class="loader"></div>
                        <p class="mt-2" id="processing-text">Uploading video...</p>
                        <div class="progress-container">
                            <div class="progress">
                                <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
                            </div>
                            <p class="text-center mt-1" id="progress-text">Initializing...</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-10 mx-auto">
                <div id="result-container" class="result-container">
                    <h3>Detection Results</h3>
                    <div class="row">
                        <div class="col-md-12">
                            <div class="ratio ratio-16x9">
                                <video id="result-video" class="video-preview" controls></video>
                            </div>
                            <div class="d-grid gap-2 mt-3">
                                <a id="download-btn" href="#" class="btn btn-success" download>Download Processed Video</a>
                            </div>
                            <div id="detection-info" class="mt-3">
                                <p><strong>Processing Time:</strong> <span id="processing-time"></span></p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let videoId = null;
        let statusCheckInterval = null;
        
        function previewVideo() {
            const preview = document.getElementById('preview');
            const file = document.getElementById('video').files[0];
            
            if (file) {
                const videoUrl = URL.createObjectURL(file);
                preview.src = videoUrl;
                preview.classList.remove('d-none');
            } else {
                preview.src = '';
                preview.classList.add('d-none');
            }
        }
        
        function updateConfValue() {
            const value = document.getElementById('confidence').value;
            document.getElementById('conf-value').textContent = value;
        }
        
        function startStatusCheck() {
            if (!videoId) return;
            
            statusCheckInterval = setInterval(() => {
                fetch(`/video_status/${videoId}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'complete') {
                            clearInterval(statusCheckInterval);
                            showResults(data.output_path, data.time_elapsed);
                        } else {
                            // Update progress UI (if needed)
                            document.getElementById('processing-text').textContent = 'Processing video...';
                            // In a real app, you'd have a way to show actual progress here
                            const randomProgress = Math.floor(Math.random() * 30) + 50; // 50-80% as placeholder
                            document.getElementById('progress-bar').style.width = `${randomProgress}%`;
                            document.getElementById('progress-text').textContent = `Processing: approximately ${randomProgress}% complete`;
                        }
                    })
                    .catch(error => {
                        console.error('Error checking status:', error);
                    });
            }, 2000); // Check every 2 seconds
        }
        
        function showResults(outputPath, processingTime) {
            // Hide loading indicator
            document.getElementById('loading').classList.add('d-none');
            
            // Set video source with cache busting
            const resultVideo = document.getElementById('result-video');
            resultVideo.src = `${outputPath}?${new Date().getTime()}`;
            
            // Set download link
            document.getElementById('download-btn').href = outputPath;
            
            // Set processing time
            document.getElementById('processing-time').textContent = processingTime;
            
            // Show result container
            document.getElementById('result-container').style.display = 'block';
        }
        
        function processVideo(fileExtension) {
            fetch(`/process_video/${videoId}?file_extension=${fileExtension}&conf_threshold=${document.getElementById('confidence').value}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('progress-bar').style.width = '50%';
                        document.getElementById('progress-text').textContent = 'Video processing started...';
                        startStatusCheck();
                    } else {
                        document.getElementById('loading').classList.add('d-none');
                        alert('Error processing video: ' + data.error);
                    }
                })
                .catch(error => {
                    document.getElementById('loading').classList.add('d-none');
                    alert('Error: ' + error);
                });
        }
        
        document.getElementById('upload-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const loading = document.getElementById('loading');
            const resultContainer = document.getElementById('result-container');
            const file = document.getElementById('video').files[0];
            
            if (!file) {
                alert('Please select a video file');
                return;
            }
            
            // Show loading indicator
            loading.classList.remove('d-none');
            resultContainer.style.display = 'none';
            
            // Reset progress
            document.getElementById('progress-bar').style.width = '10%';
            document.getElementById('progress-text').textContent = 'Uploading video...';
            
            fetch('/upload_video', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Store the video ID for status checking
                    videoId = data.video_id;
                    document.getElementById('progress-bar').style.width = '30%';
                    document.getElementById('progress-text').textContent = 'Video uploaded, preparing processing...';
                    
                    // Start processing the video
                    const fileExtension = file.name.split('.').pop();
                    processVideo(`.${fileExtension}`);
                } else {
                    loading.classList.add('d-none');
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