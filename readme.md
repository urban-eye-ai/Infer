# Urban Surveillance System using YOLO

## Overview
This project delivers a robust computer vision solution for smart city applications. The system detects zebra crossings, garbage bins, and traffic elements in urban scenes, even under challenging conditions such as motion blur, bad lighting, occlusions, or perspective shifts. It includes distortion correction, real-time performance, and automated alerts for seamless integration with smart city platforms.

## Problem Statement
The prototype addresses the following challenges:
- Detecting zebra crossings, garbage bins, and traffic elements in urban scenes.
- Handling distorted inputs caused by motion blur, low light, occlusions, or perspective shifts.
- Correcting distortions and providing enhanced images.
- Operating efficiently in real-time for smart city applications.

### Input Types
- **Images/Videos**: City surveillance feeds, CCTV footage, or street view images.
- **Distorted Data**: Inputs may include motion-blurred, low-light, occluded, or perspective-shifted images.

### Output
- **Detection Results**: Bounding boxes and labels for zebra crossings, garbage bins, and traffic elements.
- **Image Correction**: Enhanced images after distortion correction.
- **Real-Time Alerts**: Notifications for smart city control centers.

## Features
- **Robust Object Detection**: Detects entities such as zebra crossings, garbage bins, cars, bikes, people, and other traffic elements.
- **Distortion Correction**: Preprocessing pipeline to handle motion blur, low light, occlusions, and perspective shifts.
- **Real-Time Performance**: Low latency suitable for smart city applications.
- **Automated Alerts**:
  - Sends a call to traffic authorities if the number of vehicles is high.
  - Sends a call to the municipality if garbage is detected.
  - Sends a call to the police if the number of pedestrians is high or a mob is detected.
- **Clear Visualization**: Annotated outputs with bounding boxes and labels.
- **Firmware Integration**: Seamless deployment of the trained model in real-world environments.

## Dataset Sources
- **Kaggle**: Urban scene datasets and annotated images for object detection.
- **Cityscapes**: High-quality urban scene segmentation and detection.
- **COCO/Open Images**: Diverse object detection datasets.
- **Berkeley DeepDrive**: Datasets focusing on street scenes and autonomous driving.

## Prototype Highlights
- **Real-Time Alerts**: The system generates actionable alerts for smart city control centers based on detection scenarios.
- **Distortion Handling**: Preprocessing ensures accurate detection even with distorted inputs.
- **Scalability**: Designed to handle increasing data volumes and adapt to new input types.
- **Integration APIs**: Easy-to-integrate interfaces for smart city platforms.

## Evaluation
The prototype meets the following criteria:
- **Accuracy**: High precision in detecting and classifying objects.
- **Speed**: Real-time processing with minimal latency.
- **Robustness**: Reliable performance under varying input conditions.
- **Integration**: Seamless deployment with existing systems.
- **User Experience**: Intuitive outputs and actionable insights.

## Project Workflow
1. **Data Collection**:
   - Collected datasets from various sources on Roboflow.
   - Merged and preprocessed the data to ensure consistency and quality.

2. **Model Training**:
   - Used the YOLOv5s architecture for training.
   - Fine-tuned the model for optimal performance on the collected dataset.

3. **Distortion Correction**:
   - Implemented preprocessing techniques to handle motion blur, low light, occlusions, and perspective shifts.

4. **Integration**:
   - Deployed the trained model by integrating it with firmware for seamless operation.

## Requirements
- Python 3.8 or higher
- PyTorch
- Roboflow API
- YOLOv5 repository

## Installation
1. Clone the YOLOv5 repository:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install additional project-specific dependencies:
   ```bash
   pip install roboflow
   ```

## Usage
1. **Data Preparation**:
   - Use the Roboflow API to download and preprocess datasets.
   - Ensure the data is in YOLO format before training.

2. **Model Training**:
   - Run the training script:
     ```bash
     python train.py --img 640 --batch 16 --epochs 50 --data <path_to_data.yaml> --weights yolov5s.pt
     ```

3. **Firmware Integration**:
   - Export the trained model to the required format (e.g., ONNX or TensorRT).
   - Integrate the exported model with the firmware.

## Results
- Successfully detected zebra crossings, garbage bins, and traffic elements in urban scenes.
- Achieved high accuracy even with distorted inputs.
- Real-time alerts effectively notify relevant authorities.
- Distortion correction improves input quality for better detection results.

## Acknowledgments
- [Roboflow](https://roboflow.com) for dataset management.
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the object detection framework.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
