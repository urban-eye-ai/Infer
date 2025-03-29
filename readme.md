Urban Surveillance System using YOLO

## Overview
This project focuses on object detection using the YOLOv5s model. The workflow includes data collection, model training, and integration with firmware for deployment.

## Features
- **Data Collection and Preprocessing**: Merged datasets from multiple sources using Roboflow.
- **Model Training**: Trained a YOLOv5s model for object detection tasks.
- **Firmware Integration**: Integrated the trained model with firmware for real-world applications.
- **Entity Detection**: Detects entities such as cars, bikes, people, garbage cans, zebra crossings, etc.
- **Automated Alerts**:
  - Sends a call to traffic authorities if the number of vehicles is high.
  - Sends a call to the municipality if garbage is detected.
  - Sends a call to the police if the number of pedestrians is high or a mob is detected.

## Project Workflow
1. **Data Collection**:
   - Collected datasets from various sources on Roboflow.
   - Merged and preprocessed the data to ensure consistency and quality.

2. **Model Training**:
   - Used the YOLOv5s architecture for training.
   - Fine-tuned the model for optimal performance on the collected dataset.

3. **Integration**:
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
- Achieved high accuracy in object detection tasks.
- Successfully deployed the model in a firmware environment.
- Automated alert system effectively notifies relevant authorities based on detection scenarios.

## Acknowledgments
- [Roboflow](https://roboflow.com) for dataset management.
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the object detection framework.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
