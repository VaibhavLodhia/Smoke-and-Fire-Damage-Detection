# Smoke and Fire Damage Detection Using YOLOv5

## Overview
This project aims to detect smoke or fire damage in images using a custom-trained YOLOv5 model. The project involves training the YOLOv5 model on custom data and then applying the model to detect and mark regions of interest in a folder of input images. The results are saved as output images, each with bounding boxes around the detected smoke or fire.

## Project Structure
- **`yolov5`**: Directory containing the YOLOv5 code (cloned from the official repository).
- **`runs`**: Folder containing the results of training runs, including trained model weights.
- **`detect_fire_smoke.py`**: Main Python script to perform detection on images in a folder.
- **`output.jpg`**: Sample output image after detection.
- **`requirements.txt`**: Dependencies required for the project.
- **`train_data.zip`**: Custom training dataset used for training the YOLOv5 model.
- **`sample.jpg`, `sample.png`, `sample1.jpg`**: Sample input images for testing the model.

## Setup
Follow these steps to set up the environment and train the YOLOv5 model in Google Colab:

### Step 1: Clone YOLOv5 Repository and Install Dependencies
```bash
!git clone https://github.com/ultralytics/yolov5  # Clone YOLOv5 repository
%cd yolov5
%pip install -qr requirements.txt comet_ml  # Install dependencies
```

Import the required modules and verify the environment setup:
```python
import torch
import utils

display = utils.notebook_init()  # Check PyTorch and GPU availability
```

### Step 2: Prepare Dataset
Upload your training data and unzip it:
```bash
!unzip -q ../train_data.zip -d ../  # Unzip custom training data
```

### Step 3: Train YOLOv5 Model
Train the YOLOv5 model using your custom dataset:
```bash
!python train.py --img 640 --batch 16 --epochs 80 --data custom_data.yaml --weights yolov5s.pt --cache
```

### Step 4: Save Trained Model
After training, zip the trained model weights:
```bash
!zip -r trained_model.zip runs/train/exp6/
```

## Running Inference on Folder of Images
Once the model is trained, you can run inference on a folder of images using the `detect_fire_smoke.py` script. The script takes images from a specified input folder, detects smoke or fire damage, and saves the output images to an output folder.

### Python Script: `detect_fire_smoke.py`
The Python script uses the YOLOv5 model to detect smoke or fire damage in each image from the input folder.

#### How to Run the Script
1. Update the input and output folder paths in the script:
    - **Input Folder**: Folder containing the images for detection.
    - **Output Folder**: Folder where the processed images will be saved.

2. Run the script:
```bash
python detect_fire_smoke.py
```

The script will iterate over all the images in the input folder, detect smoke or fire damage, and save the output images with bounding boxes to the output folder.

### Script Details
The `detect_fire_smoke.py` script includes the following functions:
- **`load_model(weights_path)`**: Loads the YOLOv5 model with the given weights.
- **`plot_one_box(xyxy, img, label=None, ...)`**: Draws a bounding box on the image with a label.
- **`detect_smoke_fire(model, img_path, output_path)`**: Processes a single image, performs inference, and saves the output image.

### Parameters
- **Input Folder**: Path to the folder containing input images.
- **Output Folder**: Path to the folder where the processed images will be saved.

## Directory Structure
```
.
├── yolov5/                       # YOLOv5 code
├── runs/                         # Training runs, including trained model weights
├── detect_fire_smoke.py          # Python script for detection
├── output.jpg                    # Sample output image after detection
├── requirements.txt              # Project dependencies
├── train_data.zip                # Custom training dataset
├── sample.jpg                    # Sample input image for testing
├── sample.png                    # Sample input image for testing
└── sample1.jpg                   # Sample input image for testing
```

## Example Usage
- Run the `detect_fire_smoke.py` script to generate the output images with bounding boxes.

## Requirements
- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- YOLOv5 and its dependencies (as per `requirements.txt` from YOLOv5 repository)

## Acknowledgements
- This project uses the [YOLOv5](https://github.com/ultralytics/yolov5) model developed by Ultralytics.
