import sys
import pathlib
import os
import cv2
import numpy as np
import torch
from pathlib import Path

# Monkey patch to use WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

yolov5_path = 'C:/Users/Dell/Desktop/Tracker Task/yolov5'
sys.path.append(yolov5_path)
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.dataloaders import letterbox

# Function to load the model
def load_model(weights_path, device='cpu'):
    model = DetectMultiBackend(weights_path, device=device)
    return model

# Function to draw bounding boxes
def plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=3):
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line/font thickness
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Function to perform detection
def detect_smoke_fire(model, img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Couldn't load image from {img_path}")
        return


    img_processed = letterbox(img, stride=model.stride, auto=True)[0]
    img_processed = img_processed[:, :, ::-1].transpose(2, 0, 1)
    img_processed = np.ascontiguousarray(img_processed)


    img_tensor = torch.from_numpy(img_processed).to(model.device)
    img_tensor = img_tensor.float() / 255.0  
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)


    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)


    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=(0, 255, 0), line_thickness=2)

    cv2.imwrite(output_path, img)
    print(f"Output image saved at {output_path}")

if __name__ == "__main__":

    weights_path = "C:/Users/Dell/Desktop/Tracker Task/runs/train/exp6/weights/best.pt"
    img_path = "sample1.jpg"
    output_path = "output.jpg"

    model = load_model(weights_path)
    detect_smoke_fire(model, img_path, output_path)
