# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:44:54 2023

@author: Van Hoang
"""

from ultralytics import YOLO
import os
import ultralytics
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data=current_dir +'\license_data.yaml', epochs=100, imgsz=640)