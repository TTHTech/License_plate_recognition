# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 12:47:58 2023

@author: Van Hoang
"""
from ultralytics import YOLO
import os
import numpy as np
from PIL import ImageEnhance
# Model
# current_dir = os.path.dirname(os.path.abspath(__file__))# + '\\valid\\images\\'
# model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO(current_dir + '\\runs\\detect\\train10\\\weights\\best.pt')


# def DetectLicensetest(img):
#     results = model.predict(img,conf = 0.50)
#     return results[0].plot()
#     # Results



def DetectLicense(img,model):
    results = model.predict(img,conf = 0.50)

    # Results
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        if len(boxes.data)== 0:
            print('No detect')
        else:
            
            for box in boxes.xywh:
                try:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2])
                    h = int(box[3])
                    # print(x,y,w,h)
                    limitY=int(h/2)
                    limitX=int(w/2)
                    cropLicense=img.crop((x-limitX,y-limitY,x+limitX,y+limitY))
                    # cropLicense=img[y-limitY:y+limitY,x-limitX:x+limitX]
                    return cropLicense
                except ValueError as ve:
                    print("Có lỗi xảy ra: ", ve)
