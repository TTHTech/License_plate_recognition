# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 12:52:10 2023

@author: Van Hoang
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import Preprocess
from PIL import Image, ImageTk, ImageDraw
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

n = 1
Min_char = 0.01
Max_char = 0.09

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def PILImgRotated(img):
    # Chuyển đổi sang numpy array
    image_np = np.asarray(img)
    
    # Đảo ngược thứ tự kênh màu (RGB -> BGR)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Tạo ảnh OpenCV từ numpy array
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    img_copy = img.copy()
    img = cv2.medianBlur(img,5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # retval, binary_img = cv2.threshold(gray,150,255, cv2.THRESH_BINARY)

    imgGrayscaleplate, binary_img = Preprocess.preprocess(img)
    canny_image = cv2.Canny(binary_img, 250, 255)  # Canny Edge
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=2)  # Dilation
    
    dilated_image = cv2.medianBlur(dilated_image,7)
    # fig = plt.figure(dpi =300)
    # plt.subplot(1,2,1).set_title("input"), plt.xticks([]); plt.yticks([])
    # plt.imshow(dilated_image, 'gray'),plt.axis('off')
    # cv2.imshow("dilated_image",dilated_image)
    ###### Draw contour and filter out the license plate  #############
    
    dilated_image = cv2.Canny(dilated_image, 250, 255)  # Canny Edge
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất
    
    # cv2.drawContours(img, contours, -1, (255, 0, 255), 3) # Vẽ tất cả các ctour trong hình lớn
    
    
    screenCnt = []
    for c in contours:
        
        peri = cv2.arcLength(c, True)  # Tính chu vi
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h

        # cv2.putText(img, str(len(approx.copy())), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
        # cv2.putText(img, str(ratio), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
        if (len(approx) == 4):
            # print (h/w)
            # print(cv2.contourArea(c)) #130000<cv2.contourArea(c)<190000
            screenCnt.append(approx)
            # if 0.5 <=h/w and h/w < 1.2:
            cv2.drawContours(img, [c], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

            cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
    
    if screenCnt is None:
        detected = 0
        print("No plate detected")
    else:
        detected = 1

    if detected == 1:
        angles = []
        for screenCnt in screenCnt:
            cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

            ############## Find the angle of the license plate #####################
            (x1, y1) = screenCnt[0, 0]
            (x2, y2) = screenCnt[1, 0]
            (x3, y3) = screenCnt[2, 0]
            (x4, y4) = screenCnt[3, 0]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            sorted_array = array.sort(reverse=True, key=lambda x: x[1])
            (x1, y1) = array[0]
            (x2, y2) = array[1]
            doi = abs(y1 - y2)
            ke = abs(x1 - x2)
            angle = math.atan(doi / ke) * (180.0 / math.pi)
            if(angle < 11):
                angles = [0]
                break
            angles.append(angle)
            # print(angle)
    angle = np.median(angles)
    print(angle)
    if angle > 45:
        angle = angle+270
    else:
        angle = -angle
    height, width = img.shape[:2]
    # Tính toán ma trận xoay
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Xoay ảnh
    rotated = cv2.warpAffine(img_copy, M, (width, height))
    # plt.subplot(1,2,2).set_title("input"), plt.xticks([]); plt.yticks([])
    # plt.imshow(rotated, 'gray'),plt.axis('off')
    image_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    image_np = np.array(image_rgb)

    # Tạo ảnh PIL từ numpy array
    image_pil = Image.fromarray(image_np)
    return image_pil

