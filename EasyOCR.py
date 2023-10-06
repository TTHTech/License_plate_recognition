# import easyocr
# import cv2
# from matplotlib import pyplot as plt
import numpy as np
import cv2
import Preprocess
import matplotlib.pyplot as plt
import easyocr
from PIL import Image, ImageDraw, ImageFont
def OCRRESULT(image,reader):
    # Chuyển đổi sang numpy array
    
    image_np = np.asarray(image)
    # Đảo ngược thứ tự kênh màu (RGB -> BGR)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Tạo ảnh OpenCV từ numpy array
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    imgGrayscale, imgThresh = Preprocess.preprocess(img)
    current_text_license = ""

    image_np = np.array(imgGrayscale)
    result = reader.readtext(image_np)#image_np
    img = Image.fromarray(image_np)#image_np
    print(img)
    # img = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('arial.ttf', 20)

    spacer = 100
    for detection in result:
        top_left = tuple(detection[0][0])
        bottom_right = tuple(detection[0][2])
        text = detection[1]
        draw.rectangle([top_left, bottom_right], outline=(0, 255, 0), width=3)
        draw.text((20, spacer), text, font=font, fill=(0, 0, 255))
        current_text_license += text
        spacer += 15
    # print(image)
    return image, current_text_license
# reader = easyocr.Reader(['en'])
# img = Image.open('C:\\Users\\A\\.spyder-py3\\Code\\Detect-license-plate-3\\test\\images\\bienso15_png.rf.20392254825b425075cc36ab2e51ba76.jpg')
# print(img)
# OCRRESULT(img,reader)





