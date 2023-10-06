# Preprocess.py
import easyocr
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageTk,ImageDraw,ImageFont
# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5) #kích cỡ càng to thì càng mờ
ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  

###################################################################################################
def preprocess(imgOriginal):

    imgGrayscale = extractValue(imgOriginal)
    # imgGrayscale = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY) nên dùng hệ màu HSV
    # Trả về giá trị cường độ sáng ==> ảnh gray
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale) #để làm nổi bật biển số hơn, dễ tách khỏi nền
    # fig = plt.figure(dpi =300)
    # plt.subplot(1,2,2).set_title("imgMaxContrastGrayscale"), plt.xticks([]); plt.yticks([])
    # plt.imshow(imgMaxContrastGrayscale, 'gray'),plt.axis('off')
    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    # fig = plt.figure(dpi =300)
    # plt.subplot(1,2,2).set_title("imgBlurred"), plt.xticks([]); plt.yticks([])
    # plt.imshow(imgBlurred, 'gray'),plt.axis('off')
    
    #Làm mịn ảnh bằng bộ lọc Gauss 5x5, sigma = 0
    retval, imgThresh = cv2.threshold(imgBlurred,150,255, cv2.THRESH_BINARY)

    # imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    #Tạo ảnh nhị phân
    return imgGrayscale, imgThresh
#Trả về ảnh xám và ảnh nhị phân
# end function
###################################################################################################

def OCRRESULT(image,reader):
    print (image)
    # Chuyển đổi sang numpy array
    image_np = np.asarray(image)
    # Đảo ngược thứ tự kênh màu (RGB -> BGR)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Tạo ảnh OpenCV từ numpy array
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # imgGrayscale, imgThresh = Preprocess.preprocess(img)
    
    current_text_license = ""
    # image.save('cropLicense.jpg', format='JPEG')
    # reader = easyocr.Reader(['en'])
    image_np = np.array(img)
    result = reader.readtext(image_np)
    img = Image.fromarray(image_np)
    print(img)
    # img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
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
    return img, current_text_license

    
###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    
    #màu sắc, độ bão hòa, giá trị cường độ sáng
    #Không chọn màu RBG vì vd ảnh màu đỏ sẽ còn lẫn các màu khác nữa nên khó xđ ra "một màu" 
    return imgValue
# end function

###################################################################################################
def maximizeContrast(imgGrayscale):
    #Làm cho độ tương phản lớn nhất 
    height, width = imgGrayscale.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel
    
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10) #nổi bật chi tiết sáng trong nền tối
    #cv2.imwrite("tophat.jpg",imgTopHat)
    
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10) #Nổi bật chi tiết tối trong nền sáng
    #cv2.imwrite("blackhat.jpg",imgBlackHat)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat) 
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    #cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
    #Kết quả cuối là ảnh đã tăng độ tương phản 
    return imgGrayscalePlusTopHatMinusBlackHat
# end function
# reader = easyocr.Reader(['en'])
# input_path = './test.jpg'
# # img = Image.open(input_path)
# img = cv2.imread(input_path,0)
# LineDetection(img)








