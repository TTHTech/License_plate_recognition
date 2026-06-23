"""Tiền xử lý ảnh cổ điển (xử lý ảnh): tách kênh sáng HSV, tăng tương phản
bằng top-hat/black-hat và nhị phân hoá.

Module tiện ích độc lập, có thể dùng để trực quan hoá các bước xử lý ảnh
trong báo cáo. Toàn bộ ảnh đầu vào ở định dạng BGR (OpenCV).
"""
import cv2
import numpy as np

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)   # càng to càng mờ


def preprocess(img_bgr):
    """Trả về (ảnh xám, ảnh nhị phân) đã tăng tương phản."""
    grayscale = extract_value(img_bgr)
    max_contrast = maximize_contrast(grayscale)
    blurred = cv2.GaussianBlur(max_contrast, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    return grayscale, thresh


def extract_value(img_bgr):
    """Lấy kênh Value (độ sáng) từ HSV -> ảnh xám.

    Dùng HSV thay vì RGB vì kênh sáng tách nền/biển ổn định hơn.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, _, value = cv2.split(hsv)
    return value


def maximize_contrast(grayscale):
    """Tăng tương phản tối đa để làm nổi bật ký tự biển số."""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    top_hat = cv2.morphologyEx(grayscale, cv2.MORPH_TOPHAT, se, iterations=10)
    black_hat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, se, iterations=10)
    return cv2.subtract(cv2.add(grayscale, top_hat), black_hat)
