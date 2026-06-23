"""Phát hiện vùng biển số bằng YOLOv8.

Gói model trong một lớp để nạp một lần và tái dùng cho cả chế độ ảnh
(``detect_crop``) lẫn chế độ video (``detect_largest``).
"""
import cv2
from PIL import Image
from ultralytics import YOLO

from src import config


class PlateDetector:
    def __init__(self, model_path=config.MODEL_PATH, conf=config.DETECT_CONF):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_crop(self, pil_img):
        """Chế độ ảnh: trả về biển số đầu tiên phát hiện được (PIL) hoặc None."""
        results = self.model.predict(pil_img, conf=self.conf, verbose=False)
        for result in results:
            for box in result.boxes.xywh:
                x, y, w, h = (int(v) for v in box[:4])
                lx, ly = w // 2, h // 2
                return pil_img.crop((x - lx, y - ly, x + lx, y + ly))
        return None

    def detect_largest(self, frame_bgr):
        """Chế độ video: vẽ khung lên ``frame_bgr`` (tại chỗ) và trả về
        ``(crop_pil, box)`` của biển CÓ DIỆN TÍCH LỚN NHẤT, hoặc (None, None).
        """
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
        best_crop, best_box, best_area = None, None, 0
        for r in results:
            for b in r.boxes.xyxy.cpu().numpy().astype(int):
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 220, 0), 2)
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)
                    crop_bgr = frame_bgr[max(0, y1):y2, max(0, x1):x2]
                    best_crop = Image.fromarray(
                        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        return best_crop, best_box
