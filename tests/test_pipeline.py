"""Test headless: nạp model + phát hiện + OCR trên một ảnh test (không GUI).

Chạy:  python -m tests.test_pipeline   (hoặc  python tests/test_pipeline.py)
"""
import glob
import os
import sys

# Cho phép chạy trực tiếp như script: thêm thư mục gốc dự án vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from src import config, ocr
from src.detection import PlateDetector


def main():
    print("[1/4] Đang nạp mô hình YOLO...")
    detector = PlateDetector()

    imgs = sorted(glob.glob(os.path.join(config.TEST_IMAGES_DIR, "*.jpg")))
    if not imgs:
        print(f"Không tìm thấy ảnh test trong {config.TEST_IMAGES_DIR}")
        return
    print(f"[2/4] Tìm thấy {len(imgs)} ảnh. Dùng: {os.path.basename(imgs[0])}")

    print("[3/4] Phát hiện biển số...")
    crop = None
    for p in imgs[:6]:
        img = Image.open(p).convert("RGB")
        crop = detector.detect_crop(img)
        if crop is not None:
            print(f"    Phát hiện trên {os.path.basename(p)}")
            break
    if crop is None:
        print("    Không phát hiện được biển trên các ảnh thử.")
        return

    print("[4/4] Chạy OCR...")
    _, text = ocr.read_plate(crop)
    print("=" * 40)
    print("BIỂN SỐ ĐỌC ĐƯỢC:", repr(text))
    print("=" * 40)
    print("PIPELINE_OK")


if __name__ == "__main__":
    main()
