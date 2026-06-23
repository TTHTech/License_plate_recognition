# Headless test: model load + detect + OCR on one test image (no GUI)
import os, glob
from PIL import Image
from ultralytics import YOLO
import easyocr
import DetectLicense, EasyOCR

current_dir = os.path.dirname(os.path.abspath(__file__))
print("[1/5] Loading EasyOCR reader (downloads models on first run)...")
reader = easyocr.Reader(['en'])

print("[2/5] Loading YOLO model...")
model = YOLO(os.path.join(current_dir, 'runs', 'detect', 'train10', 'weights', 'best.pt'))

imgs = sorted(glob.glob(os.path.join(current_dir, 'test', 'images', '*.jpg')))
print(f"[3/5] Found {len(imgs)} test images. Using: {os.path.basename(imgs[0])}")
img = Image.open(imgs[0]).convert('RGB')

print("[4/5] Detecting license plate...")
crop = DetectLicense.DetectLicense(img, model)
if crop is None:
    print("    No plate detected on this image; trying next few...")
    for p in imgs[1:6]:
        img = Image.open(p).convert('RGB')
        crop = DetectLicense.DetectLicense(img, model)
        if crop is not None:
            print(f"    Detected on {os.path.basename(p)}")
            break

print("[5/5] Running OCR...")
out_img, text = EasyOCR.OCRRESULT(crop, reader)
print("=" * 40)
print("RECOGNIZED PLATE TEXT:", repr(text))
print("=" * 40)
print("PIPELINE_OK")
