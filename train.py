"""Huấn luyện lại mô hình phát hiện biển số (YOLOv8, 1 lớp 'license-plate').

Chuẩn bị dataset theo định dạng YOLO và cập nhật license_data.yaml, rồi chạy:
    python train.py
Trọng số sẽ được lưu trong thư mục runs/detect/.
"""
from ultralytics import YOLO

from src import config


def main():
    # Dựng kiến trúc từ YAML rồi nạp trọng số pretrained để transfer learning
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    model.train(data=config.DATASET_YAML, epochs=100, imgsz=640)


if __name__ == "__main__":
    main()
