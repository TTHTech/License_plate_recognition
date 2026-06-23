"""Cấu hình tập trung: đường dẫn, hằng số và màu giao diện.

Mọi đường dẫn được suy ra từ thư mục gốc dự án nên không phụ thuộc
vào nơi chạy lệnh. Đổi tham số ở đây thay vì rải rác trong code.
"""
import os

# ---- Đường dẫn ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train10", "weights", "best.pt")
LOGO_PATH = os.path.join(BASE_DIR, "logos", "logo.png")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test", "images")
DATASET_YAML = os.path.join(BASE_DIR, "license_data.yaml")

# ---- Phát hiện & OCR ----
DETECT_CONF = 0.50        # ngưỡng tin cậy của YOLO
OCR_EVERY = 6             # số frame tối thiểu giữa 2 lần OCR (chế độ video)
VOTE_WINDOW = 15          # số kết quả gần nhất dùng để "bỏ phiếu"

# ---- Kích thước khung hiển thị ----
ORIG_W, ORIG_H = 646, 506
PANEL_W, PANEL_H = 345, 215

# ---- Bảng màu giao diện ----
# Mỗi màu dạng (sáng, tối) để CustomTkinter tự đổi theo chế độ giao diện.
SIDEBAR = ("#e2e8f0", "#0b1220")
MAIN_BG = ("#f1f5f9", "#0f172a")
CARD = ("#ffffff", "#1e293b")
CARD_INNER = ("#e5e9f0", "#0b1220")   # nền khung ảnh bên trong card
BORDER = ("#cbd5e1", "#334155")
TEXT = ("#0f172a", "#f8fafc")         # màu chữ chính
MUTED = ("#64748b", "#94a3b8")        # màu chữ phụ
ACCENT = "#2563eb"                    # xanh dương chủ đạo
ACCENT_H = "#1d4ed8"
DANGER = "#ef4444"
DANGER_H = "#dc2626"
NEUTRAL = ("#cbd5e1", "#334155")
NEUTRAL_H = ("#94a3b8", "#475569")
CYAN = ("#0e7490", "#22d3ee")         # màu chữ biển số
OK = "#22c55e"                        # đèn trạng thái
