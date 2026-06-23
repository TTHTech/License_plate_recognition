# 🚗 Nhận diện & trích xuất biển số xe Việt Nam

Ứng dụng nhận diện biển số xe (License Plate Recognition) cho biển số Việt Nam, hỗ trợ **ảnh, video, webcam và camera điện thoại**, với giao diện đồ hoạ hiện đại kiểu dashboard (**CustomTkinter**, có chế độ Sáng / Tối).

> **Pipeline:** YOLOv8 (phát hiện vùng biển) → tự nắn thẳng + tiền xử lý ảnh → **RapidOCR (PP-OCR)** đọc ký tự → sửa lỗi theo định dạng biển số VN.

---

## ✨ Tính năng

- **Phát hiện biển số** bằng mô hình YOLOv8 đã huấn luyện (`runs/detect/train10/weights/best.pt`).
- **Đọc ký tự** bằng RapidOCR (PP-OCR chạy trên ONNX) — chính xác hơn nhiều so với OCR tiếng Anh tổng quát.
- **Tự động nhận diện** ngay khi mở ảnh (không cần bấm thêm).
- **Tự nắn thẳng (deskew)** biển nghiêng + tách dòng + sửa lỗi theo định dạng biển VN, chấm điểm chọn kết quả tốt nhất.
- **Chế độ video / webcam / camera điện thoại** (qua URL stream) tích hợp ngay trong app.
- **Đọc frame + OCR ở luồng nền** giúp video mượt, ít trễ; **bỏ phiếu theo thời gian** cho kết quả ổn định.
- **Giao diện dashboard hiện đại** (CustomTkinter): sidebar điều hướng, banner biển số nổi bật, các thẻ ảnh gốc / biển phát hiện / ảnh đã căn chỉnh / kết quả OCR; có **chế độ Sáng / Tối / Theo hệ thống**.
- **Kiến trúc tách lớp**: toàn bộ xử lý nằm trong `src/`, độc lập hoàn toàn với giao diện ở `ui/` — dễ bảo trì và mở rộng.

---

## 🧱 Cấu trúc dự án

```
.
├── main.py                  # Điểm khởi chạy ứng dụng (python main.py)
├── train.py                 # Script huấn luyện lại mô hình YOLOv8
├── run.bat                  # Chạy nhanh ứng dụng trên Windows
├── requirements.txt         # Danh sách thư viện
├── license_data.yaml        # Cấu hình dataset cho huấn luyện
├── src/                     # Lõi xử lý (không phụ thuộc giao diện)
│   ├── config.py            # Đường dẫn, hằng số, bảng màu giao diện
│   ├── detection.py         # PlateDetector — phát hiện & cắt biển bằng YOLOv8
│   ├── ocr.py               # Đọc biển bằng RapidOCR (deskew, tách dòng, sửa định dạng VN)
│   ├── preprocessing.py     # Tiền xử lý ảnh (HSV, tăng tương phản, nhị phân hoá)
│   ├── camera.py            # VideoStream — đọc frame video/webcam/cam điện thoại
│   └── recognition.py       # PlateVoter (bỏ phiếu) + ghép detect + OCR
├── ui/
│   └── app.py               # Giao diện CustomTkinter (class LicensePlateApp)
├── tests/
│   └── test_pipeline.py     # Kiểm thử pipeline ở chế độ headless (không GUI)
└── runs/detect/train10/weights/best.pt   # Trọng số mô hình đã huấn luyện
```

---

## ⚙️ Cài đặt

> Yêu cầu: **Python 3.11** (khuyến nghị), Windows.

```bash
# 1. Tải mã nguồn
git clone https://github.com/TTHTech/License_plate_recognition.git
cd License_plate_recognition

# 2. Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Cài thư viện
pip install -r requirements.txt
```

> **Lưu ý về PyTorch (CPU):** nếu bước cài `torch` báo lỗi không tìm thấy bản `+cpu`, hãy cài riêng bản CPU:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```
> Lần chạy đầu, RapidOCR sẽ tự tải model về (cần Internet một lần).

---

## ▶️ Sử dụng

Chạy ứng dụng:

```bash
python main.py
```

Hoặc trên Windows bấm đúp **`run.bat`**.

### Các nút trên sidebar
| Nút | Chức năng |
|-----|-----------|
| **Mở ảnh** | Chọn 1 ảnh → tự động phát hiện & đọc biển số. |
| **Mở video** | Chọn file video (`.mp4/.avi/.mov/.mkv`) và nhận diện theo thời gian. |
| **Webcam** | Mở webcam của máy. |
| **Camera điện thoại** | Dùng điện thoại làm camera qua URL stream (xem mục dưới). |
| **Xoay khung hình** | Xoay khung hình 90° (khi camera điện thoại bị nghiêng). |
| **Dừng** | Dừng video/camera đang phát. |
| **Xoá** | Xoá nội dung đang hiển thị. |
| **Giao diện** | Chuyển chế độ **Sáng / Tối / Theo hệ thống**. |

### Dùng camera điện thoại
1. Cài app camera IP trên điện thoại: **IP Webcam** (Android) hoặc **DroidCam** (Android/iOS).
2. Điện thoại và máy tính **cùng mạng WiFi**.
3. Mở app → bật server → lấy địa chỉ, ví dụ `http://192.168.1.19:8080`.
4. Trong app bấm **Cam ĐT**, dán URL stream (nhớ thêm `/video`):
   `http://192.168.1.19:8080/video`

---

## 🧪 Kiểm thử nhanh (không cần GUI)

```bash
python -m tests.test_pipeline
```

Script sẽ tải mô hình, chạy phát hiện + OCR trên ảnh trong `test/images/` và in kết quả ra console.

---

## 🏋️ Huấn luyện lại mô hình phát hiện (tuỳ chọn)

Mô hình phát hiện biển dùng YOLOv8 (1 lớp `license-plate`). Để huấn luyện lại:

1. Chuẩn bị dataset theo định dạng YOLO và cập nhật `license_data.yaml`.
2. Chạy:
   ```bash
   python train.py
   ```
   Trọng số sẽ được lưu trong thư mục `runs/detect/`.

---

## 📌 Ghi chú

- Dự án chạy trên **CPU** (PyTorch CPU). Có GPU NVIDIA sẽ nhanh hơn nhưng không bắt buộc.
- Phần đọc ký tự dùng model OCR **pretrained** (RapidOCR), không cần dữ liệu huấn luyện riêng.
- Độ chính xác phụ thuộc chất lượng ảnh: biển rõ, ít nghiêng, đủ sáng sẽ cho kết quả tốt nhất.

---

## 👤 Tác giả

**TTHTech – Từ Thanh Hoài**
Đồ án môn Xử lý ảnh – Nhận diện biển số xe Việt Nam.
