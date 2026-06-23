import os
import collections
import threading
import time
import RotatedImg
import DetectLicense
import Preprocess
import cv2
import numpy as np
from ultralytics import YOLO
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import tkinter as tk
from PIL import Image, ImageTk
import EasyOCR

current_dir = os.path.dirname(os.path.abspath(__file__))

# ---- Model & OCR ----
model = YOLO(os.path.join(current_dir, 'runs', 'detect', 'train10', 'weights', 'best.pt'))

# ---- Theme ----
BG        = "#0f172a"   # nền app (slate-900)
HEADER    = "#1e293b"   # thanh header / bottom bar
PANEL     = "#ffffff"
BORDER    = "#334155"
ACCENT    = "#2563eb"
ACCENT_H  = "#1d4ed8"
ACCENT2   = "#ef4444"
ACCENT2_H = "#dc2626"
MUTED     = "#94a3b8"
CARD      = "#0b1220"
CYAN      = "#22d3ee"
OK        = "#22c55e"

# ---- Panel sizes ----
ORIG_W, ORIG_H   = 646, 506
PANEL_W, PANEL_H = 345, 215

root = Tk()
root.title("SUBJECT IMAGE PROCESSING")
root.geometry("1400x760+80+10")
root.resizable(False, False)
root.configure(bg=BG)

# ---- State ----
input_path = None
current_img = None
current_text_license = ""

status_var = StringVar(value="Hãy chọn một ảnh để bắt đầu.")
result_var = StringVar(value="---")


def show_in_label(pil_img, label, box_w, box_h):
    """Hiển thị ảnh PIL vào label, co kéo vừa khung."""
    img = pil_img.resize((box_w, box_h))
    photo = ImageTk.PhotoImage(img)
    label.configure(image=photo)
    label.image = photo  # giữ tham chiếu, tránh bị thu hồi


def clear_results():
    for lb in (lbl_detect, lbl_rotate, lbl_ocr):
        lb.configure(image='')
        lb.image = None
    result_var.set("---")


def process_image():
    """Phát hiện biển số -> xoay căn chỉnh -> OCR, tự động chạy sau khi mở ảnh."""
    global current_text_license
    if current_img is None:
        return

    status_var.set("Đang phát hiện biển số...")
    root.update_idletasks()

    crop = DetectLicense.DetectLicense(current_img, model)
    if crop is None:
        clear_results()
        status_var.set("Không phát hiện được biển số trong ảnh này.")
        return

    show_in_label(crop, lbl_detect, PANEL_W, PANEL_H)

    # Căn chỉnh góc nghiêng (không bắt buộc thành công)
    rotated = None
    try:
        rotated = RotatedImg.PILImgRotated(crop)
        show_in_label(rotated, lbl_rotate, PANEL_W, PANEL_H)
    except Exception as e:
        print("Rotate error:", e)

    status_var.set("Đang đọc ký tự (OCR)...")
    root.update_idletasks()

    # OCR trên cả ảnh gốc và ảnh đã căn chỉnh, chọn kết quả tốt hơn
    ocr_img, text = EasyOCR.OCRRESULT(crop, alt=rotated)
    show_in_label(ocr_img, lbl_ocr, PANEL_W, PANEL_H)

    current_text_license = text
    if text:
        result_var.set(text)
        status_var.set("Hoàn tất.")
    else:
        result_var.set("(không đọc được)")
        status_var.set("Đã phát hiện biển số nhưng không đọc được ký tự.")


def openimage():
    global input_path, current_img
    stop_video()
    path = filedialog.askopenfilename(
        title="Open Image",
        filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                   ("PNG file", "*.png"), ("JPG file", "*.jpg"),
                   ("All files", "*.*")))
    if not path:
        return
    input_path = path
    current_img = Image.open(path).convert('RGB')
    show_in_label(current_img, lbl_original, ORIG_W, ORIG_H)
    process_image()  # tự động nhận diện ngay sau khi chọn ảnh


# ===================== Chế độ VIDEO / WEBCAM =====================
video_cap = None
video_after_id = None
video_is_stream = False                      # nguồn là stream mạng/webcam?
video_rotate = 0                             # góc xoay khung hình: 0/90/180/270
frame_count = 0
plate_votes = collections.deque(maxlen=15)   # bỏ phiếu kết quả OCR theo thời gian
OCR_EVERY = 6                                # số frame tối thiểu giữa 2 lần OCR

# Đọc frame ở luồng nền + OCR ở luồng nền để giao diện không bị treo/trễ
_frame_lock = threading.Lock()
_latest_frame = None
_reader_thread = None
_stop_flag = threading.Event()
_ocr_busy = False
_pending_ocr = None                          # kết quả OCR chờ đổ lên giao diện


def _frame_reader():
    """Luồng nền: liên tục đọc frame, chỉ giữ frame MỚI NHẤT (bỏ frame tồn đọng)."""
    global _latest_frame
    while not _stop_flag.is_set():
        cap = video_cap
        if cap is None:
            break
        try:
            ok, f = cap.read()
        except Exception:
            ok, f = False, None
        if not ok or f is None:
            time.sleep(0.01)
            continue
        with _frame_lock:
            _latest_frame = f


def _ocr_worker(crop):
    """Luồng nền: chạy OCR nặng, lưu kết quả vào _pending_ocr."""
    global _ocr_busy, _pending_ocr
    try:
        ocr_img, text = EasyOCR.OCRRESULT(crop)
        try:
            aligned = EasyOCR._deskew(crop)
        except Exception:
            aligned = crop
        _pending_ocr = (crop, aligned, ocr_img, text)
    except Exception as e:
        print("OCR worker error:", e)
    finally:
        _ocr_busy = False


def stop_video():
    """Dừng video/webcam nếu đang chạy."""
    global video_cap, video_after_id, _reader_thread, _latest_frame
    _stop_flag.set()
    if video_after_id is not None:
        try:
            root.after_cancel(video_after_id)
        except Exception:
            pass
        video_after_id = None
    if _reader_thread is not None:
        _reader_thread.join(timeout=1.0)
        _reader_thread = None
    if video_cap is not None:
        video_cap.release()
        video_cap = None
    _latest_frame = None


def start_video(src, webcam=False):
    global video_cap, frame_count, video_is_stream, _reader_thread
    global _latest_frame, _ocr_busy, _pending_ocr
    stop_video()
    if webcam:
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not cap.isOpened():        # thử backend mặc định nếu DSHOW lỗi
            cap.release()
            cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không mở được nguồn video / webcam.\n"
                             "Kiểm tra lại file hoặc thiết bị camera.")
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # giảm độ trễ cho stream
    except Exception:
        pass
    video_cap = cap
    frame_count = 0
    _latest_frame = None
    _ocr_busy = False
    _pending_ocr = None
    # stream nếu là webcam hoặc URL mạng (http/rtsp/...)
    video_is_stream = webcam or (isinstance(src, str) and '://' in src)
    plate_votes.clear()
    result_var.set("---")
    status_var.set("Đang phát video... (bấm Dừng để thoát)")
    if video_is_stream:
        # Stream: đọc frame ở luồng nền để luôn lấy frame mới nhất (giảm trễ)
        _stop_flag.clear()
        _reader_thread = threading.Thread(target=_frame_reader, daemon=True)
        _reader_thread.start()
    video_loop()


def open_video():
    stop_video()
    path = filedialog.askopenfilename(
        title="Open Video",
        filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                   ("All files", "*.*")))
    if path:
        start_video(path)


def open_webcam():
    start_video(0, webcam=True)


def cycle_rotate():
    """Xoay khung hình video 90° mỗi lần bấm (cho camera bị nghiêng)."""
    global video_rotate
    video_rotate = (video_rotate + 90) % 360
    status_var.set(f"Xoay khung hình: {video_rotate}°")


def open_phone():
    """Dùng điện thoại làm camera qua URL stream (IP Webcam / DroidCam...)."""
    stop_video()
    url = simpledialog.askstring(
        "Camera điện thoại",
        "Nhập URL stream từ app camera trên điện thoại:\n"
        "• IP Webcam (Android):  http://192.168.x.x:8080/video\n"
        "• DroidCam:              http://192.168.x.x:4747/video\n"
        "(Điện thoại và máy tính phải cùng mạng WiFi)",
        initialvalue="http://192.168.1.5:8080/video")
    if url:
        start_video(url.strip())


def video_loop():
    global video_after_id, frame_count, _ocr_busy, _pending_ocr

    if video_cap is None:
        return

    # Lấy frame: stream -> frame mới nhất từ luồng nền; file -> đọc tuần tự
    if video_is_stream:
        with _frame_lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
        if frame is None:
            video_after_id = root.after(15, video_loop)
            return
        delay = 10
    else:
        try:
            ok, frame = video_cap.read()
        except Exception:
            ok, frame = False, None
        if not ok or frame is None:
            stop_video()
            status_var.set("Đã phát hết video.")
            return
        delay = 25
    frame_count += 1

    # Xoay khung hình nếu camera bị nghiêng (vd camera điện thoại để dọc)
    if video_rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif video_rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif video_rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Phát hiện biển trên mỗi frame (nhanh)
    results = model.predict(frame, conf=0.5, verbose=False)
    best_crop, best_box, best_area = None, None, 0
    for r in results:
        for b in r.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)
                best_crop = Image.fromarray(
                    cv2.cvtColor(frame[max(0, y1):y2, max(0, x1):x2],
                                 cv2.COLOR_BGR2RGB))

    # Khởi chạy OCR ở luồng nền (không chặn video) nếu đang rảnh
    if (best_crop is not None and frame_count % OCR_EVERY == 0
            and not _ocr_busy):
        _ocr_busy = True
        threading.Thread(target=_ocr_worker, args=(best_crop,),
                         daemon=True).start()

    # Đổ kết quả OCR (nếu có) lên giao diện - chạy ở luồng chính nên an toàn
    if _pending_ocr is not None:
        crop, aligned, ocr_img, text = _pending_ocr
        _pending_ocr = None
        show_in_label(crop, lbl_detect, PANEL_W, PANEL_H)
        show_in_label(aligned, lbl_rotate, PANEL_W, PANEL_H)
        show_in_label(ocr_img, lbl_ocr, PANEL_W, PANEL_H)
        t2 = text.replace('-', '')
        if len(t2) >= 6 and t2[:2].isdigit():    # lọc nhiễu (vd "E-Y")
            plate_votes.append(text)

    # Kết quả = chuỗi xuất hiện nhiều nhất (bỏ phiếu)
    plate = ""
    if plate_votes:
        plate = collections.Counter(plate_votes).most_common(1)[0][0]
        result_var.set(plate)

    # Vẽ biển số lên frame ngay trên khung phát hiện
    if best_box is not None and plate:
        x1, y1, _, _ = best_box
        cv2.putText(frame, plate, (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    show_in_label(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                  lbl_original, ORIG_W, ORIG_H)
    video_after_id = root.after(delay, video_loop)


def removeimage():
    global input_path, current_img
    was_video = video_cap is not None
    stop_video()
    if not input_path and not was_video:
        messagebox.showinfo("Thông báo", "Chưa có ảnh hoặc video nào được mở.")
        return
    input_path = None
    current_img = None
    lbl_original.configure(image='')
    lbl_original.image = None
    clear_results()
    plate_votes.clear()
    status_var.set("Đã xoá. Hãy chọn ảnh hoặc video khác.")


# ======================= UI =======================

def style_button(btn, base, hover):
    """Nút phẳng có hiệu ứng hover."""
    btn.configure(bg=base, activebackground=hover, fg="white", bd=0,
                  relief="flat", cursor="hand2", highlightthickness=0)
    btn.bind("<Enter>", lambda e: btn.configure(bg=hover))
    btn.bind("<Leave>", lambda e: btn.configure(bg=base))


def make_panel(title, x, y):
    """Tạo một khung có tiêu đề, trả về label chứa ảnh bên trong."""
    Label(root, text=title, bg=BG, fg=MUTED, font="arial 10 bold").place(x=x + 2, y=y)
    frame = Frame(root, bg=PANEL, width=PANEL_W, height=PANEL_H,
                  highlightthickness=1, highlightbackground=BORDER, bd=0)
    frame.place(x=x, y=y + 22)
    lbl = Label(frame, bg=PANEL)
    lbl.place(x=0, y=0)
    return lbl


# ----- Header band -----
header = Frame(root, bg=HEADER, width=1400, height=96)
header.place(x=0, y=0)
logo = PhotoImage(file=os.path.join(current_dir, "logos", "logo.png"))
Label(header, image=logo, bg=HEADER).place(x=20, y=14)
Label(header, text="NHẬN DIỆN & TRÍCH XUẤT BIỂN SỐ XE",
      bg=HEADER, fg="white", font="arial 25 bold").place(x=100, y=20)
Label(header, text="Vietnamese License Plate Recognition   •   YOLOv8  +  RapidOCR",
      bg=HEADER, fg=MUTED, font="arial 12").place(x=102, y=62)
Frame(root, bg=ACCENT, width=1400, height=3).place(x=0, y=96)

# ----- Khung ảnh gốc (bên trái) -----
Label(root, text="ẢNH GỐC", bg=BG, fg=MUTED, font="arial 10 bold").place(x=20, y=112)
f_orig = Frame(root, bg="black", width=ORIG_W, height=ORIG_H,
               highlightthickness=1, highlightbackground=BORDER, bd=0)
f_orig.place(x=18, y=134)
lbl_original = Label(f_orig, bg="black")
lbl_original.place(x=0, y=0)

# ----- 3 khung kết quả (bên phải) -----
lbl_detect = make_panel("BIỂN SỐ PHÁT HIỆN", 690, 112)
lbl_rotate = make_panel("ẢNH ĐÃ CĂN CHỈNH", 1045, 112)
lbl_ocr    = make_panel("KẾT QUẢ OCR (KHOANH VÙNG)", 690, 368)

# ----- Thẻ hiển thị biển số -----
Label(root, text="BIỂN SỐ XE", bg=BG, fg=MUTED, font="arial 10 bold").place(x=1047, y=368)
card = Frame(root, bg=CARD, width=PANEL_W, height=PANEL_H,
             highlightthickness=1, highlightbackground=ACCENT, bd=0)
card.place(x=1045, y=390)
Frame(card, bg=ACCENT, width=PANEL_W, height=6).place(x=0, y=0)
Label(card, text="KẾT QUẢ NHẬN DIỆN", bg=CARD, fg=MUTED,
      font="arial 10 bold").place(x=16, y=22)
Label(card, textvariable=result_var, bg=CARD, fg=CYAN,
      font="consolas 33 bold", wraplength=PANEL_W - 24).place(
          relx=0.5, rely=0.56, anchor="center")

# ----- Thanh dưới: nút bấm + trạng thái -----
bar = Frame(root, bg=HEADER, width=1372, height=74)
bar.place(x=18, y=662)
btn_open = Button(bar, text="Mở ảnh", font="arial 12 bold", command=openimage)
btn_open.place(x=12, y=15, width=104, height=44)
style_button(btn_open, ACCENT, ACCENT_H)
btn_video = Button(bar, text="Mở video", font="arial 12 bold", command=open_video)
btn_video.place(x=122, y=15, width=104, height=44)
style_button(btn_video, ACCENT, ACCENT_H)
btn_cam = Button(bar, text="Webcam", font="arial 12 bold", command=open_webcam)
btn_cam.place(x=232, y=15, width=98, height=44)
style_button(btn_cam, ACCENT, ACCENT_H)
btn_phone = Button(bar, text="Cam ĐT", font="arial 12 bold", command=open_phone)
btn_phone.place(x=336, y=15, width=98, height=44)
style_button(btn_phone, ACCENT, ACCENT_H)
btn_rotate = Button(bar, text="Xoay", font="arial 12 bold", command=cycle_rotate)
btn_rotate.place(x=440, y=15, width=84, height=44)
style_button(btn_rotate, "#475569", "#334155")
btn_stop = Button(bar, text="Dừng", font="arial 12 bold", command=stop_video)
btn_stop.place(x=530, y=15, width=76, height=44)
style_button(btn_stop, "#475569", "#334155")
btn_remove = Button(bar, text="Xoá", font="arial 12 bold", command=removeimage)
btn_remove.place(x=612, y=15, width=76, height=44)
style_button(btn_remove, ACCENT2, ACCENT2_H)

Label(bar, text="●", bg=HEADER, fg=OK, font="arial 13").place(x=716, y=14)
Label(bar, text="TRẠNG THÁI", bg=HEADER, fg=MUTED,
      font="arial 9 bold").place(x=736, y=13)
Label(bar, textvariable=status_var, bg=HEADER, fg="white",
      font="arial 13", anchor="w").place(x=736, y=33)

root.mainloop()
