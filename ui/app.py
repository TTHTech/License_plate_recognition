"""Ứng dụng GUI (CustomTkinter): nhận diện biển số từ ảnh / video / webcam / cam điện thoại.

Giao diện kiểu dashboard: sidebar điều hướng bên trái, vùng hiển thị bên phải,
hỗ trợ chế độ Sáng / Tối / Theo hệ thống.

Lớp ``LicensePlateApp`` chỉ lo phần giao diện và điều phối; toàn bộ xử lý
nặng (phát hiện, OCR, đọc frame) nằm trong các module thuộc ``src``.

Lưu ý: vùng ảnh dùng ``tk.Label`` + ``ImageTk.PhotoImage`` (ổn định, đã kiểm
chứng) thay vì ``CTkImage`` để tránh lỗi hiển thị khi đổi ảnh liên tục.
"""
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from src import config, ocr
from src.camera import VideoStream
from src.detection import PlateDetector
from src.recognition import PlateVoter, recognize_crop

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Kích thước mặc định của ô ảnh gốc (sẽ tự cập nhật theo cửa sổ)
ORIG_W, ORIG_H = 700, 540
PANEL_W, PANEL_H = 360, 150


def _resolve(color):
    """Lấy màu phù hợp chế độ hiện tại cho widget tk thuần (không tự đổi)."""
    if isinstance(color, (tuple, list)):
        return color[0] if ctk.get_appearance_mode() == "Light" else color[1]
    return color


class LicensePlateApp:
    def __init__(self):
        self.detector = PlateDetector()

        # ---- Trạng thái chung ----
        self.input_path = None
        self.current_img = None
        self.current_text_license = ""

        # ---- Trạng thái video ----
        self.video_stream = None
        self.video_after_id = None
        self.video_rotate = 0            # góc xoay khung hình: 0/90/180/270
        self.frame_count = 0
        self.voter = PlateVoter()
        self.ocr_busy = False
        self.pending_ocr = None          # kết quả OCR chờ đổ lên giao diện

        # Kích thước ô hiển thị ảnh gốc (cập nhật theo cửa sổ)
        self._orig_box = (ORIG_W, ORIG_H)
        self._last_original = None
        self._panel_labels = []          # các tk.Label hiển thị ảnh (để đổi nền theo theme)

        self._build_ui()

    # ======================= Tiện ích hiển thị =======================

    def show_in_label(self, pil_img, label, box_w, box_h):
        """Hiển thị ảnh PIL vào label, co kéo vừa khung."""
        photo = ImageTk.PhotoImage(pil_img.resize((box_w, box_h)))
        label.configure(image=photo, text="")
        label.image = photo  # giữ tham chiếu, tránh bị thu hồi

    def show_original(self, pil_img):
        """Hiển thị ảnh gốc / khung video: lấp đầy ô tối đa nhưng GIỮ tỉ lệ."""
        self._last_original = pil_img
        box_w, box_h = self._orig_box
        iw, ih = pil_img.size
        scale = min(box_w / iw, box_h / ih)
        size = (max(int(iw * scale), 1), max(int(ih * scale), 1))
        photo = ImageTk.PhotoImage(pil_img.resize(size))
        self.lbl_original.configure(image=photo, text="")
        self.lbl_original.image = photo

    def _on_orig_resize(self, event):
        """Cập nhật kích thước ô ảnh gốc khi cửa sổ đổi cỡ, vẽ lại nếu cần."""
        self._orig_box = (max(event.width - 12, 80), max(event.height - 12, 80))
        if self._last_original is not None:
            self.show_original(self._last_original)

    def clear_results(self):
        for lb in (self.lbl_detect, self.lbl_rotate, self.lbl_ocr):
            lb.configure(image="", text="")
            lb.image = None
        self.result_var.set("---")

    # ======================= Chế độ ẢNH =======================

    def process_image(self):
        """Phát hiện biển số -> nắn thẳng -> OCR (tự chạy sau khi mở ảnh)."""
        if self.current_img is None:
            return

        self.status_var.set("Đang phát hiện biển số...")
        self.root.update_idletasks()

        crop = self.detector.detect_crop(self.current_img)
        if crop is None:
            self.clear_results()
            self.status_var.set("Không phát hiện được biển số trong ảnh này.")
            return

        self.show_in_label(crop, self.lbl_detect, PANEL_W, PANEL_H)

        # Căn chỉnh góc nghiêng (không bắt buộc thành công)
        aligned = None
        try:
            aligned = ocr.deskew(crop)
            self.show_in_label(aligned, self.lbl_rotate, PANEL_W, PANEL_H)
        except Exception as e:
            print("Deskew error:", e)

        self.status_var.set("Đang đọc ký tự (OCR)...")
        self.root.update_idletasks()

        ocr_img, text = ocr.read_plate(crop, alt=aligned)
        self.show_in_label(ocr_img, self.lbl_ocr, PANEL_W, PANEL_H)

        self.current_text_license = text
        if text:
            self.result_var.set(text)
            self.status_var.set("Hoàn tất.")
        else:
            self.result_var.set("(không đọc được)")
            self.status_var.set("Đã phát hiện biển số nhưng không đọc được ký tự.")

    def open_image(self):
        self.stop_video()
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                       ("PNG file", "*.png"), ("JPG file", "*.jpg"),
                       ("All files", "*.*")))
        if not path:
            return
        self.input_path = path
        self.current_img = Image.open(path).convert("RGB")
        self.show_original(self.current_img)
        self.process_image()  # tự động nhận diện ngay sau khi chọn ảnh

    # ======================= Chế độ VIDEO / WEBCAM =======================

    def _ocr_worker(self, crop):
        """Luồng nền: chạy OCR nặng, lưu kết quả vào self.pending_ocr."""
        try:
            self.pending_ocr = recognize_crop(crop)
        except Exception as e:
            print("OCR worker error:", e)
        finally:
            self.ocr_busy = False

    def stop_video(self):
        """Dừng video/webcam nếu đang chạy."""
        if self.video_after_id is not None:
            try:
                self.root.after_cancel(self.video_after_id)
            except Exception:
                pass
            self.video_after_id = None
        if self.video_stream is not None:
            self.video_stream.release()
            self.video_stream = None

    def start_video(self, src, webcam=False):
        self.stop_video()
        stream = VideoStream(src, webcam=webcam)
        if not stream.opened:
            stream.release()
            messagebox.showerror("Lỗi", "Không mở được nguồn video / webcam.\n"
                                 "Kiểm tra lại file hoặc thiết bị camera.")
            return
        self.video_stream = stream
        self.frame_count = 0
        self.ocr_busy = False
        self.pending_ocr = None
        self.voter.clear()
        self.result_var.set("---")
        self.status_var.set("Đang phát video... (bấm Dừng để thoát)")
        self.video_loop()

    def open_video(self):
        self.stop_video()
        path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")))
        if path:
            self.start_video(path)

    def open_webcam(self):
        self.start_video(0, webcam=True)

    def cycle_rotate(self):
        """Xoay khung hình video 90° mỗi lần bấm (cho camera bị nghiêng)."""
        self.video_rotate = (self.video_rotate + 90) % 360
        self.status_var.set(f"Xoay khung hình: {self.video_rotate}°")

    def open_phone(self):
        """Dùng điện thoại làm camera qua URL stream (IP Webcam / DroidCam...)."""
        self.stop_video()
        url = simpledialog.askstring(
            "Camera điện thoại",
            "Nhập URL stream từ app camera trên điện thoại:\n"
            "• IP Webcam (Android):  http://192.168.x.x:8080/video\n"
            "• DroidCam:              http://192.168.x.x:4747/video\n"
            "(Điện thoại và máy tính phải cùng mạng WiFi)",
            initialvalue="http://192.168.1.5:8080/video")
        if url:
            self.start_video(url.strip())

    def _rotate_frame(self, frame):
        if self.video_rotate == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.video_rotate == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.video_rotate == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def video_loop(self):
        stream = self.video_stream
        if stream is None:
            return

        ok, frame = stream.read()
        if stream.is_stream:
            if not ok:                      # chưa có frame -> thử lại sớm
                self.video_after_id = self.root.after(15, self.video_loop)
                return
            delay = 10
        else:
            if not ok or frame is None:     # hết file video
                self.stop_video()
                self.status_var.set("Đã phát hết video.")
                return
            delay = 25
        self.frame_count += 1

        frame = self._rotate_frame(frame)

        # Phát hiện biển (vẽ khung lên frame) + lấy biển lớn nhất
        best_crop, best_box = self.detector.detect_largest(frame)

        # Khởi chạy OCR ở luồng nền (không chặn video) nếu đang rảnh
        if (best_crop is not None and self.frame_count % config.OCR_EVERY == 0
                and not self.ocr_busy):
            self.ocr_busy = True
            threading.Thread(target=self._ocr_worker, args=(best_crop,),
                             daemon=True).start()

        # Đổ kết quả OCR (nếu có) lên giao diện - chạy ở luồng chính nên an toàn
        if self.pending_ocr is not None:
            crop, aligned, ocr_img, text = self.pending_ocr
            self.pending_ocr = None
            self.show_in_label(crop, self.lbl_detect, PANEL_W, PANEL_H)
            self.show_in_label(aligned, self.lbl_rotate, PANEL_W, PANEL_H)
            self.show_in_label(ocr_img, self.lbl_ocr, PANEL_W, PANEL_H)
            self.voter.add(text)

        # Kết quả = chuỗi xuất hiện nhiều nhất (bỏ phiếu)
        plate = self.voter.best()
        if plate:
            self.result_var.set(plate)

        # Vẽ biển số lên frame ngay trên khung phát hiện
        if best_box is not None and plate:
            x1, y1, _, _ = best_box
            cv2.putText(frame, plate, (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        self.show_original(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video_after_id = self.root.after(delay, self.video_loop)

    def remove_image(self):
        was_video = self.video_stream is not None
        self.stop_video()
        if not self.input_path and not was_video:
            messagebox.showinfo("Thông báo", "Chưa có ảnh hoặc video nào được mở.")
            return
        self.input_path = None
        self.current_img = None
        self._last_original = None
        self.lbl_original.configure(image="", text="Chưa có nguồn ảnh / video")
        self.lbl_original.image = None
        self.clear_results()
        self.voter.clear()
        self.status_var.set("Đã xoá. Hãy chọn ảnh hoặc video khác.")

    # ======================= Dựng giao diện =======================

    def _nav_button(self, parent, text, command, primary=True):
        if primary:
            fg, hover, txt = config.ACCENT, config.ACCENT_H, "white"
        else:
            fg, hover, txt = config.NEUTRAL, config.NEUTRAL_H, config.TEXT
        return ctk.CTkButton(parent, text=text, command=command, height=42,
                             corner_radius=10, font=ctk.CTkFont(size=14, weight="bold"),
                             fg_color=fg, hover_color=hover, text_color=txt, anchor="w")

    def _make_card(self, parent, title, img_w, img_h):
        """Thẻ bo góc có tiêu đề; trả về (thẻ, tk.Label hiển thị ảnh)."""
        card = ctk.CTkFrame(parent, corner_radius=14, fg_color=config.CARD,
                            border_width=1, border_color=config.BORDER)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=config.MUTED).pack(anchor="w", padx=16, pady=(12, 6))
        holder = ctk.CTkFrame(card, fg_color=config.CARD_INNER, corner_radius=10,
                              width=img_w + 8, height=img_h + 8)
        holder.pack(expand=True, pady=(0, 12))   # expand -> căn giữa theo chiều dọc
        holder.pack_propagate(False)
        lbl = tk.Label(holder, text="", bd=0, bg=_resolve(config.CARD_INNER),
                       fg=_resolve(config.MUTED), font=("Segoe UI", 11))
        lbl.pack(expand=True, fill="both", padx=4, pady=4)
        self._panel_labels.append(lbl)
        return card, lbl

    def _build_ui(self):
        self.root = ctk.CTk()
        self.root.title("LPR System — Nhận diện biển số xe Việt Nam")
        self.root.geometry("1500x860+60+10")
        self.root.minsize(1300, 780)
        self.root.configure(fg_color=config.MAIN_BG)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Sẵn sàng. Hãy chọn một nguồn để bắt đầu.")
        self.result_var = tk.StringVar(value="---")

        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        bar = ctk.CTkFrame(self.root, width=270, corner_radius=0, fg_color=config.SIDEBAR)
        bar.grid(row=0, column=0, sticky="nsew")
        bar.grid_propagate(False)

        # ---- Logo + tên ----
        try:
            logo_pil = Image.open(config.LOGO_PATH).convert("RGBA")
            self._logo_img = ctk.CTkImage(light_image=logo_pil, dark_image=logo_pil,
                                          size=(56, 56))
            ctk.CTkLabel(bar, image=self._logo_img, text="").pack(pady=(24, 6))
        except Exception:
            pass
        ctk.CTkLabel(bar, text="LPR SYSTEM", font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=config.TEXT).pack()
        ctk.CTkLabel(bar, text="Nhận diện biển số xe Việt Nam",
                     font=ctk.CTkFont(size=11), text_color=config.MUTED).pack(pady=(0, 18))

        # ---- Nhóm nguồn ----
        ctk.CTkLabel(bar, text="NGUỒN NHẬN DIỆN", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=config.MUTED).pack(anchor="w", padx=22, pady=(4, 6))
        for text, cmd in (("Mở ảnh", self.open_image),
                          ("Mở video", self.open_video),
                          ("Webcam", self.open_webcam),
                          ("Camera điện thoại", self.open_phone)):
            self._nav_button(bar, text, cmd, primary=True).pack(fill="x", padx=18, pady=4)

        # ---- Nhóm điều khiển ----
        ctk.CTkLabel(bar, text="ĐIỀU KHIỂN", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=config.MUTED).pack(anchor="w", padx=22, pady=(18, 6))
        for text, cmd in (("Xoay khung hình", self.cycle_rotate),
                          ("Dừng", self.stop_video)):
            self._nav_button(bar, text, cmd, primary=False).pack(fill="x", padx=18, pady=4)
        ctk.CTkButton(bar, text="Xoá", command=self.remove_image, height=42,
                      corner_radius=10, font=ctk.CTkFont(size=14, weight="bold"),
                      fg_color=config.DANGER, hover_color=config.DANGER_H,
                      text_color="white", anchor="w").pack(fill="x", padx=18, pady=4)

        # ---- Đẩy phần dưới xuống đáy ----
        ctk.CTkFrame(bar, fg_color="transparent").pack(expand=True, fill="both")

        ctk.CTkLabel(bar, text="GIAO DIỆN", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=config.MUTED).pack(anchor="w", padx=22, pady=(4, 4))
        ctk.CTkOptionMenu(bar, values=["Tối", "Sáng", "Theo hệ thống"],
                          command=self._change_appearance,
                          fg_color=config.NEUTRAL, text_color=config.TEXT,
                          button_color=config.ACCENT, button_hover_color=config.ACCENT_H,
                          dropdown_fg_color=config.CARD).pack(fill="x", padx=18, pady=(0, 12))

        ctk.CTkFrame(bar, height=1, fg_color=config.BORDER).pack(fill="x", padx=18, pady=(0, 12))
        ctk.CTkLabel(bar, text="YOLOv8  +  RapidOCR", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=config.MUTED).pack()
        ctk.CTkLabel(bar, text="Vietnamese License Plate Recognition",
                     font=ctk.CTkFont(size=10), text_color=config.MUTED).pack(pady=(0, 16))

    def _build_main(self):
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=24, pady=24)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(1, weight=1)

        # ---- Banner kết quả (toàn chiều ngang) ----
        banner = ctk.CTkFrame(main, corner_radius=16, fg_color=config.CARD,
                              border_width=1, border_color=config.ACCENT)
        banner.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 18))
        banner.grid_columnconfigure(1, weight=1)
        ctk.CTkFrame(banner, width=6, height=78, corner_radius=3,
                     fg_color=config.ACCENT).grid(row=0, column=0, rowspan=2,
                                                  padx=(16, 14), pady=14, sticky="ns")
        ctk.CTkLabel(banner, text="BIỂN SỐ NHẬN DIỆN",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=config.MUTED).grid(row=0, column=1, sticky="w", pady=(14, 0))
        ctk.CTkLabel(banner, textvariable=self.result_var,
                     font=ctk.CTkFont(family="Consolas", size=44, weight="bold"),
                     text_color=config.CYAN).grid(row=1, column=1, sticky="w", pady=(0, 14))

        # Cụm đèn + trạng thái bên phải banner
        st = ctk.CTkFrame(banner, fg_color="transparent")
        st.grid(row=0, column=2, rowspan=2, padx=20, sticky="e")
        ctk.CTkLabel(st, text="●", text_color=config.OK,
                     font=ctk.CTkFont(size=16)).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(st, textvariable=self.status_var, font=ctk.CTkFont(size=13),
                     text_color=config.TEXT, wraplength=360, justify="left").pack(side="left")

        # ---- Khung ảnh gốc (trái) ----
        left = ctk.CTkFrame(main, corner_radius=14, fg_color=config.CARD,
                            border_width=1, border_color=config.BORDER)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 18))
        ctk.CTkLabel(left, text="ẢNH GỐC / CAMERA",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=config.MUTED).pack(anchor="w", padx=18, pady=(14, 8))
        holder = ctk.CTkFrame(left, fg_color="#000000", corner_radius=10)
        holder.pack(expand=True, fill="both", padx=14, pady=(0, 14))
        holder.bind("<Configure>", self._on_orig_resize)
        self.lbl_original = tk.Label(holder, text="Chưa có nguồn ảnh / video",
                                     bd=0, bg="#000000", fg="#94a3b8",
                                     font=("Segoe UI", 14))
        self.lbl_original.pack(expand=True, fill="both")

        # ---- Cột kết quả (phải) ----
        right = ctk.CTkFrame(main, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure((0, 1, 2), weight=1, uniform="cards")  # chia đều 3 thẻ

        card1, self.lbl_detect = self._make_card(right, "BIỂN SỐ PHÁT HIỆN", PANEL_W, PANEL_H)
        card1.grid(row=0, column=0, sticky="nsew", pady=(0, 14))
        card2, self.lbl_rotate = self._make_card(right, "ẢNH ĐÃ CĂN CHỈNH", PANEL_W, PANEL_H)
        card2.grid(row=1, column=0, sticky="nsew", pady=(0, 14))
        card3, self.lbl_ocr = self._make_card(right, "KẾT QUẢ OCR (KHOANH VÙNG)", PANEL_W, PANEL_H)
        card3.grid(row=2, column=0, sticky="nsew")

    def _change_appearance(self, choice):
        mapping = {"Tối": "dark", "Sáng": "light", "Theo hệ thống": "system"}
        ctk.set_appearance_mode(mapping.get(choice, "dark"))
        # Cập nhật nền các tk.Label (widget tk thuần không tự đổi theo theme)
        inner = _resolve(config.CARD_INNER)
        muted = _resolve(config.MUTED)
        for lbl in self._panel_labels:
            lbl.configure(bg=inner, fg=muted)

    def run(self):
        self.root.mainloop()
