"""Quản lý nguồn video / webcam / camera điện thoại.

``VideoStream`` gói ``cv2.VideoCapture`` lại:
- Với stream (webcam/URL mạng): chạy một luồng nền liên tục đọc frame và chỉ
  giữ frame MỚI NHẤT (bỏ frame tồn đọng) để giảm độ trễ.
- Với file video: đọc tuần tự ngay trong ``read()``.
"""
import threading
import time

import cv2


class VideoStream:
    def __init__(self, src, webcam=False):
        if webcam:
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened():          # thử backend mặc định nếu DSHOW lỗi
                cap.release()
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)

        self.cap = cap
        self.opened = cap.isOpened()
        # stream nếu là webcam hoặc URL mạng (http/rtsp/...)
        self.is_stream = webcam or (isinstance(src, str) and "://" in src)

        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        if self.opened:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # giảm độ trễ cho stream
            except Exception:
                pass
            if self.is_stream:
                self._thread = threading.Thread(target=self._reader, daemon=True)
                self._thread.start()

    def _reader(self):
        """Luồng nền: liên tục đọc frame, chỉ giữ frame mới nhất."""
        while not self._stop.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = frame

    def read(self):
        """Trả về (ok, frame_bgr).

        - Stream: frame mới nhất; ``ok=False`` khi chưa có frame nào.
        - File: đọc tuần tự; ``ok=False`` khi hết video.
        """
        if self.is_stream:
            with self._lock:
                if self._latest is None:
                    return False, None
                return True, self._latest.copy()
        try:
            return self.cap.read()
        except Exception:
            return False, None

    def release(self):
        """Dừng luồng nền và giải phóng thiết bị."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._latest = None
