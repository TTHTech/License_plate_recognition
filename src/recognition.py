"""Lớp nghiệp vụ nối phát hiện + OCR + bỏ phiếu kết quả.

- ``PlateVoter``: gom kết quả OCR qua nhiều frame, chọn chuỗi xuất hiện
  nhiều nhất để ổn định kết quả khi chạy video.
- ``recognize_crop``: chạy OCR (nặng) trên một vùng biển đã cắt, trả về cả
  ảnh đã nắn thẳng để hiển thị.
"""
import collections

from src import config, ocr


class PlateVoter:
    """Bỏ phiếu theo thời gian: kết quả cuối là chuỗi xuất hiện nhiều nhất."""

    def __init__(self, maxlen=config.VOTE_WINDOW):
        self._votes = collections.deque(maxlen=maxlen)

    def add(self, text):
        """Thêm một kết quả OCR, đã lọc nhiễu sơ bộ (vd 'E-Y')."""
        t = text.replace("-", "")
        if len(t) >= 6 and t[:2].isdigit():
            self._votes.append(text)

    def best(self):
        if not self._votes:
            return ""
        return collections.Counter(self._votes).most_common(1)[0][0]

    def clear(self):
        self._votes.clear()


def recognize_crop(crop):
    """OCR một vùng biển đã cắt. Trả về (crop, ảnh nắn thẳng, ảnh OCR, text)."""
    ocr_img, text = ocr.read_plate(crop)
    try:
        aligned = ocr.deskew(crop)
    except Exception:
        aligned = crop
    return crop, aligned, ocr_img, text
