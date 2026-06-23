# OCR biển số xe Việt Nam bằng RapidOCR (PP-OCR chạy trên ONNX).
# Model pretrained, không cần dataset / không cần train.
# Pipeline: phóng to + viền trắng -> RapidOCR dò & đọc -> gom dòng theo toạ độ
#           -> bỏ dòng nhiễu (logo) -> sửa lỗi theo định dạng biển VN.
import re
import statistics
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR

LETTERS = 'ABCDEFGHKLMNPSTUVXYZ'
# Map chữ -> số cho các vị trí bắt buộc là số (lỗi nhầm thường gặp)
DIGIT_MAP = {
    'O': '0', 'Q': '0', 'D': '0', 'U': '0',
    'I': '1', 'J': '1', 'Z': '2', 'A': '4',
    'S': '5', 'G': '6', 'T': '7', 'B': '8',
}

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def _dg(c):
    return DIGIT_MAP.get(c, c)


def _fix_top(s):
    """Dòng trên: chỉ ép 2 ký tự đầu (mã tỉnh) về số, giữ nguyên phần series.
    (Series có thể là 1 chữ, chữ+số, hoặc 2 chữ như 29-AB, nên không ép phần này.)"""
    if not s:
        return s
    return ''.join(_dg(c) if i < 2 else c for i, c in enumerate(s))


def _fix_bottom(s):
    return ''.join(_dg(c) for c in s)


def _score(text):
    """Chấm điểm độ hợp lệ của chuỗi biển số theo định dạng VN."""
    if not text:
        return -1
    s = sum(c.isalnum() for c in text)              # càng nhiều ký tự càng tốt
    if re.match(r'^\d{2}[A-Z]{1,2}\d?-\d{4,5}$', text):
        s += 100                                    # khớp dạng biển 2 dòng chuẩn
    elif re.match(r'^\d{2}[A-Z]{1,2}\d?\d{4,5}$', text):
        s += 60                                     # đúng dạng nhưng thiếu '-'
    elif re.match(r'^\d{2}[A-Z]', text):
        s += 20
    if '-' in text:
        s += 5
    return s


def _read_one(image):
    engine = _get_engine()
    W, H = image.size
    two_line_shape = (W / max(H, 1)) < 2.2

    arr = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    big = cv2.resize(arr, (W * 3, H * 3), interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 20, 20, 20, 20,
                             cv2.BORDER_CONSTANT, value=(255, 255, 255))

    res, _ = engine(big)
    disp = Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))

    if not res:
        return disp, ''

    # Vẽ khung minh hoạ
    draw = ImageDraw.Draw(disp)
    try:
        font = ImageFont.truetype('arial.ttf', 26)
    except Exception:
        font = ImageFont.load_default()

    items = []
    for box, text, score in res:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        cleaned = ''.join(ch for ch in text.upper() if ch.isalnum())
        if not cleaned:
            continue
        cy = sum(ys) / 4.0
        cx = sum(xs) / 4.0
        h = max(ys) - min(ys)
        items.append([cy, cx, h, cleaned])
        draw.rectangle([min(xs), min(ys), max(xs), max(ys)],
                       outline=(0, 255, 0), width=2)
        draw.text((min(xs), max(0, min(ys) - 28)), cleaned,
                  font=font, fill=(255, 0, 0))

    if not items:
        return disp, ''

    # Gom thành các dòng theo toạ độ y
    items.sort()
    mh = statistics.median([it[2] for it in items])
    lines, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it[0] - cur[-1][0]) <= mh * 0.6:
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    lines.append(cur)

    # Mỗi dòng: ghép theo x; chỉ giữ dòng có chứa số (loại logo chữ thuần)
    cand = []
    for ln in lines:
        ln.sort(key=lambda x: x[1])
        s = ''.join(x[3] for x in ln)
        if any(ch.isdigit() for ch in s):
            cand.append([statistics.mean(x[0] for x in ln), s,
                         statistics.median([x[2] for x in ln])])

    if len(cand) > 2:                       # giữ 2 dòng cao nhất (là biển số)
        cand = sorted(cand, key=lambda x: -x[2])[:2]
    cand.sort(key=lambda x: x[0])           # thứ tự trên -> dưới
    texts = [c[1] for c in cand]

    if len(texts) == 2:
        return disp, _fix_top(texts[0]) + '-' + _fix_bottom(texts[1])
    if len(texts) == 1:
        s = texts[0]
        # Biển vuông nhưng 2 dòng bị gộp -> tách 5 số cuối làm dòng dưới
        if two_line_shape and len(s) >= 8:
            return disp, _fix_top(s[:-5]) + '-' + _fix_bottom(s[-5:])
        return disp, _fix_top(s)
    return disp, ''


def _deskew(image):
    """Nắn thẳng biển nghiêng: xoay thử nhiều góc, chọn góc làm hàng chữ ngang nhất."""
    rgb = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    th = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = gray.shape
    best_angle, best_score = 0, -1.0
    for ang in range(-15, 16):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        rot = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_NEAREST)
        proj = rot.sum(axis=1).astype(np.float64)
        score = float(((proj[1:] - proj[:-1]) ** 2).sum())  # hàng càng sắc nét
        if score > best_score:
            best_score, best_angle = score, ang
    if best_angle == 0:
        return image
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
    rot = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return Image.fromarray(rot)


def _rec_line(bgr):
    """OCR chỉ-nhận-diện trên một dải ảnh 1 dòng."""
    out = _get_engine()(bgr, use_det=False, use_cls=False, use_rec=True)
    res = out[0] if isinstance(out, tuple) else out
    if not res:
        return ''
    first = res[0]
    txt = first[0] if isinstance(first, (list, tuple)) else str(first)
    return ''.join(ch for ch in txt.upper() if ch.isalnum())


def _read_split(image):
    """Nắn thẳng -> cắt 2 dòng theo projection -> OCR từng dòng."""
    rgb = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    bgr = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (W * 3, H * 3),
                     interpolation=cv2.INTER_CUBIC)
    disp = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if W / max(H, 1) >= 2.2:                 # biển 1 dòng
        return disp, _fix_top(_rec_line(bgr))

    th = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    rs = th.sum(axis=1).astype(np.float64)
    lo, hi = int(H * 0.35), int(H * 0.65)
    sp = (lo + int(np.argmin(rs[lo:hi]))) / float(H)
    top = bgr[:int(sp * H * 3)]
    bot = bgr[int(sp * H * 3):]
    t, b = _fix_top(_rec_line(top)), _fix_bottom(_rec_line(bot))
    return disp, (t + '-' + b if t and b else t + b)


def OCRRESULT(image, alt=None, reader=None):
    """Thử nhiều cách đọc (gốc, căn chỉnh, tự nắn thẳng, cắt dòng) rồi chọn bản hợp lệ nhất."""
    desk = None
    try:
        desk = _deskew(image)
    except Exception as e:
        print("Deskew error:", e)

    candidates = []
    for img in (image, alt, desk):
        if img is None:
            continue
        try:
            candidates.append(_read_one(img))
        except Exception as e:
            print("OCR error:", e)
    if desk is not None:
        try:
            candidates.append(_read_split(desk))
        except Exception as e:
            print("Split-OCR error:", e)

    best = None
    for disp, text in candidates:
        sc = _score(text)
        if best is None or sc > best[0]:
            best = (sc, disp, text)

    if best is None:
        return image, ''
    return best[1], best[2]
