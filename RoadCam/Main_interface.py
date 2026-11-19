import sys
import os
import csv
from datetime import datetime
import cv2
import numpy as np
import json
import math
import traceback

from PyQt6.QtCore import Qt, QTimer, QPoint, QPointF, QRectF, QEvent, QSize, QSettings
from PyQt6.QtGui import (
    QImage, QPixmap, QAction, QKeySequence, QPainter, QPen, QColor,
    QBrush, QPolygonF
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QFileDialog, QMessageBox, QSlider, QSpinBox, QComboBox,
    QListWidget, QListWidgetItem, QSplitter, QSizePolicy, QCheckBox,
    QDialog, QGridLayout, QTabWidget, QToolBar, QButtonGroup, QStackedWidget, QStyle, QDoubleSpinBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


from M2_adjustment import *
from M3_filters import *
from M4_noise import *
from M5_color import hsv_channels, range_mask, lane_mask
from M6_morphology import binarize, erode, dilate, opening, closing, remove_small_regions
from M7_edges import sobel, canny, detect_lane, hough_all
from M8_video import *
OPERATION_ERRORS = (
    cv2.error,
    TypeError,
    ValueError,
    ZeroDivisionError
)
# =========================================================
# BẢNG MÀU GIAO DIỆN (tối / sáng) — một nơi duy nhất
# =========================================================

PALETTES = {
    "dark": dict(BG="#0d1420", PANEL="#151e2d", PANEL_2="#1a2537", BORDER="#29364c",
                 TEXT="#dce5f3", MUTED="#8a9ab1", ACCENT="#4485ff",
                 SELECTED="#273d62", CANVAS="#0b0e14"),
    "light": dict(BG="#f1f4f8", PANEL="#ffffff", PANEL_2="#edf2f7", BORDER="#d9e1eb",
                  TEXT="#182639", MUTED="#66768a", ACCENT="#2563eb",
                  SELECTED="#e3ecff", CANVAS="#e4e9f0"),
}
P = dict(PALETTES["dark"])   # bảng màu đang dùng; apply_theme() cập nhật tại chỗ


def canvas_style():
    return f"background: transparent; color: {P['MUTED']}; font-size: 16px;"


# =========================================================
# HỖ TRỢ ĐỌC / GHI ẢNH CÓ ĐƯỜNG DẪN TIẾNG VIỆT
# =========================================================

def read_image_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        if image.ndim == 3 and image.shape[2] == 2:
            return None
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image, alpha=255.0 / max(1, image.max()))
        return image
    except Exception:
        return None


def write_image_unicode(path, image):
    ext = os.path.splitext(path)[1].lower() or ".png"
    if ext in (".jpg", ".jpeg") and image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("OpenCV không thể mã hóa ảnh ở định dạng này.")
    encoded.tofile(path)


def image_channels(image):
    if image is None:
        return 0
    return 1 if image.ndim == 2 else image.shape[2]


def pixel_text(image, x, y):
    """Giá trị điểm ảnh (x, y) dạng chữ; dùng chung cho thanh trạng thái và tệp CSV."""
    px = image[y, x]
    if image.ndim == 2:
        return f"(Gray) = {int(px)}"
    return f"(B, G, R) = ({int(px[0])}, {int(px[1])}, {int(px[2])})"



def to_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def to_qpixmap(image):
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    if image.ndim == 2:
        qimg = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_Grayscale8)
    elif image.shape[2] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format.Format_RGBA8888)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# =========================================================
# XỬ LÝ ẢNH CƠ BẢN (G1)
# =========================================================

ASPECT_RATIOS = [("Tự do (pixel)", None), ("Gốc của ảnh", "original"),
                 ("1:1  vuông", (1, 1)), ("3:4  dọc", (3, 4)), ("4:3  ngang", (4, 3)),
                 ("2:3  dọc", (2, 3)), ("16:9  ngang", (16, 9)), ("9:16  dọc", (9, 16)),
                 ("3:2  ngang", (3, 2)), ("4:5  dọc", (4, 5)), ("5:4  ngang", (5, 4)),
                 ("21:9  rộng", (21, 9))]


def ratio_size(width, height, ratio):
    """Khung lớn nhất có tỉ lệ `ratio` nằm gọn trong ảnh width × height."""
    rw, rh = ratio
    if width * rh > height * rw:
        return max(1, round(height * rw / rh)), height
    return width, max(1, round(width * rh / rw))


def resize_photo(image, width, height, mode, interpolation):
    h, w = image.shape[:2]
    if mode == "crop":
        if w * height > h * width:
            crop_w = max(1, round(h * width / height))
            left = (w - crop_w) // 2
            image = image[:, left:left + crop_w]
        else:
            crop_h = max(1, round(w * height / width))
            top = (h - crop_h) // 2
            image = image[top:top + crop_h, :]
    elif mode == "contain":
        scale = min(width / w, height / h)
        nw, nh = min(width, max(1, round(w * scale))), min(height, max(1, round(h * scale)))
        resized = cv2.resize(image, (nw, nh), interpolation=interpolation)
        result = np.full((height, width) + image.shape[2:], 255, dtype=image.dtype)
        left, top = (width - nw) // 2, (height - nh) // 2
        result[top:top + nh, left:left + nw] = resized
        return result
    return cv2.resize(image, (width, height), interpolation=interpolation)


def keep_alpha(func):
    """Tách kênh alpha (nếu có) trước khi xử lý rồi ghép lại sau."""
    def wrapper(image, **kw):
        alpha = None
        if image.ndim == 3 and image.shape[2] == 4:
            alpha, image = image[..., 3], image[..., :3]
        out = func(np.ascontiguousarray(image), **kw)
        info = None
        if isinstance(out, tuple):      # hàm trả (ảnh, {tên: giá trị}) cho Bảng kết quả
            out, info = out
        if alpha is not None:
            if out.ndim == 2:           # mặt nạ / ảnh biên: đổi sang BGR để ghép lại với alpha
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            out = np.dstack([out, alpha])
        return out if info is None else (out, info)
    return wrapper


# =========================================================
# HÀM XỬ LÝ TỰ VIẾT TRONG FILE NÀY
# (hàm của M2 / M3 / M4 được nối thẳng vào bảng lệnh bên dưới, không cần hàm bọc)
# =========================================================

def cmd_brightness(image, value):
    return np.clip(image.astype(np.float32) + value, 0, 255).astype(np.uint8)


def cmd_contrast(image, a, b):
    if b <= a:
        b = a + 1
    return contrast_stretching(img=image, a=a, b=b)


def cmd_temperature(image, value):
    if image.ndim != 3:
        return image
    f = image.astype(np.float32)
    f[..., 2] += value * 0.6   # R
    f[..., 0] -= value * 0.6   # B
    return np.clip(f, 0, 255).astype(np.uint8)


def cmd_saturation(image, value):
    if image.ndim != 3:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def cmd_levels(image, value):
    levels = int(value)
    if not 2 <= levels < 256:
        return image
    q = np.floor(image.astype(np.float32) * levels / 256) * (255 / (levels - 1))
    return np.clip(q, 0, 255).astype(np.uint8)


def cmd_invert(image):
    return 255 - image


def _fixed_seed(func, seed=12345):
    """Cùng một nhiễu cho xem trước và Áp dụng, không làm đổi random toàn cục."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return func()
    finally:
        np.random.set_state(state)


def cmd_gaussian_noise(image, sigma):
    return _fixed_seed(lambda: app_gaussian_noise(image, mean=0, sigma=float(sigma)))


def cmd_salt_pepper_noise(image, p):
    return _fixed_seed(lambda: app_salt_pepper_noise(image, p=float(p)))


def cmd_denoise(image, method, ksize, q):
    return denoising(image, method=method, ksize=int(ksize), q=float(q))


# ---------------------------------------------------------
# NỐI M5 / M6 / M7 (module của nhóm) VÀO GIAO DIỆN
# Các hàm cmd_* chỉ đổi "giá trị thanh trượt" thành đối số đúng của hàm trong module,
# không thêm thuật toán nào. Thuật toán nằm ở M5_color.py, M6_morphology.py, M7_edges.py.
# ---------------------------------------------------------

def to_gray(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _need_color(image):
    if image.ndim != 3:
        raise ValueError("Cần ảnh màu (3 kênh) để chọn vùng theo màu HSV.")


def _mask_or_image(image, mask, view):
    """view='mask' -> mặt nạ trắng/đen; còn lại -> ảnh chỉ giữ vùng được chọn."""
    return mask if view == "mask" else cv2.bitwise_and(image, image, mask=mask)


# ---- M5: Màu ----
def cmd_hsv_range(image, hmin, hmax, smin, smax, vmin, vmax, view):
    _need_color(image)
    return _mask_or_image(image, range_mask(image, (hmin, smin, vmin), (hmax, smax, vmax)), view)


def cmd_lane_mask(image, w_smax, w_vmin, y_hmin, y_hmax, y_smin, y_vmin, view, roi_mask=None):
    _need_color(image)
    mask = lane_mask(image, roi_mask,
                     white_range=((0, 0, w_vmin), (179, w_smax, 255)),
                     yellow_range=((y_hmin, y_smin, y_vmin), (y_hmax, 255, 255)))
    return _mask_or_image(image, mask, view)


# ---- M6: Hình thái (Co / Giãn / Mở / Đóng gọi thẳng erode, dilate, opening, closing) ----
def cmd_binarize(image, threshold, otsu):
    binary, used = binarize(image, int(threshold), bool(otsu))
    return binary, {"Ngưỡng đã dùng": int(used)}


def cmd_remove_small(image, min_area):
    cleaned, remain, removed = remove_small_regions(image, min_area)
    return cleaned, {"Vùng còn lại": remain, "Vùng đã xóa": removed}


# ---- M7: Biên ----
def cmd_sobel(image):
    return sobel(to_gray(image))


def cmd_canny(image, canny_low, canny_high):
    return canny(to_gray(image), canny_low, canny_high)


def cmd_hough(image, canny_low, canny_high, threshold, min_len, max_gap, mode):
    kw = dict(canny_low=canny_low, canny_high=canny_high,
              threshold=threshold, min_len=min_len, max_gap=max_gap)
    if mode == "classify":
        return detect_lane(image, roi=None, **kw)       # phân loại trái (xanh lá) / phải (đỏ)
    return hough_all(image, roi=None, **kw)             # vẽ mọi đoạn


def cmd_detect_lane(image, canny_low, canny_high, threshold, min_len, max_gap, roi_points=None):
    if roi_points is None:
        raise ValueError("Chưa có ROI hợp lệ cho ảnh này: chọn công cụ 'ROI hình thang' "
                         "và click 4 đỉnh (hoặc Nạp ROI).")
    return detect_lane(image, roi=roi_points, canny_low=canny_low, canny_high=canny_high,
                       threshold=threshold, min_len=min_len, max_gap=max_gap)


def mse_psnr(a, b):
    """MSE và PSNR (dB) giữa 2 ảnh uint8 cùng kích thước — nguồn duy nhất cho mọi nơi cần PSNR."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return mse, (float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse))


def similarity_metrics(a, b):
    """Độ giống nhau giữa 2 ảnh uint8 cùng kích thước."""
    mse, psnr = mse_psnr(a, b)
    return {
        "mse": mse,
        "psnr": psnr,
        "max_diff": int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()),
        "same_pct": float(np.mean(a == b) * 100),
    }


# =========================================================
# BẢNG LỆNH — mỗi lệnh là MỘT dòng command(...)
# Tham số dạng slider: (key, label, min, max, default, scale, decimals)
# Tham số dạng combo:  (key, label, "combo", [(text, data), ...], default)
# `key` trùng tên tham số của hàm xử lý, nên nối thẳng được vào hàm của M2/M3/M4.
# =========================================================

def _img(func):
    """Bọc hàm của M2/M3/M4 (gọi bằng img=...) thành hàm nhận ảnh ở đối số đầu, có giữ alpha."""
    return keep_alpha(lambda image, **kw: func(img=image, **kw))


def command(name, func, *params, lib=False, **extra):
    """lib=True: `func` là hàm M2/M3/M4. lib=False: hàm tự viết ở trên. Cả hai đều được giữ alpha."""
    return {"name": name, "func": _img(func) if lib else keep_alpha(func),
            "params": list(params), **extra}


equalize_cv2 = _img(hist_balance_openCV)   # dùng cho nút "So sánh với OpenCV"

KSIZE_ITEMS = [(f"{k} × {k}", k) for k in (3, 5, 7, 9)]
MORPH_SHAPES = [("Hình chữ nhật", cv2.MORPH_RECT),
                ("Hình elip", cv2.MORPH_ELLIPSE),
                ("Hình chữ thập", cv2.MORPH_CROSS)]
ITERATION_ITEMS = [(str(i), i) for i in (1, 2, 3, 4, 5)]
DENOISE_METHODS = [("Trung bình hình học", "geometric"),
                   ("Phản điều hòa ", "contraharmonic")]


def ksize_param(default=5, label="Kích thước nhân"):
    return ("ksize", label, "combo", KSIZE_ITEMS, default)


KSIZE_MORPH = [(f"{k} × {k}", k) for k in (3, 5, 7, 9, 11, 13, 15)]   # đề: kích thước 3–15
MORPH_PARAMS = [("ksize", "Kích thước nhân", "combo", KSIZE_MORPH, 3),
                ("shape", "Hình dạng nhân", "combo", MORPH_SHAPES, cv2.MORPH_RECT),
                ("iterations", "Số lần lặp", "combo", ITERATION_ITEMS, 1)]

COMMANDS = {
    "invert": command("Âm bản (Invert)", cmd_invert),
    "brightness": command("Độ sáng", cmd_brightness, ("value", "Độ sáng", -100, 100, 0, 1, 0)),
    "logarit": command("Biến đổi Log", log_adjust, lib=True),
    "gamma": command("Gamma", gamma_adjust, ("gamma", "Gamma", 10, 300, 1.0, 100, 2), lib=True),
    "contrast": command("Độ tương phản", cmd_contrast,
                        ("a", "a", 0, 255, 0, 1, 2), ("b", "b", 0, 255, 255, 1, 2), editable=True),
    "balance histogram": command("Cân bằng Histogram", hist_balance, lib=True,
                                 buttons=[("So sánh với OpenCV", "compare_equalize_opencv")]),
    "clahe": command("CLAHE", clahe_adjust,
                     ("clip_limit", "Clip limit", 10, 50, 2.0, 10, 1),
                     ("tile", "Kích thước lưới ô", 4, 16, 8, 1, 0), lib=True),
    "temperature": command("Nhiệt độ màu", cmd_temperature,
                           ("value", "Nhiệt độ màu", -100, 100, 0, 1, 0)),
    "saturation": command("Độ bão hòa", cmd_saturation, ("value", "Độ bão hòa", 0, 300, 1.0, 100, 2)),
    "levels": command("Lượng tử hóa", cmd_levels, ("value", "Số mức", 2, 256, 256, 1, 0)),
}

FILTERS = {
    "mean_blur": command("Bộ lọc trung bình", mean_filter, ksize_param(), lib=True),
    "gaussian_blur": command("Bộ lọc Gaussian", gaussian_filter, ksize_param(),
                             ("sigma", "Sigma", 50, 500, 1.0, 100, 2), lib=True),
    "median_blur": command("Bộ lọc trung vị", median_filter, ksize_param(), lib=True),
    "laplacian": command("Làm sắc nét Laplacian", laplacian_filter,
                         ("c", "Cường độ", 0, 100, 0.5, 100, 2), lib=True),
    "unsharp": command("Unsharp Masking", unsharp_masking,
                       ("k", "Cường độ", 0, 300, 0.5, 100, 2), lib=True),
    # section="Nhiễu": thành menu con trong menu và có nhãn riêng trong tab
    "gaussian_noise": command("Thêm nhiễu Gaussian", cmd_gaussian_noise,
                              ("sigma", "Sigma (σ)", 5, 50, 10, 1, 0), section="Nhiễu"),
    "salt_pepper_noise": command("Thêm nhiễu muối–tiêu", cmd_salt_pepper_noise,
                                 ("p", "Tỉ lệ điểm nhiễu (%)", 1, 20, 5, 1, 0), section="Nhiễu"),
    "denoise": command("Giảm nhiễu", cmd_denoise,
                       ("method", "Bộ lọc", "combo", DENOISE_METHODS, "contraharmonic"),
                       ksize_param(3, "Kích thước cửa sổ"),
                       ("q", "Q (chỉ cho phản điều hòa)", -200, 200, 1.5, 100, 2), section="Nhiễu"),
}

# ---- M5 (menu Màu) ----
VIEW_PARAM = ("view", "Hiển thị", "combo",
              [("Ảnh chỉ giữ vùng được chọn", "image"), ("Mặt nạ trắng / đen", "mask")], "image")
SIDE_BY_SIDE = [("Xem mặt nạ | ảnh song song", "show_mask_and_image")]

COLORS = {
    "hsv_range": command("Chọn theo dải màu", cmd_hsv_range,
                         ("hmin", "H min (0–179)", 0, 179, 0, 1, 0),
                         ("hmax", "H max (0–179)", 0, 179, 179, 1, 0),
                         ("smin", "S min", 0, 255, 0, 1, 0), ("smax", "S max", 0, 255, 255, 1, 0),
                         ("vmin", "V min", 0, 255, 0, 1, 0), ("vmax", "V max", 0, 255, 255, 1, 0),
                         VIEW_PARAM, buttons=SIDE_BY_SIDE),
    "lane": command("Tách vạch làn", cmd_lane_mask,
                    ("w_smax", "Vạch trắng: S tối đa", 0, 255, 40, 1, 0),
                    ("w_vmin", "Vạch trắng: V tối thiểu", 0, 255, 200, 1, 0),
                    ("y_hmin", "Vạch vàng: H min", 0, 179, 15, 1, 0),
                    ("y_hmax", "Vạch vàng: H max", 0, 179, 35, 1, 0),
                    ("y_smin", "Vạch vàng: S min", 0, 255, 80, 1, 0),
                    ("y_vmin", "Vạch vàng: V min", 0, 255, 100, 1, 0),
                    VIEW_PARAM, needs_roi="mask", buttons=SIDE_BY_SIDE),
}

# ---- M7 (menu Biên) ---- (giá trị mặc định lấy theo ví dụ trong M7_edges.py của nhóm)
CANNY_PARAMS = [("canny_low", "Canny: ngưỡng thấp", 0, 255, 50, 1, 0),
                ("canny_high", "Canny: ngưỡng cao", 0, 255, 150, 1, 0)]
HOUGH_PARAMS = [("threshold", "Hough: threshold (số phiếu)", 1, 200, 15, 1, 0),
                ("min_len", "Hough: minLineLength", 1, 300, 40, 1, 0),
                ("max_gap", "Hough: maxLineGap", 0, 300, 20, 1, 0)]

EDGES = {
    "sobel": command("Sobel", cmd_sobel),
    "canny": command("Canny", cmd_canny, *CANNY_PARAMS),
    "hough": command("Tìm đường thẳng (Hough)", cmd_hough, *CANNY_PARAMS, *HOUGH_PARAMS,
                     ("mode", "Cách vẽ", "combo",
                      [("Phân loại trái (xanh lá) / phải (đỏ)", "classify"),
                       ("Vẽ mọi đoạn", "all")], "classify")),
    "lane": command("Phát hiện làn (tự động)", cmd_detect_lane, *CANNY_PARAMS, *HOUGH_PARAMS,
                    needs_roi="points"),
}

# Mục menu không phải "lệnh có tham số" (mở hộp thoại): {nhóm: [(tên, tên phương thức)]}
MENU_ACTIONS = {"Màu": [("Xem kênh H, S, V", "show_hsv_channels")]}

MORPHOLOGY = {
    "binarize": command("Nhị phân hóa", cmd_binarize,
                        ("threshold", "Ngưỡng", 0, 255, 127, 1, 0),
                        ("otsu", "Cách chọn ngưỡng", "combo",
                         [("Ngưỡng tay (thanh trượt)", 0), ("Tự động Otsu", 1)], 0)),
    "erode": command("Co (Erode)", erode, *MORPH_PARAMS),
    "dilate": command("Giãn (Dilate)", dilate, *MORPH_PARAMS),
    "opening": command("Mở (Opening)", opening, *MORPH_PARAMS),
    "closing": command("Đóng (Closing)", closing, *MORPH_PARAMS),
    "remove_small": command("Xóa vùng nhỏ", cmd_remove_small,
                            ("min_area", "Diện tích tối thiểu (px)", 0, 5000, 50, 1, 0)),
}

# ---- MỤC 2: nhóm lệnh dùng chung cho menu, tab Thông số và tra cứu cấu hình ----
GROUPS = {"Điều chỉnh": COMMANDS, "Bộ lọc": FILTERS, "Màu": COLORS,
          "Hình thái": MORPHOLOGY, "Biên": EDGES}

TRANSFORMS = [("Xoay 90° phải", lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)),
              ("Xoay 90° trái", lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)),
              ("Lật ngang", lambda im: cv2.flip(im, 1)),
              ("Lật dọc", lambda im: cv2.flip(im, 0))]

# =========================================================
# VÙNG HIỂN THỊ ẢNH: zoom theo chuột, vừa khung, ROI, thước, kéo
# =========================================================

# =========================================================
# ROI HÌNH THANG (M1 - RoadCam)
# =========================================================

def order_quad(points):
    """Sắp 4 điểm theo góc quanh tâm để không bị tứ giác 'thắt nơ' dù click thứ tự nào."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def roi_mask(shape, points):
    """Mặt nạ cùng kích thước ảnh: trong đa giác = 255, ngoài = 0."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


def roi_apply(image, points):
    """Chỉ giữ vùng trong ROI (ngoài ROI thành đen)."""
    mask = roi_mask(image.shape, points)
    return cv2.bitwise_and(image, image, mask=mask)

class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.zoom = 1.0
        self.fit_mode = True
        self.scroll = None              # QScrollArea chứa canvas
        self.mouse_callback = None
        self.zoom_callback = None
        self.interaction_callback = None
        self.interaction_mode = "none"  # none / roi / ruler / pan
        self.is_dragging = False # ROI
        self.drag_start = None
        self.drag_end = None
        self.ruler_points = []
        self.pan_last = None
        self.roi_poly_points = []     # các đỉnh ROI hình thang (tọa độ ảnh)
        self.roi_poly_size = None      # (w, h) của ảnh lúc vẽ ROI
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet(canvas_style())

    # ---------- hiển thị ----------
    def set_image(self, image):
        size_changed = (
            self.image is None or image is None
            or self.image.shape[:2] != image.shape[:2]
        )
        self.image = image
        if size_changed and self.fit_mode:
            self.fit_to_view()
        else:
            self.refresh()

    def refresh(self):
        if self.image is None:
            self.setPixmap(QPixmap())
            self.setText("MỞ ẢNH ĐỂ BẮT ĐẦU  (Ctrl+O)")
            if self.scroll:
                vp = self.scroll.viewport()
                self.resize(vp.width(), vp.height())
            return
        h, w = self.image.shape[:2]
        dw, dh = max(1, int(w * self.zoom)), max(1, int(h * self.zoom))
        mode = (Qt.TransformationMode.FastTransformation if self.zoom >= 2
                else Qt.TransformationMode.SmoothTransformation)
        pixmap = to_qpixmap(self.image).scaled(
            dw, dh, Qt.AspectRatioMode.IgnoreAspectRatio, mode)
        self.setText("")
        self.setPixmap(pixmap)
        self.resize(dw, dh)
        if self.zoom_callback:
            self.zoom_callback(self.zoom)

    def fit_to_view(self):
        """Ảnh khớp với vùng hiển thị (giữ tỉ lệ)."""
        self.fit_mode = True
        if self.image is None or self.scroll is None:
            self.refresh()
            return
        vp = self.scroll.viewport()
        h, w = self.image.shape[:2]
        self.zoom = max(0.02, min((vp.width() - 8) / w, (vp.height() - 8) / h))
        self.refresh()

    def set_zoom(self, zoom, anchor=None):
        """Đổi zoom, giữ điểm ảnh dưới `anchor` (tọa độ canvas) đứng yên."""
        if self.image is None or self.scroll is None:
            return
        vp = self.scroll.viewport()
        if anchor is None:
            vpos = QPoint(vp.width() // 2, vp.height() // 2)
            cpos = self.mapFrom(vp, vpos)
        else:
            cpos = anchor
            vpos = self.mapTo(vp, cpos)
        ix, iy = cpos.x() / self.zoom, cpos.y() / self.zoom
        self.fit_mode = False
        self.zoom = min(max(zoom, 0.02), 16.0)
        self.refresh()
        self.scroll.horizontalScrollBar().setValue(int(ix * self.zoom - vpos.x()))
        self.scroll.verticalScrollBar().setValue(int(iy * self.zoom - vpos.y()))

    # ---------- tương tác ----------
    def set_interaction_mode(self, mode):
        self.interaction_mode = mode
        self.drag_start = self.drag_end = None
        self.is_dragging = False
        self.ruler_points = []
        self.pan_last = None
        cursors = {
            "roi": Qt.CursorShape.CrossCursor,
            "ruler": Qt.CursorShape.PointingHandCursor,
            "pan": Qt.CursorShape.OpenHandCursor,
            "roi_poly": Qt.CursorShape.CrossCursor,
        }
        self.setCursor(cursors.get(mode, Qt.CursorShape.ArrowCursor))
        self.update()

    def image_point(self, pos, clamp=False):
        if self.image is None:
            return None
        h, w = self.image.shape[:2]
        x, y = int(pos.x() / self.zoom), int(pos.y() / self.zoom)
        if clamp:
            return (min(max(x, 0), w), min(max(y, 0), h))
        if 0 <= x < w and 0 <= y < h:
            return (x, y)
        return None

    def mousePressEvent(self, event):
        if self.image is None:
            return
        if event.button() == Qt.MouseButton.MiddleButton or self.interaction_mode == "pan":
            self.pan_last = event.globalPosition()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.interaction_mode == "roi":
            # bấm ngoài ảnh vẫn kéo được, điểm bị kẹp vào mép ảnh
            self.drag_start = self.drag_end = self.image_point(
                event.position(), clamp=True
            )
            self.is_dragging = True
            self.update()
            return
        if self.interaction_mode == "roi_poly":
            h, w = self.image.shape[:2]
            point = self.image_point(event.position(), clamp=True)
            # bắt đầu ROI mới nếu đã đủ 4 điểm hoặc ảnh đổi kích thước
            if self.roi_poly_size != (w, h) or len(self.roi_poly_points) >= 4:
                self.roi_poly_points = []
            self.roi_poly_size = (w, h)
            self.roi_poly_points.append(point)
            self.update()
            if len(self.roi_poly_points) == 4 and self.interaction_callback:
                self.interaction_callback("roi_poly", list(self.roi_poly_points), None)
            return
        point = self.image_point(event.position())
        if point is None:
            return
        if self.interaction_mode == "ruler":
            self.ruler_points.append(point)
            if len(self.ruler_points) > 2:
                self.ruler_points = [point]
            self.update()
            if self.interaction_callback:
                self.interaction_callback("point", point, None)
            if len(self.ruler_points) == 2 and self.interaction_callback:
                self.interaction_callback("ruler", *self.ruler_points)

    def mouseMoveEvent(self, event):
        # đang kéo ảnh: chỉ kéo khi còn giữ chuột trái hoặc chuột giữa
        if self.pan_last is not None:
            held = event.buttons() & (Qt.MouseButton.LeftButton | Qt.MouseButton.MiddleButton)
            if not held:                      # đã nhả chuột -> dừng kéo
                self.pan_last = None
                self.set_interaction_mode(self.interaction_mode)   # trả con trỏ về đúng chế độ
            elif self.scroll:
                delta = event.globalPosition() - self.pan_last
                self.pan_last = event.globalPosition()
                hb, vb = self.scroll.horizontalScrollBar(), self.scroll.verticalScrollBar()
                hb.setValue(hb.value() - int(delta.x()))
                vb.setValue(vb.value() - int(delta.y()))
                return
        point = self.image_point(event.position())
        if point is not None and self.mouse_callback:
            self.mouse_callback(*point)
        if self.interaction_mode == "roi" and self.is_dragging:
            self.drag_end = self.image_point(event.position(), clamp=True)
            self.update()

    def mouseReleaseEvent(self, event):
        if (
            self.interaction_mode == "roi"
            and event.button() == Qt.MouseButton.LeftButton
            and self.is_dragging
        ):
            self.drag_end = self.image_point(
                event.position(), clamp=True
            )

            self.is_dragging = False

            if self.interaction_callback:
                self.interaction_callback(
                    "roi", self.drag_start, self.drag_end
                )

            self.update()

    def wheelEvent(self, event):
        if self.image is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.set_zoom(self.zoom * factor, event.position().toPoint())
        event.accept()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        z = self.zoom
        if self.interaction_mode == "roi" and self.drag_start and self.drag_end:
            (x1, y1), (x2, y2) = self.drag_start, self.drag_end
            rect = QRectF(min(x1, x2) * z, min(y1, y2) * z, abs(x2 - x1) * z, abs(y2 - y1) * z)
            painter.fillRect(rect, QColor(47, 124, 246, 40))
            painter.setPen(QPen(QColor("#4da3ff"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(rect)
            painter.setPen(QColor("white"))
            painter.drawText(rect.topLeft() + QPointF(4, -4), f"{abs(x2 - x1)} × {abs(y2 - y1)}")
        if self.interaction_mode == "ruler" and self.ruler_points:
            painter.setPen(QPen(QColor("#ff4d6d"), 2))
            pts = [QPointF(x * z, y * z) for x, y in self.ruler_points]
            for p in pts:
                painter.drawEllipse(p, 4, 4)
            if len(pts) == 2:
                painter.drawLine(pts[0], pts[1])
        if (self.interaction_mode == "roi_poly" and self.roi_poly_points
                and self.roi_poly_size == (self.image.shape[1], self.image.shape[0])):
            pts = [QPointF(x * z, y * z) for x, y in self.roi_poly_points]
            painter.setPen(QPen(QColor("#4da3ff"), 2))
            if len(pts) == 4:
                painter.setBrush(QBrush(QColor(47, 124, 246, 50)))
                painter.drawPolygon(QPolygonF(pts))
            else:
                for i in range(len(pts) - 1):
                    painter.drawLine(pts[i], pts[i + 1])
            painter.setBrush(QBrush(QColor("#4da3ff")))
            for i, p in enumerate(pts):
                painter.drawEllipse(p, 4, 4)
                painter.drawText(p + QPointF(7, -7), str(i + 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.end()


# =========================================================
# HISTOGRAM RGB (nền tối, tô màu)
# =========================================================

class HistogramCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(3, 1.6), facecolor=P["PANEL"])
        self.figure.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMinimumHeight(110)
        self.mode = "RGB"
        self.update_histogram(None)

    def update_histogram(self, image):
        ax = self.axes
        ax.clear()
        ax.set_facecolor(P["PANEL"])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if image is None:
            ax.text(0.5, 0.5, "Chưa có ảnh", ha="center", va="center",
                    color=P["MUTED"], transform=ax.transAxes)
            self.draw_idle()
            return
        x = np.arange(256)
        if image.ndim == 2 or self.mode == "Xám":
            gray = image if image.ndim == 2 else cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
            ax.fill_between(x, hist, color="#c9d2e0", alpha=0.6, linewidth=0)
        else:
            bgr = to_bgr(image)
            for ch, color, name in [(2, "#ff3b3b", "R"), (1, "#2ee66b", "G"), (0, "#3b7bff", "B")]:
                if self.mode not in ("RGB", name):
                    continue
                hist = cv2.calcHist([bgr], [ch], None, [256], [0, 256]).ravel()
                ax.fill_between(x, hist, color=color, alpha=0.45, linewidth=0)
                ax.plot(x, hist, color=color, linewidth=0.8)
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        self.draw_idle()


# =========================================================
# SO SÁNH TRƯỚC / SAU (thanh trượt kéo được)
# =========================================================

class CompareWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.before = None
        self.after = None
        self.split = 0.5
        self.setMinimumHeight(140)
        self.setCursor(Qt.CursorShape.SizeHorCursor)

    def set_images(self, before, after):
        self.before = to_qpixmap(before) if before is not None else None
        self.after = to_qpixmap(after) if after is not None else None
        self.update()

    def target_rect(self):
        pm = self.before or self.after
        if pm is None:
            return QRectF(self.rect())
        s = min(self.width() / pm.width(), self.height() / pm.height())
        w, h = pm.width() * s, pm.height() * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(P["PANEL"]))
        if self.before is None or self.after is None:
            p.setPen(QColor(P["MUTED"]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Chưa có ảnh")
            return
        r = self.target_rect()
        p.drawPixmap(r, self.after, QRectF(self.after.rect()))
        sx = r.left() + r.width() * self.split
        p.save()
        p.setClipRect(QRectF(r.left(), r.top(), sx - r.left(), r.height()))
        p.drawPixmap(r, self.before, QRectF(self.before.rect()))
        p.restore()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("white"), 2))
        p.drawLine(QPointF(sx, r.top()), QPointF(sx, r.bottom()))
        p.setBrush(QBrush(QColor(P["ACCENT"])))
        p.drawEllipse(QPointF(sx, r.center().y()), 9, 9)

    def _move(self, event):
        r = self.target_rect()
        if r.width() > 0:
            self.split = min(max((event.position().x() - r.left()) / r.width(), 0), 1)
            self.update()

    def mousePressEvent(self, event):
        self._move(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._move(event)


# =========================================================
# Ô PANEL CÓ TIÊU ĐỀ (kiểu card tối)
# =========================================================

class Card(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        self.header = QHBoxLayout()
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.header.addWidget(self.title)
        self.header.addStretch()
        lay.addLayout(self.header)
        self.body = lay


class ParamSlider(QWidget):
    """Một dòng tham số: tên ............ giá trị  + slider.

    editable=True: thay nhãn giá trị bằng ô nhập số, ô nhập và thanh trượt đồng bộ hai chiều.
    """

    def __init__(self, name, minimum, maximum, default, scale=1, decimals=0,
                 editable=False, parent=None):
        super().__init__(parent)
        self.scale, self.decimals, self.default = scale, decimals, default
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        row = QHBoxLayout()
        row.addWidget(QLabel(name))
        row.addStretch()
        self.value_label = QLabel()
        self.spin = None
        if editable:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(decimals)
            self.spin.setRange(minimum / scale, maximum / scale)
            self.spin.setSingleStep(1 / scale)
            self.spin.setKeyboardTracking(False)   # chỉ nhận số khi nhấn Enter / rời ô
            self.spin.setFixedWidth(84)
            self.spin.valueChanged.connect(self._spin_changed)
            row.addWidget(self.spin)
        else:
            row.addWidget(self.value_label)
        lay.addLayout(row)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.valueChanged.connect(self._update_label)
        lay.addWidget(self.slider)
        self.reset()

    def value(self):
        return self.slider.value() / self.scale

    def reset(self):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(self.default * self.scale)))
        self.slider.blockSignals(False)
        self._update_label()

    def _spin_changed(self, v):
        self.slider.setValue(int(round(v * self.scale)))   # kích hoạt valueChanged → xem trước

    def _update_label(self, *_):
        v = self.value()
        if self.spin is not None:
            self.spin.blockSignals(True)
            self.spin.setValue(v)
            self.spin.blockSignals(False)
        else:
            self.value_label.setText(f"{v:.{self.decimals}f}" if self.decimals else f"{int(v)}")

class ParamCombo(QWidget):
    def __init__(self, name, items, default, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        self.default = default
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        lay.addWidget(QLabel(name))
        self.combo = QComboBox()
        for text, data in items:
            self.combo.addItem(text, data)
        lay.addWidget(self.combo)
        self.reset()
    
    def value(self):
        return self.combo.currentData()
    
    def reset(self):
        index = self.combo.findData(self.default)
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(index if index >= 0 else 0)
        self.combo.blockSignals(False)
 

# ====================================================
# ỨNG DỤNG CHÍNH
# =========================================================

class ImageStudio(VideoModuleM8, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageStudio 2 — Chỉnh sửa ảnh & ảnh thẻ")
        self.resize(1500, 900)
        self.setMinimumSize(1000, 700)
        self.settings = QSettings("ImageStudio", "ImageStudio")
        self.theme_name = self.settings.value("theme", "dark")

        self.original_image = None
        self.current_image = None
        self.preview_image = None
        self.command_info = {}   # số liệu do lệnh trả về, hiện ở Bảng kết quả
        self.image_path = ""
        self.roi_points = None
        self.roi_polygon = None   # ROI hình thang: 4 điểm đã sắp thứ tự
        self.roi_size = None   
        
        # G3: lịch sử dạng danh sách trạng thái + con trỏ (giống Photoshop)
        self.history = []          # [(tên, ảnh)]
        self.history_index = -1
        self.max_history = 20

        # Video
        self.init_video_state()

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(40)
        self.preview_timer.timeout.connect(self.update_preview)

        self.apply_theme()
        self.create_menu()
        self.create_toolbar()
        self.build_interface()
        self.create_status_bar()
        self.refresh_history_list()
        self.canvas.refresh()
        self.update_photo_preset()

    # -----------------------------------------------------
    # THEME
    # -----------------------------------------------------

    def apply_theme(self):
        P.update(PALETTES[self.theme_name if self.theme_name in PALETTES else "dark"])
        BG, PANEL, PANEL_2, BORDER, TEXT, MUTED, ACCENT, selected, canvas_bg = (
            P[k] for k in ("BG", "PANEL", "PANEL_2", "BORDER", "TEXT", "MUTED",
                           "ACCENT", "SELECTED", "CANVAS"))
        self.setStyleSheet(f"""
            QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI'; font-size: 11px; }}
            QMenuBar, QToolBar, QStatusBar {{ background: {PANEL}; border: none; padding: 5px; }}
            QMenuBar::item {{ padding: 5px 9px; background: transparent; }}
            QMenu, QComboBox QAbstractItemView {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 5px; }}
            QMenu::item {{ padding: 7px 20px; }}
            QMenu::item:selected, QListWidget::item:selected {{ background: {selected}; color: {TEXT}; }}
            QToolBar {{ spacing: 5px; border-bottom: 1px solid {BORDER}; }}
            QToolBar#tools {{ border-right: 1px solid {BORDER}; padding: 8px 5px; }}
            QLabel#logo {{ font-size: 14px; font-weight: 600; padding: 5px 12px; }}
            QLabel#toolTitle {{ font-size: 11px; font-weight: 700; color: {MUTED}; padding: 8px 5px; }}
            QLabel#toolSection {{ font-size: 10px; font-weight: 600; color: {MUTED}; padding: 12px 5px 4px; }}
            QLabel#muted {{ color: {MUTED}; }}
            QLabel#cardTitle {{ font-size: 13px; font-weight: 600; }}
            QLabel#resultValue {{ font-weight: 600; }}
            QFrame#welcome {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 18px; }}
            QFrame#welcome QLabel {{ background: transparent; border: none; }}
            QLabel#welcomeTitle {{ font-size: 32px; font-weight: 700; }}
            QLabel#welcomeSubtitle {{ font-size: 14px; color: {MUTED}; }}
            QLabel#eyebrow {{ color: {ACCENT}; font-size: 11px; font-weight: 700; }}
            QFrame#card QPushButton#ratio {{ padding: 4px 2px; font-size: 11px; font-weight: 600; }}
            QPushButton#openWelcome {{ background: {ACCENT}; color: white; border: none; padding: 13px 26px; font-weight: 600; }}
            QFrame#card, QFrame#resultCard {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; border-top: 2px solid {ACCENT}; }}
            QFrame#resultCard {{ border-top-color: #e9a34e; }}
            QFrame#card QWidget, QFrame#resultCard QWidget {{ background: transparent; }}
            QFrame#card QPushButton#adjustHead {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 7px; padding: 0 12px; text-align: left; }}
            QFrame#card QPushButton#adjustHead:hover {{ background: {selected}; border-color: {ACCENT}; }}
            QFrame#card QPushButton#adjustHead:checked {{ background: {selected}; border-color: {ACCENT}; }}
            QPushButton, QToolButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 7px; padding: 6px 8px; }}
            QPushButton:hover, QToolButton:hover {{ background: {selected}; border-color: {ACCENT}; }}
            QPushButton:checked, QToolButton:checked {{ background: {selected}; border-color: {ACCENT}; color: {ACCENT}; }}
            QPushButton#tool {{ text-align: left; background: transparent; border: 1px solid transparent; padding: 8px 7px; }}
            QPushButton#tool:checked {{ background: {selected}; color: {ACCENT}; border-color: {BORDER}; }}
            QPushButton#primary, QFrame#card QPushButton#primary {{ background: {ACCENT}; color: white; border-color: {ACCENT}; font-weight: 600; }}
            QPushButton:disabled, QToolButton:disabled {{ color: {MUTED}; }}
            QComboBox, QSpinBox, QDoubleSpinBox {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 7px; min-height: 20px; }}
            QComboBox QAbstractItemView {{ selection-background-color: {selected}; selection-color: {TEXT}; }}
            QTabWidget::pane {{ border: none; padding-top: 7px; }}
            QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 7px 7px; border-bottom: 2px solid transparent; }}
            QTabBar::tab:selected {{ color: {ACCENT}; border-bottom-color: {ACCENT}; font-weight: 600; }}
            QScrollArea {{ background: {canvas_bg}; border: 1px solid {BORDER}; border-radius: 10px; }}
            QScrollArea > QWidget > QWidget {{ background: {canvas_bg}; }}
            QScrollArea#inspectorScroll {{ background: transparent; border: none; }}
            QScrollArea#inspectorScroll > QWidget > QWidget {{ background: {BG}; }}
            QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
            QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -5px 0; border-radius: 7px; }}
            QScrollBar {{ background: transparent; width: 8px; height: 8px; }}
            QScrollBar::handle {{ background: {BORDER}; border-radius: 4px; min-height: 25px; min-width: 25px; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
            QListWidget {{ background: transparent; border: none; outline: none; }}
            QListWidget::item {{ padding: 9px 6px; border-radius: 5px; }}
            QSplitter::handle {{ background: {BG}; }}
            QStatusBar {{ border-top: 1px solid {BORDER}; }}
            QStatusBar QLabel {{ color: {MUTED}; padding: 2px 8px; }}
            
        """)
        if hasattr(self, "histogram"):
            self.histogram.figure.set_facecolor(PANEL)
            self.histogram.update_histogram(self.display_source())
            self.canvas.setStyleSheet(canvas_style())
            self.compare_widget.update()

    def change_theme(self, index):
        self.theme_name = "light" if index == 1 else "dark"
        self.settings.setValue("theme", self.theme_name)
        self.apply_theme()

    # -----------------------------------------------------
    # MENU (Tệp, Ảnh, Điều chỉnh, Bộ lọc, Màu, Hình thái, Biên, Video, AI, Trợ giúp)
    # -----------------------------------------------------

    def add_action(self, menu, text, slot, shortcut=None):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def create_menu(self):
        bar = self.menuBar()

        m = bar.addMenu("Tệp")
        self.add_action(m, "Mở ảnh...", self.open_image, "Ctrl+O")
        self.add_action(m, "Lưu", self.save_image, "Ctrl+S")
        self.add_action(m, "Lưu thành...", self.save_image_as, "Ctrl+Shift+S")
        self.add_action(m, "Lưu vùng chọn ROI...", self.save_roi)
        m.addSeparator()
        self.add_action(m, "Thoát", self.close, "Ctrl+Q")

        m = bar.addMenu("Ảnh")
        self.add_action(m, "Thông tin ảnh", self.show_image_info, "Ctrl+I")
        self.add_action(m, "Đổi kích thước...", lambda: self.show_parameter_tab(self.resize_tab_index))
        self.add_action(m, "Cắt ROI ra ảnh riêng (tự lưu)", self.crop_roi, "Ctrl+Shift+X")
        self.add_action(m, "Cắt ảnh hiện tại theo vùng chọn", self.crop_into_image)
        self.add_action(m, "Vẽ ROI hình thang (click 4 điểm)", self.start_roi_polygon)
        self.add_action(m, "Áp dụng ROI (giữ vùng)", self.apply_roi_polygon)
        self.add_action(m, "Lưu ROI...", self.save_roi_polygon)
        self.add_action(m, "Nạp ROI...", self.load_roi_polygon)
        self.add_action(m, "Xóa ROI", self.clear_roi_polygon)        
        m.addSeparator()
        for text, fn in TRANSFORMS:
            self.add_action(m, text, lambda _=False, t=text, f=fn: self.apply_op(t, f))
        m.addSeparator()
        self.add_action(m, "Khôi phục ảnh gốc", self.restore_original)

        menus = {}
        for group_name, commands in GROUPS.items():
            m = menus[group_name] = bar.addMenu(group_name)
            for text, slot_name in MENU_ACTIONS.get(group_name, []):
                self.add_action(m, text, getattr(self, slot_name))
            submenus = {}
            for key, config in commands.items():
                section = config.get("section")
                if section and section not in submenus:
                    m.addSeparator()
                    submenus[section] = m.addMenu(section)
                self.add_action(submenus.get(section, m), config["name"],
                                lambda _=False, g=group_name, k=key: self.open_command(g, k))
        menus["Điều chỉnh"].addSeparator()
        self.add_action(menus["Điều chỉnh"], "Đổi kích thước...",
                        lambda: self.show_parameter_tab(self.resize_tab_index))

        self.create_video_menu(bar)

        m = bar.addMenu("AI")
        a = self.add_action(m, "(Sẽ bổ sung)", lambda: None)
        a.setEnabled(False)

        m = bar.addMenu("Trợ giúp")
        self.add_action(m, "Phím tắt", lambda: QMessageBox.information(
            self, "Phím tắt",
            "Ctrl+O: Mở ảnh    Ctrl+S: Lưu\nCtrl+Z / Ctrl+Y: Undo / Redo\n"
            "Lăn chuột: Zoom theo vị trí con trỏ\nNút giữa chuột: Kéo ảnh\n"
            "Ctrl+0: Vừa khung    Ctrl+1: 100%"))

    @staticmethod
    def gray_of(image):
        if image.ndim == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------
    # THANH CÔNG CỤ TRÊN + HỘP CÔNG CỤ TRÁI
    # -----------------------------------------------------

    def create_toolbar(self):
        tb = QToolBar("Thanh công cụ")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        logo = QLabel("  Image<b>Studio</b> <span style='font-size:10px'>2.0</span>  ")
        logo.setObjectName("logo")
        tb.addWidget(logo)
        tb.addSeparator()
        for label, slot in [("Mở ảnh", self.open_image), ("Lưu ảnh", self.save_image)]:
            button = QPushButton(label)
            if label == "Mở ảnh":
                button.setObjectName("primary")
            button.clicked.connect(slot)
            tb.addWidget(button)
        tb.addSeparator()

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        self.undo_action.triggered.connect(self.undo)
        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        self.redo_action.triggered.connect(self.redo)
        tb.addAction(self.undo_action)
        tb.addAction(self.redo_action)
        tb.addSeparator()

        self.zoom_combo = QComboBox()
        self.zoom_combo.setEditable(True)
        self.zoom_combo.setFixedWidth(130)
        self.zoom_combo.addItems(["Vừa khung", "25%", "50%", "75%", "100%", "150%", "200%", "400%"])
        self.zoom_combo.activated.connect(self.zoom_combo_changed)
        self.zoom_combo.lineEdit().returnPressed.connect(self.zoom_combo_changed)
        tb.addWidget(self.zoom_combo)

        for text, slot, sc in [("-", lambda: self.zoom_by(1 / 1.25), "Ctrl+-"),
                               ("+", lambda: self.zoom_by(1.25), "Ctrl+="),
                               ("Vừa khung", lambda: self.canvas.fit_to_view(), "Ctrl+0"),
                               ("1:1", lambda: self.canvas.set_zoom(1.0), "Ctrl+1")]:
            a = QAction(text, self)
            a.setShortcut(QKeySequence(sc))
            a.triggered.connect(slot)
            tb.addAction(a)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self.compare_action = QAction("So sánh trước/sau", self)
        self.compare_action.setCheckable(True)
        self.compare_action.setChecked(True)
        self.compare_action.toggled.connect(lambda on: self.set_comparison_visible(on))
        tb.addAction(self.compare_action)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentIndex(1 if self.theme_name == "light" else 0)
        self.theme_combo.setToolTip("Giao diện tối / sáng")
        self.theme_combo.currentIndexChanged.connect(self.change_theme)
        tb.addWidget(self.theme_combo)
        photo_action = QAction("Ảnh thẻ / Cỡ in", self)
        photo_action.triggered.connect(lambda: self.show_parameter_tab(self.resize_tab_index))
        tb.insertAction(self.compare_action, photo_action)

        # Hộp công cụ dọc bên trái
        tools = QToolBar("Hộp công cụ")
        tools.setObjectName("tools")
        tools.setMovable(False)
        tools.setOrientation(Qt.Orientation.Vertical)
        tools.setFixedWidth(154)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tools)

        tool_content = QWidget()
        tool_layout = QVBoxLayout(tool_content)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.setSpacing(2)
        tool_scroll = QScrollArea()
        tool_scroll.setObjectName("inspectorScroll")
        tool_scroll.setWidgetResizable(True)
        tool_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tool_scroll.setWidget(tool_content)
        tool_scroll.setFixedWidth(144)
        tools.addWidget(tool_scroll)

        def section(text):
            lbl = QLabel(text)
            lbl.setObjectName("toolSection")
            tool_layout.addWidget(lbl)

        SP = QStyle.StandardPixmap

        def tool_button(icon, text, tip=None, slot=None, checkable=False):
            b = QPushButton(text)
            b.setObjectName("tool")
            b.setFixedHeight(34)
            b.setIcon(self.style().standardIcon(icon))
            b.setToolTip(tip or text)
            b.setCheckable(checkable)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if slot:
                b.clicked.connect(slot)
            tool_layout.addWidget(b)
            return b

        title = QLabel("HỘP CÔNG CỤ")
        title.setObjectName("toolTitle")
        tool_layout.addWidget(title)

        section("CHỌN & ĐO")
        group = QButtonGroup(self)
        group.setExclusive(True)
        self.tool_actions = {}
        for icon, text, tip, mode in [
            (SP.SP_ArrowUp, "Con trỏ", "Con trỏ: xem giá trị pixel", "none"),
            (SP.SP_FileDialogDetailedView, "Chọn vùng", "Chọn vùng ROI (kéo chuột)", "roi"),
            (SP.SP_FileDialogDetailedView, "ROI hình thang", "Click 4 đỉnh vùng mặt đường", "roi_poly"),
            (SP.SP_FileDialogInfoView, "Thước đo", "Click điểm A rồi điểm B", "ruler"),
            (SP.SP_ArrowRight, "Kéo ảnh", "Kéo ảnh (Pan)", "pan"),
        ]:
            b = tool_button(icon, text, tip, checkable=True,
                            slot=lambda _=False, md=mode: self.set_tool(md))
            group.addButton(b)
            self.tool_actions[mode] = b
        self.tool_actions["none"].setChecked(True)

        section("XEM")
        tool_button(SP.SP_ArrowUp, "Phóng to", slot=lambda: self.zoom_by(1.25))
        tool_button(SP.SP_ArrowDown, "Thu nhỏ", slot=lambda: self.zoom_by(1 / 1.25))
        tool_button(SP.SP_TitleBarMaxButton, "Vừa khung", slot=lambda: self.canvas.fit_to_view())
        tool_button(SP.SP_DesktopIcon, "100%", slot=lambda: self.canvas.set_zoom(1.0))

        section("VÙNG CHỌN")
        tool_button(SP.SP_DialogApplyButton, "Cắt ROI (tự lưu)",
                    "Cắt vùng chọn ra ảnh riêng, tự lưu vào thư mục ROI", self.crop_roi)
        tool_button(SP.SP_ArrowDown, "Cắt vào ảnh", "Cắt ảnh hiện tại theo ROI (Undo được)",
                    self.crop_into_image)
        tool_button(SP.SP_DialogSaveButton, "Lưu ROI thành...", "Chọn nơi lưu ROI", self.save_roi)
        tool_button(SP.SP_DialogApplyButton, "Áp dụng ROI",
                    "Giữ lại vùng trong ROI hình thang (Undo được)", self.apply_roi_polygon)
        tool_button(SP.SP_DialogSaveButton, "Lưu ROI", "Lưu 4 tọa độ ROI ra tệp .json",
                    self.save_roi_polygon)
        tool_button(SP.SP_ArrowDown, "Nạp ROI", "Nạp ROI từ tệp .json", self.load_roi_polygon)

        section("ẢNH")
        tool_button(SP.SP_BrowserReload, "Xoay 90°", slot=lambda: self.apply_op(*TRANSFORMS[0]))
        tool_button(SP.SP_FileDialogContentsView, "Ảnh xám", slot=self.convert_gray)
        tool_button(SP.SP_DirIcon, "Tách kênh B/G/R", slot=self.split_channels)
        tool_button(SP.SP_MessageBoxInformation, "Thông tin ảnh", slot=self.show_image_info)
        tool_button(SP.SP_DialogResetButton, "Khôi phục gốc", slot=self.restore_original)

        tool_layout.addStretch()

    # -----------------------------------------------------
    # BỐ CỤC CHÍNH
    # -----------------------------------------------------

    def build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(5, 5, 5, 5)

        # --- Vùng hiển thị ảnh ---
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(10)
        self.canvas = ImageCanvas()
        self.canvas.mouse_callback = self.update_pixel_status
        self.canvas.zoom_callback = self.zoom_changed
        self.canvas.interaction_callback = self.handle_canvas_interaction
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.viewport().installEventFilter(self)
        self.canvas.scroll = self.scroll_area
        self.workspace_stack = QStackedWidget()
        welcome = QFrame()
        welcome.setObjectName("welcome")
        wl = QVBoxLayout(welcome)
        wl.setContentsMargins(40, 32, 40, 32)
        wl.setSpacing(16)
        wl.addStretch()
        for text, name in [("IMAGESTUDIO / KHÔNG GIAN SÁNG TẠO", "eyebrow"),
                           ("Bắt đầu với một bức ảnh", "welcomeTitle"),
                           ("Chỉnh màu, cắt ảnh và tạo ảnh thẻ trong một không gian.", "welcomeSubtitle")]:
            label = QLabel(text)
            label.setObjectName(name)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            wl.addWidget(label)
        open_row = QHBoxLayout()
        open_row.addStretch()
        open_button = QPushButton("Mở ảnh từ máy tính")
        open_button.setObjectName("openWelcome")
        open_button.setCursor(Qt.CursorShape.PointingHandCursor)
        open_button.clicked.connect(self.open_image)
        open_row.addWidget(open_button)
        open_row.addStretch()
        wl.addLayout(open_row)
        tip = QLabel("Ctrl + O để mở ảnh  •  PNG, JPG, WEBP, BMP, TIFF")
        tip.setObjectName("muted")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl.addWidget(tip)
        wl.addStretch()
        self.workspace_stack.addWidget(welcome)
        self.workspace_stack.addWidget(self.scroll_area)
        cl.addWidget(self.workspace_stack, 1)
        cl.addWidget(self.build_video_bar())
        bottom = QHBoxLayout()
        self.image_title = QLabel("Chưa mở ảnh")
        self.image_title.setObjectName("muted")
        bottom.addWidget(self.image_title)
        bottom.addStretch()
        cl.addLayout(bottom)

        # --- Cột phải 1: Histogram + Thông số ---
        col1 = QWidget()
        c1 = QVBoxLayout(col1)
        c1.setContentsMargins(0, 0, 0, 0)
        c1.setSpacing(6)

        hist_card = Card("Histogram")
        self.hist_mode = QComboBox()
        self.hist_mode.addItems(["RGB", "R", "G", "B", "Xám"])
        self.hist_mode.currentTextChanged.connect(self.hist_mode_changed)
        hist_card.header.addWidget(self.hist_mode)
        self.histogram = HistogramCanvas()
        hist_card.body.addWidget(self.histogram)
        hist_card.setMinimumHeight(180)
        hist_card.setMaximumHeight(270)
        c1.addWidget(hist_card, 2)

        param_card = Card("Thông số")
        self.param_tabs = QTabWidget()
        self.param_tabs.setDocumentMode(True)
        

        self.active_group = None
        self.active_command = None
        self.operation_sliders = {}   # {(nhóm, key): {parameter: widget}}
        self.operation_pages = {}    # {(nhóm, key): page index}
        self.operation_stacks = {}

        # Mỗi nhóm có một trang danh sách và các trang tham số riêng.
        for group_name, commands in GROUPS.items():
            self._add_operation_tab(group_name, commands)

        detail = QWidget()
        dl = QGridLayout(detail)
        dl.setContentsMargins(0, 8, 0, 0)
        self.photo_preset = QComboBox()
        for label, _ in ASPECT_RATIOS:
            self.photo_preset.addItem(label)
        self.photo_fit = QComboBox()
        self.photo_fit.addItem("Cắt giữa vừa khung", "crop")
        self.photo_fit.addItem("Giữ trọn ảnh • nền trắng", "contain")
        self.photo_fit.addItem("Kéo giãn theo kích thước", "stretch")
        dl.addWidget(QLabel("Tỉ lệ khung"), 0, 0)
        dl.addWidget(self.photo_preset, 0, 1)
        ratio_row = QHBoxLayout()   # nút chọn nhanh tỉ lệ
        for i, (label, _) in list(enumerate(ASPECT_RATIOS))[2:8]:
            b = QPushButton(label.split()[0])
            b.setObjectName("ratio")
            b.setMinimumWidth(30) 
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=i: self.photo_preset.setCurrentIndex(k))
            ratio_row.addWidget(b)
        dl.addLayout(ratio_row, 1, 0, 1, 2)
        dl.addWidget(QLabel("Cách căn ảnh"), 3, 0)
        dl.addWidget(self.photo_fit, 3, 1)
        self.resize_w = QSpinBox()
        self.resize_h = QSpinBox()
        for sp in (self.resize_w, self.resize_h):
            sp.setRange(1, 20000)
            sp.setSuffix(" px")
        self.resize_w.valueChanged.connect(lambda v: self.resize_linked("w", v))
        self.resize_h.valueChanged.connect(lambda v: self.resize_linked("h", v))
        dl.addWidget(QLabel("Rộng"), 4, 0)
        dl.addWidget(self.resize_w, 4, 1)
        dl.addWidget(QLabel("Cao"), 5, 0)
        dl.addWidget(self.resize_h, 5, 1)
        self.keep_ratio = QCheckBox("Giữ tỉ lệ khung hình")
        self.keep_ratio.setChecked(True)
        dl.addWidget(self.keep_ratio, 6, 0, 1, 2)
        self.resize_percent = QComboBox()
        self.resize_percent.addItems(["Theo %…", "25%", "50%", "75%", "150%", "200%"])
        self.resize_percent.activated.connect(self.resize_by_percent)
        dl.addWidget(QLabel("Nhanh"), 7, 0)
        dl.addWidget(self.resize_percent, 7, 1)
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["Bilinear", "Bicubic", "Nearest", "Area", "Lanczos"])
        dl.addWidget(QLabel("Nội suy"), 8, 0)
        dl.addWidget(self.interp_combo, 8, 1)
        dl.setRowStretch(12, 1)
        self.photo_hint = QLabel("Chọn tỉ lệ khung hoặc nhập kích thước pixel.")
        self.photo_hint.setObjectName("muted")
        self.photo_hint.setWordWrap(True)
        dl.addWidget(self.photo_hint, 9, 0, 1, 2)
        resize_button = QPushButton("Đổi kích thước ảnh")
        resize_button.setObjectName("primary")
        resize_button.clicked.connect(self.resize_image)
        dl.addWidget(resize_button, 10, 0, 1, 2)
        self.photo_preset.currentIndexChanged.connect(self.update_photo_preset)
        self.resize_tab_index = self.param_tabs.addTab(detail, "Kích thước")
        self._linking = False
        
        param_card.body.addWidget(self.param_tabs, 1)
        btn_row = QHBoxLayout()
        self.apply_button = QPushButton("Áp dụng")
        self.apply_button.setObjectName("primary")
        self.apply_button.clicked.connect(self.apply_preview)
        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.clicked.connect(self.cancel_preview)
        self.preview_button = QPushButton("Xem trước")
        self.preview_button.setCheckable(True)
        self.preview_button.setChecked(True)
        self.preview_button.setToolTip("Bật: kéo thanh trượt là xem trước ngay. Tắt: xem ảnh hiện tại")
        self.preview_button.toggled.connect(lambda _: self.update_preview())
        btn_row.addWidget(self.preview_button, 1)
        btn_row.addWidget(self.apply_button, 1)
        btn_row.addWidget(self.cancel_button, 1)
        param_card.body.addLayout(btn_row)
        c1.addWidget(param_card, 4)
        col2 = QWidget()
        c2 = QVBoxLayout(col2)
        c2.setContentsMargins(0, 0, 0, 0)
        c2.setSpacing(6)

        # --- Cột phải 2: Lịch sử + So sánh ---
        hist_list_card = Card("Lịch sử thao tác")
        self.history_count = QLabel("0/0")
        self.history_count.setObjectName("muted")
        hist_list_card.header.addWidget(self.history_count)
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.history_item_clicked)
        hist_list_card.body.addWidget(self.history_list, 1)
        ur = QHBoxLayout()
        b_undo = QPushButton("Undo")
        b_undo.clicked.connect(self.undo)
        b_redo = QPushButton("Redo")
        b_redo.clicked.connect(self.redo)
        ur.addWidget(b_undo)
        ur.addWidget(b_redo)
        hist_list_card.body.addLayout(ur)
        c2.addWidget(hist_list_card, 3)

        result_card = Card("Bảng kết quả")
        result_card.setObjectName("resultCard")
        grid = QGridLayout()
        grid.setVerticalSpacing(4)
        self.result_values = {}
        for r, key in enumerate(["Số pixel", "Kích thước", "PSNR (so với gốc)", "Trung bình",
                                 "Độ lệch chuẩn", "Min / Max", "Số bước lịch sử", "Kết quả lệnh"]):
            k = QLabel(key)
            k.setObjectName("muted")
            k.setWordWrap(True)
            v = QLabel("--")
            v.setObjectName("resultValue")
            v.setWordWrap(True)
            v.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(k, r, 0)
            grid.addWidget(v, r, 1)
            self.result_values[key] = v
        result_card.body.addLayout(grid)
        c2.addWidget(result_card)

        self.compare_card = Card("So sánh trước - sau")
        self.compare_widget = CompareWidget()
        self.compare_card.body.addWidget(self.compare_widget, 1)
        lbl = QHBoxLayout()
        for t in ("Trước", "Sau"):
            x = QLabel(t)
            x.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.addWidget(x)
        self.compare_card.body.addLayout(lbl)
        self.compare_card.setMinimumHeight(195)
        self.compare_card.setVisible(self.compare_action.isChecked())
        c2.addWidget(self.compare_card, 2)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.addWidget(center)
        for column, minimum in [(col1, 275), (col2, 255)]:
            scroll = QScrollArea()
            scroll.setObjectName("inspectorScroll")
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setMinimumWidth(minimum)
            scroll.setWidget(column)
            self.main_splitter.addWidget(scroll)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([780, 295, 275])
        root.addWidget(self.main_splitter)

    def _add_operation_tab(self, group_name, commands):
        """Tạo tab danh sách thao tác và trang tham số cho từng thao tác."""
        stack = QStackedWidget()
        self.operation_stacks[group_name] = stack

        list_page = QWidget()
        layout = QVBoxLayout(list_page)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(6)
        hint = QLabel("Chọn thao tác để xem và thay đổi thông số.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        for text, slot_name in MENU_ACTIONS.get(group_name, []):
            action_btn = QPushButton(text)
            action_btn.setObjectName("adjustHead")
            action_btn.setFixedHeight(36)
            action_btn.clicked.connect(getattr(self, slot_name))
            layout.addWidget(action_btn)
        seen = set()
        for key, config in commands.items():
            section = config.get("section")
            if section and section not in seen:
                seen.add(section)
                sep = QLabel(section.upper())
                sep.setObjectName("toolSection")
                layout.addWidget(sep)
            button = QPushButton(config["name"])
            button.setObjectName("adjustHead")
            button.setFixedHeight(36)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, g=group_name, k=key: self.open_command(g, k))
            layout.addWidget(button)
        if group_name == "Điều chỉnh":
            gray_btn = QPushButton("Chuyển ảnh xám")
            gray_btn.clicked.connect(self.convert_gray)
            layout.addWidget(gray_btn)
        layout.addStretch()
        stack.addWidget(list_page)

        for key, config in commands.items():
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(0, 6, 0, 0)
            page_layout.setSpacing(8)
            top = QHBoxLayout()
            back = QPushButton("← Danh sách")
            back.clicked.connect(self.close_command)
            title = QLabel(config["name"])
            title.setObjectName("cardTitle")
            top.addWidget(back)
            top.addStretch()
            top.addWidget(title)
            page_layout.addLayout(top)

            widgets = {}
            for spec in config.get("params", []):
                if len(spec) == 5 and spec[2] == "combo":
                    pkey, label, _, items, default = spec
                    widget = ParamCombo(label, items, default)
                    widget.combo.currentIndexChanged.connect(self.schedule_preview)
                else:
                    pkey, label, minimum, maximum, default, scale, decimals = spec
                    widget = ParamSlider(label, minimum, maximum, default, scale, decimals,
                                         editable=config.get("editable", False))
                    widget.slider.valueChanged.connect(self.schedule_preview)
                widgets[pkey] = widget
                page_layout.addWidget(widget)

            if widgets:
                reset_btn = QPushButton("Đặt lại")
                reset_btn.clicked.connect(
                    lambda _=False, group=widgets: ([w.reset() for w in group.values()], self.schedule_preview()))
                page_layout.addWidget(reset_btn)
            else:
                note = QLabel("Thao tác này không có tham số. Nhấn Áp dụng để thực hiện.")
                note.setObjectName("muted")
                note.setWordWrap(True)
                page_layout.addWidget(note)
            for text, slot_name in config.get("buttons", []):
                extra_btn = QPushButton(text)
                extra_btn.clicked.connect(getattr(self, slot_name))
                page_layout.addWidget(extra_btn)
            page_layout.addStretch()
            ref = (group_name, key)
            self.operation_sliders[ref] = widgets
            self.operation_pages[ref] = stack.addWidget(page)

        self.param_tabs.addTab(stack, group_name)

    def show_parameter_tab(self, index):
        self.param_tabs.setCurrentIndex(index)

    def open_command(self, group, key):
        if self.canvas.interaction_mode == "roi_poly":      # <-- thêm
            self.tool_actions["none"].setChecked(True)
            self.set_tool("none")
        self.clear_preview(reset_sliders=True)
        self.active_group = group
        self.active_command = key
        self.param_tabs.setCurrentIndex(list(self.operation_stacks).index(group))
        self.operation_stacks[group].setCurrentIndex(self.operation_pages[(group, key)])
        self.display_image()
        self.schedule_preview()

    def close_command(self):
        self.cancel_preview()
        if self.active_group in self.operation_stacks:
            self.operation_stacks[self.active_group].setCurrentIndex(0)
        self.active_command = None
        self.active_group = None

    def set_comparison_visible(self, visible):
        self.compare_card.setVisible(visible)

    def eventFilter(self, obj, event):
        if obj is self.scroll_area.viewport() and event.type() == QEvent.Type.Resize:
            if self.canvas.fit_mode:
                QTimer.singleShot(0, self.canvas.fit_to_view)
        return super().eventFilter(obj, event)

    # -----------------------------------------------------
    # THANH TRẠNG THÁI
    # -----------------------------------------------------

    def create_status_bar(self):
        sb = self.statusBar()
        self.status_message = QLabel("● Ready")
        self.status_info = QLabel()
        sb.addWidget(self.status_message, 1)
        sb.addPermanentWidget(self.status_info)
        self.last_pixel = None
        self.update_status_details()

    def update_pixel_status(self, x, y):
        self.last_pixel = (x, y)
        self.update_status_details()

    def zoom_changed(self, zoom):
        text = f"{zoom * 100:.0f}%"
        self.zoom_combo.setEditText(f"Vừa khung ({text})" if self.canvas.fit_mode else text)
        if hasattr(self, "status_info"):
            self.update_status_details()

    def update_status_details(self):
        """x = 312, y = 128 · giá trị (B, G, R) = (140, 87, 201) · 640×480, 3 kênh · phóng 150%"""
        image = self.display_source()
        if image is None:
            self.status_info.setText("x = --, y = -- · giá trị = -- · chưa có ảnh")
            return
        h, w = image.shape[:2]
        if self.last_pixel and self.last_pixel[1] < h and self.last_pixel[0] < w:
            x, y = self.last_pixel
            pos, value = f"x = {x}, y = {y}", "giá trị " + pixel_text(image, x, y)
        else:
            pos, value = "x = --, y = --", "giá trị = --"
        self.status_info.setText(
            f"{pos}  ·  {value}  ·  {w}×{h}, {image_channels(image)} kênh  ·  "
            f"phóng {self.canvas.zoom * 100:.0f}%")

    # -----------------------------------------------------
    # BẢNG KẾT QUẢ: số đếm, PSNR, thông số
    # -----------------------------------------------------

    def update_results(self):
        image = self.display_source()
        if image is None:
            for v in self.result_values.values():
                v.setText("--")
            return
        h, w = image.shape[:2]
        c = image_channels(image)
        ref = self.original_image
        psnr = "-- (khác kích thước)"
        if ref is not None and ref.shape == image.shape:
            value = mse_psnr(ref, image)[1]
            psnr = "∞ (giống ảnh gốc)" if value == float("inf") else f"{value:.2f} dB"
        data = {
            "Số pixel": f"{w * h:,}",
            "Kích thước": f"{w} × {h} · {c} kênh",
            "PSNR (so với gốc)": psnr,
            "Trung bình": f"{image.mean():.2f}",
            "Độ lệch chuẩn": f"{image.std():.2f}",
            "Min / Max": f"{int(image.min())} / {int(image.max())}",
            "Số bước lịch sử": f"{max(0, self.history_index)}",
            "Kết quả lệnh": " · ".join(f"{k}: {v}" for k, v in self.command_info.items()) or "--",
        }
        for k, v in data.items():
            self.result_values[k].setText(v)

    # -----------------------------------------------------
    # LỊCH SỬ (G3)
    # -----------------------------------------------------

    def reset_history(self, name, image):
        self.history = [(name, image.copy())]
        self.history_index = 0
        self.refresh_history_list()

    def push_state(self, name, image):
        """Thêm trạng thái mới; bỏ các bước Redo phía sau (giống Photoshop)."""
        self.history = self.history[: self.history_index + 1]
        self.history.append((name, image.copy()))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.history_index = len(self.history) - 1
        self.current_image = image.copy()
        self.after_image_changed()

    def goto_state(self, index):
        if not (0 <= index < len(self.history)) or self.is_video_open:
            return
        self.clear_preview(reset_sliders=True)
        self.history_index = index
        self.current_image = self.history[index][1].copy()
        self.after_image_changed()

    def undo(self):
        if self.preview_image is not None:
            self.cancel_preview()
            return
        self.step_history(-1, "Undo")

    def redo(self):
        self.step_history(+1, "Redo")

    def step_history(self, delta, label):
        target = self.history_index + delta
        if 0 <= target < len(self.history):
            self.goto_state(target)
            self.status_message.setText(f"{label} → " + self.history[self.history_index][0])
        else:
            self.status_message.setText(f"Không còn thao tác để {label}")

    def history_item_clicked(self, item):
        self.goto_state(item.data(Qt.ItemDataRole.UserRole))

    def refresh_history_list(self):
        """Thứ tự trên → dưới: mới → cũ. Bước hiện tại được tô, bước đã Undo hiện mờ."""
        self.history_list.blockSignals(True)
        self.history_list.clear()
        for i, (name, _) in reversed(list(enumerate(self.history))):
            icon = "📂" if i == 0 else ("✓" if i <= self.history_index else "↷")
            item = QListWidgetItem(f"{icon}  {name}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            if i > self.history_index:
                item.setForeground(QColor("#5c677a"))
            self.history_list.addItem(item)
        if self.history_index >= 0:
            self.history_list.setCurrentRow(len(self.history) - 1 - self.history_index)
            self.history_list.scrollToItem(self.history_list.currentItem())
        self.history_list.blockSignals(False)
        self.history_count.setText(f"{self.history_index + 1}/{len(self.history)}")
        self.undo_action.setEnabled(self.history_index > 0)
        self.redo_action.setEnabled(self.history_index < len(self.history) - 1)

    # -----------------------------------------------------
    # HIỂN THỊ
    # -----------------------------------------------------

    def display_source(self):
        return self.preview_image if self.preview_image is not None else self.current_image

    def display_image(self):
        image = self.display_source()
        self.workspace_stack.setCurrentIndex(0 if image is None else 1)
        self.canvas.set_image(image)
        self.histogram.update_histogram(image)
        before = self.original_image if self.preview_image is None else self.current_image
        self.compare_widget.set_images(before, image)
        self.update_status_details()
        self.update_results()

    def after_image_changed(self):
        self.display_image()
        self.refresh_history_list()
        self.sync_resize_spins()
        if not self.is_video_open:
            title = os.path.basename(self.image_path) if self.image_path else "Ảnh"
            self.image_title.setText(f"{title} — {self.history[self.history_index][0]}")
            self.setWindowTitle(f"ImageStudio 2 — {title}")

    def hist_mode_changed(self, mode):
        self.histogram.mode = mode
        self.histogram.update_histogram(self.display_source())

    # -----------------------------------------------------
    # MỞ / LƯU
    # -----------------------------------------------------

    def load_source(self, path, image, history_name):
        """Đặt `image` làm ảnh gốc mới (dùng chung cho mở ảnh và mở video)."""
        self.reset_roi_state()
        self.image_path = path
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.clear_preview(reset_sliders=True)
        self.canvas.fit_mode = True
        self.canvas.image = None   # buộc vừa khung lại
        self.reset_history(history_name, image)


    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở ảnh", "", "Tệp ảnh (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if not path:
            return
        image = read_image_unicode(path)
        if image is None:
            QMessageBox.critical(self, "Lỗi mở ảnh",
                                 "Tệp không hợp lệ hoặc định dạng ảnh chưa được hỗ trợ.")
            return
        self.stop_video(silent=True)
        self.load_source(path, image, f"Mở ảnh ({os.path.basename(path)})")
        self.after_image_changed()
        self.status_message.setText("Đã mở ảnh")

    def can_edit(self):
        """Điều kiện chung trước mọi thao tác sửa ảnh: có ảnh, không phát video, không xem trước dở."""
        if self.current_image is None:
            QMessageBox.information(self, "Thông báo", "Hãy mở ảnh trước.")
            return False
        if self.is_video_open:
            QMessageBox.information(self, "Thông báo", "Hãy dừng video (Video → Dừng) trước khi chỉnh sửa.")
            return False
        return self.ensure_no_preview()


    def ensure_no_preview(self):
        if self.preview_image is not None:
            QMessageBox.information(self, "Thông báo", "Hãy nhấn Áp dụng hoặc Hủy trước.")
            return False
        return True

    def save_image(self):
        if self.current_image is None or not self.ensure_no_preview():
            return
        if not self.image_path or self.is_video_open:
            self.save_image_as()
        else:
            self.write_to(self.image_path, self.current_image)

    def ask_save_path(self, title, suggested=""):
        path, flt = QFileDialog.getSaveFileName(
            self, title, suggested, "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)")
        if not path:
            return None
        if not os.path.splitext(path)[1]:
            path += {"JPEG": ".jpg", "BMP": ".bmp", "TIFF": ".tif"}.get(flt.split()[0], ".png")
        return path

    def save_image_as(self):
        if self.current_image is None or not self.ensure_no_preview():
            return
        path = self.ask_save_path("Lưu ảnh thành")
        if path and self.write_to(path, self.current_image):
            self.image_path = path

    def write_to(self, path, image):
        try:
            write_image_unicode(path, image)
            self.status_message.setText("Đã lưu: " + path)
            return True
        except Exception as error:
            QMessageBox.critical(self, "Lỗi lưu ảnh", str(error))
            return False

    # -----------------------------------------------------
    # VIDEO
    # -----------------------------------------------------

    
    # -----------------------------------------------------
    # ZOOM
    # -----------------------------------------------------

    def zoom_by(self, factor):
        self.canvas.set_zoom(self.canvas.zoom * factor)

    def zoom_combo_changed(self, *_):
        text = self.zoom_combo.currentText().strip()
        if text.startswith("Vừa"):
            self.canvas.fit_to_view()
            return
        try:
            self.canvas.set_zoom(float(text.rstrip("%")) / 100)
        except ValueError:
            pass

    # -----------------------------------------------------
    # CÔNG CỤ: ROI / THƯỚC / KÉO
    # -----------------------------------------------------

    def set_tool(self, mode):
        if self.current_image is None and mode != "none":
            self.tool_actions["none"].setChecked(True)
            return
        self.canvas.set_interaction_mode(mode)
        self.status_message.setText({
            "none": "Ready",
            "roi": "Chọn vùng: kéo chuột tạo hình chữ nhật → ✂ để cắt, 💾 để lưu vùng chọn",
            "pan": "Kéo ảnh: giữ chuột trái và kéo (nút giữa chuột dùng được ở mọi công cụ)",
            "ruler": "Thước đo: click điểm A rồi điểm B",
            "roi_poly": "ROI hình thang: click lần lượt 4 đỉnh (thứ tự nào cũng được)",
        }[mode])

    def reset_roi_state(self):
        """Xóa toàn bộ trạng thái ROI/tương tác của ảnh hoặc video cũ."""
        self.roi_points = None
        self.set_roi(None, None)
        self.canvas.set_interaction_mode("none")   # đồng thời xóa kéo chuột / thước đo / pan đang dở
        self.tool_actions["none"].setChecked(True)
        
    def handle_canvas_interaction(self, mode, a, b):
        if mode == "roi":
            (xa, ya), (xb, yb) = a, b
            self.roi_points = (a, b)
            self.status_message.setText(
                f"ROI: ({min(xa, xb)}, {min(ya, yb)}) → ({max(xa, xb)}, {max(ya, yb)}) "
                f"= {abs(xb - xa)} × {abs(yb - ya)} px   |   ✂ cắt   💾 lưu")
        elif mode == "roi_poly":
            self.set_roi(order_quad(a), self.canvas.roi_poly_size)
            self.status_message.setText(
                "ROI hình thang đã có 4 đỉnh — bấm ✂ Áp dụng ROI, 💾 Lưu ROI, hoặc click lại để vẽ mới")            
        elif mode == "point":
            x, y = a
            path = self.save_point_row("Điểm", x, y)
            self.status_message.setText(f"Điểm ({x}, {y}) {pixel_text(self.current_image, x, y)} — đã tự lưu vào {path}")
        else:
            (x1, y1), (x2, y2) = a, b
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            euclid = (dx * dx + dy * dy) ** 0.5
            path = self.save_point_row("Đo 2 điểm", x1, y1, x2, y2, f"{euclid:.2f}", dx + dy, max(dx, dy))
            self.status_message.setText(
                f"A({x1},{y1}) B({x2},{y2}) | Euclid={euclid:.2f} px | D4={dx + dy} | D8={max(dx, dy)}"
                f" — đã tự lưu vào {path}")

    @property
    def image_base(self):
        """Tên ảnh không có đuôi, dùng để đặt tên tệp kết quả."""
        return os.path.splitext(os.path.basename(self.image_path))[0] if self.image_path else "anh"


    def output_folder(self, sub):
        folder = os.path.dirname(self.image_path) if self.image_path else os.getcwd()
        folder = os.path.join(folder, sub)
        os.makedirs(folder, exist_ok=True)
        return folder

    def save_point_row(self, kind, x1, y1, x2="", y2="", euclid="", d4="", d8=""):
        """Tự động ghi điểm / phép đo vào <thư mục ảnh>/KetQua/<tên ảnh>_diem_do.csv"""
        base = self.image_base
        path = os.path.join(self.output_folder("KetQua"), f"{base}_diem_do.csv")
        new_file = not os.path.exists(path)
        try:
            with open(path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(["Thời gian", "Loại", "xA", "yA", "Giá trị A", "xB", "yB",
                                     "Euclid (px)", "D4 (px)", "D8 (px)"])
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), kind, x1, y1,
                                 pixel_text(self.current_image, x1, y1), x2, y2, euclid, d4, d8])
        except OSError as error:
            QMessageBox.warning(self, "Lưu điểm", f"Không ghi được file điểm: {error}")
        return path

    def roi_image(self):
        if self.current_image is None:
            return None, None
        if not self.roi_points:
            QMessageBox.information(self, "Vùng chọn", "Hãy chọn công cụ ⬚ và kéo chuột trên ảnh trước.")
            return None, None
        (xa, ya), (xb, yb) = self.roi_points
        x1, x2 = sorted((xa, xb))
        y1, y2 = sorted((ya, yb))
        if x2 == x1:   # cho phép cắt mọi kích thước, kể cả 1 pixel
            x2 = x1 + 1
        if y2 == y1:
            y2 = y1 + 1
        h, w = self.current_image.shape[:2]
        x1, x2 = min(x1, w - 1), min(x2, w)
        y1, y2 = min(y1, h - 1), min(y2, h)
        if x2 <= x1 or y2 <= y1:
            QMessageBox.warning(self, "Vùng chọn", "Vùng chọn nằm ngoài ảnh.")
            return None, None
        return self.current_image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def crop_roi(self):
        """Cắt vùng ROI ra và TỰ ĐỘNG lưu thành ảnh riêng; ảnh đang sửa giữ nguyên."""
        cropped, box = self.roi_image()
        if cropped is None:
            return
        x1, y1, x2, y2 = box
        base = self.image_base
        name = f"{base}_roi_{x1}_{y1}_{x2 - x1}x{y2 - y1}_{datetime.now():%H%M%S}.png"
        path = os.path.join(self.output_folder("ROI"), name)
        if self.write_to(path, cropped):
            self.status_message.setText(f"✂ Đã cắt {x2 - x1}×{y2 - y1} px và lưu: {path}")
            self.roi_preview(cropped, path)

    def roi_preview(self, image, path):
        dialog = QDialog(self)
        dialog.setWindowTitle("Vùng ROI đã cắt")
        lay = QVBoxLayout(dialog)
        pic = QLabel()
        pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pm = to_qpixmap(image)
        if pm.width() > 600 or pm.height() > 450:
            pm = pm.scaled(600, 450, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        elif pm.width() < 120 and pm.height() < 120:   # ROI nhỏ thì phóng to để dễ nhìn
            pm = pm.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.FastTransformation)
        pic.setPixmap(pm)
        lay.addWidget(pic)
        info = QLabel(f"{image.shape[1]} × {image.shape[0]} px\nĐã lưu: {path}")
        info.setWordWrap(True)
        lay.addWidget(info)
        row = QHBoxLayout()
        open_btn = QPushButton("Mở thư mục")
        open_btn.clicked.connect(lambda: os.startfile(os.path.dirname(path)))
        use_btn = QPushButton("Dùng làm ảnh hiện tại")
        use_btn.clicked.connect(lambda: (dialog.accept(), self.crop_into_image()))
        close_btn = QPushButton("Đóng")
        close_btn.clicked.connect(dialog.accept)
        for b in (open_btn, use_btn, close_btn):
            row.addWidget(b)
        lay.addLayout(row)
        dialog.exec()

    def crop_into_image(self):
        """Cắt ảnh hiện tại theo ROI (có trong lịch sử, Undo được)."""
        if not self.can_edit():
            return
        cropped, box = self.roi_image()
        if cropped is None:
            return
        x1, y1, x2, y2 = box
        self.roi_points = None
        self.canvas.set_interaction_mode("none")
        self.tool_actions["none"].setChecked(True)
        self.canvas.fit_mode = True
        self.push_state(f"Cắt ảnh ({x2 - x1}×{y2 - y1})", cropped)

    def save_roi(self):
        cropped, _ = self.roi_image()
        if cropped is not None:
            self.save_crop_file(cropped)

    def save_crop_file(self, image):
        base = self.image_base
        folder = os.path.dirname(self.image_path) if self.image_path else ""
        path = self.ask_save_path("Lưu vùng cắt ROI", os.path.join(folder, f"{base}_roi.png"))
        if path:
            self.write_to(path, image)
            
    # -----------------------------------------------------
    # ROI HÌNH THANG (M1 - RoadCam)
    # -----------------------------------------------------

    def start_roi_polygon(self):
        if self.current_image is None:
            QMessageBox.information(self, "Thông báo", "Hãy mở ảnh trước.")
            return
        self.tool_actions["roi_poly"].setChecked(True)
        self.set_tool("roi_poly")

    def current_roi_points(self, image=None, warn=True):
        """Trả về 4 đỉnh ROI nếu hợp lệ với kích thước ảnh, ngược lại None."""
        image = self.current_image if image is None else image
        if image is None:
            return None
        h, w = image.shape[:2]
        if not self.roi_polygon or self.roi_size != (w, h):
            if warn:
                QMessageBox.information(
                    self, "ROI",
                    "Chưa có ROI hợp lệ cho ảnh này.\n"
                    "Chọn công cụ 'ROI hình thang' rồi click 4 đỉnh, hoặc Nạp ROI.")
            return None
        return self.roi_polygon

    def get_roi_mask(self, image):
        """Dùng cho M5, M7, M8: mặt nạ ROI (255/0) cùng kích thước `image`, hoặc None."""
        pts = self.current_roi_points(image, warn=False)
        return None if pts is None else roi_mask(image.shape, pts)

    def apply_roi_polygon(self):
        pts = self.current_roi_points()
        if pts is None:
            return
        self.apply_op("ROI hình thang", lambda im: roi_apply(im, pts))

    def set_roi(self, points, size):
        """Một nơi duy nhất cập nhật ROI hình thang: cửa sổ chính và canvas luôn đồng bộ."""
        self.roi_polygon, self.roi_size = points, size
        self.canvas.roi_poly_points = list(points) if points else []
        self.canvas.roi_poly_size = size
        self.canvas.update()

    def clear_roi_polygon(self):
        self.set_roi(None, None)
        self.status_message.setText("Đã xóa ROI")

    def save_roi_polygon(self):
        pts = self.current_roi_points()
        if pts is None:
            return
        w, h = self.roi_size
        base = self.image_base
        path, _ = QFileDialog.getSaveFileName(self, "Lưu ROI", f"{base}_roi.json", "ROI (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"width": w, "height": h,
                           "points": [[int(x), int(y)] for x, y in pts]},
                          f, ensure_ascii=False, indent=2)
            self.status_message.setText("Đã lưu ROI: " + path)
        except OSError as error:
            QMessageBox.critical(self, "Lỗi lưu ROI", str(error))

    def load_roi_polygon(self):
        if self.current_image is None:
            QMessageBox.information(self, "Thông báo", "Hãy mở ảnh hoặc video trước.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Nạp ROI", "", "ROI (*.json)")
        if not path:
            return
        h, w = self.current_image.shape[:2]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sw, sh = int(data["width"]), int(data["height"])
            pts = [(int(x), int(y)) for x, y in data["points"]]
            if len(pts) != 4:
                raise ValueError("ROI phải có đúng 4 điểm")
            if (sw, sh) != (w, h):   # ảnh/video khác độ phân giải → quy đổi tỉ lệ
                pts = [(round(x * w / sw), round(y * h / sh)) for x, y in pts]
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            QMessageBox.critical(self, "Lỗi nạp ROI", f"Tệp ROI không hợp lệ: {error}")
            return
        self.set_roi(order_quad(pts), (w, h))
        self.tool_actions["roi_poly"].setChecked(True)
        self.set_tool("roi_poly")
        self.status_message.setText(
            "Đã nạp ROI: " + path + " — bấm ✂ Áp dụng ROI để giữ vùng này")

    # -----------------------------------------------------
    # THAO TÁC TRỰC TIẾP (menu bộ lọc / màu / hình thái / biên)
    # -----------------------------------------------------

    def apply_op(self, name, func):
        if not self.can_edit():
            return
        try:
            result = func(self.current_image)
        except cv2.error as error:
            QMessageBox.warning(self, name, f"Không áp dụng được: {error}")
            return
        if result.shape[:2] != self.current_image.shape[:2]:
            self.canvas.fit_mode = True
        self.push_state(name, result)
        self.status_message.setText("Đã áp dụng: " + name)

    def convert_gray(self):
        if self.current_image is not None and self.current_image.ndim == 2:
            QMessageBox.information(self, "Thông báo", "Ảnh đã là ảnh xám.")
            return
        self.apply_op("Chuyển ảnh xám", self.gray_of)

    def restore_original(self):
        if self.original_image is None or self.is_video_open:
            return
        self.clear_preview(reset_sliders=True)
        self.canvas.fit_mode = True
        self.push_state("Khôi phục ảnh gốc", self.original_image)

    # -----------------------------------------------------
    # G2: XEM TRƯỚC THAM SỐ (realtime) / ÁP DỤNG / HỦY
    # -----------------------------------------------------

    def current_operation_config(self):
        if self.active_group is None or self.active_command is None:
            return None
        return GROUPS[self.active_group][self.active_command]

    def run_command(self, config, image, params):
        """Chạy một lệnh trong bảng.
        needs_roi="mask"   -> truyền thêm roi_mask (mặt nạ 255/0 của ROI hình thang, hoặc None)
        needs_roi="points" -> truyền thêm roi_points (4 đỉnh ROI, hoặc None)
        Lệnh có thể trả (ảnh, {tên: giá trị}); số liệu được giữ ở self.command_info."""
        need = config.get("needs_roi")
        extra = {}
        if need == "mask":
            extra["roi_mask"] = self.get_roi_mask(image)
        elif need == "points":
            extra["roi_points"] = self.current_roi_points(image, warn=False)
        out = config["func"](image, **params, **extra)
        self.command_info = {}
        if isinstance(out, tuple):
            out, self.command_info = out
        return out

    def show_hsv_channels(self):
        image = self.current_image
        if image is None:
            QMessageBox.information(self, "Thông báo", "Hãy mở ảnh trước.")
            return
        if image.ndim == 2:
            QMessageBox.information(self, "Kênh HSV", "Ảnh xám không có kênh màu HSV.")
            return
        h, s, v = hsv_channels(to_bgr(image))
        h_view = (h.astype(np.float32) * 255 / 179).astype(np.uint8)   # H chỉ 0–179: kéo giãn để dễ nhìn
        self.show_images("Kênh H, S, V (ảnh xám)",
                         [("H (kéo giãn 0–179 → 0–255)", h_view), ("S", s), ("V", v)])

    def show_mask_and_image(self):
        """Nút 'Xem mặt nạ | ảnh song song' của lệnh đang mở (dùng đúng giá trị thanh trượt hiện tại)."""
        config = self.current_operation_config()
        if config is None or self.current_image is None:
            return
        params = self.current_params()
        try:
            mask = self.run_command(config, self.current_image, {**params, "view": "mask"})
            kept = self.run_command(config, self.current_image, {**params, "view": "image"})
        except OPERATION_ERRORS as error:
            QMessageBox.warning(self, config["name"], f"Không hiển thị được: {error}")
            return
        self.show_images(config["name"],
                         [("Mặt nạ (trắng = được chọn)", mask), ("Ảnh chỉ giữ vùng được chọn", kept)],
                         size=(440, 380))

    def current_params(self):
        """Đọc giá trị từ slider hoặc combobox của thao tác đang mở."""
        if self.active_command is None or self.active_group is None:
            return {}
        widgets = self.operation_sliders[(self.active_group, self.active_command)]
        return {key: widget.value() for key, widget in widgets.items()}

    def schedule_preview(self, *_):
        if self.current_image is not None and not self.is_video_open:
            self.preview_timer.start()

    def update_preview(self):
        key = self.active_command
        config = self.current_operation_config()
        if (key is None or config is None or self.current_image is None
                or not self.preview_button.isChecked()):
            self.preview_image = None
        else:
            params = self.current_params()
            try:
                self.preview_image = self.run_command(config, self.current_image, params)
            except OPERATION_ERRORS as error:
                self.preview_image = None
                self.display_image()
                self.status_message.setText(
                    f"Preview thất bại: {error}"
                )
                return
        self.display_image()
        if self.preview_image is None:
            self.status_message.setText("Ready")
        else:
            no_roi = config.get("needs_roi") and self.get_roi_mask(self.current_image) is None
            self.status_message.setText("Xem trước — nhấn Áp dụng hoặc Hủy" + (
                " — chưa có ROI hợp lệ, đang xử lý toàn ảnh" if no_roi else ""))

    def describe_command(self, params):
        config = self.current_operation_config()
        if not params:
            return config["name"]
        return f"{config['name']}: " + ", ".join(
            f"{name}={value}" for name, value in params.items())

    def apply_preview(self):
        if self.param_tabs.currentIndex() == self.resize_tab_index:
            self.resize_image()
            return
        self.preview_timer.stop()
        key = self.active_command
        config = self.current_operation_config()
        if key is None or config is None or self.current_image is None or self.is_video_open:
            return
        params = self.current_params()
        try:
            result = self.run_command(config, self.current_image, params)

        except OPERATION_ERRORS as error:
            self.preview_image = None
            self.display_image()

            QMessageBox.warning(
                self,
                config["name"],
                f"Không áp dụng được: {error}"
            )

            return
        info = self.command_info
        self.clear_preview(reset_sliders=True)
        name = self.describe_command(params)
        self.push_state(name, result)
        self.command_info = info          # giữ số liệu của lệnh vừa áp dụng trên Bảng kết quả
        self.update_results()
        self.status_message.setText("Đã áp dụng: " + name)

    def clear_preview(self, reset_sliders=False):
        self.preview_timer.stop()
        self.preview_image = None
        self.command_info = {}
        if reset_sliders:
            for group in self.operation_sliders.values():
                for widget in group.values():
                    widget.reset()
                    
    def cancel_preview(self):
        self.clear_preview(reset_sliders=True)
        self.sync_resize_spins()
        self.display_image()
        self.status_message.setText("Đã hủy")

    # -----------------------------------------------------
    # ĐỔI KÍCH THƯỚC ẢNH (tab Chi tiết → Áp dụng)
    # -----------------------------------------------------

    def current_ratio(self):
        """Tỉ lệ (rộng, cao) đang chọn, hoặc None nếu tự do."""
        ratio = ASPECT_RATIOS[max(0, self.photo_preset.currentIndex())][1]
        if ratio == "original":
            if self.current_image is None:
                return None
            h, w = self.current_image.shape[:2]
            return (w, h)
        return ratio

    def update_photo_preset(self, *_):
        ratio = self.current_ratio()
        fixed = ratio is not None
        self.keep_ratio.setEnabled(not fixed)
        self.photo_fit.setEnabled(fixed)
        if not fixed:
            self.photo_hint.setText("Nhập kích thước pixel; bật giữ tỉ lệ để tránh méo ảnh.")
            self.sync_resize_spins()
            return
        if self.current_image is None:
            self.photo_hint.setText("Mở ảnh để tính kích thước theo tỉ lệ.")
            return
        h, w = self.current_image.shape[:2]
        width, height = ratio_size(w, h, ratio)
        self._linking = True
        self.resize_w.setValue(width)
        self.resize_h.setValue(height)
        self._linking = False
        self.photo_hint.setText(
            f"Tỉ lệ {self.photo_preset.currentText().split()[0]} → {width} × {height} px. "
            "Sửa Rộng hoặc Cao, ô còn lại tự theo tỉ lệ. "
            "'Cắt giữa' bỏ phần rìa; 'Giữ trọn' thêm nền trắng.")

    def sync_resize_spins(self):
        if self.current_ratio() is not None and self.photo_preset.currentIndex() != 1:
            self.update_photo_preset()
            return
        if self.current_image is None:
            return
        h, w = self.current_image.shape[:2]
        self._linking = True
        self.resize_w.setValue(w)
        self.resize_h.setValue(h)
        self._linking = False

    def resize_linked(self, which, value):
        if self._linking or self.current_image is None:
            return
        ratio = self.current_ratio()
        if ratio is None:
            if not self.keep_ratio.isChecked():
                return
            h, w = self.current_image.shape[:2]
        else:
            w, h = ratio
        self._linking = True
        if which == "w":
            self.resize_h.setValue(max(1, round(value * h / w)))
        else:
            self.resize_w.setValue(max(1, round(value * w / h)))
        self._linking = False

    def resize_by_percent(self, index):
        if index == 0 or self.current_image is None:
            return
        pct = float(self.resize_percent.currentText().rstrip("%")) / 100
        h, w = self.current_image.shape[:2]
        self._linking = True
        self.resize_w.setValue(max(1, round(w * pct)))
        self.resize_h.setValue(max(1, round(h * pct)))
        self._linking = False
        self.resize_percent.setCurrentIndex(0)

    def resize_image(self):
        if not self.can_edit():
            return
        h, w = self.current_image.shape[:2]
        nw, nh = self.resize_w.value(), self.resize_h.value()
        if (nw, nh) == (w, h):
            self.status_message.setText("Kích thước không đổi")
            return
        interp = {"Bilinear": cv2.INTER_LINEAR, "Bicubic": cv2.INTER_CUBIC,
                  "Nearest": cv2.INTER_NEAREST, "Area": cv2.INTER_AREA,
                  "Lanczos": cv2.INTER_LANCZOS4}[self.interp_combo.currentText()]
        fixed = self.current_ratio() is not None
        mode = self.photo_fit.currentData() if fixed else "stretch"
        result = resize_photo(self.current_image, nw, nh, mode, interp)
        self.canvas.fit_mode = True
        label = f" (tỉ lệ {self.photo_preset.currentText().split()[0]})" if fixed else ""
        self.push_state(f"Đổi kích thước {w}×{h} → {nw}×{nh}{label}", result)
        self.status_message.setText(f"Đã đổi kích thước: {nw} × {nh}")

    # -----------------------------------------------------
    # THÔNG TIN / TÁCH KÊNH
    # -----------------------------------------------------

    def show_image_info(self):
        image = self.current_image
        if image is None:
            QMessageBox.information(self, "Thông tin ảnh", "Hãy mở ảnh trước.")
            return
        h, w = image.shape[:2]
        c = image_channels(image)
        QMessageBox.information(
            self, "Thông tin ảnh",
            f"Tệp: {os.path.basename(self.image_path)}\n"
            f"Kích thước: {w} × {h} pixel ({w * h:,} pixel)\n"
            f"Số kênh: {c} ({'ảnh xám' if c == 1 else 'ảnh màu'})\n"
            f"Kiểu dữ liệu: {image.dtype}\n"
            f"Dung lượng chưa nén: {w * h * c:,} byte")

    def show_images(self, title, items, footer="", size=(300, 280)):
        """Hộp thoại xếp các ảnh thành một hàng, mỗi ảnh có chú thích; `footer` là dòng chữ bên dưới."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        grid = QGridLayout(dialog)
        for col, (caption, img) in enumerate(items):
            cap, pic = QLabel(caption), QLabel()
            cap.setObjectName("cardTitle")
            for w in (cap, pic):
                w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pic.setPixmap(to_qpixmap(img).scaled(
                *size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            grid.addWidget(cap, 0, col)
            grid.addWidget(pic, 1, col)
        row = 2
        if footer:
            info = QLabel(footer)
            info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(info, row, 0, 1, len(items))
            row += 1
        close_btn = QPushButton("Đóng")
        close_btn.clicked.connect(dialog.accept)
        grid.addWidget(close_btn, row, 0, 1, len(items))
        dialog.exec()

    def split_channels(self):
        image = self.current_image
        if image is None:
            return
        if image.ndim == 2:
            QMessageBox.information(self, "Tách kênh màu", "Ảnh xám không có 3 kênh B/G/R.")
            return
        b, g, r = cv2.split(to_bgr(image))
        self.show_images("Tách kênh màu B / G / R", [("Kênh B", b), ("Kênh G", g), ("Kênh R", r)])

    def compare_equalize_opencv(self):
        image = self.current_image
        if image is None:
            QMessageBox.information(self, "Thông báo", "Hãy mở ảnh trước.")
            return
        mine = COMMANDS["balance histogram"]["func"](image)   # tự cài [TC]
        ref = equalize_cv2(image)                              # OpenCV
        if mine.shape != ref.shape:
            QMessageBox.warning(self, "So sánh",
                                "Hai kết quả khác kích thước/số kênh nên không so sánh được.")
            return
        m = similarity_metrics(mine, ref)
        psnr_text = "∞ (giống hệt nhau)" if m["psnr"] == float("inf") else f'{m["psnr"]:.2f} dB'
        footer = (
            f'Độ giống nhau của 2 thuật toán\n'
            f'• Số pixel trùng khớp: {m["same_pct"]:.2f}%\n'
            f'• PSNR: {psnr_text}\n'
            f'• MSE: {m["mse"]:.4f}\n'
            f'• Chênh lệch lớn nhất: {m["max_diff"]} mức xám')
        self.show_images("Cân bằng histogram: tự cài vs OpenCV",
                         [("Tự cài (NumPy)", mine), ("cv2.equalizeHist", ref)],
                         footer, size=(440, 380))
    
    def closeEvent(self, event):
        self.video_timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
        event.accept()
        
def excepthook(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    QMessageBox.critical(None, "Lỗi chương trình", f"{exc_type.__name__}: {exc}")

if __name__ == "__main__":
    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageStudio()
    window.show()
    sys.exit(app.exec())