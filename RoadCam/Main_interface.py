import sys
import os
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QScrollArea,
    QFileDialog, QMessageBox, QSlider, QDoubleSpinBox,
    QSpinBox, QComboBox, QListWidget, QSplitter,
    QSizePolicy, QCheckBox, QDialog, QGridLayout
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from M2_adjustment import *

# =========================================================
# HỖ TRỢ ĐỌC ẢNH CÓ ĐƯỜNG DẪN TIẾNG VIỆT
# =========================================================

def read_image_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        if image.ndim == 3 and image.shape[2] == 2:
            return None
        return image
    except Exception:
        return None


def image_channels(image):
    if image is None:
        return 0
    return 1 if image.ndim == 2 else image.shape[2]


# =========================================================
# VÙNG HIỂN THỊ ẢNH
# =========================================================

class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.zoom = 1.0
        self.mouse_callback = None
        self.interaction_mode = "none"  # none / roi / ruler
        self.drag_start = None
        self.drag_end = None
        self.ruler_points = []
        self.interaction_callback = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 250)
        self.setMouseTracking(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored
        )
        self.setStyleSheet(
            "background-color: #f4f8fc; color: #263b53;"
            "font-size: 16px;"
        )

    def set_image(self, image):
        self.image = image
        self.refresh()

    def refresh(self):
        if self.image is None:
            self.setPixmap(QPixmap())
            self.setText("MỞ ẢNH ĐỂ BẮT ĐẦU")
            self.resize(600, 400)
            return

        image = self.image
        height, width = image.shape[:2]

        if image.ndim == 2:
            qimage = QImage(
                image.data, width, height, image.strides[0],
                QImage.Format.Format_Grayscale8
            ).copy()
        elif image.shape[2] == 4:
            rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            qimage = QImage(
                rgba.data, width, height, rgba.strides[0],
                QImage.Format.Format_RGBA8888
            ).copy()
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimage = QImage(
                rgb.data, width, height, rgb.strides[0],
                QImage.Format.Format_RGB888
            ).copy()

        pixmap = QPixmap.fromImage(qimage)
        display_width = max(1, int(width * self.zoom))
        display_height = max(1, int(height * self.zoom))
        pixmap = pixmap.scaled(
            display_width, display_height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setText("")
        self.setPixmap(pixmap)
        self.resize(display_width, display_height)

    def set_interaction_mode(self, mode):
        self.interaction_mode = mode
        self.drag_start = None
        self.drag_end = None
        self.ruler_points = []
        self.update()
        if mode == "roi":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == "ruler":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def image_point(self, event):
        x = int(event.position().x() / self.zoom)
        y = int(event.position().y() / self.zoom)
        if self.image is None:
            return None
        h, w = self.image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return (x, y)
        return None

    def mousePressEvent(self, event):
        if self.image is None or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        point = self.image_point(event)
        if point is None:
            return
        if self.interaction_mode == "roi":
            self.drag_start = point
            self.drag_end = point
            self.update()
        elif self.interaction_mode == "ruler":
            self.ruler_points.append(point)
            if len(self.ruler_points) > 2:
                self.ruler_points = [point]
            self.update()
            if len(self.ruler_points) == 2 and self.interaction_callback:
                self.interaction_callback("ruler", self.ruler_points[0], self.ruler_points[1])

    def mouseMoveEvent(self, event):
        point = self.image_point(event)
        if point is not None and self.mouse_callback:
            x, y = point
            pixel = self.image[y, x]
            if self.image.ndim == 2:
                value = f"Gray={int(pixel)}"
            elif len(pixel) == 4:
                value = f"B={int(pixel[0])}, G={int(pixel[1])}, R={int(pixel[2])}, A={int(pixel[3])}"
            else:
                value = f"B={int(pixel[0])}, G={int(pixel[1])}, R={int(pixel[2])}"
            self.mouse_callback(x, y, value)
        if self.interaction_mode == "roi" and self.drag_start is not None and point is not None:
            self.drag_end = point
            self.update()

    def mouseReleaseEvent(self, event):
        if self.interaction_mode == "roi" and event.button() == Qt.MouseButton.LeftButton and self.drag_start is not None:
            point = self.image_point(event)
            if point is not None:
                self.drag_end = point
                if self.interaction_callback:
                    self.interaction_callback("roi", self.drag_start, self.drag_end)
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.interaction_mode == "roi" and self.drag_start and self.drag_end:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end
            painter.setPen(QPen(QColor("#39a7ff"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(int(min(x1,x2)*self.zoom), int(min(y1,y2)*self.zoom),
                             max(1,int(abs(x2-x1)*self.zoom)), max(1,int(abs(y2-y1)*self.zoom)))
        if self.interaction_mode == "ruler" and self.ruler_points:
            painter.setPen(QPen(QColor("#ff4d6d"), 2))
            x1, y1 = self.ruler_points[0]
            painter.drawEllipse(int(x1*self.zoom)-4, int(y1*self.zoom)-4, 8, 8)
            if len(self.ruler_points) == 2:
                x2, y2 = self.ruler_points[1]
                painter.drawLine(int(x1*self.zoom), int(y1*self.zoom), int(x2*self.zoom), int(y2*self.zoom))
                painter.drawEllipse(int(x2*self.zoom)-4, int(y2*self.zoom)-4, 8, 8)
        painter.end()

    def wheelEvent(self, event):
        if self.image is None:
            return
        if event.angleDelta().y() > 0:
            self.zoom = min(self.zoom * 1.1, 5.0)
        else:
            self.zoom = max(self.zoom / 1.1, 0.1)
        self.refresh()
        if self.mouse_callback:
            self.mouse_callback(None, None, None)



# =========================================================
# HISTOGRAM
# =========================================================

class HistogramCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(4, 2), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMinimumHeight(125)
        self.show_empty()

    def show_empty(self):
        self.axes.clear()
        self.axes.set_title("Histogram")
        self.axes.text(
            0.5, 0.5, "Chưa có ảnh",
            ha="center", va="center",
            transform=self.axes.transAxes
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.draw()

    def update_histogram(self, image):
        self.axes.clear()
        if image is None:
            self.show_empty()
            return

        if image.ndim == 2:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            self.axes.plot(hist, color="gray", label="Gray")
        else:
            for channel, color, name in [
                (0, "blue", "Blue"),
                (1, "green", "Green"),
                (2, "red", "Red")
            ]:
                hist = cv2.calcHist(
                    [image], [channel], None, [256], [0, 256]
                )
                self.axes.plot(hist, color=color, label=name)

        self.axes.set_title("Histogram")
        self.axes.set_xlim([0, 256])
        self.axes.set_xlabel("Giá trị pixel")
        self.axes.set_ylabel("Số pixel")
        self.axes.legend(fontsize=7, loc="upper right")
        self.draw()


# =========================================================
# ỨNG DỤNG CHÍNH G1-G4
# =========================================================

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ARIN03020 - Xử lý ảnh số và video số | G1-G4")
        self.resize(1450, 850)
        self.setMinimumSize(1050, 650)

        self.original_image = None
        self.current_image = None
        self.preview_image = None
        self.preview_active = False
        self.image_path = ""
        self.updating_controls = False

        # G3: các trạng thái ảnh để Undo/Redo
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 20
        self.current_state_name = "Mở ảnh"

        # G4: video
        self.video_capture = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.read_video_frame)
        self.video_fps = 25
        self.video_frame_number = 0
        self.is_video_open = False

        # G2: so sánh ảnh trước/sau
        self.compare_mode = False
        self.compare_before = None
        self.compare_after = None

        self.create_menu()
        self.build_interface()
        self.create_status_bar()
        self.apply_theme()
        self.update_history_buttons()

    # -----------------------------------------------------
    # MENU
    # -----------------------------------------------------

    def create_menu(self):
        bar = self.menuBar()
        self.file_menu = bar.addMenu("Tệp")
        image_menu = bar.addMenu("Ảnh")
        adjust_menu = bar.addMenu("Điều chỉnh")
        self.history_menu = bar.addMenu("Lịch sử")
        filter_menu = bar.addMenu("Bộ lọc")
        bar.addMenu("Màu")
        bar.addMenu("Hình thái")
        bar.addMenu("Biên")
        bar.addMenu("Video")
        bar.addMenu("AI")
        bar.addMenu("Trợ giúp")

        self.open_action = QAction("Mở ảnh...", self)
        self.open_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_action.triggered.connect(self.open_image)
        self.file_menu.addAction(self.open_action)

        self.save_action = QAction("Lưu", self)
        self.save_action.setShortcut(QKeySequence("Ctrl+S"))
        self.save_action.triggered.connect(self.save_image)
        self.file_menu.addAction(self.save_action)

        self.save_as_action = QAction("Lưu thành...", self)
        self.save_as_action.triggered.connect(self.save_image_as)
        self.file_menu.addAction(self.save_as_action)

        self.file_menu.addSeparator()

        self.open_video_action = QAction("Mở video...", self)
        self.open_video_action.triggered.connect(self.open_video)
        self.file_menu.addAction(self.open_video_action)

        self.file_menu.addSeparator()
        exit_action = QAction("Thoát", self)
        exit_action.triggered.connect(self.close)
        self.file_menu.addAction(exit_action)

        gray_action = QAction("Chuyển ảnh xám", self)
        gray_action.triggered.connect(self.convert_gray)
        image_menu.addAction(gray_action)

        info_action = QAction("Thông tin ảnh", self)
        info_action.triggered.connect(self.show_image_info)
        image_menu.addAction(info_action)

        channels_action = QAction("Tách kênh màu B / G / R", self)
        channels_action.triggered.connect(self.split_channels)
        image_menu.addAction(channels_action)

        self.crop_action = QAction("Cắt theo vùng chọn", self)
        self.crop_action.triggered.connect(self.crop_roi)
        image_menu.addAction(self.crop_action)

        restore_action = QAction("Khôi phục ảnh gốc", self)
        restore_action.triggered.connect(self.restore_original)
        image_menu.addAction(restore_action)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self.undo)
        self.history_menu.addAction(undo_action)
        self.undo_action = undo_action

        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self.redo)
        self.history_menu.addAction(redo_action)
        self.redo_action = redo_action

        self.history_menu.addSeparator()
        clear_history_action = QAction("Xóa danh sách lịch sử", self)
        clear_history_action.triggered.connect(self.clear_history_display)
        self.history_menu.addAction(clear_history_action)

        #Menu adjust
        for name in ["Độ sáng", "Âm bản", "Log", "Gamma", "Kéo giãn tương phản", "Cân bằng Histogram","Lượng tử hóa"]:
            action = QAction(name, self)
            if name == "Âm bản":
                action.triggered.connect(self.apply_negative)
            elif name == "Log":
                action.triggered.connect(self.apply_log)
            elif name == "Cân bằng Histogram":
                action.triggered.connect(self.apply_equalize)
            else:
                action.triggered.connect(
                        lambda checked=False, n=name: self.select_operation(n)
                    )
            adjust_menu.addAction(action)

        #Menu Filter
        for name in ["Trung bình", "Gaussian", "Trung vị", "Sắc nét Laplacian", "Unshaprp masking"]:
            action = QAction(name, self)
            filter_menu.addAction(action)

    # -----------------------------------------------------
    # GIAO DIỆN
    # -----------------------------------------------------

    def build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Hộp công cụ trái
        tools_group = QGroupBox("HỘP CÔNG CỤ")
        tools_group.setMinimumWidth(140)
        tools_group.setMaximumWidth(180)
        tools = QVBoxLayout(tools_group)

        for label, callback in [
            ("Mở ảnh", self.open_image),
            ("Chuyển ảnh xám", self.convert_gray),
            ("Khôi phục ảnh gốc", self.restore_original),
            ("Phóng to", self.zoom_in),
            ("Thu nhỏ", self.zoom_out),
            ("100% / Vừa khung", self.zoom_reset),
            ("Thông tin ảnh", self.show_image_info),
            ("Tách kênh B/G/R", self.split_channels),
            ("Chọn vùng (ROI)", self.activate_roi),
            ("Cắt vùng đã chọn", self.crop_roi),
            ("Thước đo 2 điểm", self.activate_ruler),
            ("Kết thúc công cụ", self.deactivate_tool),
            ("Undo  (Ctrl+Z)", self.undo),
            ("Redo  (Ctrl+Y)", self.redo),
        ]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            tools.addWidget(button)

        tools.addSpacing(10)
        tools.addWidget(QLabel("Thước đo / Chọn điểm"))
        tools.addStretch()

        # Vùng ảnh giữa
        image_group = QGroupBox("VÙNG HIỂN THỊ ẢNH")
        image_layout = QVBoxLayout(image_group)
        self.image_title = QLabel("Mở ảnh để bắt đầu")
        self.image_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(self.image_title)

        self.canvas = ImageCanvas()
        self.canvas.mouse_callback = self.update_pixel_status
        self.canvas.interaction_callback = self.handle_canvas_interaction
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(self.scroll_area, 1)

        compare_row = QHBoxLayout()
        self.compare_checkbox = QCheckBox("So sánh trước / sau")
        self.compare_checkbox.toggled.connect(self.toggle_compare)
        compare_row.addWidget(self.compare_checkbox)
        compare_row.addStretch()
        image_layout.addLayout(compare_row)

        # Panel phải
        right_panel = QWidget()
        right_panel.setMinimumWidth(275)
        right_panel.setMaximumWidth(360)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        histogram_group = QGroupBox("HISTOGRAM")
        histogram_layout = QVBoxLayout(histogram_group)
        self.histogram = HistogramCanvas()
        histogram_layout.addWidget(self.histogram)
        right_layout.addWidget(histogram_group, 3)

        param_group = QGroupBox("THAM SỐ - G2")
        param = QVBoxLayout(param_group)
        self.operation_combo = QComboBox()
        self.operation_combo.addItems(
            ["Độ sáng", "Gamma", "Kéo giãn tương phản", "Lượng tử hóa"]
        )
        self.operation_combo.currentIndexChanged.connect(self.operation_changed)
        param.addWidget(QLabel("Chọn thao tác"))
        param.addWidget(self.operation_combo)

        self.value_label = QLabel("Giá trị: 0")
        param.addWidget(self.value_label)
        self.value_slider = QSlider(Qt.Orientation.Horizontal)
        self.value_slider.valueChanged.connect(self.slider_changed)
        param.addWidget(self.value_slider)
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setDecimals(2)
        self.value_spin.valueChanged.connect(self.spin_changed)
        param.addWidget(self.value_spin)
        
        #  Mức đen a / mức trắng b 
        self.levels_widget = QWidget()
        levels_layout = QVBoxLayout(self.levels_widget)
        levels_layout.setContentsMargins(0, 0, 0, 0)

        # Mức đen a
        levels_layout.addWidget(QLabel("Mức đen a:"))
        black_row = QHBoxLayout()
        self.black_slider = QSlider(Qt.Orientation.Horizontal)
        self.black_slider.setRange(0, 255)
        self.black_slider.setValue(0)
        self.black_spin = QSpinBox()
        self.black_spin.setRange(0, 255)
        self.black_spin.setValue(0)
        black_row.addWidget(self.black_slider, 1)
        black_row.addWidget(self.black_spin)
        levels_layout.addLayout(black_row)

        # Mức trắng b
        levels_layout.addWidget(QLabel("Mức trắng b:"))
        white_row = QHBoxLayout()
        self.white_slider = QSlider(Qt.Orientation.Horizontal)
        self.white_slider.setRange(0, 255)
        self.white_slider.setValue(255)
        self.white_spin = QSpinBox()
        self.white_spin.setRange(0, 255)
        self.white_spin.setValue(255)
        white_row.addWidget(self.white_slider, 1)
        white_row.addWidget(self.white_spin)
        levels_layout.addLayout(white_row)

        for spin in (self.black_spin, self.white_spin):
            spin.setKeyboardTracking(False)
        
        # Slider và ô nhập cùng gọi levels_changed
        for w in (self.black_slider, self.black_spin,
                self.white_slider, self.white_spin):
            w.valueChanged.connect(self.levels_changed)

        self.levels_widget.setVisible(False)
        param.addWidget(self.levels_widget)
        #---------------------------------------------
        button_row = QHBoxLayout()
        self.preview_button = QPushButton("Xem trước")
        self.preview_button.clicked.connect(self.preview_operation)
        button_row.addWidget(self.preview_button)
        self.apply_button = QPushButton("Áp dụng")
        self.apply_button.clicked.connect(self.apply_preview)
        button_row.addWidget(self.apply_button)
        self.cancel_button = QPushButton("Hủy")
        self.cancel_button.clicked.connect(self.cancel_preview)
        button_row.addWidget(self.cancel_button)
        param.addLayout(button_row)
        right_layout.addWidget(param_group, 3)

        history_group = QGroupBox("LỊCH SỬ THAO TÁC - G3")
        history_layout = QVBoxLayout(history_group)
        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(75)
        history_layout.addWidget(self.history_list)
        right_layout.addWidget(history_group, 2)

        result_group = QGroupBox("BẢNG KẾT QUẢ")
        result_layout = QVBoxLayout(result_group)
        self.result_label = QLabel(
            "Số pixel: --\nPSNR: --\nKích thước: --\nSố kênh: --"
        )
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        right_layout.addWidget(result_group, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(tools_group)
        splitter.addWidget(image_group)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([155, 820, 320])
        main_layout.addWidget(splitter)

        self.operation_changed()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f4f8fc;
                color: #243247;
                font-size: 12px;
            }
            QMenuBar {
                background-color: #a9c9ed;
                color: #263b53;
                padding: 5px;
            }
            QMenuBar::item:selected { background-color: #d4e6fa; }
            QMenu {
                background-color: white;
                color: #263b53;
                border: 1px solid #cbd5e1;
            }
            QMenu::item:selected { background-color: #dceafa; }
            QGroupBox {
                background-color: #ffffff;
                border: 1px solid #d5e3f2;
                border-radius: 8px;
                margin-top: 10px;
                padding: 7px;
                font-weight: bold;
                color: #17365d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #dceafa;
                color: #263b53;
                border: 1px solid #bfd5ee;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton:hover { background-color: #c5ddf7; }
            QPushButton:pressed { background-color: #a9c9ed; }
            QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
                background-color: white;
                border: 1px solid #cbd5e1;
                border-radius: 3px;
                padding: 4px;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #dce7f3;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #79a9df;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
            QStatusBar {
                background-color: #a9c9ed;
                color: #263b53;
            }
            QStatusBar QLabel { color: #263b53; padding: 2px; }
        """)

    # -----------------------------------------------------
    # THANH TRẠNG THÁI
    # -----------------------------------------------------

    def create_status_bar(self):
        self.status_message = QLabel("Sẵn sàng")
        self.status_details = QLabel("x = --, y = -- | Giá trị pixel: --")
        self.statusBar().addWidget(self.status_message, 1)
        self.statusBar().addPermanentWidget(self.status_details)

    def update_pixel_status(self, x, y, value):
        image = self.get_display_image()
        if image is None:
            return
        height, width = image.shape[:2]
        if x is None or y is None:
            self.update_status_details()
            return
        self.status_details.setText(
            f"x={x}, y={y} | {value} | {width}×{height} | "
            f"{image_channels(image)} kênh | Zoom {self.canvas.zoom * 100:.0f}%"
        )

    def update_status_details(self):
        image = self.get_display_image()
        if image is None:
            self.status_details.setText("x = --, y = -- | Giá trị pixel: --")
            return
        height, width = image.shape[:2]
        self.status_details.setText(
            f"{width}×{height} | {image_channels(image)} kênh | "
            f"Zoom {self.canvas.zoom * 100:.0f}%"
        )

    # -----------------------------------------------------
    # MỞ ẢNH / TỆP G4
    # -----------------------------------------------------

    def stop_video(self):
        self.video_timer.stop()
        self.is_video_open = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở ảnh", "",
            "Tệp ảnh (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return

        image = read_image_unicode(path)
        if image is None:
            QMessageBox.critical(
                self, "Lỗi mở ảnh",
                "Tệp không hợp lệ hoặc định dạng ảnh chưa được hỗ trợ."
            )
            return

        self.stop_video()
        self.image_path = path
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.preview_image = None
        self.preview_active = False
        self.compare_before = None
        self.compare_after = None
        self.compare_checkbox.setChecked(False)
        self.canvas.zoom = 1.0

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current_state_name = "Mở ảnh"
        self.history_list.clear()
        self.history_list.addItem("Mở ảnh: " + os.path.basename(path))
        self.update_history_buttons()
        self.display_image()
        self.update_result()
        self.status_message.setText("Đã mở ảnh")

    def save_image(self):
        if self.current_image is None:
            QMessageBox.information(self, "Thông báo", "Chưa có ảnh để lưu.")
            return
        if self.preview_active:
            QMessageBox.information(
                self, "Thông báo",
                "Hãy nhấn Áp dụng hoặc Hủy trước khi lưu."
            )
            return
        if not self.image_path or self.is_video_open:
            self.save_image_as()
        else:
            self.write_image(self.image_path)

    def save_image_as(self):
        if self.current_image is None:
            QMessageBox.information(self, "Thông báo", "Chưa có ảnh để lưu.")
            return
        if self.preview_active:
            QMessageBox.information(
                self, "Thông báo",
                "Hãy nhấn Áp dụng hoặc Hủy trước khi lưu."
            )
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Lưu ảnh thành", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".jpg" if "JPEG" in selected_filter else ".png"
        ext = os.path.splitext(path)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg"]:
            QMessageBox.warning(
                self, "Định dạng không hợp lệ",
                "Chỉ hỗ trợ lưu PNG hoặc JPG/JPEG."
            )
            return
        self.write_image(path)

    def write_image(self, path):
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                ext = ".png"
                path += ext
            image_to_save = self.current_image
            if ext in [".jpg", ".jpeg"] and image_to_save.ndim == 3 and image_to_save.shape[2] == 4:
                image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_BGRA2BGR)
            success, encoded = cv2.imencode(ext, image_to_save)
            if not success:
                raise ValueError("OpenCV không thể mã hóa ảnh ở định dạng này.")
            encoded.tofile(path)
            self.image_path = path
            self.status_message.setText("Đã lưu: " + os.path.basename(path))
        except Exception as error:
            QMessageBox.critical(self, "Lỗi lưu ảnh", str(error))

    # -----------------------------------------------------
    # G4: MỞ VIDEO VÀ ĐỌC KHUNG HÌNH
    # -----------------------------------------------------

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở video", "",
            "Tệp video (*.mp4 *.avi *.mov *.mkv *.wmv *.mpeg *.mpg)"
        )
        if not path:
            return

        self.stop_video()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            QMessageBox.critical(
                self, "Lỗi mở video",
                "Không mở được video. Tệp có thể không hợp lệ hoặc codec không được hỗ trợ."
            )
            return

        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            QMessageBox.critical(
                self, "Lỗi video",
                "Video không có khung hình đọc được."
            )
            return

        self.video_capture = capture
        self.is_video_open = True
        self.image_path = path
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps and fps > 0 else 25
        self.video_frame_number = 1

        self.original_image = frame.copy()
        self.current_image = frame.copy()
        self.preview_image = None
        self.preview_active = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.current_state_name = "Video - khung hình 1"
        self.history_list.clear()
        self.history_list.addItem("Mở video: " + os.path.basename(path))
        self.update_history_buttons()
        self.canvas.zoom = 1.0
        self.display_image()
        self.update_result()
        self.status_message.setText("Đang phát video")

        interval = max(1, int(1000 / self.video_fps))
        self.video_timer.start(interval)

    def read_video_frame(self):
        if self.video_capture is None:
            self.stop_video()
            return

        ok, frame = self.video_capture.read()
        if not ok or frame is None:
            self.stop_video()
            self.status_message.setText("Video đã phát hết")
            return

        self.video_frame_number += 1
        self.original_image = frame.copy()
        self.current_image = frame.copy()
        self.preview_image = None
        self.preview_active = False
        self.compare_before = None
        self.compare_after = None
        self.display_image()
        self.update_result()
        self.image_title.setText(
            f"{os.path.basename(self.image_path)} — "
            f"Khung hình {self.video_frame_number}"
        )
        self.status_message.setText(
            f"Đang phát video — khung hình {self.video_frame_number}"
        )

    # -----------------------------------------------------
    # HIỂN THỊ / HISTOGRAM / KẾT QUẢ
    # -----------------------------------------------------

    def get_display_image(self):
        if self.compare_mode and self.compare_before is not None and self.compare_after is not None:
            return self.compare_after
        if self.preview_active and self.preview_image is not None:
            return self.preview_image
        return self.current_image

    def display_image(self):
        image = self.get_display_image()
        self.canvas.set_image(image)
        if image is None:
            self.image_title.setText("Mở ảnh để bắt đầu")
            self.histogram.update_histogram(None)
            self.update_status_details()
            return

        height, width = image.shape[:2]
        if self.compare_mode and self.compare_before is not None:
            self.image_title.setText("SO SÁNH TRƯỚC / SAU — đang xem ảnh sau")
        elif self.preview_active:
            self.image_title.setText("XEM TRƯỚC — Chưa áp dụng")
        elif self.is_video_open:
            self.image_title.setText(
                f"Video: {os.path.basename(self.image_path)} — "
                f"Khung hình {self.video_frame_number}"
            )
        else:
            self.image_title.setText(
                os.path.basename(self.image_path) if self.image_path else "Ảnh đang xử lý"
            )

        self.histogram.update_histogram(image)
        self.update_status_details()

    def update_result(self):
        image = self.current_image
        if image is None:
            self.result_label.setText(
                "Số pixel: --\nPSNR: --\nKích thước: --\nSố kênh: --"
            )
            return
        height, width = image.shape[:2]
        channels = image_channels(image)
        self.result_label.setText(
            f"Số pixel: {width * height:,}\n"
            f"PSNR: -- (chưa có ảnh tham chiếu)\n"
            f"Kích thước: {width} × {height}\n"
            f"Số kênh: {channels}\n"
            f"Kiểu ảnh: {'Ảnh xám' if channels == 1 else 'Ảnh màu'}"
        )

    # -----------------------------------------------------
    # ZOOM
    # -----------------------------------------------------

    def zoom_in(self):
        self.canvas.zoom = min(self.canvas.zoom * 1.2, 5.0)
        self.canvas.refresh()
        self.update_status_details()

    def zoom_out(self):
        self.canvas.zoom = max(self.canvas.zoom / 1.2, 0.1)
        self.canvas.refresh()
        self.update_status_details()

    def zoom_reset(self):
        self.canvas.zoom = 1.0
        self.canvas.refresh()
        self.update_status_details()

    # -----------------------------------------------------
    # CHUYỂN ẢNH XÁM
    # -----------------------------------------------------

    def convert_gray(self):
        if self.current_image is None:
            QMessageBox.information(self, "Thông báo", "Hãy mở ảnh trước.")
            return
        image = self.current_image
        if image.ndim == 2:
            QMessageBox.information(self, "Thông báo", "Ảnh đã là ảnh xám.")
            return
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.preview_image = gray
        self.preview_active = True
        self.compare_before = self.current_image.copy()
        self.compare_after = gray.copy()
        self.display_image()
        self.status_message.setText("Xem trước chuyển ảnh xám — nhấn Áp dụng hoặc Hủy")

    # -----------------------------------------------------
    # CHỨC NĂNG ẢNH TIÊU BẢN: THÔNG TIN / KÊNH / ROI / THƯỚC
    # -----------------------------------------------------

    def show_image_info(self):
        image = self.get_display_image()
        if image is None:
            QMessageBox.information(self, "Thông tin ảnh", "Hãy mở ảnh trước.")
            return
        h, w = image.shape[:2]
        channels = image_channels(image)
        size_bytes = w * h * channels
        QMessageBox.information(
            self, "Thông tin ảnh",
            f"Chiều rộng: {w} pixel\n"
            f"Chiều cao: {h} pixel\n"
            f"Số kênh: {channels} ({'ảnh xám' if channels == 1 else 'ảnh màu'})\n"
            f"Kiểu dữ liệu: {image.dtype}\n"
            f"Dung lượng chưa nén: {w} × {h} × {channels} = {size_bytes:,} byte"
        )

    def split_channels(self):
        image = self.current_image
        if image is None:
            QMessageBox.information(self, "Tách kênh màu", "Hãy mở ảnh trước.")
            return
        if image.ndim == 2:
            QMessageBox.information(self, "Tách kênh màu", "Ảnh hiện tại là ảnh xám, không có 3 kênh B/G/R.")
            return
        if image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image
        b, g, r = cv2.split(bgr)
        dialog = QDialog(self)
        dialog.setWindowTitle("Tách kênh màu B / G / R")
        dialog.resize(900, 360)
        layout = QGridLayout(dialog)
        for col, (name, channel) in enumerate([("Kênh B (Blue)", b), ("Kênh G (Green)", g), ("Kênh R (Red)", r)]):
            title = QLabel(name)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview = QLabel()
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            qimg = QImage(channel.data, channel.shape[1], channel.shape[0], channel.strides[0], QImage.Format.Format_Grayscale8).copy()
            pix = QPixmap.fromImage(qimg).scaled(280, 260, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            preview.setPixmap(pix)
            layout.addWidget(title, 0, col)
            layout.addWidget(preview, 1, col)
        dialog.exec()

    def activate_roi(self):
        if self.current_image is None:
            QMessageBox.information(self, "Chọn vùng", "Hãy mở ảnh trước.")
            return
        self.canvas.set_interaction_mode("roi")
        self.status_message.setText("Chọn vùng ROI: kéo chuột tạo hình chữ nhật, rồi chọn 'Cắt vùng đã chọn'.")

    def activate_ruler(self):
        if self.current_image is None:
            QMessageBox.information(self, "Thước đo", "Hãy mở ảnh trước.")
            return
        self.canvas.set_interaction_mode("ruler")
        self.status_message.setText("Thước đo: click điểm A rồi click điểm B trên ảnh.")

    def deactivate_tool(self):
        self.canvas.set_interaction_mode("none")
        self.status_message.setText("Đã kết thúc công cụ chọn vùng / thước đo.")

    def handle_canvas_interaction(self, mode, point_a, point_b):
        if mode == "roi":
            self.roi_points = (point_a, point_b)
            x1, y1 = point_a
            x2, y2 = point_b
            self.status_message.setText(f"ROI: ({min(x1,x2)}, {min(y1,y2)}) → ({max(x1,x2)}, {max(y1,y2)}). Nhấn 'Cắt vùng đã chọn'.")
        elif mode == "ruler":
            x1, y1 = point_a
            x2, y2 = point_b
            dx, dy = abs(x2-x1), abs(y2-y1)
            euclid = (dx*dx + dy*dy) ** 0.5
            d4 = dx + dy
            d8 = max(dx, dy)
            self.status_message.setText(f"A({x1},{y1}) B({x2},{y2}) | Euclid={euclid:g} px | D4={d4} px | D8={d8} px")
            QMessageBox.information(self, "Kết quả thước đo", f"A({x1}, {y1}) → B({x2}, {y2})\n\nEuclid = {euclid:g} pixel\nD4 = {d4} pixel\nD8 = {d8} pixel")

    def crop_roi(self):
        if self.current_image is None:
            QMessageBox.information(self, "Cắt ROI", "Hãy mở ảnh trước.")
            return
        points = getattr(self, "roi_points", None)
        if not points:
            QMessageBox.information(self, "Cắt ROI", "Hãy chọn công cụ 'Chọn vùng (ROI)' và kéo chuột trên ảnh trước.")
            return
        (xa, ya), (xb, yb) = points
        x1, x2 = sorted((xa, xb))
        y1, y2 = sorted((ya, yb))
        if x2 <= x1 or y2 <= y1:
            QMessageBox.warning(self, "Cắt ROI", "Vùng chọn quá nhỏ. Hãy kéo tạo hình chữ nhật lớn hơn.")
            return
        cropped = self.current_image[y1:y2, x1:x2].copy()
        self.save_undo_state()
        self.current_image = cropped
        self.current_state_name = f"Cắt ROI ({x1},{y1})–({x2},{y2})"
        self.preview_image = None
        self.preview_active = False
        self.compare_mode = False
        self.roi_points = None
        self.canvas.set_interaction_mode("none")
        self.canvas.zoom = 1.0
        self.display_image()
        self.update_result()
        self.history_list.addItem(self.current_state_name)
        self.history_list.scrollToBottom()
        self.update_history_buttons()
        self.status_message.setText("Đã cắt ảnh theo vùng chọn ROI.")

    def restore_original(self):
        if self.original_image is None:
            QMessageBox.information(self, "Thông báo", "Hãy mở ảnh trước.")
            return
        if self.is_video_open:
            QMessageBox.information(
                self, "Thông báo",
                "Đang mở video; chuyển tới khung hình khác để xem ảnh tiếp theo."
            )
            return
        if self.current_image is not None:
            self.save_undo_state()
        self.current_image = self.original_image.copy()
        self.preview_image = None
        self.preview_active = False
        self.compare_mode = False
        self.compare_checkbox.blockSignals(True)
        self.compare_checkbox.setChecked(False)
        self.compare_checkbox.blockSignals(False)
        self.current_state_name = "Khôi phục ảnh gốc"
        self.display_image()
        self.update_result()
        self.history_list.addItem("Khôi phục ảnh gốc")
        self.history_list.scrollToBottom()
        self.update_history_buttons()
        self.status_message.setText("Đã khôi phục ảnh gốc")

    # -----------------------------------------------------
    # G2: CHỌN THAO TÁC / THAM SỐ
    # -----------------------------------------------------

    def select_operation(self, name):
        index = self.operation_combo.findText(name)
        if index >= 0:
            self.operation_combo.setCurrentIndex(index)

    def operation_changed(self, *_):
        if not hasattr(self, "value_slider"):
            return
        operation = self.operation_combo.currentText()
        is_levels = operation == "Kéo giãn tương phản"
        self.levels_widget.setVisible(is_levels)
        self.value_label.setVisible(not is_levels)
        self.value_slider.setVisible(not is_levels)
        self.value_spin.setVisible(not is_levels)
        if is_levels:
            return
        
        self.updating_controls = True

        if operation == "Độ sáng":
            minimum, maximum, value, decimals, step = -100, 100, 0, 0, 1
        elif operation == "Gamma":
            minimum, maximum, value, decimals, step = 10, 300, 100, 2, 0.1
        else:
            minimum, maximum, value, decimals, step = 2, 256, 256, 0, 1

        self.value_slider.setRange(minimum, maximum)
        self.value_slider.setValue(value)
        self.value_spin.setDecimals(decimals)
        self.value_spin.setSingleStep(step)

        if decimals:
            self.value_spin.setRange(minimum / 100, maximum / 100)
            self.value_spin.setValue(value / 100)
        else:
            self.value_spin.setRange(minimum, maximum)
            self.value_spin.setValue(value)

        self.updating_controls = False
        self.update_value_label()

    def slider_changed(self, value):
        if self.updating_controls:
            return
        operation = self.operation_combo.currentText()
        self.updating_controls = True
        self.value_spin.setValue(value / 100 if operation in ("Độ tương phản", "Gamma") else value)
        self.updating_controls = False
        self.update_value_label()
        self.preview_operation()
        
    #Mức sáng
    def levels_changed(self, *_):
        sender = self.sender()
        # Lấy giá trị từ widget vừa được người dùng thao tác
        if sender in (self.black_slider, self.black_spin):
            a, b = sender.value(), self.white_slider.value()
            if a >= b:
                a = max(0, b - 1)
        else:
            a, b = self.black_slider.value(), sender.value()
            if b <= a:
                b = min(255, a + 1)

        # Cập nhật cả 4 widget mà không kích hoạt lại tín hiệu
        for w, v in ((self.black_slider, a), (self.black_spin, a),
                    (self.white_slider, b), (self.white_spin, b)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

        self.preview_operation()

    def spin_changed(self, value):
        if self.updating_controls:
            return
        
        operation = self.operation_combo.currentText()
        self.updating_controls = True
        self.value_slider.setValue(round(value * 100) if operation in ("Độ tương phản", "Gamma") else round(value))
        self.updating_controls = False
        self.update_value_label()
        self.preview_operation()

    def get_parameter_value(self):
        operation = self.operation_combo.currentText()
        if operation == "Kéo giãn tương phản":
            return (self.black_slider.value(), self.white_slider.value())
        value = self.value_spin.value()
        return float(value) if operation in ("Độ tương phản", "Gamma") else int(round(value))

    def update_value_label(self):
        operation = self.operation_combo.currentText()
        value = self.get_parameter_value()
        if operation in ("Độ tương phản", "Gamma"):
            self.value_label.setText(f"{operation}: {value:.2f}")
        elif operation == "Lượng tử hóa":
            self.value_label.setText(f"Số mức lượng tử hóa: {value}")
        else:
            self.value_label.setText(f"{operation}: {value}")

    def process_image(self, image, operation, value):
        if image is None:
            return None
        source = image.astype(np.float32)

        if operation == "Độ sáng":
            result = source + value
            return np.clip(result, 0, 255).astype(np.uint8)

        if operation == "Độ tương phản":
            result = (source - 127.5) * float(value) + 127.5
            return np.clip(result, 0, 255).astype(np.uint8)

        if operation == "Gamma":
            gamma = max(float(value), 0.01)
            return gamma_adjust(img=image, gamma=gamma)

        if operation == "Lượng tử hóa":
            levels = max(2, min(int(value), 256))
            result = np.floor(source * levels / 256) * (255 / (levels - 1))
            return np.clip(result, 0, 255).astype(np.uint8)

        if operation == "Kéo giãn tương phản":
            a, b = value
            return contrast_stretching(img=image, a=a, b=b)
        
        if operation == "Log":
            return log_adjust(img=image)
                
        if operation == "Âm bản":
            return negative_img(img=image)
        
        if operation == "Cân bằng Histogram":
            return hist_balance(img=image)
        
        return image.copy()
        
    def preview_operation(self):
        if self.current_image is None or self.is_video_open:
            return
        operation = self.operation_combo.currentText()
        value = self.get_parameter_value()
        self.preview_image = self.process_image(self.current_image, operation, value)
        self.preview_active = True
        self.compare_before = self.current_image.copy()
        self.compare_after = self.preview_image.copy()
        self.display_image()
        self.status_message.setText(f"Xem trước: {operation}")

    def apply_preview(self, checked=False, operation=None, value=None):
        if self.current_image is None:
            QMessageBox.information(self, "Thông báo", "Hãy mở ảnh trước.")
            return
        if self.is_video_open:
            QMessageBox.information(
                self, "Thông báo",
                "Tạm dừng/đóng video trước khi áp dụng thao tác ảnh."
            )
            return
        
        if operation is not None:
            # Áp dụng trực tiếp (Âm bản, Log...): tự xử lý ảnh, không cần xem trước
            result = self.process_image(self.current_image, operation, value)
            if result is None:
                return
            self.preview_image = result
        else:
            if not self.preview_active or self.preview_image is None:
                return
            operation = self.operation_combo.currentText()
            value = self.get_parameter_value()

        operation = self.operation_combo.currentText()
        value = self.get_parameter_value()
        if operation in ("Độ tương phản", "Gamma"):
            command_name = f"{operation}: {value:.2f}"
        elif operation == "Lượng tử hóa":
            command_name = f"{operation}: {value} mức"
        elif operation == "Kéo giãn tương phản":
            command_name = f"{operation}: a={value[0]}, b={value[1]}"
        elif operation == "Âm bản":
            command_name = f"{operation}"
        else:
            command_name = f"{operation}: {value}"

        self.save_undo_state()
        self.current_image = self.preview_image.copy()
        self.current_state_name = command_name
        self.preview_image = None
        self.preview_active = False
        self.compare_mode = False
        self.compare_checkbox.blockSignals(True)
        self.compare_checkbox.setChecked(False)
        self.compare_checkbox.blockSignals(False)
        self.compare_before = None
        self.compare_after = None

        self.display_image()
        self.update_result()
        self.history_list.addItem(command_name)
        self.history_list.scrollToBottom()
        self.update_history_buttons()
        self.status_message.setText(f"Đã áp dụng: {command_name}")
    
    def apply_negative(self):
        self.apply_preview(operation="Âm bản")

    def apply_log(self):
        self.apply_preview(operation="Log")
        
    def apply_equalize(self):
        self.apply_preview(operation="Cân bằng Histogram")
    
    def cancel_preview(self):
        if self.current_image is None:
            return
        self.preview_image = None
        self.preview_active = False
        self.compare_before = None
        self.compare_after = None
        self.compare_mode = False
        self.compare_checkbox.blockSignals(True)
        self.compare_checkbox.setChecked(False)
        self.compare_checkbox.blockSignals(False)
        self.display_image()
        self.status_message.setText("Đã hủy xem trước")

    def toggle_compare(self, checked):
        if self.current_image is None:
            self.compare_checkbox.blockSignals(True)
            self.compare_checkbox.setChecked(False)
            self.compare_checkbox.blockSignals(False)
            return

        if checked:
            if self.preview_active and self.preview_image is not None:
                self.compare_before = self.current_image.copy()
                self.compare_after = self.preview_image.copy()
            else:
                self.compare_before = self.original_image.copy()
                self.compare_after = self.current_image.copy()

            self.compare_mode = True
            # Ghép ảnh trước/sau cạnh nhau để so sánh trực quan
            before = self.compare_before
            after = self.compare_after
            if before.ndim == 2:
                before = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
            elif before.shape[2] == 4:
                before = cv2.cvtColor(before, cv2.COLOR_BGRA2BGR)
            if after.ndim == 2:
                after = cv2.cvtColor(after, cv2.COLOR_GRAY2BGR)
            elif after.shape[2] == 4:
                after = cv2.cvtColor(after, cv2.COLOR_BGRA2BGR)

            target_h = min(before.shape[0], after.shape[0])
            before = cv2.resize(before, (max(1, int(before.shape[1] * target_h / before.shape[0])), target_h))
            after = cv2.resize(after, (max(1, int(after.shape[1] * target_h / after.shape[0])), target_h))
            gap = np.full((target_h, 12, 3), 45, dtype=np.uint8)
            self.compare_display_image = np.hstack((before, gap, after))
            self.canvas.set_image(self.compare_display_image)
            self.histogram.update_histogram(after)
            self.image_title.setText("SO SÁNH TRƯỚC (trái) / SAU (phải)")
            self.status_message.setText("Đang so sánh ảnh trước và sau")
        else:
            self.compare_mode = False
            self.display_image()

    # -----------------------------------------------------
    # G3: UNDO / REDO / LỊCH SỬ
    # -----------------------------------------------------

    def save_undo_state(self):
        if self.current_image is None:
            return
        self.undo_stack.append((
            self.current_image.copy(),
            self.current_state_name
        ))
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.update_history_buttons()

    def update_history_buttons(self):
        if hasattr(self, "undo_action"):
            self.undo_action.setEnabled(len(self.undo_stack) > 0)
            self.redo_action.setEnabled(len(self.redo_stack) > 0)

    def undo(self):
        if self.current_image is None or self.is_video_open:
            return
        if self.preview_active:
            self.cancel_preview()
        if not self.undo_stack:
            self.status_message.setText("Không còn thao tác để Undo")
            return

        self.redo_stack.append((
            self.current_image.copy(), self.current_state_name
        ))
        self.current_image, self.current_state_name = self.undo_stack.pop()
        self.current_image = self.current_image.copy()
        self.preview_image = None
        self.preview_active = False
        self.compare_mode = False
        self.compare_checkbox.blockSignals(True)
        self.compare_checkbox.setChecked(False)
        self.compare_checkbox.blockSignals(False)
        self.display_image()
        self.update_result()
        self.history_list.addItem("Undo → " + self.current_state_name)
        self.history_list.scrollToBottom()
        self.update_history_buttons()
        self.status_message.setText("Đã Undo")

    def redo(self):
        if self.current_image is None or self.is_video_open:
            return
        if self.preview_active:
            self.cancel_preview()
        if not self.redo_stack:
            self.status_message.setText("Không còn thao tác để Redo")
            return

        self.undo_stack.append((
            self.current_image.copy(), self.current_state_name
        ))
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)

        self.current_image, self.current_state_name = self.redo_stack.pop()
        self.current_image = self.current_image.copy()
        self.preview_image = None
        self.preview_active = False
        self.compare_mode = False
        self.compare_checkbox.blockSignals(True)
        self.compare_checkbox.setChecked(False)
        self.compare_checkbox.blockSignals(False)
        self.display_image()
        self.update_result()
        self.history_list.addItem("Redo → " + self.current_state_name)
        self.history_list.scrollToBottom()
        self.update_history_buttons()
        self.status_message.setText("Đã Redo")

    def clear_history_display(self):
        self.history_list.clear()
        self.history_list.addItem("Đã xóa danh sách hiển thị lịch sử")
        # Không xóa undo_stack/redo_stack, để không làm mất khả năng Undo/Redo.

    # -----------------------------------------------------
    # ĐÓNG ỨNG DỤNG
    # -----------------------------------------------------

    def closeEvent(self, event):
        self.stop_video()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec())
