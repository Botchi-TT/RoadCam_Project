# TODO:
"""
    1. Phát video
    2. Tạm dừng và tiếp tục video
    3. Xử lý từng khung hình
    4. Phân tích video theo yêu cầu bài thực hành
"""
import os

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QHBoxLayout, QFrame,
    QFileDialog, QMessageBox, QSlider, QSpinBox, QDialog, QGridLayout,
    QProgressDialog, QDialogButtonBox
)
from M7_edges import detect_lane

LANE_DEFAULTS = {"canny_low": 50, "canny_high": 150,
                 "threshold": 15, "min_len": 40, "max_gap": 20}

LANE_ERRORS = (cv2.error, ValueError, TypeError, ZeroDivisionError)


# =========================================================
# M8 — VIDEO MIXIN (menu, transport bar, overlays, export)
# =========================================================

class VideoModuleM8:
    """Video features of M8, mixed into the main window class.

    The host window must already provide: add_action(), status_message,
    image_title, canvas, display_image(), reset_history(), clear_preview(),
    output_folder(), image_path, current_image, original_image,
    current_roi_points() and operation_sliders.
    """

    def init_video_state(self):
        """Create every attribute the video module needs. Call it in __init__."""
        self.video_capture = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.read_video_frame)
        self.video_fps = 25
        self.video_frame_number = 0      # 1-based index shown to the user
        self.video_total_frames = 0
        self.video_path = ""
        self.is_video_open = False
        self.video_at_end = False        # True when playback reached the last frame
        self.video_overlay = "none"      # none | lane | diff | flow
        self.video_raw_frame = None      # untouched frame, overlays are computed from it
        self.video_shown = None          # last frame drawn on screen (overlay included)
        self.video_prev_gray = None      # previous frame (gray) for absdiff / optical flow
        self.flow_points = None          # tracked corners for Lucas-Kanade
        self.flow_trail = None           # accumulated motion trails
        self.diff_threshold = 25         # threshold used by the change detector
        self._lane_sliders_hooked = False

    def create_video_menu(self, bar):
        """Build the "Video" menu on the given menu bar."""
        m = bar.addMenu("Video")
        self.add_action(m, "Mở video...", self.open_video)
        self.add_action(m, "Phát / Tạm dừng", self.toggle_video, "Space")
        self.add_action(m, "Khung trước", self.prev_frame, "Ctrl+Left")
        self.add_action(m, "Khung sau", self.next_frame, "Ctrl+Right")
        m.addSeparator()
        self.add_action(m, "Trích khung hiện tại (sang vùng ảnh)", self.extract_frame, "Ctrl+E")
        m.addSeparator()
        self.overlay_actions = {}
        for key, text in [("lane", "Phát hiện làn trên video"),
                          ("diff", "Phát hiện thay đổi"),
                          ("flow", "Theo vết chuyển động")]:
            act = QAction(text, self)
            act.setCheckable(True)
            act.triggered.connect(lambda _=False, k=key: self.set_video_overlay(k))
            m.addAction(act)
            self.overlay_actions[key] = act
        self.add_action(m, "Tắt lớp phủ (xem video gốc)", lambda: self.set_video_overlay("none"))
        m.addSeparator()
        self.add_action(m, "Ngưỡng phát hiện thay đổi...", self.diff_threshold_dialog)
        self.add_action(m, "Xuất video phát hiện làn (toàn bộ khung)...", self.export_lane_video)
        m.addSeparator()
        self.add_action(m, "Dừng video (để chỉnh sửa khung hình)", self.stop_video)

    def build_video_bar(self):
        """Transport bar under the viewer: play, step, seek slider, frame/fps label."""
        self.video_bar = QFrame()
        self.video_bar.setObjectName("videoBar")
        row = QHBoxLayout(self.video_bar)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(6)

        self.video_play_button = QPushButton("▶")
        self.video_play_button.setFixedWidth(34)
        self.video_play_button.setToolTip("Phát / Tạm dừng (Space)")
        self.video_play_button.clicked.connect(self.toggle_video)
        row.addWidget(self.video_play_button)

        prev_button = QPushButton("⏮")
        prev_button.setFixedWidth(34)
        prev_button.setToolTip("Khung trước (Ctrl+←)")
        prev_button.clicked.connect(self.prev_frame)
        row.addWidget(prev_button)

        next_button = QPushButton("⏭")
        next_button.setFixedWidth(34)
        next_button.setToolTip("Khung sau (Ctrl+→)")
        next_button.clicked.connect(self.next_frame)
        row.addWidget(next_button)

        # seek slider: 1..total frames. Only valueChanged is used (drag, click on the
        # groove and keyboard all go through it); show_video_frame() blocks its signals
        # when it moves the handle by code, so there is no feedback loop.
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setRange(1, 1)
        self.video_slider.valueChanged.connect(self.seek_video)
        row.addWidget(self.video_slider, 1)

        self.video_info = QLabel("khung 0 / 0 · 0.0 fps")
        self.video_info.setObjectName("muted")
        row.addWidget(self.video_info)

        grab_button = QPushButton("Trích khung")
        grab_button.setToolTip("Đưa khung hình đang xem sang vùng ảnh để dùng M1–M7 (Ctrl+E)")
        grab_button.clicked.connect(self.extract_frame)
        row.addWidget(grab_button)

        self.video_bar.setVisible(False)
        return self.video_bar

    # ----- open / close -----------------------------------

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Mở video", "", "Tệp video (*.mp4 *.avi *.mov *.mkv *.wmv *.mpeg *.mpg)")
        if not path:
            return
        self.open_video_path(path)

    def open_video_path(self, path):
        """Open `path` and show its first frame (also usable without the file dialog)."""
        self.stop_video(silent=True)
        capture = cv2.VideoCapture(path)
        ok, frame = capture.read() if capture.isOpened() else (False, None)
        if not ok or frame is None:
            capture.release()
            QMessageBox.critical(self, "Lỗi mở video",
                                 "Không mở được video hoặc codec không được hỗ trợ.")
            return
        self.video_capture = capture
        self.is_video_open = True
        self.image_path = path
        self.video_path = path
        # cv2.CAP_PROP_FPS and cv2.CAP_PROP_FRAME_COUNT: playback speed + length
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps if fps and fps > 0 else 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_total_frames = max(total, 1)
        self.video_frame_number = 1
        self.video_at_end = False
        self.video_prev_gray = None
        self.flow_points = None
        self.flow_trail = None
        self.clear_preview(reset_sliders=True)
        self.canvas.fit_mode = True
        self.canvas.image = None
        self.video_slider.blockSignals(True)
        self.video_slider.setRange(1, self.video_total_frames)
        self.video_slider.setValue(1)
        self.video_slider.blockSignals(False)
        self.video_bar.setVisible(True)
        self.show_video_frame(frame)
        self.reset_history(f"Mở video ({os.path.basename(path)})", self.current_image)
        self.status_message.setText(
            f"Đã mở video: {os.path.basename(path)} — {self.video_total_frames} khung, "
            f"{self.video_fps:.1f} fps (Space để phát)")

    def stop_video(self, silent=False):
        """Release the capture; the frame on screen stays editable by M1-M7."""
        self.video_timer.stop()
        was_open = self.is_video_open
        self.is_video_open = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.video_prev_gray = None
        self.flow_points = None
        self.flow_trail = None
        if hasattr(self, "video_bar"):
            self.video_bar.setVisible(False)
            self.video_play_button.setText("▶")
        if was_open and not silent and self.current_image is not None:
            self.original_image = self.current_image.copy()
            self.reset_history(f"Khung hình {self.video_frame_number}", self.current_image)
            self.status_message.setText("Đã dừng video — có thể chỉnh sửa khung hình hiện tại")

    # ----- playback / navigation --------------------------

    def toggle_video(self):
        if self.video_capture is None:
            return
        if self.video_timer.isActive():
            self.video_timer.stop()
            self.video_play_button.setText("▶")
            self.status_message.setText("Tạm dừng video")
        else:
            if self.video_at_end:            # played to the end before: start again from frame 1
                self.goto_frame(1)
            self.is_video_open = True
            self.video_timer.start(max(1, int(1000 / self.video_fps)))
            self.video_play_button.setText("⏸")
            self.status_message.setText("Đang phát video (Space: tạm dừng)")

    def read_video_frame(self):
        """Timer slot: read and display the next frame while playing."""
        if self.video_capture is None:
            return
        ok, frame = self.video_capture.read()
        if not ok or frame is None:
            self.video_timer.stop()
            self.video_at_end = True
            self.video_play_button.setText("▶")
            self.status_message.setText("Video đã phát hết (bấm Phát để xem lại từ đầu)")
            return
        self.video_frame_number = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        self.show_video_frame(frame)

    def goto_frame(self, number):
        """Seek to a 1-based frame number and display it (playback paused)."""
        if self.video_capture is None:
            return
        number = max(1, min(int(number), self.video_total_frames))
        if number == self.video_frame_number and not self.video_at_end:
            return                           # already on that frame
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, number - 1)
        ok, frame = self.video_capture.read()
        if not ok or frame is None:
            return
        # Stepping one frame keeps the previous frame, so change detection and optical
        # flow still work when the user clicks "next / previous frame". Only a real
        # jump breaks the continuity and restarts them.
        keep = self.video_prev_gray is not None and abs(number - self.video_frame_number) == 1
        self.is_video_open = True
        self.video_at_end = False
        self.video_frame_number = number
        if not keep:
            self.video_prev_gray = None
            self.flow_points = None
            self.flow_trail = None
        self.show_video_frame(frame)

    def seek_video(self, value):
        self.video_timer.stop()
        self.video_play_button.setText("▶")
        self.goto_frame(value)

    def prev_frame(self):
        self.video_timer.stop()
        self.video_play_button.setText("▶")
        self.goto_frame(self.video_frame_number - 1)

    def next_frame(self):
        self.video_timer.stop()
        self.video_play_button.setText("▶")
        self.goto_frame(self.video_frame_number + 1)

    def show_video_frame(self, frame, shown=None):
        """Store the raw frame, apply the active overlay (unless `shown` is given) and refresh."""
        self.video_raw_frame = frame
        if shown is None:
            shown = self.apply_video_overlay(frame)
        self.video_shown = shown
        self.current_image = shown
        self.original_image = shown
        self.display_image()
        if not self.video_slider.isSliderDown():
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(min(self.video_frame_number, self.video_total_frames))
            self.video_slider.blockSignals(False)
        self.video_info.setText(
            f"khung {self.video_frame_number} / {self.video_total_frames} · "
            f"{self.video_fps:.1f} fps")
        self.image_title.setText(
            f"{os.path.basename(self.image_path)} — khung {self.video_frame_number}"
            f"/{self.video_total_frames}")

    def refresh_video_frame(self):
        """Redraw the frame on screen after the ROI or a lane parameter changed (while paused).

        Lane detection is recomputed from the raw frame. Change detection and tracking
        depend on the previous frame, so their last result is only shown again:
        recomputing them here would compare the frame with itself.
        """
        if (self.video_capture is None or self.video_raw_frame is None
                or self.video_timer.isActive()):
            return
        if self.video_overlay in ("diff", "flow") and self.video_shown is not None:
            self.show_video_frame(self.video_raw_frame, shown=self.video_shown)
        else:
            self.show_video_frame(self.video_raw_frame)

    # ----- M8 overlays ------------------------------------

    def set_video_overlay(self, mode):
        """Switch the live overlay: lane detection, change detection or tracking."""
        self.video_overlay = mode
        for key, act in self.overlay_actions.items():
            act.setChecked(key == mode)
        self.video_prev_gray = None
        self.flow_points = None
        self.flow_trail = None
        names = {"none": "Tắt lớp phủ", "lane": "Phát hiện làn",
                 "diff": "Phát hiện thay đổi", "flow": "Theo vết chuyển động"}
        self.status_message.setText("Video: " + names.get(mode, mode))
        if self.video_raw_frame is not None and self.video_capture is not None:
            self.show_video_frame(self.video_raw_frame)

    def apply_video_overlay(self, frame):
        if self.video_overlay == "lane":
            return self.lane_frame(frame)
        if self.video_overlay == "diff":
            return self.diff_frame(frame)
        if self.video_overlay == "flow":
            return self.flow_frame(frame)
        return frame.copy()

    # ----- lane detection: M7 function + ROI and parameters of the main window -----

    def lane_roi(self, frame):
        """4 vertices of the trapezoid ROI that fits this frame, or None."""
        getter = getattr(self, "current_roi_points", None)
        return getter(frame, warn=False) if getter else None

    def lane_command_params(self):
        """Canny / Hough values = the sliders of Bien > Phat hien lan (tu dong)."""
        params = dict(LANE_DEFAULTS)
        widgets = getattr(self, "operation_sliders", {}).get(("Biên", "lane"), {})
        for key in params:
            widget = widgets.get(key)
            if widget is not None:
                params[key] = int(round(widget.value()))
        self._hook_lane_sliders(widgets)
        return params

    def _hook_lane_sliders(self, widgets):
        """Once: moving a lane slider redraws the paused frame."""
        if self._lane_sliders_hooked or not widgets:
            return
        for widget in widgets.values():
            slider = getattr(widget, "slider", None)
            if slider is not None:
                slider.valueChanged.connect(lambda _=0: self.refresh_video_frame())
        self._lane_sliders_hooked = True

    def draw_lanes(self, frame, roi, params, draw_roi=False):
        """M7 `detect_lane` on one frame -> BGR frame with the left (green) / right (red) lanes."""
        bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        result = detect_lane(bgr, roi=roi, **params)
        if isinstance(result, tuple):        # tolerate a version of M7 that also returns info
            result = result[0]
        out = result.copy()
        if draw_roi:
            cv2.polylines(out, [np.array(roi, np.int32).reshape(-1, 1, 2)], True, (255, 200, 0), 1)
        return out

    def lane_frame(self, frame):
        """Lane overlay for the frame on screen (needs a valid ROI, like the Edge menu)."""
        roi = self.lane_roi(frame)
        out = frame.copy()
        if roi is None:
            cv2.putText(out, "Chua co ROI hop le", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.status_message.setText(
                "Chưa có ROI hợp lệ cho video: chọn 'ROI hình thang' và click 4 đỉnh, hoặc Nạp ROI")
            return out
        try:
            return self.draw_lanes(frame, roi, self.lane_command_params(), draw_roi=True)
        except LANE_ERRORS as error:
            cv2.putText(out, "Loi phat hien lan", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.status_message.setText(f"Lỗi phát hiện làn: {error}")
            return out

    # ----- change detection / motion tracking -------------

    def diff_frame(self, frame):
        """Change detection: absdiff(current, previous) -> threshold -> boxes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        out = frame.copy()
        if self.video_prev_gray is not None:
            diff = cv2.absdiff(gray, self.video_prev_gray)
            _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
            # tint the changed pixels in red, then box the significant blobs
            out[mask > 0] = (0.4 * out[mask > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # a blob is only boxed when it covers at least 0.15% of the frame,
            # otherwise a moving camera fills the screen with tiny boxes
            min_area = max(300.0, 0.0015 * mask.size)
            changed = 0
            for c in contours:
                if cv2.contourArea(c) < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
                changed += 1
            ratio = 100.0 * float(np.count_nonzero(mask)) / mask.size
            cv2.putText(out, f"Thay doi: {changed} vung ({ratio:.1f}%)  nguong={self.diff_threshold}",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.video_prev_gray = gray
        return out

    def flow_frame(self, frame):
        """Lucas-Kanade tracking of up to 100 corners, drawing their trails."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = frame.copy()
        if self.flow_trail is None or self.flow_trail.shape[:2] != frame.shape[:2]:
            self.flow_trail = np.zeros_like(frame)
        if self.video_prev_gray is None or self.flow_points is None or len(self.flow_points) < 10:
            # (re)seed the tracker with at most 100 good corners
            self.flow_points = cv2.goodFeaturesToTrack(
                gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            self.flow_trail = np.zeros_like(frame)
        elif self.flow_points is not None and len(self.flow_points):
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.video_prev_gray, gray, self.flow_points, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            if new_points is not None:
                good_new = new_points[status.flatten() == 1]
                good_old = self.flow_points[status.flatten() == 1]
                for new, old in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(self.flow_trail, (int(a), int(b)), (int(c), int(d)),
                             (0, 255, 255), 2)
                    cv2.circle(out, (int(a), int(b)), 3, (0, 0, 255), -1)
                self.flow_points = good_new.reshape(-1, 1, 2)
        out = cv2.add(out, self.flow_trail)
        count = 0 if self.flow_points is None else len(self.flow_points)
        cv2.putText(out, f"Theo vet: {count} diem", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.video_prev_gray = gray
        return out

    def diff_threshold_dialog(self):
        """Threshold used by "Phat hien thay doi" (applies from the next frame)."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Ngưỡng phát hiện thay đổi")
        grid = QGridLayout(dialog)
        spin = QSpinBox()
        spin.setRange(1, 255)
        spin.setValue(int(self.diff_threshold))
        grid.addWidget(QLabel("Ngưỡng |khung hiện tại − khung trước| (1–255)"), 0, 0)
        grid.addWidget(spin, 0, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        grid.addWidget(buttons, 1, 0, 1, 2)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.diff_threshold = spin.value()
        self.status_message.setText(f"Ngưỡng phát hiện thay đổi = {self.diff_threshold}")

    # ----- frame grabbing ---------------------------------

    def extract_frame(self):
        """Hand the ORIGINAL frame on screen to the image pipeline so M1-M7 can edit it.

        The raw frame is used (not the picture with lane lines / boxes drawn on it),
        the overlay is switched off and the video is stopped like "Dung video".
        """
        if self.video_capture is None or self.video_raw_frame is None:
            QMessageBox.information(self, "Trích khung hình", "Hãy mở video trước.")
            return
        frame = self.video_raw_frame.copy()
        number = self.video_frame_number
        self.video_overlay = "none"
        for act in self.overlay_actions.values():
            act.setChecked(False)
        self.current_image = frame            # stop_video() copies it into original_image + history
        self.clear_preview(reset_sliders=True)
        self.stop_video()
        self.video_raw_frame = None
        self.video_shown = None
        self.canvas.fit_mode = True
        self.display_image()
        self.status_message.setText(
            f"Đã trích khung {number} (ảnh gốc, không có lớp phủ) — "
            f"có thể dùng các menu Ảnh / Bộ lọc / Biên")

    # ----- export -----------------------------------------

    def export_lane_video(self):
        """Run lane detection on every frame and write the result with VideoWriter."""
        if not self.video_path:
            QMessageBox.information(self, "Xuất video", "Hãy mở video trước.")
            return
        probe = cv2.VideoCapture(self.video_path)
        if not probe.isOpened():
            QMessageBox.critical(self, "Xuất video", "Không mở lại được video nguồn.")
            return
        width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or self.video_total_frames
        probe.release()

        # same ROI as the Edge menu: it must be a saved/drawn ROI of this frame size
        roi = self.lane_roi(np.empty((height, width), np.uint8))
        if roi is None:
            QMessageBox.warning(
                self, "Xuất video",
                f"Chưa có ROI hợp lệ cho video {width}×{height}.\n"
                "Hãy vẽ ROI hình thang (4 đỉnh) hoặc Nạp ROI (.json) rồi xuất lại.")
            return
        params = self.lane_command_params()

        base = os.path.splitext(os.path.basename(self.video_path))[0]
        suggested = os.path.join(self.output_folder("KetQua"), f"{base}_lan.mp4")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu video đã phát hiện làn", suggested, "Video MP4 (*.mp4);;Video AVI (*.avi)")
        if not out_path:
            return

        self.video_timer.stop()
        self.video_play_button.setText("▶")
        source = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if out_path.lower().endswith(".mp4") else "XVID"))
        writer = cv2.VideoWriter(out_path, fourcc, self.video_fps, (width, height))
        if not source.isOpened() or not writer.isOpened():
            source.release()
            writer.release()
            QMessageBox.critical(self, "Xuất video", "Không tạo được tệp video đầu ra.")
            return

        progress = QProgressDialog("Đang phát hiện làn trên từng khung hình...",
                                   "Hủy", 0, total, self)
        progress.setWindowTitle("Xuất video")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        done = 0
        failure = None
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                break
            try:
                writer.write(self.draw_lanes(frame, roi, params))
            except LANE_ERRORS as error:
                failure = error
                break
            done += 1
            if done % 3 == 0 or done == total:
                progress.setValue(min(done, total))
                QApplication.processEvents()
            if progress.wasCanceled():
                break
        canceled = progress.wasCanceled()
        writer.release()
        source.release()
        progress.setValue(total)

        if failure is not None:
            QMessageBox.critical(self, "Xuất video", f"Dừng ở khung {done + 1}: {failure}")
            self.status_message.setText(f"Xuất video bị lỗi sau {done} khung: {out_path}")
            return
        if canceled:
            self.status_message.setText(f"Đã hủy — video tạm có {done} khung: {out_path}")
            return
        self.status_message.setText(f"Đã xuất {done} khung có vẽ làn: {out_path}")
        if QMessageBox.question(
                self, "Xuất video",
                f"Đã xuất {done} khung hình.\n{out_path}\n\nMở thư mục chứa video?"
        ) == QMessageBox.StandardButton.Yes and hasattr(os, "startfile"):
            os.startfile(os.path.dirname(out_path))