# RoadCam
M2 - M8
NODE 30/10
- Thêm các chức năng hiệu chỉnh log và cân bằng histogram
    +Chưa thêm so sánh
- Sửa lại ROI để bấm chuột sẽ dừng việc cắt ROI ( tường minh hành động):
    + Thêm self.is_dragging = False ở ImageCaves.__init__() và set_interaction_mode()
    + Sửa phần mousePressEvent()
    + mouseMoveEven() sửa điều kiện cuối cùng if self.interaction_mode == "roi" and self.is_dragging:
    + Sửa mouseReleaseEvent()