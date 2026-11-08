# TODO:
"""
    1. Xây dựng chức năng phát hiện biên
    2. Cho phép lựa chọn phương pháp phát hiện biên
    3. Điều chỉnh tham số
    4. Hiển thị kết quả
"""
from Libary import *
from M3_filters import gaussian_filter

G = (0, 255, 0)     # vạch trái (BGR)
R = (0, 0, 255)       # vạch phải (BGR)
BLUE = (255, 128, 0)

def sobel(gray_img):
    gx=cv2.Sobel(gray_img,cv2.CV_64F,1,0)
    gy=cv2.Sobel(gray_img,cv2.CV_64F,0,1)
    dolon=np.sqrt(gx**2+gy**2)
    dolon=np.split(dolon,0,255)
    return dolon.astype(np.uint8)

def canny(gray_img,low,high):
    low,high= int(low), int(high)
    return cv2.Canny(gray_img,low,high)

def hough_lines(edge_img,threshold,min_len,max_gap):
    line=cv2.HoughLinesP(edge_img,1,np.pi/180,int(threshold),min_len,max_gap)
    if line is None:
        return []
    else:
        return line.reshape(-1, 4)

def classify_line(line, slope_thresh=0.5):
    left,right=[],[]
    for x1,y1,x2,y2 in line:
        if x2==x1:
            continue
        m=(y2-y1)/(x2-x1)
        if m <-slope_thresh:
            left.append((x1,y1,x2,y2))
        elif m>slope_thresh:
            right.append((x1,y1,x2,y2))
    return left,right

def draw_lane(img,left,right,thickness=3):
     # Nếu ảnh xám thì chuyển sang BGR
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Nếu ảnh màu thì giữ nguyên ảnh gốc
    else:
        out = img.copy()
    for x1,y1,x2,y2 in left:
        cv2.line(out,(int(x1),int(y1)),(int(x2),int(y2)),G,thickness)
    for x1,y1,x2,y2 in right:
        cv2.line(out,(int(x1),int(y1)),(int(x2),int(y2)),R,thickness)
    return out

#pipeline
def apply_roi(edges,roi_points):
    if roi_points is None or len(roi_points) < 3:
        raise ValueError("ROI phải có ít nhất 3 điểm")

    mask = np.zeros(edges.shape[:2], dtype=np.uint8)
    points = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return cv2.bitwise_and(edges, mask)

def edges_in_roi(img,canny_low,canny_high,blur_ksize=5,blur_sigma=1.5,roi=None):
    blur=gaussian_filter(img,blur_ksize,blur_sigma)
    return apply_roi(canny(blur,canny_low,canny_high),roi)

def hough_all(img,threshold,min_len,max_gap,canny_low,canny_high,roi):
    # Vẽ mọi đoạn tìm được
    edges=edges_in_roi(img,canny_low,canny_high,roi)
    return draw_lane(img,hough_lines(edges,threshold,min_len,max_gap))

def lane_split(img,roi,threshold,min_len,max_gap,canny_low,canny_high):
    return detect_lane(
        img,
        roi=roi,
        canny_low=canny_low,
        canny_high=canny_high,
        threshold=threshold,
        min_len=min_len,
        max_gap=max_gap)

def detect_lane(img,roi,canny_low,canny_high,threshold,min_len,max_gap):
    edges=edges_in_roi(img,canny_low,canny_high,roi=roi)
    left,right=classify_line(hough_lines(edges,threshold,min_len,max_gap))
    return draw_lane(img,left,right)

if __name__ == "__main__":
    import glob
    roi = np.array([
    (0, 719),       # Góc dưới trái
    (300, 350),     # Góc trên trái
    (980, 350),     # Góc trên phải
    (1279, 719)     # Góc dưới phải
    ], dtype=np.int32)
    for path in glob.glob("test_images/*.jpg"):
        img = cv2.imread(path)
        res = detect_lane(img,roi,canny_low=50,canny_high=150,threshold=15,min_len=40,max_gap=20)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.show()