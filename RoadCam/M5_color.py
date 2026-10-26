# TODO:
import cv2,os,glob
import numpy as np

WHITE_RANGE = ((0, 0, 200), (179, 40, 255)) #(h,s,v,)min, (h,s,v)max
YELLOW_RANGE = ((15, 80, 100), (35, 255, 255))
#tách 3 kênh màu h s v
def hsv_channels(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    return h,s,v

def range_mask(image,lower,upper):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv,np.array(lower,dtype=np.uint8),np.array(upper,np.uint8))

def lane_mask(image_bgr, roi_mask=None,white_range=WHITE_RANGE, yellow_range=YELLOW_RANGE):
    """Mặt nạ vạch làn = (trắng OR vàng) AND ROI."""
    white = range_mask(image_bgr, *white_range)
    yellow = range_mask(image_bgr, *yellow_range)
    mask = cv2.bitwise_or(white, yellow)
    if roi_mask is not None:                      # M1 chưa làm thì bỏ qua ROI
        mask = cv2.bitwise_and(mask, roi_mask)
    return mask

"""if __name__=='__main__':
    os.makedirs("out", exist_ok=True)
    h, w = 540, 960
    roi = np.zeros((h, w), np.uint8)
    cv2.fillPoly(roi, [np.array([(50, h), (440, 320), (520, 320), (w-50, h)])], 255)
    
    img=cv2.imread("data/solidWhiteCurve.jpg")
    m=lane_mask(img)
    cv2.imshow('original',img)
    cv2.imshow('umbala',m)  
    cv2.waitKey(0)
   for path in sorted(glob.glob("data/*.jpg")):
        img = cv2.imread(path)
        m = lane_mask(img, roi)
        name = os.path.basename(path)
        print(f"{name:28s} điểm ảnh vạch: {cv2.countNonZero(m)}")
        both = np.hstack([img, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)])
        cv2.imwrite(f"out/m5_{name}", both)"""