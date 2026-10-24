from Libary import *

def average_filter(img, ksize):
    return cv2.blur(img, (ksize, ksize), cv2.BORDER_DEFAULT)

def gaussian_filter(img, ksize, sigma):
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError("ksize phải là số lẻ >= 3")
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_DEFAULT)

def median_filter(img, ksize):
    return cv2.medianBlur(img, (ksize, ksize), cv2.BORDER_DEFAULT)

def laplacian_filter(img, c):
    laplacion = cv2.Laplacian(img, cv2.CV_64F)
    
    sharpened = img.astype(np.float64) - c * laplacion
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def unsharp_masking(img, k):
    sharpened = img.astype(np.float64) - gaussian_filter(img, 5).astypy(np.float64)
    result = img.astype(np.float64) + k * sharpened
    return np.clip(result, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    img = cv2.imread("solidDarkRight.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)