# TODO:
""" 1. Tạo mặt nạ
    2. Thực hiện phép toán hình thái học
    3. Làm sạch ảnh
    4. Đếm tế bào theo yêu cầu bài thực hành
"""
from Libary import *

MORPH_SHAPES = [("Hình chữ nhật", cv2.MORPH_RECT),
                ("Hình elip", cv2.MORPH_ELLIPSE),
                ("Hình chữ thập", cv2.MORPH_CROSS)]


#Nhị phân hóa
def binarize(img, threshold_val=127, use_otsu=False):
    if img is None:
        return None,0
    if len(img.shape)==3:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    else:
        gray=img
    if use_otsu:
        actual_thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        actual_thresh=threshold_val
        _,binary=cv2.threshold(gray,threshold_val,255,cv2.THRESH_BINARY)
    return binary, actual_thresh

#Mặt nạ
def make_kernel(shape,ksize):
    ksize=int(np.clip(ksize,3,15))
    ksize+=1-ksize%2
    return cv2.getStructuringElement(shape,(ksize,ksize))

#Cac hinh thai
def erode(img,ksize,shape,iterations):
    return cv2.erode(img,make_kernel(shape,ksize),iterations=int(np.clip(iterations,1,5)))

def dilate(img,ksize,shape,iterations):
    return cv2.dilate(img,make_kernel(shape,ksize),int(np.clip(iterations,1,5)))

def opening(img,ksize,shape,iterations):
    return cv2.morphologyEx(img,cv2.MORPH_OPEN,make_kernel(shape,ksize),int(np.clip(iterations,1,5)))

def closing(img,ksize,shape,iterations):
    return cv2.morphologyEx(img,cv2.MORPH_CLOSE,make_kernel(shape,ksize),int(np.clip(iterations,1,5)))
    
def remove_small_regions(img,min_area):
     binary,_=binarize(img)
     n,labels,stats,centroids=cv2.connectedComponentsWithStats(binary,connectivity=8)
     if n<=1:
         return binary,0,0
     keep=stats[1:,cv2.CC_STAT_AREA]>= int(min_area) #stats index 0 chứa thông tin của nền
     lut= np.zeros(n,dtype=np.uint8)
     lut[1:][keep]=255
     remain=int(keep.sum())
     return lut[labels], remain, int(len(keep)-remain)

def count_regions(img):
    #Số vùng trắng liên thông (8-lân cận) của mặt nạ
    binary,_=binarize(img)
    n, _ = cv2.connectedComponents(binary, connectivity=8)
    return n - 1

if __name__=="__main__":
    # from M5_color import lane_mask
    # # 1. Đọc ảnh
    # image_path = "test_images/solidWhiteCurve.jpg"
    # img = cv2.imread(image_path)

    # if img is None:
    #     print("Không thể đọc ảnh:", image_path)

    # # 2. M5 - Tạo mask vạch làn
    # mask = lane_mask(img)

    # # 3. M6 - Nhị phân hóa
    # binary, threshold = binarize(mask, threshold_val=127)

    # print("===== BINARIZE =====")
    # print("Threshold:", threshold)

    # # 4. M6 - Các phép toán hình thái học
    # kernel_size = 3
    # shape = cv2.MORPH_RECT
    # iterations = 1

    # eroded = erode(binary, kernel_size, shape, iterations)
    # dilated = dilate(binary, kernel_size, shape, iterations)
    # opened = opening(binary, kernel_size, shape, iterations)
    # closed = closing(binary, kernel_size, shape, iterations)

    # # 5. M6 - Xóa vùng nhỏ
    # min_area = 50

    # cleaned, remaining, removed = remove_small_regions(
    #     binary, min_area
    # )

    # # 6. M6 - Đếm vùng liên thông
    # regions_before = count_regions(binary)
    # regions_after = count_regions(cleaned)

    # print("\n===== CONNECTED COMPONENTS =====")
    # print("Số vùng trước khi làm sạch:", regions_before)
    # print("Số vùng sau khi làm sạch:", regions_after)
    # print("Số vùng còn lại:", remaining)
    # print("Số vùng đã xóa:", removed)

    # # 7. Hiển thị kết quả
    # results = {
    #     "Original": img,
    #     "M5 - Lane Mask": mask,
    #     "M6 - Binary": binary,
    #     "M6 - Erosion": eroded,
    #     "M6 - Dilation": dilated,
    #     "M6 - Opening": opened,
    #     "M6 - Closing": closed,
    #     "M6 - Remove Small": cleaned
    # }

    # for name, result in results.items():
    #     cv2.imshow(name, result)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    main()