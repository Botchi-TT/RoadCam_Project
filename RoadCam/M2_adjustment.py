"""
    M2 gồm các chứng năng:
        1 Tạo ảnh âm bản
        2 Điều chỉnh Gamma
        3 Điều chỉnh Log
        4 Kéo giãn tương phản
        5 Cân bằng Histogram: công thức s = 255 * CDF(r)
        6 Cân bằng Histogram Clahe   [chưa làm]
        7 Thể hiện ảnh màu xử lý kênh màu RGB
    =========================================================
    1 - 4 Đã làm khá hoàn thiện
    5 - 6 Cần điều chỉnh thêm
    7 Chưa thực hiện được
"""

from Libary import *

def negative_img(img):
    ""
    negative_image = 255 - img
    return negative_image

def gamma_adjust(img, gamma):
    """
        Tại sao lại có phần normalized_img
        Tại sao có phần np.clip() và astype(np.uint8)
    """
    normalized_img = img / 255.0 
    #tính toán này có thể bị vượt qua 0-255
    img_gamma = 255.0 * np.power(normalized_img, gamma)
    
    img_gamma = np.clip(np.round(img_gamma), 0, 255).astype(np.uint8)
    return img_gamma

def log_adjust(img):
    """
        Kiểm tra có ngoại lệ là hàm c * np.log() có nằm ngoài mức sáng hay không 0 - 255
        Và nếu là log2 log10 thì ảnh có bị ảnh hưởng gì không
    """
    c = 255 / np.log(256)#đây là ln()
    img_log = img.astype(float)
    img_log = (c * np.log(1.0 + img_log)).astype(np.uint8)
    return img_log

def contrast_stretching(img, a, b):
    """
        Giải thích rõ a và b đại diện cho điều gì
        Việc tăng a, b có ý nghĩa gì
    """
    
    # Check giá trị a - b nên để ở đâu
    img_contr = img.astype(float)
    img_contr = ((img_contr - a) * 255) / (b - a)
    return np.clip(img_contr, 0, 255).astype(np.uint8)

#Vẫn cần cải thiện thêm
def hist_balance(img):
    #có thể dùng np.bincount()
    """
        CHỈ XỬ DỤNG ĐƯỢC VỚI ẢNH XÁM KHÔNG XÀI VỚI ẢNH MÀU
        total_pixel =  tổng pixel của ảnh
        count_pixe; = danh sách tần số xuống hiện của điểm ảnh 
        Giải thích lut[img]
    """
    total_pixel = img.shape[0] * img.shape[1]
    count_pixel = [float(np.sum(img == i)/ total_pixel) for i in range(256)]
    cdf = np.cumsum(count_pixel)
    lut = np.clip(np.round(255 * cdf), 0, 255).astype(np.uint8)
    return lut[img]

def hist_balance_openCV(img):
    return cv2.equalizeHist(img)

def draw_hist(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

if __name__ == "__main__":
    # img = cv2.imread("solidDarkRight.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_neg = negative_img(img)
    # for i in np.arange(0.1, 3, 0.5):
    #     img_gamma = gamma_adjust(img, i)
    #     plt.imshow(img_gamma)
    #     plt.title(f'{i}')
    #     plt.show()
    
    # img_log = log_adjust(img=img)
    # fig, ax = plt.subplots(1, 2,figsize=(10,6))
    # ax[0].imshow(img)
    # ax[1].imshow(img_log)
    # plt.show()
    
    # img_ctr = contrast_stretching(img, 10, 15)
    # plt.imshow(img_ctr)
    # plt.show()
    
    # img_balance = hist_balance(img)
    # img_bal_cv2 = hist_balance_openCV(img)
    # fig, ax = plt.subplots(1, 2,figsize=(10,6))
    # ax[0].imshow(img_balance, cmap='gray')
    # ax[1].imshow(img_bal_cv2, cmap='gray')
    # plt.show()
    pass
    