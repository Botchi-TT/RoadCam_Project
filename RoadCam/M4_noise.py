from Libary import *
from M3_filters import *
# TODO:
""" 
    1. Thêm nhiễu vào ảnh
    2. Xây dựng các phương pháp giảm nhiễu
    3. Cho phép điều chỉnh tham số
    4. Đánh giá ảnh sau xử lý
"""
def app_gaussian_noise(img, mean=0, sigma=5):
    noise = np.random.normal(mean, sigma, img.shape)

    noisy_img = img.astype(np.float64) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def app_salt_pepper_noise(img, p, salt_vs_pepper = 0.5):
    noisy_img = img.copy()
    p = p / 100.0
    
    h, w =img[:2]
    
    num_noise = int(p * h *w)
    
    num_salt = int(num_noise * salt_vs_pepper)
    num_pepper = num_noise - num_salt
    
    salt_y = np.random.randint(0, h, num_salt)
    salt_x = np.random.randint(0, w, num_salt)
    
    pepper_y = np.random.randint(0, h, num_pepper)
    pepper_x = np.random.randint(0, w, num_pepper)
    
    noisy_img[salt_y, salt_x] = 255
    noisy_img[pepper_y, pepper_x] = 0
    
    return noisy_img
    
def denoising():
    pass
if __name__ == "__main__":
    pass