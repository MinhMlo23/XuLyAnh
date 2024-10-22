import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
input_image_path = 'phongcanh.png'  # Thay đổi đường dẫn tới ảnh của bạn
image = cv2.imread(input_image_path)

# Kiểm tra xem ảnh có được đọc thành công không
if image is None:
    print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn.")
    exit()

# Chuyển đổi sang ảnh xám
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Ảnh âm tính
negative_image = 255 - gray_image

# 2. Tăng độ tương phản (Sử dụng CLAHE)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))  # Tăng clipLimit để tăng cường độ tương phản
contrast_image = clahe.apply(gray_image)

# 3. Biến đổi log
# Normalize gray_image to range [0, 1] before applying log transformation
normalized_gray_image = gray_image / 255.0
log_image = np.log1p(normalized_gray_image)  # log1p tính log(x + 1)
log_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 4. Cân bằng Histogram
hist_equalized_image = cv2.equalizeHist(gray_image)



# Hiển thị kết quả
plt.figure(figsize=(18, 12))

# Hiển thị ảnh gốc và histogram của nó
plt.subplot(3, 4, 1)
plt.title('Ảnh gốc')
plt.imshow(gray_image, cmap='gray')

plt.subplot(3, 4, 2)
plt.title('Histogram ảnh gốc')
plt.hist(gray_image.ravel(), bins=256, range=[0, 256], color='black')

# Hiển thị ảnh âm tính và histogram của nó
plt.subplot(3, 4, 3)
plt.title('Ảnh âm tính')
plt.imshow(negative_image, cmap='gray')

plt.subplot(3, 4, 4)
plt.title('Histogram ảnh âm tính')
plt.hist(negative_image.ravel(), bins=256, range=[0, 256], color='black')

# Hiển thị ảnh tăng độ tương phản và histogram của nó
plt.subplot(3, 4, 5)
plt.title('Tăng độ tương phản')
plt.imshow(contrast_image, cmap='gray')

plt.subplot(3, 4, 6)
plt.title('Histogram tăng độ tương phản')
plt.hist(contrast_image.ravel(), bins=256, range=[0, 256], color='black')

# Hiển thị ảnh biến đổi log và histogram của nó
plt.subplot(3, 4, 7)
plt.title('Biến đổi log')
plt.imshow(log_image, cmap='gray')

plt.subplot(3, 4, 8)
plt.title('Histogram biến đổi log')
plt.hist(log_image.ravel(), bins=256, range=[0, 256], color='black')

# Hiển thị ảnh cân bằng histogram và histogram của nó
plt.subplot(3, 4, 9)
plt.title('Cân bằng Histogram')
plt.imshow(hist_equalized_image, cmap='gray')

plt.subplot(3, 4, 10)
plt.title('Histogram cân bằng Histogram')
plt.hist(hist_equalized_image.ravel(), bins=256, range=[0, 256], color='black')



plt.tight_layout()
plt.show()