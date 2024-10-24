import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh
image = cv2.imread('Beckham.png', cv2.IMREAD_GRAYSCALE)

# Khai báo toán tử Sobel
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

# Dò biên bằng toán tử Sobel
gradient_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
gradient_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)
sobel_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# Khai báo kernel Laplace Gaussian
def laplacian_of_gaussian(sigma):
    size = int(2 * (np.ceil(3 * sigma)) + 1)
    log_kernel = np.zeros((size, size))
    mean = size // 2
    for x in range(size):
        for y in range(size):
            x_val = x - mean
            y_val = y - mean
            log_kernel[x, y] = (1 / (np.pi * sigma**4)) * (sigma**2 - (x_val**2 + y_val**2)) * \
                               np.exp(- (x_val**2 + y_val**2) / (2 * sigma**2))
    return log_kernel

log_kernel = laplacian_of_gaussian(sigma=1.0)
laplacian = cv2.filter2D(image, cv2.CV_64F, log_kernel)

# Hiển thị kết quả
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Sobel Magnitude')
plt.imshow(sobel_magnitude, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Laplacian of Gaussian')
plt.imshow(laplacian, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
