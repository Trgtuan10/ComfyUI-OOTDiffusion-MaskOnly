import numpy as np
import cv2
import matplotlib.pyplot as plt

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width=1, mode='constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

# Tạo hình vuông có 4 cạnh trắng và vùng giữa rỗng
size = 200
img = np.zeros((size, size), dtype=np.uint8)
cv2.rectangle(img, (0, 0), (size-1, size-1), 255, 2)  # Hình chữ nhật viền trắng

# Lưu ảnh trước khi lấp đầy
cv2.imwrite('before_fill.png', img)

# Hiển thị hình ảnh ban đầu
plt.imshow(img, cmap='gray')
plt.title('Hình ảnh trước khi lấp đầy')
plt.show()

# Lấp đầy vùng giữa
filled_img = hole_fill(img)

# Lưu ảnh sau khi lấp đầy
cv2.imwrite('after_fill.png', filled_img)

# Hiển thị hình ảnh sau khi lấp đầy
plt.imshow(filled_img, cmap='gray')
plt.title('Hình ảnh sau khi lấp đầy')
plt.show()
