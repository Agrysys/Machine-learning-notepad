import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('dataset\Train\Matang\Copy of m8.png', 0)

# Display original image
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.show()

# Noise Reduction
img_noise = cv2.GaussianBlur(img, (5, 5), 0)

# Display image after noise reduction
plt.imshow(img_noise, cmap='gray')
plt.title('Image after Noise Reduction')
plt.show()

# Finding Intensity Gradient of the Image
sobelx = cv2.Sobel(img_noise, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_noise, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(sobelx**2 + sobely**2)
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Display image after finding intensity gradient
plt.imshow(mag, cmap='gray')
plt.title('Image after Finding Intensity Gradient')
plt.show()

# Non-maximum Suppression
theta = np.arctan2(sobely, sobelx)
theta = np.rad2deg(theta) % 180
rows, cols = img_noise.shape
for i in range(1, rows-1):
    for j in range(1, cols-1):
        if (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] <= 180):
            q, r = mag[i, j+1], mag[i, j-1]
        elif (22.5 <= theta[i,j] < 67.5):
            q, r = mag[i+1, j-1], mag[i-1, j+1]
        elif (67.5 <= theta[i,j] < 112.5):
            q, r = mag[i+1, j], mag[i-1, j]
        else:
            q, r = mag[i-1, j-1], mag[i+1, j+1]
        if (mag[i,j] >= q) and (mag[i,j] >= r):
            mag[i,j] = mag[i,j]
        else:
            mag[i,j] = 0

# Display image after non-maximum suppression
plt.imshow(mag, cmap='gray')
plt.title('Image after Non-maximum Suppression')
plt.show()
