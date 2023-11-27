import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Load image
image = cv2.imread('dataset\Train\Mentah\Copy of Tm1.png', cv2.IMREAD_GRAYSCALE)

# Sobel
sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
sobel = np.hypot(sobelx, sobely)

# Prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittx = ndimage.convolve(image, kernelx)
prewitty = ndimage.convolve(image, kernely)
prewitt = np.hypot(prewittx, prewitty)

# Robert
robertx = np.array([[1, 0],[0, -1]])
roberty = np.array([[0, 1],[-1, 0]])
robertx = ndimage.convolve(image, robertx)
roberty = ndimage.convolve(image, roberty)
robert = np.hypot(robertx, roberty)

# Laplacian of Gaussian
log = ndimage.gaussian_laplace(image, sigma=3)

# Canny
canny = cv2.Canny(image, 100, 200)

# Kirsh
kernel = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
kirsh = ndimage.convolve(image, kernel)

# Plotting
plt.figure(figsize=(20, 15))
plt.subplot(231), plt.imshow(sobel, cmap='gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(prewitt, cmap='gray')
plt.title('Prewitt'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(robert, cmap='gray')
plt.title('Robert'), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(log, cmap='gray')
plt.title('Laplacian of Gaussian'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(canny, cmap='gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(kirsh, cmap='gray')
plt.title('Kirsh'), plt.xticks([]), plt.yticks([])
plt.show()
