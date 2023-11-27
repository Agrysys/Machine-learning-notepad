import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

paths = ["dataset\Train\shadow\IMG_20231113_105957.jpg","dataset\Train\shadow\IMG_20231113_110833.jpg","dataset\Train\shadow\IMG_20231113_111048.jpg"]

# Load the image
image = Image.open(paths[1])

# Convert the image data to an array
data = np.array(image)

imagecv2 = cv2.imread(paths[1])
gray = cv2.cvtColor(imagecv2, cv2.COLOR_BGR2GRAY)

# Get the RGB values
r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]

# Create a figure with 4 subplots: one for each color channel, and one for the histogram
fig, axs = plt.subplots(4, 2, figsize=(12, 12))

# Display the red channel in the first subplot
axs[0, 0].imshow(r, cmap='gray' )
axs[0, 0].set_title('Red Channel')
axs[0, 0].axis('off')

# Display the green channel in the second subplot
axs[1, 0].imshow(g, cmap='gray')
axs[1, 0].set_title('Green Channel')
axs[1, 0].axis('off')

# Display the blue channel in the third subplot
axs[2, 0].imshow(b, cmap='gray')
axs[2, 0].set_title('Blue Channel')
axs[2, 0].axis('off')

# Create the histogram in the fourth subplot
axs[0, 1].hist(r.ravel(), bins=256, color='red', alpha=0.5)
axs[0, 1].set_title('Histogram_red')
axs[1, 1].hist(g.ravel(), bins=256, color='green', alpha=0.5)
axs[1, 1].set_title('Histogram_grees')
axs[2, 1].hist(b.ravel(), bins=256, color='blue', alpha=0.5)
axs[2, 1].set_title('Histogram_blue')

axs[3,0].imshow(gray, cmap="gray")

# Show the plot
plt.show()


