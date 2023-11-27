import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

paths = ["dataset\Train\shadow\IMG_20231113_105957.jpg","dataset\Train\shadow\IMG_20231113_110833.jpg","dataset\Train\shadow\IMG_20231113_111048.jpg"]

# Load the image
image = cv2.imread(paths[1])

# Convert the image to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image.reshape((-1, 3))

# Perform k-means clustering to find the most dominant color
kmeans = KMeans(n_clusters=3)
kmeans.fit(pixels)

# Replace all pixels in the image that are not the dominant color with white
dominant_color = kmeans.cluster_centers_[kmeans.predict([np.mean(pixels, axis=0)])]
mask = kmeans.predict(pixels) != kmeans.predict([dominant_color])
pixels[mask] = [255, 255, 255]

# Reshape the pixels back to the original image dimensions
image = pixels.reshape(image.shape)

# Display the original and segmented images side by side
fig, ax = plt.subplots(1, 2)
ax[0].imshow(cv2.cvtColor(cv2.imread('image_path.jpg'), cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[1].imshow(image)
ax[1].set_title('Segmented Image')
plt.show()
