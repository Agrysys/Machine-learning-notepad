import cv2
import numpy as np
import os

# Directory containing your images
image_directory = 'dataset\Train\Bukan'
output_dir = 'dataset\\tracehold\\Train\\Bukan'

# Iterate over each image in the directory
for filename in os.listdir(image_directory):
    # Only process .jpg images
    img = cv2.imread(os.path.join(image_directory, filename))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Save the processed image
    cv2.imwrite(os.path.join(output_dir, 'processed_' + filename), edges)
    print(filename)
        
