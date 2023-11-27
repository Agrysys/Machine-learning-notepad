import cv2


path_M = "dataset\Test\Matang\Copy of m1.png"
path_TM = "dataset\Test\Mentah\Copy of Tm21.png"
# Read the image as a BGR array
image = cv2.imread(path_M)
# Slice the array to get the red channel
red = image[:,:,2]
# Convert the red channel to grayscale
gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
# Show the grayscale image
cv2.imshow('Gray', gray)
cv2.waitKey(0)
