import sys
import os
import cv2
import numpy as np
sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
from utility.feature_extractor import extract_glcm_feature,extract_hsv_features

image = cv2.imread('dataset\Train\Matang\Copy of m6.png')
glcm = extract_glcm_feature(image)
hsv = extract_hsv_features(image)

print("==============")
print("|     HSV     |")
print("==============")
print(hsv)

print("==============")
print("|     GLCM    |")
print("==============")
print(glcm)
np.savetxt('arr.txt', hsv, fmt='%d', delimiter=',')