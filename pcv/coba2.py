import pandas as pd
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

image = cv2.imread('tomat.jpg')
image = np.array(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
DF = pd.DataFrame(image)
DF.to_csv("data2.csv")
glcm = graycomatrix(image, distances=[2], angles=[0], levels=256, symmetric=True, normed=True)
cs = graycoprops(glcm, 'contrast')[0,0]
hom = graycoprops(glcm, 'homogeneity')[0,0]
eng = graycoprops(glcm, 'energy')[0,0]
kor = graycoprops(glcm, 'correlation')[0,0]
fitur = [cs, hom, eng, kor]
# while True:
#     cv2.imshow('gray',gray)
#     cv2.imshow('gray1',gray1)
#     key = cv2.waitKey(0)
#     if key == ord("q"):
#         break
# cv2.destroyAllWindows()

# print("image", image)
# print("gray", gray)
print("fitur" , fitur)