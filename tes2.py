import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
from utility.feature_extractor import extract_glcm_feature,extract_hsv_features, extract_glcm_features_all_angles

path = "dataset\Test\Matang\Copy of m1.png"
image = cv2.imread(path)

glcm = extract_glcm_features_all_angles(image)

print(str(glcm))
print("length : "+ str(len(glcm)))

data = glcm

properties = list(data.keys())
angles = list(data[properties[0]].keys())
distances = list(data[properties[0]][angles[0]].keys())

fig, axs = plt.subplots(len(properties), len(angles), figsize=(15, 15))

for i, prop in enumerate(properties):
    fig.suptitle(prop, fontsize=16)
    fig, axs = plt.subplots(1, len(angles), figsize=(15, 3))
    for j, angle in enumerate(angles):
        axs[j].plot(distances, [data[prop][angle][d] for d in distances])
        axs[j].set_title(f'angle {angle}')
        axs[j].set_xlabel('Distance')
        axs[j].set_ylabel(prop)
    plt.tight_layout()
    

plt.show()