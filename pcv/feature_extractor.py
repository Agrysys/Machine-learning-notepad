import pandas as pd
import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
from utility.feature_extractor import extract_hsv_features, extract_glcm_features_all_angles

dataset_path = 'dataset/Train'
categories = ['Matang', 'Mentah']
# Menyiapkan list untuk menyimpan fitur, label, dan nama file
features = []
labels = []
file_names = []

# Loop melalui setiap kategori (matang dan mentah)
for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)  # Label kategori
    
    # Loop melalui setiap gambar dalam setiap kategori
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (100, 100))  # Resize gambar ke ukuran yang sama
        print("extracting "+ img_name,end=", var size : ")
        # Ekstraksi fitur warna dan edge
        glcm_feature = extract_glcm_features_all_angles(image)
        color_features = extract_hsv_features(image)
        # Gabungkan fitur-fitur
        all_features = np.concatenate([color_features, glcm_feature])
        # Simpan fitur, label, dan nama file
        features.append(all_features)
        labels.append(label)
        file_names.append(img_name)
        print(str(sys.getsizeof(features)))

# Convert features, labels, and file names to DataFrame
features_df = pd.DataFrame(features)
labels_df = pd.DataFrame(labels, columns=['Label'])
file_names_df = pd.DataFrame(file_names, columns=['File Name'])

# Concatenate features, labels, and file names
data = pd.concat([file_names_df, features_df, labels_df], axis=1)

# Save to CSV
data.to_csv('melon_features.csv', index=False)
