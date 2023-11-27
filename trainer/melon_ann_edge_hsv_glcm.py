import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


def extract_GLCM(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    cs = graycoprops(glcm, 'contrast')[0,0]
    hom = graycoprops(glcm, 'homogeneity')[0,0]
    eng = graycoprops(glcm, 'energy')[0,0]
    kor = graycoprops(glcm, 'correlation')[0,0]
    fitur = [cs, hom, eng, kor]
    return fitur

# Fungsi untuk ekstraksi fitur warna
def extract_color_features(image):
    # Ubah gambar ke format HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Ambil histogram dari channel Hue (warna)
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    
    return hist_hue.flatten()

# Fungsi untuk ekstraksi fitur urat melon menggunakan edge detection
def extract_edge_features(image):
    # Ubah gambar ke skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Terapkan Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Hitung jumlah piksel tepi yang terdeteksi
    edge_pixel_count = np.sum(edges) / 255  # Normalisasi
    
    return edge_pixel_count

# Mendefinisikan path ke dataset melon (direktori 'melon_dataset' berisi gambar melon)
dataset_path = 'dataset/Train'
categories = ['Matang', 'Mentah']

# Menyiapkan list untuk menyimpan fitur dan label
features = []
labels = []

# Loop melalui setiap kategori (matang dan mentah)
for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)  # Label kategori
    
    # Loop melalui setiap gambar dalam setiap kategori
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (100, 100))  # Resize gambar ke ukuran yang sama
        
        # Ekstraksi fitur warna dan edge
        glcm_feature = extract_GLCM(image)
        color_features = extract_color_features(image)
        edge_features = extract_edge_features(image)
        
        # Gabungkan fitur-fitur
        all_features = np.concatenate([color_features, [edge_features], glcm_feature])
        
        # Simpan fitur dan label
        features.append(all_features)
        labels.append(label)

# Ubah ke format numpy
features = np.array(features)
labels = to_categorical(labels)

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Buat model CNN sederhana untuk klasifikasi
model = Sequential()
model.add(Dense(128, input_shape=(features.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Latih model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Simpan model setelah pelatihan
model.save('model\\model_melon_edge_hsv_glcm.h5')
