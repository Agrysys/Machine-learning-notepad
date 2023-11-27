import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from skimage.feature import graycomatrix, graycoprops
import sys

sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
from utility.feature_extractor import extract_glcm_feature,extract_hsv_features, extract_glcm_features_all_angles

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
        print("extracting "+ img_name,end=", var size : ")
        # Ekstraksi fitur warna dan edge
        glcm_feature = extract_glcm_features_all_angles(image)
        color_features = extract_hsv_features(image)
        # Gabungkan fitur-fitur
        all_features = np.concatenate([color_features, glcm_feature])
        # Simpan fitur dan label
        features.append(all_features)
        labels.append(label)
        print(str(sys.getsizeof(features)))


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

model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Latih model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Simpan model setelah pelatihan
model.save('models/model_melon_hsv_glcm_28.h5')
