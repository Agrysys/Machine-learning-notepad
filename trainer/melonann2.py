import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Function to extract RGB color features
def extract_color_features(image):
    # Split the image into its Red, Green, and Blue components
    b, g, r = cv2.split(image)

    # Calculate histograms for Red, Green, and Blue channels
    hist_red = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_blue = cv2.calcHist([b], [0], None, [256], [0, 256])

    # Flatten and concatenate the histograms
    all_features = np.concatenate([hist_red.flatten(), hist_green.flatten(), hist_blue.flatten()])

    return all_features

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
        color_features = extract_color_features(image)
        edge_features = extract_edge_features(image)
        
        # Gabungkan fitur-fitur
        all_features = np.concatenate([color_features, [edge_features]])
        
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
model.save('model_melon_edge_RGB.h5')
