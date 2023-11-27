import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# ... (Previous code for feature extraction remains the same)
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

# Define dataset path and categories
dataset_path = 'dataset/Train'
categories = ['Matang', 'Mentah','Bukan']

# Prepare lists to store features and labels
data = []
labels = []

for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (100, 100))  # Resize image to a fixed size
        
        # Ekstraksi fitur warna dan edge
        color_features = extract_color_features(image)
        edge_features = extract_edge_features(image)
        
        # Gabungkan fitur-fitur
        all_features = np.concatenate([color_features, [edge_features]])

        # Preprocess the image and store it along with the label
        data.append(all)
        labels.append(label)

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize pixel values of the images
data = data / 255.0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build a simple CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('model_melon_cnn.h5')
