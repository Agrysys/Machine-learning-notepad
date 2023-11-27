import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the saved model
model = load_model('model_melon.h5')

# Define functions for feature extraction (similar to the training code)
def extract_color_features(image):
    # Ubah gambar ke format HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Ambil histogram dari channel Hue (warna)
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    
    return hist_hue.flatten()

def extract_edge_features(image):
    # Ubah gambar ke skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Terapkan Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Hitung jumlah piksel tepi yang terdeteksi
    edge_pixel_count = np.sum(edges) / 255  # Normalisasi
    
    return edge_pixel_count

# Function to predict on a single image
def predict_single_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))

    # Extract features from the single image
    color_features = extract_color_features(image)
    edge_features = extract_edge_features(image)
    all_features = np.concatenate([color_features, [edge_features]])

    # Reshape the features to match the input shape of the model
    input_features = all_features.reshape(1, -1)

    # Perform prediction
    prediction = model.predict(input_features)
    predicted_class = np.argmax(prediction)  # Get the index of the class with the highest probability
    return predicted_class  # Return the predicted class index

def predict_images(image_paths):
    predictions = []
    # categories = ['Matang', 'Mentah']

    for image_path in image_paths:
        predict = predict_single_image(image_path)
        predictions.append("M" if predict == 0 else "TM")

    return predictions

def predict_images_in_folder(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return predict_images(image_paths)
# Path to the new image you want to predict
# Paths to the new images you want to predict
image_paths = ['dataset\\Train\\Matang\\Copy of m7.png','dataset\\Test\\Mentah\\Copy of Tm101.png','dataset\\Test\\Matang\\Copy of m90.png', 'dataset\\try\\melon.jpeg','dataset\\try\\images3.jpeg']  # Add more paths as needed
folder_path = "dataset\\try"
# Make prediction on the list of images
predicted_classes = predict_images_in_folder(folder_path)
for i, predicted_class in enumerate(predicted_classes):
    print(f"Predicted category for Image {i + 1}: {predicted_class}")