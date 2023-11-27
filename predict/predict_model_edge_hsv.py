import cv2
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
from utility.feature_extractor import extract_edge_features,extract_hsv_features

# Load the saved model
model_path = os.path.join('models','model_melon_edge_hsv.h5')
model_edge_hsv = load_model(model_path)

# Function to predict on a single image
def predict_single_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))

    # Extract features from the single image
    color_features = extract_hsv_features(image)
    edge_features = extract_edge_features(image)
    all_features = np.concatenate([color_features, [edge_features]])

    # Reshape the features to match the input shape of the model
    input_features = all_features.reshape(1, -1)

    # Perform prediction
    prediction = model_edge_hsv.predict(input_features)
    predicted_class = np.argmax(prediction)  # Get the index of the class with the highest probability
    return predicted_class  # Return the predicted class index

def predict_images(image_paths):
    predictions = []
    # categories = ['Matang', 'Mentah']

    for image_path in image_paths:
        predict = predict_single_image(image_path)
        predictions.append("TM" if predict == 1 else "M")

    return predictions

def predict_images_in_folder(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return predict_images(image_paths)