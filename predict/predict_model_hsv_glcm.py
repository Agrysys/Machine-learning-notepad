import os.path
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
from utility.feature_extractor import extract_glcm_feature,extract_hsv_features

model_Path = os.path.join('models','model_melon_hsv_glcm.h5')
model_hsv_glcm = load_model(model_Path)

def predict_single_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100, 100))

    # Ekstraksi fitur warna dan edge
    glcm_feature = extract_glcm_feature(image)
    color_features = extract_hsv_features(image)
        
    # Gabungkan fitur-fitur
    all_features = np.concatenate([color_features, glcm_feature])
    # Reshape the features for prediction
    features_for_prediction = np.expand_dims(all_features, axis=0)

    # Make a prediction
    prediction = model_hsv_glcm.predict(features_for_prediction)

    # Get the category with the highest probability
    predicted_category = np.argmax(prediction)    # Get the index of the class with the highest probability
    return predicted_category  # Return the predicted class index

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

