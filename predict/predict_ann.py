# Load the model
import cv2
import numpy as np
from keras.models import load_model
model = load_model("model_terbaik.h5")

# Load your image
img_path = 'dataset\Test\Bukan\IMG_20231114_145714.jpg'
img_path = 'dataset\Test\Matang\Copy of m3.png'
img_path = 'dataset\Test\Mentah\Copy of Tm24.png'
img = cv2.imread(img_path)
canny = cv2.Canny(img,100,200)
resized_canny = cv2.resize(canny, (150, 150))

# Reshape your data to match the input shape of the model
input_data = np.array([resized_canny])  # Model expects a batch of images as input

# Make a prediction
predictions = model.predict(input_data)

# Get the predicted category
predicted_category_index = np.argmax(predictions)
categories = ['Bukan','Matang','Mentah']
predicted_category = categories[predicted_category_index]

print(predicted_category)
