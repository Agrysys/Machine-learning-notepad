import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Function to extract edge features
def extract_edge_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_pixel_count = np.sum(edges) / 255  # Normalization
    return edge_pixel_count

# Defining the path to the melon dataset ('melon_dataset' directory contains melon images)
dataset_path = 'dataset/Train'
categories = ['Matang', 'Mentah']

features = []
labels = []

# Loop through each category (Matang and Mentah)
for category in categories:
    path = os.path.join(dataset_path, category)
    label = categories.index(category)  # Category label
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (100, 100))  # Resize images to the same size

        # Extract edge features
        edge_features = extract_edge_features(image)

        # Save features and labels
        features.append([edge_features])
        labels.append(label)

# Convert to numpy format
features = np.array(features)
labels = to_categorical(labels)

# Split the dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a simple neural network model for classification
model = Sequential()
model.add(Dense(128, input_shape=(1,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('model_melon_edge.h5')
