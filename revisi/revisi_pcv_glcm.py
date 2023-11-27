import cv2
from skimage.feature import graycoprops, graycomatrix
import numpy as np
import matplotlib.pyplot as plt

def extract_glcm_features_all_angles(image):
    image = np.array(image)
    distances = [1]
    angles = [0, 45, 90, 270]
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = {}
    for prop in properties:
        features[prop] = {}
        for index in range(len(angles)):
            features[prop][angles[index]] = graycoprops(glcm, prop)[0,index]
    return features

def convert_image_2_edge(image):
    edges = cv2.Canny(image, 100, 200)
    
    return  edges

def show(data):
    # Create a figure with 4 subplots, one for each feature
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Loop through the features and plot the histograms
    for i, feature in enumerate(data.keys()):
    # Get the row and column index of the subplot
        row = i // 2
        col = i % 2

    # Get the values and angles for the feature
        values = list(data[feature].values())
        angles = list(data[feature].keys())

    # Plot the histogram on the subplot
        axes[row, col].bar(angles, values, width=20)
        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel("Angle")
        axes[row, col].set_ylabel("Value")

# Adjust the layout and show the figure
    plt.tight_layout()
    plt.show()

path_M = "dataset\Test\Matang\Copy of m1.png"
path_TM = "dataset\Test\Mentah\Copy of Tm21.png"

img_m = cv2.imread(path_M)
img_tm = cv2.imread(path_TM)

edge_m = convert_image_2_edge(img_m)
edge_tm = convert_image_2_edge(img_tm)

gray_m = cv2.cvtColor(img_m,cv2.COLOR_BGR2GRAY)
gray_tm = cv2.cvtColor(img_tm,cv2.COLOR_BGR2GRAY)

gray_glcm_m = extract_glcm_features_all_angles(gray_m)
gray_glcm_tm = extract_glcm_features_all_angles(gray_tm)

glcm_m = extract_glcm_features_all_angles(edge_m)
glcm_tm = extract_glcm_features_all_angles(edge_tm)

print("===========")
print("GLCM MATANG")
print("===========")
print(glcm_m)

print("===========")
print("GLCM MENTAH")
print("===========")
print(glcm_tm)



# Show the plot