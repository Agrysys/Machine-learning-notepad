import cv2
import tensorflow
import os
import sys
import csv
sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
from utility.feature_extractor import extract_glcm_features_all_angles

path_train_matang = "dataset\Train\Matang"
path_train_mentah = "dataset\Train\Mentah"
path_train_bukan = "dataset\Train\Bukan"
path_test_matang = "dataset\Test\Matang"
path_test_mentah = "dataset\Test\Mentah"
path_test_bukan = "dataset\Test\Bukan"

categories = ['matang','mentah', "bukan"]

training_folders = [path_train_matang,path_train_mentah,path_train_bukan]
test_folders = [path_test_matang, path_test_mentah,path_test_bukan]
features = []
labels = []

features_test = []
labels_test = []

props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
angles = [0, 45, 90, 135]

try:
    f = open("fitur/refisi/glcm_edge_train_data.csv","x")
    ft = open("fitur/refisi/glcm_edge_test_data.csv","x")
except(FileExistsError):
    f = open("fitur/glcm_edge_train_data.csv","w")
    ft = open("fitur/glcm_edge_test_data.csv","w")
finally:
    f.write("nama citra , ")
    for prop in props:
        for angle in angles:
            f.write(f"{prop}({angle}) , ")
    f.write("label\n")
    print("load train")
    for folder in range(len(training_folders)):
        label = categories[folder]
        folder = training_folders[folder]
        print(folder)
        row = []
        f.newlines
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image,(100,100))
            
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(image, threshold1=30, threshold2=100)
            glcm = extract_glcm_features_all_angles(edge)
            
            features.append(glcm.flatten())
            labels.append(label)
            
            f.write(image_name+" , ")
            for feature in glcm:
                f.write(str(round(feature,4))+" , ")
            
            f.write(label+"\n")
        f.flush
    print("load test")
    for folder in range(len(test_folders)):
        label = categories[folder]
        folder = test_folders[folder]
        row = []
        ft.newlines
        print(folder)
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image,(100,100))
            
            image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(image, threshold1=30, threshold2=100)
            glcm = extract_glcm_features_all_angles(edge)
            
            features.append(glcm.flatten())
            labels.append(label)
            
            ft.write(image_name+" , ")
            for feature in glcm:
                ft.write(str(round(feature,4))+" , ")
            
            ft.write(label+"\n")
        ft.flush
    ft.close()