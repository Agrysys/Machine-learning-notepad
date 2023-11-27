import cv2
import tensorflow
import os
import sys
import csv
sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
from utility.feature_extractor import extract_glcm_selected_feature

path_train_matang = "dataset\Train\Matang"
path_train_mentah = "dataset\Train\Mentah"
path_train_bukan = "dataset\Train\Bukan"
path_test_matang = "dataset\Test\Matang"
path_test_mentah = "dataset\Test\Mentah"
path_test_bukan = "dataset\Test\Bukan"

categories = ['matang','mentah','bukan']

training_folders = [path_train_matang,path_train_mentah,path_train_bukan]
test_folders = [path_test_matang, path_test_mentah, path_test_bukan]
features = []
labels = []

features_test = []
labels_test = []

properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
angles = [0, 45, 90, 135]
selected = [
        [angles[0],properties[0]],
        [angles[3],properties[0]],
        [angles[0],properties[1]],
        [angles[3],properties[1]],
        [angles[3],properties[2]],
        [angles[0],properties[4]],
        [angles[1],properties[4]],
        [angles[3],properties[4]]
    ]

path_train_csv = "fitur/selected/glcm_edge_train_data1.csv"
path_test_csv = "fitur/selected/glcm_edge_test_data1.csv"

try:
    f = open(path_train_csv,"x")
    ft = open(path_test_csv,"x")
except(FileExistsError):
    f = open(path_train_csv,"w")
    ft = open(path_test_csv,"w")
finally:
    f.write("nama citra , ")
    ft.write("nama citra , ")
    for angle, prop in selected:
        f.write(f"{prop} {angle}, ")
        ft.write(f"{prop} {angle}, ")
    f.write("label\n")
    ft.write("label\n")
    
    for folder in range(len(training_folders)):
        label = categories[folder]
        folder = training_folders[folder]
        row = []
        f.newlines
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image,(100,100))
            
            # image_edge = cv2.Canny(image,100,200)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            glcm = extract_glcm_selected_feature(image_gray)
            
            features.append(glcm.flatten())
            labels.append(label)
            print(f"{image_name} : {glcm[0]}")
            
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
        for image_name in os.listdir(folder):
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image,(100,100))
            
            image_edge = cv2.Canny(image,100,200)
            glcm = extract_glcm_selected_feature(image_edge)
            
            features.append(glcm.flatten())
            labels.append(label)
            print(glcm[0])
            
            ft.write(image_name+" , ")
            for feature in glcm:
                ft.write(str(round(feature,4))+" , ")
            
            ft.write(label+"\n")
        ft.flush
    ft.close()