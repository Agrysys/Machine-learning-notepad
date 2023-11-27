from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import cv2
import numpy as np
import os


# Ukuran gambar
img_width, img_height = 150, 150

# train_data_dir = 'dataset\\tracehold\\Train'
# validation_data_dir = 'dataset\\tracehold\\Test'
train_data_dir = 'dataset\\Train'
validation_data_dir = 'dataset\\Test'
nb_train_samples = 251
nb_validation_samples = 72
epochs = 60
batch_size = 16

Train_labels = np.array([])
Train_data = np.array([])

for kelas in os.listdir(train_data_dir):
    Train_labels = np.append(Train_labels,kelas)
    dir_kelas = os.path.join(train_data_dir,kelas)
    for img_name in os.listdir(dir_kelas):
        img_path = os.path.join(dir_kelas,img_name)
        print(img_path)
        img = cv2.imread(img_path)
        canny = cv2.Canny(img,100,200)
        Train_data = np.append(Train_data,canny)
        


