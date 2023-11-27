import cv2
import os

validation_data_dir = '/content/drive/MyDrive/kelompok E1S5/data/umum/dataset/Test'
labels = ['Bukan','Matang','Mentah']

Validation_labels = []
Validation_data = []

for kelas in os.listdir(validation_data_dir):

    dir_kelas = os.path.join(train_data_dir,kelas)
    for img_name in os.listdir(dir_kelas):
        img_path = os.path.join(dir_kelas,img_name)
        img = cv2.imread(img_path)
        canny = cv2.Canny(img,100,200)
        resized_canny = cv2.resize(canny, (150, 150))
        Validation_data.append(resized_canny)
        if kelas == labels[0]:
          Validation_labels.append(0)
        elif kelas == labels[1]:
          Validation_labels.append(1)
        else:
          Validation_labels.append(2)