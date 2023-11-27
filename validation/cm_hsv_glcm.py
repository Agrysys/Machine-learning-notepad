import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import sys
sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
import predict.predict_model_hsv_glcm as fungsi

# Data dummy
# Misalkan 1 mewakili melon matang dan 0 mewakili melon mentah
actual = np.array([])
predicted = np.array([])

data_test_dir = "dataset\\Test"
dataset_matang_dir = os.path.join(data_test_dir,'Matang')
dataset_mentah_dir = os.path.join(data_test_dir,'Mentah')

for matang in os.listdir(dataset_matang_dir):
    actual = np.append(actual,"M")

for mentah in os.listdir(dataset_mentah_dir):
    actual = np.append(actual,"TM")

predicsM = fungsi.predict_images_in_folder(dataset_matang_dir)
predicsTM = fungsi.predict_images_in_folder(dataset_mentah_dir)

predicted = np.append(predicted,predicsM)
predicted = np.append(predicted,predicsTM)

print(actual)
print("length : "+str(len(actual)))
print(predicted)
print("length : "+str(len(predicted)))
# Membuat confusion matrix
cm = confusion_matrix(actual, predicted)

# calculate the acuracy
TN, FP, FN, TP = cm.ravel()

accurasy = (TP+TN)/(TP+TN+FP+FN)

print(accurasy)

# Labels untuk confusion matrix
labels = ['Melon Mentah', 'Melon Matang']

# Create confusion matrix plot using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted \n Accuracy: {:.2f}'.format(accurasy))
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.title('hsv dan glcm \n Confusion Matrix')

# Menambahkan akurasi ke dalam plot
# plt.text(0.5, 1.1, , horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

plt.show()