from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
import predict.predict_model_hsv_glcm as fungsi

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
# Konversi ke nilai biner
actual_binary = np.where(actual == 'M', 1, 0)
predicted_binary = np.where(predicted == 'M', 1, 0)

# Menghitung nilai ROC
fpr, tpr, thresholds = roc_curve(actual_binary, predicted_binary)
roc_auc = auc(fpr, tpr)

# Membuat plot
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
