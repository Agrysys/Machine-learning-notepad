from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = load_model('model\inn\epo-60_modelmodel-edge_ann.h5')

# Fungsi untuk melakukan prediksi
def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_batch/255.
    prediction = model.predict(img_preprocessed)
    return np.argmax(prediction)

# Fungsi untuk mendapatkan label sebenarnya dari path file
def get_true_label(img_path):
    label = os.path.basename(os.path.dirname(img_path))
    # Mengubah label string menjadi numerik
    if label == 'Matang':
        return 0
    elif label == 'Mentah':
        return 1
    elif label == 'Bukan':
        return 2

# List untuk menyimpan prediksi dan label sebenarnya
predictions = []
true_labels = []

# Melakukan prediksi pada setiap file dalam folder
folders = ['dataset/Test/Bukan', 'dataset/Test/Matang', 'dataset/Test/Mentah']

for folder in folders:
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            predictions.append(predict_image(model, img_path))
            true_labels.append(get_true_label(img_path))

# Menghitung confusion matrix
cm = confusion_matrix(true_labels, predictions)

# Menghitung akurasi
accuracy = np.trace(cm) / np.sum(cm)
print('Akurasi:', accuracy)

# Menampilkan confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title(f"akurasi {accuracy * 100}")
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
