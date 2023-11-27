import pandas as pd
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



labels_train = []
labels_test = []
features_train = None
features_test = None
catecories = [' matang',' mentah'," bukan"]
with open("fitur\selected\glcm_edge_train_data1.csv","r") as train_ds:
    train_ds_reader = csv.reader(train_ds)
    
    next(train_ds_reader)
    for row in train_ds_reader:
        count = 0
        feature = np.array([])
        for cell in row:
            if (count != 0 ) and (count != len(row)-1):
                feature = np.append(feature,float(cell))
            elif (count == len(row)-1):
                if cell == ' matang':
                    label = 0
                elif cell == ' bukan':
                    label = 2
                else:
                    label = 1
            count += 1
        if features_train is None:
            features_train = feature
        else:
            features_train = np.vstack((features_train, feature))
        labels_train.append(label)
        
with open("fitur\selected\glcm_edge_test_data1.csv","r") as test_ds:
    test_ds_reader = csv.reader(test_ds)
    
    next(test_ds_reader)
    for i,row in enumerate(test_ds_reader):
        count = 0
        feature = np.array([])
        for cell in row:
            if (count != 0 ) and (count != len(row)-1):
                feature = np.append(feature,float(cell))
            elif (count == len(row)-1):
                if cell == ' matang':
                    label = 0
                elif cell == ' bukan':
                    label = 2
                else:
                    label = 1
            count += 1
        if features_test is None:
            features_test = feature
        else:
            try:
                features_test = np.vstack((features_test, feature))
            except ValueError as e:
                print(f"Error at row {i}: {e}")
        labels_test.append(label)
        
# Get shuffled indices

features_train, labels_train = shuffle(features_train, labels_train)



features_train_np = np.array(features_train)
features_test_np = np.array(features_test)
labels_train_np = np.array(labels_train)
labels_test_np = np.array(labels_test)

print(labels_test_np)
print(labels_train_np)
print(type(features_train[0]))
# Membagi data menjadi data latih dan data uji
# X_train, X_test, y_train, y_test = train_test_split(features_train_np, label_train_np, test_size=0.2, random_state=42)


epoch = 40
batch_size = 16
optimizers = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
optimizers_index = 0
layers = [3,64]
# Membuat model ANN
model = Sequential()
model.add(Dense(layers[0], input_dim=features_train_np.shape[1], activation='relu'))
model.add(Dense(3, activation='softmax'))  # Jumlah neuron sesuai dengan jumlah kelas

# Mengompilasi model
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers[2], metrics=['accuracy'])

# Melatih model
model.fit(features_train_np, labels_train_np, epochs=epoch, batch_size=batch_size, validation_data=(features_test_np,labels_test_np))

# Mengevaluasi model
y_pred = np.argmax(model.predict(features_test_np), axis=-1)
cm = confusion_matrix(labels_test_np, y_pred)
accuracy = np.trace(cm) / float(np.sum(cm))
confidence = np.max(y_pred, axis=-1)
print(f'donfidence {confidence}' )
print(f'Accuracy: {accuracy * 100}%')
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title(f"acurasy : {accuracy}")
plt.show()


model_file_name = str(f'model\\refisi\\ann\\model_melon_gray_glcm_ep-{epoch}_btsz-{batch_size}_optz-{optimizers[optimizers_index]}_lyr-')

for layer in layers:
    model_file_name = model_file_name + f"({layer})"

# Simpan model setelah pelatihan
model.save(model_file_name+f"-acrsy{accuracy * 100}.h5")
