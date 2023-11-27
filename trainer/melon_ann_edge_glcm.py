import csv
import numpy as np
from sklearn.utils import shuffle
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

labels_train = []
labels_test = []
features_train = None
features_test = None
catecories = [' matang',' mentah',' bukan']
with open("fitur\\refisi\\glcm_gray_train_data.csv","r") as train_ds:
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
                else:
                    label = 1
            count += 1
        print(feature.shape)
        if features_train is None:
            features_train = feature
        else:
            features_train = np.vstack((features_train, feature))
        labels_train.append(label)
        
with open("fitur\\refisi\\glcm_gray_test_data.csv","r") as test_ds:
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
                else:
                    label = 1
            count += 1
        print(feature.shape)
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
# labels_train = to_categorical(labels_train)
# labels_test_test = to_categorical(labels_test)

# Convert labels to categorical one-hot encoding
y_train_encoded = to_categorical(labels_train, num_classes=2)
y_test_encoded = to_categorical(labels_test, num_classes=2)

model = Sequential()
model.add(Dense(64, input_shape=(features_train_np.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(len(features_train_np))
print(len(y_train_encoded))
print(len(features_test_np))
print(len(y_test_encoded))

# Latih model
model.fit(features_train_np, y_train_encoded, epochs=10, batch_size=32, validation_data=(features_test_np, y_test_encoded))

# Simpan model setelah pelatihan
model.save('models\\revisi\\model_melon_edge_glcm.h5')