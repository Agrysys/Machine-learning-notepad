import numpy as np
import csv
from tensorflow.keras.models import load_model


model = load_model('model\gray\model_melon_gray_glcm_ep-30_btsz-32_optz-Adam_lyr-(32)(16)-acrsy91.07142686843872.h5')
labels_train = []
labels_test = []
features_train = None
features_test = None
with open("fitur\edge_gray_train_data3.csv","r") as train_ds:
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
                elif cell == ' mentah':
                    label = 1
                else:
                    label = 2
            count += 1
        if features_train is None:
            features_train = feature
        else:
            features_train = np.vstack((features_train, feature))
        labels_train.append(label)

np_features = np.array(features_train)
np_label = np.array(labels_train)

# for feature in np_features:
#     # print(feature)
#     feature = np.expand_dims(feature, axis=0)  # Expanding dimensions
#     predict = model.predict(feature)
#     print(predict)


# Use the model to make predictions
predictions = model.predict(np_features)

# The output, `predictions`, is an array of probabilities for each class.
# To get the predicted class, you can use `np.argmax` function:
predicted_classes = np.argmax(predictions, axis=1)
label_names = ['matang', 'mentah', 'unknown']
for i in range(len(predicted_classes)):
    print(f"{np_label[i]} {i}: {predicted_classes[i]}")
