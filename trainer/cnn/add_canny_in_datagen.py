from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import cv2
import numpy as np


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

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
print(input_shape)

# Fungsi untuk mengubah gambar menjadi edge canny
def edge_canny(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 100, 200)
    return edges

# Fungsi untuk mengaplikasikan edge canny ke setiap gambar dalam generator
def preprocess_input(img):
    np.multiply(img, 255, out=img, casting='unsafe')
    img = img.astype(np.uint8)
    img = edge_canny(img)
    img = np.expand_dims(img, axis=2)
    return img

# Augmentasi data
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255,preprocessing_function=preprocess_input)



train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

print(f"jumlah data : {len(train_generator)}")


# Membuat model
model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3)) # Jumlah kategori
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# Melatih model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# Menyimpan model
model.save('model/inn/epo-60_citra-edge-prep_modelmodel-edge_ann.h5')
