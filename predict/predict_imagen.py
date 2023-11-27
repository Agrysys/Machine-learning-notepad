from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

# Dimensi gambar
img_width, img_height = 150, 150

# Memuat model yang telah dilatih
model = load_model('model\inn\modelmodel-edge_ann.h5')


img_path = 'dataset\\tracehold\Test\Bukan\processed_IMG_20231114_145714.jpg'
img_path = 'dataset\\tracehold\Test\Matang\processed_Copy of m1.png'
img_path = 'dataset\\tracehold\Test\Mentah\processed_Copy of Tm24.png'

img_path = 'dataset\Test\Matang\Copy of m30.png'
img_path = 'dataset\Test\Bukan\IMG_20231114_150800.jpg'
# img_path = 'dataset\Test\Mentah\Copy of Tm25.png'
# Mengubah gambar menjadi array
img_cv = cv2.imread(img_path)
edge = cv2.Canny(img_cv,100,200)
edge_rgb = cv2.cvtColor(edge,cv2.COLOR_GRAY2RGB)
edge_rgb_sized = cv2.resize(edge_rgb,(150,150))
# img = image.load_img(img_path, target_size=(img_width, img_height))
img = image.img_to_array(edge_rgb_sized)
img = np.expand_dims(img, axis=0)

# Memprediksi kelas gambar
classes = model.predict(img)
predicted_category_index = np.argmax(classes)
categories = ['Bukan','Matang','Mentah']
predicted_category = categories[predicted_category_index]

print(predicted_category)
