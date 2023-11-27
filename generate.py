from tensorflow.keras.utils import plot_model

# Load the model
from tensorflow.keras.models import load_model
model = load_model('models/model_melon_hsv_glcm_laporan.h5')

# Plot the model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
