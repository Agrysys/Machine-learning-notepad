import cv2
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix,graycoprops
import numpy as np

def convert_image_2_edge(image):
    edges = cv2.Canny(image, 100, 200)
    
    return  edges

def extract_glcm_features(edge_image):
    # Compute the GLCM of the image
    glcm = graycomatrix(edge_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    
    # Compute GLCM properties
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    
    # Return the features as a dictionary
    features = {'contrast': np.mean(contrast),
                'dissimilarity': np.mean(dissimilarity),
                'homogeneity': np.mean(homogeneity),
                'energy': np.mean(energy),
                'correlation': np.mean(correlation)}
    
    return features

path_M = "dataset\Test\Matang\Copy of m1.png"
path_TM = "dataset\Test\Mentah\Copy of Tm21.png"
path_BM = "dataset\Train\Bukan\IMG_20231114_145935.jpg"

img_m = cv2.imread(path_M)
img_tm = cv2.imread(path_TM)
img_bm = cv2.imread(path_BM)

edge_m = convert_image_2_edge(img_m)
edge_tm = convert_image_2_edge(img_tm)
edge_bm = convert_image_2_edge(img_bm)

glcm_m = extract_glcm_features(edge_m)

print(glcm_m)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

axes[0, 0].imshow(img_m)
axes[0, 0].set_title('matang_raw')
axes[0, 0].axis('off')

axes[0, 1].imshow(edge_m,'gray')
axes[0, 1].set_title('matang_edge')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(img_m,cv2.COLOR_BGR2GRAY),'gray')
axes[0, 2].set_title('matang_gray')
axes[0, 2].axis('off')

axes[1, 0].imshow(img_tm)
axes[1, 0].set_title('mentah')
axes[1, 0].axis('off')

axes[1, 1].imshow(edge_tm,'gray')
axes[1, 1].set_title('mentah_edge')
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(img_tm,cv2.COLOR_BGR2GRAY),'gray')
axes[1, 2].set_title('mentah_gray')
axes[1, 2].axis('off')

axes[2, 0].imshow(img_bm,)
axes[2, 0].set_title('bukan melon')
axes[2, 0].axis('off')

axes[2, 1].imshow(edge_bm,'gray')
axes[2, 1].set_title('bukan melon edge')
axes[2, 1].axis('off')

axes[2, 2].imshow(cv2.cvtColor(img_bm,cv2.COLOR_BGR2GRAY),'gray')
axes[2, 2].set_title('bukan melon gray')
axes[2, 2].axis('off')

plt.show()