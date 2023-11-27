import sys
import os
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("D:\\kuliah\\semester 5\\tugas akhir\\Machine Learning"))
from utility.feature_extractor import extract_glcm_feature,extract_hsv_features

image_matang_path = 'dataset\Train\Matang\Copy of m6.png'
image_mentah_path = 'dataset\Train\Mentah\Copy of Tm1.png'

image_matang = cv2.imread(image_matang_path)
image_mentah = cv2.imread(image_mentah_path)

glcm_matang = extract_glcm_feature(image_matang)
glcm_mentah = extract_glcm_feature(image_mentah)

print(glcm_matang)
print(glcm_mentah)

 # Plot image and GLCM properties
fig, axs = plt.subplots(3, 2, figsize=(16, 6))
fig.tight_layout()

    
label = ['citra melon matang', 'citra melon mentah']

axs[0,0].imshow(image_matang)
axs[0,0].set_title('citra matang')
axs[0,1].imshow(image_mentah)
axs[0,1].set_title('citra mentah')

    # Display image
axs[1,0].bar(label,[glcm_matang[0],glcm_mentah[0]])
axs[1,0].set_title('contras')

axs[1,1].bar(label,[glcm_matang[1],glcm_mentah[1]])
axs[1,1].set_title('homogenity')

axs[2,0].bar(label,[glcm_matang[2],glcm_mentah[2]])
axs[2,0].set_title('energy')

axs[2,1].bar(label,[glcm_matang[3],glcm_mentah[3]])
axs[2,1].set_title('correlation')
    
fig.subplots_adjust(hspace=0.5)  # Adjust as needed
    
plt.show()