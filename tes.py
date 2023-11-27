import matplotlib.pyplot as plt
import cv2
from skimage.feature import graycomatrix, graycoprops
import statistics


def extract_hsv_features(image):
    # Ubah gambar ke format HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Ambil histogram dari channel Hue (warna)
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    
    # Plot gambar dan histogram
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Tampilkan gambar
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Image')
    axs[0].axis('off')
    
    # Plot histogram
    axs[1].plot(hist_hue)
    axs[1].set_title('Histogram of Hue')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Frequency')
    
    plt.show()
    
    return hist_hue.flatten()

def extract_glcm_feature(image):
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate GLCM
    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Calculate GLCM properties
    cs = graycoprops(glcm, 'contrast')[0, 0]
    hom = graycoprops(glcm, 'homogeneity')[0, 0]
    eng = graycoprops(glcm, 'energy')[0, 0]
    kor = graycoprops(glcm, 'correlation')[0, 0]
    
    # Plot image and GLCM properties
    fig, axs = plt.subplots(2, 2, figsize=(16, 6))
    
    label = []

    # Display image
    axs[0,0].bar(['image1','image'])
    
    
    # Display GLCM properties
    properties = ['Contrast', 'Homogeneity']
    values = [cs, hom]
    
    axs[0,1].bar(properties, values)
    axs[0,1].set_title('GLCM Properties')
    
    plt.show()
    
    return [cs, hom, eng, kor]

image = cv2.imread('dataset\Train\Matang\Copy of m6.png')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
