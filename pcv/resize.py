import cv2
import matplotlib.pyplot as plt

path_image_mentah = "dataset\Train\Mentah\Copy of Tm1.png"
path_image_matang = "dataset\Train\Matang\Copy of m6.png"
path_image_bukan = "dataset\Train\Bukan\IMG_20231114_145935.jpg"

# Membaca citra
image_mentah = cv2.imread(path_image_mentah)
image_matang = cv2.imread(path_image_matang)
image_bukan = cv2.imread(path_image_bukan)

# Mengubah ukuran citra
image_mentah_rs = cv2.resize(image_mentah, (150, 150))
image_matang_rs = cv2.resize(image_matang, (150, 150))
image_bukan_rs = cv2.resize(image_bukan, (150, 150))

# Mendeteksi tepi Canny
image_mentah_canny = cv2.Canny(image_mentah_rs, 100, 200)
image_matang_canny = cv2.Canny(image_matang_rs, 100, 200)
image_bukan_canny = cv2.Canny(image_bukan_rs, 100, 200)

# Menampilkan citra asli, citra yang telah diubah ukurannya, dan citra hasil deteksi tepi Canny
plt.figure(figsize=(15, 15))

plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(image_mentah, cv2.COLOR_BGR2RGB))
plt.title('Citra Mentah Asli')

plt.subplot(3, 3, 2)
plt.imshow(cv2.cvtColor(image_mentah_rs, cv2.COLOR_BGR2RGB))
plt.title('Citra Mentah Resize')

plt.subplot(3, 3, 3)
plt.imshow(image_mentah_canny, cmap='gray')
plt.title('Citra Mentah Canny')

plt.subplot(3, 3, 4)
plt.imshow(cv2.cvtColor(image_matang, cv2.COLOR_BGR2RGB))
plt.title('Citra Matang Asli')

plt.subplot(3, 3, 5)
plt.imshow(cv2.cvtColor(image_matang_rs, cv2.COLOR_BGR2RGB))
plt.title('Citra Matang Resize')

plt.subplot(3, 3, 6)
plt.imshow(image_matang_canny, cmap='gray')
plt.title('Citra Matang Canny')

plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(image_bukan, cv2.COLOR_BGR2RGB))
plt.title('Citra Bukan Asli')

plt.subplot(3, 3, 8)
plt.imshow(cv2.cvtColor(image_bukan_rs, cv2.COLOR_BGR2RGB))
plt.title('Citra Bukan Resize')

plt.subplot(3, 3, 9)
plt.imshow(image_bukan_canny, cmap='gray')
plt.title('Citra Bukan Canny')

plt.tight_layout()
plt.show()