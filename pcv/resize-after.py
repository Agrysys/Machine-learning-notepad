import cv2
import matplotlib.pyplot as plt

path_image_mentah = "dataset\\tracehold\Train\Bukan\processed_IMG_20231114_150121.jpg"
path_image_matang = "dataset\\tracehold\Train\Matang\processed_Copy of m7.png"
path_image_bukan = "dataset\\tracehold\Train\Mentah\processed_Copy of Tm5.png"

# Membaca citra
image_mentah = cv2.imread(path_image_mentah)
image_matang = cv2.imread(path_image_matang)
image_bukan = cv2.imread(path_image_bukan)

# Mengubah ukuran citra
image_mentah_rs = cv2.resize(image_mentah, (150, 150))
image_matang_rs = cv2.resize(image_matang, (150, 150))
image_bukan_rs = cv2.resize(image_bukan, (150, 150))


# Menampilkan citra asli, citra yang telah diubah ukurannya, dan citra hasil deteksi tepi Canny
plt.figure(figsize=(15, 15))

plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(image_mentah, cv2.COLOR_BGR2RGB))
plt.title('Citra Mentah Asli')

plt.subplot(3, 3, 2)
plt.imshow(cv2.cvtColor(image_mentah_rs, cv2.COLOR_BGR2RGB))
plt.title('Citra Mentah Resize')

plt.subplot(3, 3, 4)
plt.imshow(cv2.cvtColor(image_matang, cv2.COLOR_BGR2RGB))
plt.title('Citra Matang Asli')

plt.subplot(3, 3, 5)
plt.imshow(cv2.cvtColor(image_matang_rs, cv2.COLOR_BGR2RGB))
plt.title('Citra Matang Resize')

plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(image_bukan, cv2.COLOR_BGR2RGB))
plt.title('Citra Bukan Asli')

plt.subplot(3, 3, 8)
plt.imshow(cv2.cvtColor(image_bukan_rs, cv2.COLOR_BGR2RGB))
plt.title('Citra Bukan Resize')


plt.tight_layout()
plt.show()