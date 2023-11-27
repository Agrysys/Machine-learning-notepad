import cv2
import matplotlib.pyplot as plt

path_M = "dataset\Test\Matang\Copy of m63.png"
path_TM = "dataset\Test\Mentah\Copy of Tm67.png"
# Membaca gambar
gambar_m = cv2.imread(path_M)
gambar_tm = cv2.imread(path_TM)

red = gambar_m[:,:,0]

# Mengubah gambar ke ruang warna RGB
gambar_rgb = cv2.cvtColor(gambar_m, cv2.COLOR_BGR2RGB)
gambar_rgb_tm = cv2.cvtColor(gambar_tm, cv2.COLOR_BGR2RGB)

# Mengubah gambar ke ruang warna HSV
gambar_hsv_m= cv2.cvtColor(gambar_m, cv2.COLOR_BGR2HSV)
gambar_hsv_tm = cv2.cvtColor(gambar_tm, cv2.COLOR_RGB2HSV)

print(str(gambar_m[:,:,0].flatten()))
# Membuat plot
fig, axs = plt.subplots(2, 4, figsize=(10, 5))

# Menampilkan gambar RGB
axs[0,0].imshow(gambar_rgb)
axs[0,0].set_title('Gambar Matang')
axs[0,0].axis('off')

# Menampilkan gambar HSV
axs[0,1].imshow(gambar_hsv_m[:,:,0],"gray")
axs[0,1].set_title('Gambar Hue')
axs[0,1].axis('off')

axs[0,2].imshow(gambar_hsv_m[:,:,1],"gray")
axs[0,2].set_title('Gambar Saturation')
axs[0,2].axis('off')

axs[0,3].imshow(gambar_hsv_m[:,:,2],"gray")
axs[0,3].set_title('Gambar value')
axs[0,3].axis('off')

# Menampilkan gambar RGB
axs[1,0].imshow(gambar_rgb_tm)
axs[1,0].set_title('Gambar Mentah')
axs[1,0].axis('off')

# Menampilkan gambar HSV
axs[1,1].imshow(gambar_hsv_tm[:,:,0],"gray")
axs[1,1].set_title('Gambar Hue')
axs[1,1].axis('off')

axs[1,2].imshow(gambar_hsv_tm[:,:,1],"gray")
axs[1,2].set_title('Gambar Saturation')
axs[1,2].axis('off')

axs[1,3].imshow(gambar_hsv_tm[:,:,2],"gray")
axs[1,3].set_title('Gambar value')
axs[1,3].axis('off')

axs[1,0].imshow(gambar_rgb_tm)
axs[1,0].set_title('Gambar Mentah')
axs[1,0].axis('off')

# # Menampilkan gambar RGB
# axs[2,1].imshow(gambar_rgb[:,:,0],"gray")
# axs[2,1].set_title('Gambar red')
# axs[2,1].axis('off')

# axs[2,2].imshow(gambar_rgb[:,:,1],"gray")
# axs[2,2].set_title('Gambar green')
# axs[2,2].axis('off')

# axs[2,3].imshow(gambar_rgb[:,:,2],"gray")
# axs[2,3].set_title('Gambar blue')
# axs[2,3].axis('off')

# axs[3,1].imshow(cv2.cvtColor(red,cv2.COLOR_RGB2GRAY),"gray")
# axs[3,1].set_title('Gambar red')
# axs[3,1].axis('off')

# axs[3,2].imshow(gambar_rgb_tm[:,:,1],"gray")
# axs[3,2].set_title('Gambar green')
# axs[3,2].axis('off')

# axs[3,3].imshow(gambar_rgb_tm[:,:,2],"gray")
# axs[3,3].set_title('Gambar blue')
# axs[3,3].axis('off')
# Menampilkan plot
plt.show()
