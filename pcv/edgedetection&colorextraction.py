import cv2
import matplotlib.pyplot as plt

def extract_color_and_edges(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    
    # Ubah warna dari BGR (OpenCV) ke RGB (Matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Deteksi tepi menggunakan metode Canny
    edges = cv2.Canny(image, 150, 200)

    return image_rgb, edges

# Ganti dengan path foto buah melon yang Anda miliki
melon_image_path_1 = 'dataset\Test\Matang\Copy of m1.png'
melon_image_path_2 = 'dataset\Test\Mentah\Copy of Tm21.png'
bukan_image_path = 'dataset\Train\Bukan\IMG_20231114_145935.jpg'

# Ekstraksi warna dan deteksi tepi dari gambar pertama
image1, edges1 = extract_color_and_edges(melon_image_path_1)

# Ekstraksi warna dan deteksi tepi dari gambar kedua
image2, edges2 = extract_color_and_edges(melon_image_path_2)

image3, edges3 = extract_color_and_edges(bukan_image_path)

# Tampilkan kedua gambar dalam satu plot
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

print(edges1.shape)

print(image1.flatten())

# Plot gambar pertama
axes[0, 0].imshow(image1)
axes[0, 0].set_title('matang')
axes[0, 0].axis('off')

axes[0, 1].imshow(edges1, cmap='gray')
axes[0, 1].set_title('matang edge')
axes[0, 1].axis('off')

# axes[0, 2].imshow(image1.flatten())
# axes[0, 2].set_title('rgb flaten')
# axes[0, 2].axis('off')

# Plot gambar kedua
axes[1, 0].imshow(image2)
axes[1, 0].set_title('metah')
axes[1, 0].axis('off')

axes[1, 1].imshow(edges2, cmap='gray')
axes[1, 1].set_title('mentah edge')
axes[1, 1].axis('off')


# plot gambar 3
axes[2, 0].imshow(image3)
axes[2, 0].set_title('bukan melon')
axes[2, 0].axis('off')

axes[2, 1].imshow(edges3, cmap='gray')
axes[2, 1].set_title('bukan melon edge')
axes[2, 1].axis('off')

plt.tight_layout()
plt.show()
