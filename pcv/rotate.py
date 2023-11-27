from PIL import Image
import os

# specify the directory
directory = 'dataset\Test\Bukan'

# iterate over files in that directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = Image.open(os.path.join(directory, filename))
        for i in range(1, 4):
            # rotate image
            img_rotated = img.rotate(i * 90)
            # save rotated image
            img_rotated.save(os.path.join(directory, f'rotated_{i*90}_{filename}'))
