import os
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist


(train_images, train_labels), (_, _) = mnist.load_data()

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dir_path = os.path.join(base_dir, 'dataset/images/')
output_dir = 'dataset/images'
os.makedirs(output_dir, exist_ok=True)


saved_images_count = {i: 0 for i in range(10)}

for image, label in zip(train_images, train_labels):
    if saved_images_count[label] < 1:
        img = Image.fromarray(image)
        img = img.point(lambda p: p > 128 and 255)
        img.save(os.path.join(output_dir, f'{label}.png'))
        saved_images_count[label] += 1

    if all(count >= 1 for count in saved_images_count.values()):
        break

print("Images saved successfully.")