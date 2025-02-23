from src.utils.utils import image_to_vector
from src.hopfield.model import HopfieldNetwork
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import pandas as pd

dir_path = os.path.join('dataset/images/')
label_names = ['0','1']


image_vectors = []
for number in label_names:
    image_path = os.path.join(dir_path, number + '.png')
    image_vector = image_to_vector(image_path)
    binary_image_vector = np.where(image_vector == 0, -1, image_vector).flatten()
    if number == '0':
        app = binary_image_vector
    image_vectors.append(binary_image_vector)

patterns = np.vstack(image_vectors)


model = HopfieldNetwork(patterns)


rand = 0.3
test = patterns[1].copy()
corrupted = model.get_corrupted(test,rand)
plt.imshow(corrupted.reshape(28, 28))
plt.title("Corrupted Image")
plt.show()
temperature = 0.1
predicted, m = model.predict(corrupted,temperature)
x = np.arange(0,len(m[0,:])+1)
plt.plot(x,np.insert(m[0,:], 0, 0),label="Mattis magnetization")
plt.xlabel("MC step")
plt.ylabel(r"$m_{\mu}$")
plt.title("Mattis magnetization wrt target pattern")
plt.show()
plt.imshow(predicted.reshape(28, 28), cmap='gray')
plt.title("Predicted Image")
plt.show()