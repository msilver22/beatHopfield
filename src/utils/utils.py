import numpy as np
from PIL import Image

def image_to_vector(image_path):
    """
    Takes an image from a given path and returns a flattened vector of the image.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    np.ndarray: Flattened vector of the image.
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    image_array[image_array == 255] = 1
    return image_array