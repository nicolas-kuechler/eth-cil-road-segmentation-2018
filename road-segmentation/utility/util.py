from PIL import Image
import numpy as np

"""
collection of utility functions
"""

def to_iter(obj):
    """
    converts obj to iterable if it is not already
    """
    try:
        iter(obj)
    except TypeError:
        obj = [obj]
    return obj

def to_image(img_array):
    """
    create PIL Image from numpy array with values \in [0,1]
    """
    img_array = (img_array * 255).round().astype(np.uint8)
    return Image.fromarray(img_array)

def to_array(img):
    """
    create numpy array with values \in [0,1] from PIL Image
    """
    img_array = np.asarray(img)
    return img_array / 255.0
