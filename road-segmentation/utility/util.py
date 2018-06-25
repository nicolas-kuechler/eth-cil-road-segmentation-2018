from PIL import Image
import numpy as np

# converts obj to iterable if it is not already
def to_iter(obj):
    try:
        iter(obj)
    except TypeError:
        obj = [obj]
    return obj


# create PIL Image from numpy array with values \in [0,1]
def to_image(img_array):
    img_array = (img_array * 255).round().astype(np.uint8)
    return Image.fromarray(img_array)

def to_array(img):
    img_array = np.asarray(img)
    return img_array / 255.0
