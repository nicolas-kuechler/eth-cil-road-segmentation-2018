import tensorflow as tf
import matplotlib.pyplot as plt
from config import config
from load_data import Dataset
from PIL import Image
import numpy as np
from utility import image_split_and_merge

# utility function to display the dataset
def display(img, gt):
    f = plt.figure(figsize=(5, 10))
    ax = f.add_subplot(3,2,1)
    ax.set_title('Image')
    plt.imshow(img)

    ax = f.add_subplot(3,2,2)
    ax.set_title('Groundtruth')

    plt.imshow(gt.reshape((gt.shape[0],gt.shape[1])))
    plt.show()

# build the dataset
dataset = Dataset(config)

split = 'test' # 'train', 'valid', 'test'
ds = dataset.build_dataset(split) # tensorflow dataset

# ds = ds.batch(config['batch_size']) # could also batch where

value = ds.make_one_shot_iterator().get_next()

with tf.Session() as sess:

    if split == 'test':
        print(split)
        image, index = sess.run(value)
        print('Test Image Shape: ', image.shape)
        plt.imshow(image)
        plt.show()
        print('Index: ', index)

    else:
        x1, y1 = sess.run(value)
        display(x1, y1)
        print('x1: ', x1.shape)
        print('y1: ', y1.shape)

        x2, y2 = sess.run(value)
        display(x2, y2)
        print('x2: ', x2.shape)
        print('y2: ', y2.shape)
