import tensorflow as tf
import matplotlib.pyplot as plt
from config import config
from load_data import Dataset

dataset = Dataset(config)

ds = dataset.build_dataset()

value = ds.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    x1, y1 = sess.run(value)
    plt.imshow(x1[0,:,:,:])
    plt.show()

    print('x1: ', x1.shape)
    print('y1: ', y1.shape)

    x2, y2 = sess.run(value)
    plt.imshow(x2[0,:,:,:])
    plt.show()

    print('x2: ', x2.shape)
    print('y2: ', y2.shape)
