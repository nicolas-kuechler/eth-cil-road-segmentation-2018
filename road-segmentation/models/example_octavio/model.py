import tensorflow as tf
from core.abstract_model import AbstractModel

class Model(AbstractModel):

    def __init__(self, config, dataset, mode):
        self.is_training = mode == 'train'
        super().__init__(config, dataset, mode)

    def build_model(self):
        self.images = self.dataset.img_batch
        self.labels = tf.cast(self.dataset.labels, tf.float32)

        input = tf.cast(self.images, tf.float32)

        with tf.variable_scope('encoding'):
            conv = tf.layers.conv2d(inputs=input, filters=32, kernel_size=7, padding='same',
                                    activation=tf.nn.leaky_relu)
            conv = tf.layers.conv2d(inputs=conv, filters=64, kernel_size=3, padding='same',
                                    activation=tf.nn.leaky_relu)
            conv = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, padding='same')
            conv = tf.layers.conv2d(inputs=conv, filters=32, kernel_size=3, padding='same',
                                    activation=tf.nn.leaky_relu)
            conv = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, padding='same')

            conv = tf.layers.conv2d(inputs=conv, filters=256, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu)
            conv = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, padding='same')

        with tf.variable_scope('decoding'):
            conv = tf.layers.conv2d(inputs=conv, filters=256, kernel_size=3, padding='same',
                                    activation=tf.nn.leaky_relu)
            conv = tf.layers.conv2d_transpose(conv, filters=128, kernel_size=3, padding='same', strides=2)
            conv = tf.layers.conv2d_transpose(conv, filters=64, kernel_size=3, padding='same', strides=2)
            conv = tf.layers.conv2d_transpose(conv, filters=64, kernel_size=3, padding='same', strides=2)
            conv = tf.layers.conv2d(conv, filters=32, kernel_size=3, strides=1, padding='same')
            conv = tf.layers.conv2d(conv, filters=1, kernel_size=3, strides=1, padding='same')

        output = tf.nn.sigmoid(conv)

        # Define predictions, train_op, loss
        self.predictions = output
        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)

        # optimize
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize()
