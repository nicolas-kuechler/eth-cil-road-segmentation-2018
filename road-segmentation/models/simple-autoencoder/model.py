import tensorflow as tf
from core.abstract_model import AbstractModel

# SqueezeNet Architecture https://arxiv.org/pdf/1602.07360.pdf
class Model(AbstractModel):

    def __init__(self, config, dataset, mode):
        super().__init__(config, dataset, mode)

    def build_model(self):
        self.images = self.dataset.img_batch
        self.labels = self.dataset.labels

        # Build Neural Network Architecture (here single convolution layer)
        input = tf.cast(self.images, tf.float32)

        # Squeeze
        with tf.name_scope(name='Squeeze'):
            conv1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=1, padding='same', activation=tf.nn.relu,
                                     name='squeeze_1')



        # Define predictions, train_op, loss
        self.predictions = conv
        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)

        # optimize
        self.optimize()

    def fire_module(self, input, number):
        # n filters in squeeze 1x1 should be smaller than the total of filters in expand 1x1 + expand 3x3 c.f. paper
        assert(self.config.N_FILTERS_SQUEEZE[number] <=
               (self.config.N_FILTERS_EXPAND1[number] + self.config.N_FILTERS_EXPAND3[number]))
        with tf.name_scope(name='fire_module' + str(number)):
            with tf.name_scope(name='squeeze'):
                input = tf.layers.conv2d(inputs=input, filters=self.config.N_FILTERS_SQUEEZE[number],
                                         kernel_size=1, strides=1, activation=tf.nn.relu)
            with tf.name_scope(name='expand'):
                input = tf.layers.conv2d(inputs=input, filters=self.config.N_FILTERS_EXPAND1[number],
                                         kernel_size=1, strides=1)
                input = tf.layers.conv2d(inputs=input, filters=self.config.N_FILTERS_EXPAND3[number],
                                         kernel_size=3, strides=1)
                input = tf.nn.relu(input)
        return input