import tensorflow as tf
from core.abstract_model import AbstractModel

# SqueezeNet Architecture https://arxiv.org/pdf/1602.07360.pdf
class Model(AbstractModel):

    def __init__(self, config, dataset, mode):
        super().__init__(config, dataset, mode)

    def build_model(self):
        self.check_configs()
        self.images = self.dataset.img_batch
        self.labels = tf.cast(self.dataset.labels, tf.float32)

        # Build Neural Network Architecture (here single convolution layer)
        input = tf.cast(self.images, tf.float32)

        # first conv layer
        input = tf.layers.conv2d(inputs=input, filters=96, kernel_size=7, strides=2,
                                 padding='same', name='conv1')

        # fire modules
        connections = []
        for layer_number in range(self.config.N_fire_modules):
            input, connections = self.fire_module(input, layer_number, connections)

        # inverted fire modules
        for layer_number in range(self.config.N_fire_modules):
            input = self.inverse_fire_module(input, layer_number, connections)

        # final conv layer
        self.predictions = tf.layers.conv2d_transpose(inputs=input, filters=1, kernel_size=7, strides=2,
                                                      padding='same', name='final_conv', activation=tf.sigmoid)

        # Define predictions, train_op, loss
        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)

        # optimize
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize()

    def fire_module(self, input, layer_number, connections):
        if self.config.MAX_POOLS[layer_number]:
            with tf.variable_scope('maxpool' + str(layer_number)):
                input = tf.layers.max_pooling2d(inputs=input, pool_size=3, strides=2, padding='same')

        with tf.variable_scope('fire_module' + str(layer_number)):
            with tf.name_scope(name='squeeze'):
                input = tf.layers.conv2d(inputs=input, filters=self.config.FILTERS_SQUEEZE1[layer_number],
                                         kernel_size=1, strides=1, activation=tf.nn.leaky_relu, padding='same')
                input = tf.layers.batch_normalization(input, training=self.is_training)
            with tf.name_scope(name='expand'):
                input = tf.layers.conv2d(inputs=input, filters=self.config.FILTERS_EXPAND1[layer_number],
                                         kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu)
                input = tf.layers.conv2d(inputs=input, filters=self.config.FILTERS_EXPAND3[layer_number],
                                         kernel_size=3, strides=1, padding='same', activation=tf.nn.leaky_relu)
                input = tf.layers.batch_normalization(input, training=self.is_training)
                connections.append(input)

        if self.config.DROPOUTS[layer_number]:
            with tf.name_scope(name='dropout' + str(layer_number)):
                input = tf.nn.dropout(input, keep_prob=0.5)

        return input, connections

    def inverse_fire_module(self, input, layer_number, connections):
        with tf.variable_scope('inverse_fire_module' + str(layer_number)):
            with tf.name_scope(name='expand'):
                input = tf.layers.conv2d(inputs=input, filters=self.config.FILTERS_EXPAND3[-layer_number-1],
                                         kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
                input = connections[-layer_number-1] + input
                input = tf.layers.conv2d(inputs=input, filters=self.config.FILTERS_EXPAND1[-layer_number-1],
                                         kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
            with tf.name_scope(name='squeeze'):
                input = tf.layers.conv2d(inputs=input, filters=self.config.FILTERS_SQUEEZE1[-layer_number-1],
                                         kernel_size=1, strides=1, activation=tf.nn.relu, padding='same')

        if self.config.MAX_POOLS[-layer_number-1]:
            with tf.variable_scope('upsampling' + str(layer_number)):
                input = tf.layers.conv2d_transpose(inputs=input, filters=self.config.FILTERS_SQUEEZE1[-layer_number-1],
                                                   kernel_size=3, strides=2, padding='same')
        return input

    def check_configs(self):
        assert (self.config.N_fire_modules == len(self.config.FILTERS_SQUEEZE1))
        assert (self.config.N_fire_modules == len(self.config.FILTERS_EXPAND1))
        assert (self.config.N_fire_modules == len(self.config.FILTERS_EXPAND3))

        # n filters in squeeze 1x1 should be smaller than the total of filters in expand 1x1 + expand 3x3 c.f. paper
        for layer_number in range(self.config.N_fire_modules):
            assert (self.config.FILTERS_SQUEEZE1[layer_number] <=
                    (self.config.FILTERS_EXPAND1[layer_number] + self.config.FILTERS_EXPAND3[layer_number]))
