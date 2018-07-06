import tensorflow as tf
import numpy as np
from core.abstract_model import AbstractModel

class Model(AbstractModel):

    """
    UNet Model Implementation
    """

    def __init__(self, config, dataset, mode):
        self.is_training = mode == 'train'
        super().__init__(config, dataset, mode)


    def build_model(self):
        """
        build the model by putting together the encoder part with the bridge and the decoder
        """
        self.images = self.dataset.img_batch
        self.labels = tf.cast(self.dataset.labels, tf.float32)

        # Build Neural Network Architecture
        input = tf.cast(self.images, tf.float32)

        encoder, skip = self.encoder(input=input)
        self.predictions = self.decoder(input=encoder, skip=skip)

        # Define predictions, train_op, loss
        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize()


    def encoder(self, input):
        """
        build the encoder part of the network
        """
        skip = {}

        with tf.name_scope(name='level_1'):
            path = tf.layers.conv2d(input, filters=64, kernel_size=(3, 3), strides=1, padding='same', name='conv_1.1')
            path = tf.nn.relu(path, name='relu_1.2')

        with tf.name_scope(name='level_2'):
            path = self.bn_conv_relu(path, filters=64, name='bn_conv_relu_2.1')
            skip['2'] = path
            path = self.bn_conv_relu(path, filters=64, name='bn_conv_relu_2.2')
            path = tf.layers.max_pooling2d(path, pool_size=2, strides=2, name='max_pooling_2.3')

        with tf.name_scope(name='level_3'):
            path, skip['3'] = self.encoder_block(path, '3')

        with tf.name_scope(name='level_4'):
            path, skip['4'] = self.encoder_block(path, '4')

        with tf.name_scope(name='level_5'):
            path, skip['5'] = self.encoder_block(path, '5')

        with tf.name_scope(name='level_6'):
            output, skip['6'] = self.encoder_block(path, '6')

        return output, skip

    def decoder(self, input, skip):
        """
        build the decoder part of the network by also connecting the skip connections
        """

        with tf.name_scope(name='bridge'):
            path = self.bn_conv_relu(input, filters=64, name='bn_conv_relu_bridge_1')
            path = self.bn_conv_relu(path, filters=64, name='bn_conv_relu_bridge_2')
            path = self.bn_conv_t_relu(path, filters=64, name='bn_conv_t_relu_bridge_3')

        with tf.name_scope(name='level_6'):
            path = self.decoder_block(path, skip['6'], 'lvl6')

        with tf.name_scope(name='level_5'):
            path = self.decoder_block(path, skip['5'], 'lvl5')

        with tf.name_scope(name='level_4'):
            path = self.decoder_block(path, skip['4'], 'lvl4')

        with tf.name_scope(name='level_3'):
            path = self.decoder_block(path, skip['3'], 'lvl3')

        with tf.name_scope(name='output'):
            path = tf.concat([path, skip['2']], axis=3)
            path = self.bn_conv_relu(path, filters=96, name='bn_conv_relu_out_2')
            path = self.bn_conv_relu(path, filters=64, name='bn_conv_relu_out_3')
            output = tf.layers.conv2d(path, filters=1, kernel_size=(1, 1), strides=1, padding='valid', activation=tf.sigmoid, name='convout')

        return output

    def encoder_block(self, input, id):
        """
        build an encoder block
        """
        path = self.bn_conv_relu(input, filters=64, name='bn_conv_relu_' + id + '.1')
        path = self.bn_conv_relu(path, filters=64, name='bn_conv_relu_' + id + '.2')
        skip = path
        path = self.bn_conv_relu(path, filters=64, name='bn_conv_relu_' + id + '.3')
        output = tf.layers.max_pooling2d(path, pool_size=2, strides=2, name='max_pooling_' + id + '.4')
        return output, skip

    def decoder_block(self, input, skip, id):
        """
        build a decoder block
        """
        path = tf.concat([input, skip], axis=3)
        path = self.bn_conv_relu(path, filters=96, name='bn_conv_relu_' + id + '.2')
        path = self.bn_conv_relu(path, filters=64, name='bn_conv_relu_' + id + '.3')
        output = self.bn_conv_t_relu(path, filters=64, name='bn_conv_t_relu_' + id + '.4')
        return output

    def bn_conv_relu(self, input, filters, name):
        path = tf.layers.batch_normalization(input, training=self.is_training, name=name+'_bn')
        return tf.layers.conv2d(path, filters=filters, kernel_size=(3, 3), strides=1, padding='same', activation=tf.nn.relu, name=name+'_conv')

    def bn_conv_t_relu(self, input, filters, name):
        path = tf.layers.batch_normalization(input, training=self.is_training, name=name+'_bn')
        return tf.layers.conv2d_transpose(path, filters=filters, kernel_size=(3, 3), strides=2, padding='same', activation=tf.nn.relu, name=name+'_conv_t')
