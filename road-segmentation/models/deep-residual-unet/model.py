import tensorflow as tf
import numpy as np
from core.abstract_model import AbstractModel

class Model(AbstractModel):

    def __init__(self, config, dataset, mode):
        self.is_training = mode == 'train'
        super().__init__(config, dataset, mode)


    def build_model(self):
        self.images = self.dataset.img_batch
        self.labels = self.dataset.labels

        # Build Neural Network Architecture
        input = tf.cast(self.images, tf.float32)

        encoding1, encoding2, encoding3 = self.encoder(input=input)
        bridge = self.bridge(input=encoding3)
        decoder = self.decoder(input=bridge, shortcut1=encoding3, shortcut2=encoding2, shortcut3=encoding1)

        output = tf.layers.conv2d(decoder, filters=1, kernel_size=(1, 1), strides=1, activation=tf.sigmoid, name='output')

        # Define predictions, train_op, loss
        self.predictions = output
        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize()


    def encoder(self, input):
        # Encoding 1
        conv1 = tf.layers.conv2d(input, filters=64, kernel_size=(3, 3), strides=1, padding='same', name='encoding1_conv1')
        bn1 = tf.layers.batch_normalization(conv1, training=self.is_training, name='encoding1_bn1')
        relu1 = tf.nn.relu(bn1, name='encoding1_relu1')

        conv2 = tf.layers.conv2d(relu1, filters=64, kernel_size=(3, 3), strides=1, padding='same', name='encoding1_conv2')
        input_adjusted = tf.layers.conv2d(input, filters=64, kernel_size=(1, 1), strides=1 ,name='encoding1_adjust')
        input_adjusted = tf.layers.batch_normalization(input_adjusted, training=self.is_training,name='encoding1_adjust_bn')
        encoding1 = tf.add(conv2, input_adjusted, name='encoding1_add')

        # Encoding 2
        encoding2 = self.res_unit(encoding1, filters=[128,128], strides=[2,1], name='encoding2')

        # Encoding 3
        encoding3 = self.res_unit(encoding2, filters=[256,256], strides=[2,1], name='encoding3')

        return encoding1, encoding2, encoding3

    def bridge(self, input):
        return self.res_unit(input, filters=[512,512], strides=[2,1], name='bridge')


    def decoder(self, input, shortcut1, shortcut2, shortcut3):
        ups1 = tf.keras.layers.UpSampling2D(size=(2,2))(input) # TODO [nku] try transpose conv
        concat1 = tf.concat([shortcut1, ups1], axis=3)
        decoding1 = self.res_unit(concat1, filters=[256,256], strides=[1,1], name='decoding1')

        ups2 = tf.keras.layers.UpSampling2D(size=(2,2))(decoding1) # TODO [nku] try transpose conv
        concat2 = tf.concat([shortcut2, ups2], axis=3)
        decoding2 = self.res_unit(concat2, filters=[128,128], strides=[1,1], name='decoding2')

        ups3 = tf.keras.layers.UpSampling2D(size=(2,2))(decoding2) # TODO [nku] try transpose conv
        concat3 = tf.concat([shortcut3, ups3], axis=3)
        decoding3 = self.res_unit(concat3, filters=[64,64], strides=[1,1], name='decoding3')

        return decoding3

    def res_unit(self, input, filters, strides, name):

        bn1_out = tf.layers.batch_normalization(input, training=self.is_training, name=name+'_bn1')
        relu1_out = tf.nn.relu(bn1_out, name=name+'_relu1')
        conv1_out = tf.layers.conv2d(relu1_out, filters=filters[0], kernel_size=(3, 3), strides=strides[0], padding='same', name=name+'_conv1')

        bn2_out = tf.layers.batch_normalization(conv1_out, training=self.is_training,name=name+'_bn2')
        relu2_out = tf.nn.relu(bn2_out, name=name+'_relu2')
        conv2_out = tf.layers.conv2d(relu2_out, filters=filters[1], kernel_size=(3, 3), strides=strides[1], padding='same',name=name+'_conv2')

        # need to adjust input such that it can be added with conv2_out
        input_adjusted = tf.layers.conv2d(input, filters=filters[1], kernel_size=(1, 1), strides=strides[0]*strides[1] ,name=name+'_adjust')
        input_adjusted = tf.layers.batch_normalization(input_adjusted, training=self.is_training,name=name+'_adjust_bn')

        output = tf.add(conv2_out, input_adjusted, name=name+'_add')

        return output
