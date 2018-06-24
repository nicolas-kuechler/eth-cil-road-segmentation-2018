import tensorflow as tf
from core.abstract_model import AbstractModel

class Model(AbstractModel):

    def __init__(self, config, dataset, mode):
        self.is_training = mode == 'train'
        super().__init__(config, dataset, mode)

    def build_model(self):
        self.images = self.dataset.img_batch
        self.labels = tf.cast(self.dataset.labels, tf.float32)

        # Build Neural Network Architecture (here single convolution layer)
        input = tf.cast(self.images, tf.float32)
        conv = tf.layers.conv2d(inputs=input, filters=1, kernel_size=[3, 3], padding='same')
        conv_bn = tf.layers.batch_normalization(conv, training=self.is_training)
        output = tf.nn.sigmoid(conv_bn)

        # Define predictions, train_op, loss
        self.predictions = output
        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)

        # optimize
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimize()
