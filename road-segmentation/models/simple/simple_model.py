import tensorflow as tf
from models.abstract_model import AbstractModel

class SimpleModel(AbstractModel):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def build_model(self):
        img_batch = self.dataset.img_batch
        self.labels = self.dataset.labels

        # Build Neural Network Architecture (here single convolution layer)
        img_batch = tf.cast(img_batch, tf.float32)
        conv = tf.layers.conv2d(inputs=img_batch, filters=1, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)

        # Define predictions, train_op, loss
        self.predictions = conv
        self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
