from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

class AbstractModel(ABC):
    def __init__(self, config, dataset, mode):
        self.config = config
        self.dataset = dataset

        assert(mode in ['train', 'test']), "mode must be either train or test"
        self.mode = mode

        self.images = None      # input of neural network
        self.predictions = None # output layer of neural network
        self.labels = None # can be groundtruth in case of training or id's in case of test
        self.loss = None
        self.train_op = None
        self.mse = None
        self.init_global_step_counter()
        self.init_epoch_counter()
        self.init_learning_rate()
        self.build_model()
        self.build_mse() # for a fair comparison the validation always uses mse
        self.build_summaries()
        self.n_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        self.init_saver()

        # ensure that model defines predictions, loss and train_op
        assert(self.images is not None), "images (input) must be defined by model"
        assert(self.predictions is not None), "predictions (output layer) must be defined by model"
        assert(self.labels is not None), "labels must be defined by model"
        assert(self.loss is not None), "loss must be defined by model"
        assert(self.train_op is not None), "train_op must be defined by model"
        assert(self.mse is not None)


    @abstractmethod
    def build_model(self):
        pass

    def init_global_step_counter(self):
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def init_epoch_counter(self):
        with tf.variable_scope('epoch'):
            self.epoch = tf.Variable(0, trainable=False, name='epoch')
            self.epoch_increment_op = tf.assign_add(self.epoch, 1, name='increment_epoch')

    def init_saver(self):
        var_list = tf.trainable_variables()
        var_list.append(self.global_step)
        var_list.append(self.epoch)

        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.config.MAX_CHECKPOINTS_TO_KEEP, save_relative_paths=True)

    def init_learning_rate(self):
        # configure learning rate
        if self.config.LEARNING_RATE_TYPE == 'exponential':
            self.lr = tf.train.exponential_decay(self.config.LEARNING_RATE,
                                            global_step=self.global_step,
                                            decay_steps=self.config.LEARNING_RATE_DECAY_STEPS,
                                            decay_rate=self.config.LEARNING_RATE_DECAY_RATE,
                                            staircase=False)
            self.lr_decay_op = tf.identity(self.lr)
        elif self.config.LEARNING_RATE_TYPE == 'linear':
            self.lr = tf.Variable(self.config.LEARNING_RATE, trainable=False)
            self.lr_decay_op = lr.assign(tf.multiply(self.lr, self.config.LEARNING_RATE_DECAY_RATE))
        elif self.config.LEARNING_RATE_TYPE == 'fixed':
            self.lr = self.config.LEARNING_RATE
            self.lr_decay_op = tf.identity(self.lr)
        else:
            raise ValueError('learning rate type "{}" unknown.'.format(self.config.LEARNING_RATE_TYPE))

    def build_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss, collections=['train'])
            tf.summary.scalar('mean_squarred_error', self.mse, collections=['train', 'valid'])
            tf.summary.scalar('learning_rate', self.lr, collections=['train'])

            self.rmse_valid_pl = tf.placeholder(tf.float32, name='rmse_valid_pl')
            rmse_valid_s = tf.summary.scalar('rmse_valid_s', self.rmse_valid_pl)
            self.summary_valid_rmse = tf.summary.merge([rmse_valid_s])

            tf.summary.image('image', self.images, max_outputs=self.config.SUMMARY_IMAGE_MAX_OUTPUTS, collections=['train', 'valid', 'test'])
            tf.summary.image('prediction', self.predictions, max_outputs=self.config.SUMMARY_IMAGE_MAX_OUTPUTS, collections=['train', 'valid', 'test'])
            tf.summary.image('groundtruth', self.labels, max_outputs=self.config.SUMMARY_IMAGE_MAX_OUTPUTS, collections=['train', 'valid'])

            self.summary_train = tf.summary.merge(tf.get_collection('train'))
            self.summary_valid = tf.summary.merge(tf.get_collection('valid'))
            self.summary_test = tf.summary.merge(tf.get_collection('test'))


    def build_mse(self):
        self.mse = tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions)

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.CHECKPOINT_DIR, self.global_step)
        print("Model saved")

    def load(self, sess):
        checkpoint_id = self.config.CHECKPOINT_ID

        if checkpoint_id is None:
            checkpoint_path =  tf.train.latest_checkpoint(self.config.CHECKPOINT_DIR)
        else:
            checkpoint_path = os.path.join(os.path.abspath(self.config.CHECKPOINT_DIR), 'model-{}'.format(checkpoint_id))

        if checkpoint_path:
            print("Loading model checkpoint {} ...\n".format(checkpoint_path))
            self.saver.restore(sess, checkpoint_path)
            print("Model loaded")
