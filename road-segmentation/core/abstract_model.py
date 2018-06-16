from abc import ABC, abstractmethod
import tensorflow as tf

class AbstractModel(ABC):
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

        self.predictions = None # output layer of neural network
        self.labels = None # can be groundtruth in case of training or id's in case of test
        self.loss = None
        self.train_op = None
        self.mse = None
        self.init_global_step_counter()
        self.init_epoch_counter()
        self.init_saver()
        self.init_learning_rate()
        self.build_model()
        self.build_mse() # for a fair comparison the validation always uses mse

        # ensure that model defines predictions, loss and train_op
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

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.MAX_CHECKPOINTS_TO_KEEP)

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
