from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

class AbstractModel(ABC):
    """
    Every model must inherit this class which provides the shared functionality

    Overview:
    initializes global step counter, epoch counter, learning rate, gradient clipping,
    functionality to save and load a model, different metrics and tensorboard summaries
    """

    def __init__(self, config, dataset, mode):
        self.config = config
        self.dataset = dataset

        assert(mode in ['train', 'test']), "mode must be either train or test"
        self.mode = mode

        self.images = self.dataset.img_batch   # input of neural network
        self.labels = tf.cast(self.dataset.labels, tf.float32)
        self.ids = self.dataset.id_batch
        self.predictions = None # output layer of neural network
        self.loss = None
        self.train_op = None
        self.mse = None
        self.init_global_step_counter()
        self.init_epoch_counter()
        self.init_learning_rate()
        self.build_model()

        self.build_placeholder()
        self.build_metrics()
        self.build_summaries()

        self.n_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        self.init_saver()

        # ensure that model defines predictions, loss and train_op
        assert(self.predictions is not None), "predictions (output layer) must be defined by model"
        assert(self.loss is not None), "loss must be defined by model"
        assert(self.train_op is not None), "train_op must be defined by model"
        assert(self.mse is not None)

    @abstractmethod
    def build_model(self):
        pass

    def init_global_step_counter(self):
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(1, trainable=False, name='global_step')

    def init_epoch_counter(self):
        with tf.variable_scope('epoch'):
            self.epoch = tf.Variable(0, trainable=False, name='epoch')
            self.epoch_increment_op = tf.assign_add(self.epoch, 1, name='increment_epoch')

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.MAX_CHECKPOINTS_TO_KEEP)

    def init_learning_rate(self):
        """
        configure the learning rate according to the config
        """
        if self.config.LEARNING_RATE_TYPE == 'exponential':
            self.lr = tf.train.exponential_decay(self.config.LEARNING_RATE,
                                            global_step=self.global_step,
                                            decay_steps=self.config.LEARNING_RATE_DECAY_STEPS,
                                            decay_rate=self.config.LEARNING_RATE_DECAY_RATE,
                                            staircase=self.config.LEARNING_RATE_STAIRCASE)
            self.lr_decay_op = tf.identity(self.lr)
        elif self.config.LEARNING_RATE_TYPE == 'linear':
            self.lr = tf.Variable(self.config.LEARNING_RATE, trainable=False)
            self.lr_decay_op = self.lr.assign(tf.multiply(self.lr, self.config.LEARNING_RATE_DECAY_RATE))
        elif self.config.LEARNING_RATE_TYPE == 'fixed':
            self.lr = self.config.LEARNING_RATE
            self.lr_decay_op = tf.identity(self.lr)
        else:
            raise ValueError('learning rate type "{}" unknown.'.format(self.config.LEARNING_RATE_TYPE))


    def build_placeholder(self):
        """
        builds the placeholder that by default use the dataset, allow however
        also to directly feed images, labels and predictions
        (this is necessary for the patch based validation where we want to calculate
        all the metrics also on the merged images)
        """
        self.images_pl = tf.placeholder_with_default(input=self.images, shape=[None, None, None, 3], name='images_pl')
        self.labels_pl = tf.placeholder_with_default(input=self.labels, shape=[None, None, None, 1], name='labels_pl')
        self.predictions_pl = tf.placeholder_with_default(input=self.predictions, shape=[None, None, None, 1] , name='predictions_pl')
        self.rmse_valid_pl = tf.placeholder(tf.float32, name='rmse_valid_pl')

    def build_metrics(self):
        """
        build precision, recall and f1 score metrics at different thresholds
        (at what kind of threshold value is a pixel considered to be a road)
        and also build the mean squared error metric
        """
        # Define the metric and update operations
        self.thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.precision, self.precision_update = tf.metrics.precision_at_thresholds(labels=self.labels_pl, predictions=self.predictions_pl, thresholds=self.thresholds, name='precision')
        self.recall, self.recall_update = tf.metrics.recall_at_thresholds(labels=self.labels_pl, predictions=self.predictions_pl, thresholds=self.thresholds, name='recall')

        self.f1 = tf.div(tf.multiply(self.precision, self.recall), tf.add(self.precision, self.recall))
        self.mse = tf.losses.mean_squared_error(labels=self.labels_pl, predictions=self.predictions_pl)

        # Isolate the variables stored behind the scenes by the metric operation
        precision_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision")
        recall_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall")

        # Define initializer to initialize/reset running variables
        self.precision_running_vars_initializer = tf.variables_initializer(var_list=precision_running_vars)
        self.recall_running_vars_initializer = tf.variables_initializer(var_list=recall_running_vars)


    def build_summaries(self):
        """
        builds the tensorboard summaries for training and validation
        """

        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss, family='01_general', collections=['train_img','train'])
            tf.summary.scalar('mean_squarred_error', self.mse, family='01_general', collections=['train_img','train', 'valid'])
            tf.summary.scalar('learning_rate', self.lr, family='01_general', collections=['train_img','train'])

            tf.summary.scalar('rmse_valid', self.rmse_valid_pl, family='01_general', collections=['valid_end'])

            for i, t in enumerate(self.thresholds):
                tf.summary.scalar('f1_' + str(t), self.f1[i], family='02_f1' ,collections=['valid_end', 'train_end'])
                tf.summary.scalar('precision_' + str(t), self.precision[i], family='03_precision', collections=['valid_end', 'train_end'])
                tf.summary.scalar('recall_' + str(t), self.recall[i], family='04_recall', collections=['valid_end', 'train_end'])

            tf.summary.image('image', self.images_pl,  family='img', max_outputs=self.config.SUMMARY_IMAGE_MAX_OUTPUTS, collections=['train_img'])
            tf.summary.image('prediction', self.predictions_pl, family='pred', max_outputs=self.config.SUMMARY_IMAGE_MAX_OUTPUTS, collections=['train_img'])
            tf.summary.image('groundtruth', self.labels_pl, family='gt', max_outputs=self.config.SUMMARY_IMAGE_MAX_OUTPUTS, collections=['train_img'])

            tf.summary.image('image', self.images_pl, family='img', max_outputs=self.config.SUMMARY_FULL_IMAGE_MAX_OUTPUTS, collections=['valid', 'test'])
            tf.summary.image('prediction', self.predictions_pl, family='pred', max_outputs=self.config.SUMMARY_FULL_IMAGE_MAX_OUTPUTS, collections=['valid', 'test'])
            tf.summary.image('groundtruth', self.labels_pl, family='gt', max_outputs=self.config.SUMMARY_FULL_IMAGE_MAX_OUTPUTS, collections=['valid', 'test'])


            self.summary_train = tf.summary.merge(tf.get_collection('train'))
            self.summary_train_img = tf.summary.merge(tf.get_collection('train_img'))
            self.summary_train_end = tf.summary.merge(tf.get_collection('train_end'))

            self.summary_valid = tf.summary.merge(tf.get_collection('valid'))
            self.summary_valid_end = tf.summary.merge(tf.get_collection('valid_end'))
            self.summary_test = tf.summary.merge(tf.get_collection('test'))


    def optimize(self):
        optimizer = self.config.OPTIMIZER(learning_rate=self.lr)

        if self.config.USE_GRADIENT_CLIPPING:
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        else:
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config.CHECKPOINT_DIR + self.config.MODEL_NAME, self.global_step)
        print("Model saved")

    def load(self, sess):
        checkpoint_id = self.config.CHECKPOINT_ID

        if checkpoint_id is None:
            checkpoint_path =  tf.train.latest_checkpoint(self.config.CHECKPOINT_DIR)
        else:
            checkpoint_path = os.path.join(os.path.abspath(self.config.CHECKPOINT_DIR), 'model-{}'.format(checkpoint_id))

        if checkpoint_path:
            print("Loading model checkpoint {} ...".format(checkpoint_path))
            self.saver.restore(sess, checkpoint_path)
            print("Model loaded")
