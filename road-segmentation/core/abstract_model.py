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
            self.global_step = tf.Variable(1, trainable=False, name='global_step')

    def init_epoch_counter(self):
        with tf.variable_scope('epoch'):
            self.epoch = tf.Variable(0, trainable=False, name='epoch')
            self.epoch_increment_op = tf.assign_add(self.epoch, 1, name='increment_epoch')

    def init_saver(self):
        var_list = tf.trainable_variables()
        var_list.append(self.global_step)
        var_list.append(self.epoch)

        # TODO [nku] saving and loading of learning rate that makes sense for all types
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
            self.lr_decay_op = self.lr.assign(tf.multiply(self.lr, self.config.LEARNING_RATE_DECAY_RATE))
        elif self.config.LEARNING_RATE_TYPE == 'fixed':
            self.lr = self.config.LEARNING_RATE
            self.lr_decay_op = tf.identity(self.lr)
        else:
            raise ValueError('learning rate type "{}" unknown.'.format(self.config.LEARNING_RATE_TYPE))

    def build_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss, family='01_general', collections=['train_img','train'])
            tf.summary.scalar('mean_squarred_error', self.mse, family='01_general', collections=['train_img','train', 'valid'])
            tf.summary.scalar('learning_rate', self.lr, family='01_general', collections=['train_img','train'])

            self.rmse_valid_pl = tf.placeholder(tf.float32, name='rmse_valid_pl')
            tf.summary.scalar('rmse_valid', self.rmse_valid_pl, family='01_general', collections=['valid_end'])

            # Define the metric and update operations

            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            precision, self.precision_update = tf.metrics.precision_at_thresholds(labels=self.labels, predictions=self.predictions, thresholds=thresholds, name='precision')
            recall, self.recall_update = tf.metrics.recall_at_thresholds(labels=self.labels, predictions=self.predictions, thresholds=thresholds, name='recall')

            f1 = tf.div(tf.multiply(precision, recall), tf.add(precision, recall))
            for i, t in enumerate(thresholds):
                tf.summary.scalar('f1_' + str(t), f1[i], family='02_f1' ,collections=['valid_end', 'train_end'])
                tf.summary.scalar('precision_' + str(t), precision[i], family='03_precision', collections=['valid_end', 'train_end'])
                tf.summary.scalar('recall_' + str(t), recall[i], family='04_recall', collections=['valid_end', 'train_end'])



            # Isolate the variables stored behind the scenes by the metric operation
            # TODO [nku] decide if want to implement accuracy later
            # valid_accuracy, valid_accuracy_update = tf.metrics.accuracy(self.labels, self.predictions_binary, name="valid_accuracy", updates_collections=['valid'], metrics_collections=['valid_end'])
            # train_accuracy, train_accuracy_update = tf.metrics.accuracy(self.labels, self.predictions_binary, name="train_accuracy", updates_collections=['train_img','train'], metrics_collections=['train_end'])
            # valid_accuracy_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="valid_accuracy")
            # train_accuracy_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train_accuracy")
            # -> would have to add them to valid_running_vars and train_running_vars

            precision_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision")
            recall_running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall")

            # Define initializer to initialize/reset running variables
            self.precision_running_vars_initializer = tf.variables_initializer(var_list=precision_running_vars)
            self.recall_running_vars_initializer = tf.variables_initializer(var_list=recall_running_vars)

            tf.summary.image('image', self.images,  family='img', max_outputs=self.config.SUMMARY_IMAGE_MAX_OUTPUTS, collections=['train_img'])
            tf.summary.image('prediction', self.predictions, family='pred', max_outputs=self.config.SUMMARY_IMAGE_MAX_OUTPUTS, collections=['train_img'])
            tf.summary.image('groundtruth', self.labels, family='gt', max_outputs=self.config.SUMMARY_IMAGE_MAX_OUTPUTS, collections=['train_img'])

            tf.summary.image('image', self.images, family='img', max_outputs=self.config.SUMMARY_FULL_IMAGE_MAX_OUTPUTS, collections=['valid', 'test'])
            tf.summary.image('prediction', self.predictions, family='pred', max_outputs=self.config.SUMMARY_FULL_IMAGE_MAX_OUTPUTS, collections=['valid', 'test'])
            tf.summary.image('groundtruth', self.labels, family='gt', max_outputs=self.config.SUMMARY_FULL_IMAGE_MAX_OUTPUTS, collections=['valid', 'test'])


            self.summary_train = tf.summary.merge(tf.get_collection('train'))
            self.summary_train_img = tf.summary.merge(tf.get_collection('train_img'))
            self.summary_train_end = tf.summary.merge(tf.get_collection('train_end'))

            self.summary_valid = tf.summary.merge(tf.get_collection('valid'))
            self.summary_valid_end = tf.summary.merge(tf.get_collection('valid_end'))
            self.summary_test = tf.summary.merge(tf.get_collection('test'))


    def build_mse(self):
        self.mse = tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions)

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
