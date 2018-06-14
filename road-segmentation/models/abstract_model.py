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

    def build_mse(self):
        self.mse = tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions)


    def save(self, sess):
        # TODO [nku] implement save model
        raise NotImplementedError()

    def load(self, sess):
        # TODO [nku] implement save model
        raise NotImplementedError()
