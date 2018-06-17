import tensorflow as tf

class Evaluation():

    def __init__(self, sess, config, model):
        self.sess = sess
        self.config = config
        self.model = model

    def eval(self):
        self.sess.run(self.model.dataset.init_op_test) # switch to test dataset
        # TODO [nku] Implement Evaluation of Test Set

        # loop through test dataset and get predictions
        while(True):
            try:
                # process batch
                predictions, labels = self.sess.run([self.model.predictions, self.model.labels])
                print('Lables: ', labels.shape)
                # TODO [nku] add datastructure that keeps all predictions and labels and allows to combine them
            except tf.errors.OutOfRangeError:
                print('End of Test Set')
                break

        # TODO [nku] if patched combine to complete image

        # TODO [nku] write submission file or maybe simply write output masks
