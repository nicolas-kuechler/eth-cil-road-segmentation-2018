

class Evaluation():

    def __init__(self, sess, config, model):
        self.sess = sess
        self.config = config
        self.model = model

    def eval(self):
        self.sess.run(model.dataset.init_op_test) # switch to test dataset

        # TODO [nku] Implement Evaluation of Test Set

        # loop through test dataset and get predictions
        while(True):

            # process batch
            predictions, labels = self.sess.run(model.predictions, self.labels)

            # TODO [nku] add datastructure that keeps all predictions and labels and allows to combine them
            # TODO [nku] add try except to catch when dataset is finished


        # if patched combine to complete image

        # write submission file
