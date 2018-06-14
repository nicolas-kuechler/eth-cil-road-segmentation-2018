import tensorflow as tf
import numpy as np

class Training():

    def __init__(self, sess, training_config, model):
        self.sess = sess
        self.config = training_config
        self.model = model

    def train(self):

        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.config['n_epochs']):
            print('EPOCH: ', epoch)

            # Training
            print('Start Training')
            self.sess.run(self.model.dataset.init_op_train) # switch to training dataset
            for i in range(self.config['n_batches_per_epoch']):
                # train step
                loss, _, mse = self.sess.run([self.model.loss, self.model.train_op, self.model.mse])
                print('Loss: ', loss)
                # TODO [nku] write to summary
            print('End of Training Set')


            # Validation
            print('Start Validation')
            self.sess.run(self.model.dataset.init_op_valid) # switch to validation dataset
            mse_sum = 0
            count = 0
            while True:
                # validation step
                try:
                    # TODO [nku] consider that here the patch based approach
                    # has a disadvantage because for an image he has "more border"
                    # with the test set this disadvantage is countered by
                    # overlapping and averaged patches -> maybe can do this here aswell?
                    mse_sum += self.sess.run(self.model.mse)
                    count += 1
                except tf.errors.OutOfRangeError:
                    print('End of Validation Set')
                    break

            # Calculate Root Mean Squared Error
            rmse = np.sqrt(mse_sum / float(count))
            print('RMSE: ', rmse)

            # TODO [nku] write to summary
