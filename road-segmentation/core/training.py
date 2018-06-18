import tensorflow as tf
import numpy as np
from tqdm import tqdm

class Training():

    def __init__(self, sess, config, model):
        self.sess = sess
        self.config = config
        self.model = model

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)

        self.model.load(self.sess) # load the model if it exists

        self.summary_writer_train = tf.summary.FileWriter(self.config.SUMMARY_TRAIN_DIR, self.sess.graph)
        self.summary_writer_valid = tf.summary.FileWriter(self.config.SUMMARY_VALID_DIR, self.sess.graph)

    def train(self):

        for epoch in range(self.sess.run(self.model.epoch), self.config.N_EPOCHS):
            print('Start Epoch: ', epoch)

            # Training
            self.sess.run(self.model.dataset.init_op_train) # switch to training dataset

            for i in tqdm(range(self.config.N_BATCHES_PER_EPOCH)):
                step = tf.train.global_step(self.sess, self.model.global_step)

                # train step
                fetches = {
                    'train_op': self.model.train_op,
                    'loss': self.model.loss,
                    'summary': self.model.summary_train
                }
                train_output = self.sess.run(fetches)

                self.summary_writer_train.add_summary(train_output['summary'], global_step=step)


            # Validation
            print('Start Validation...')
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

                    fetches = {
                        'mse': self.model.mse,
                        #'summary': self.model.summary_valid
                    }
                    valid_output = self.sess.run(fetches)

                    self.summary_writer_valid.add_summary(valid_output['summary'], global_step=step)

                    mse_sum += valid_output['mse']
                    count += 1
                except tf.errors.OutOfRangeError:
                    break # end of dataset

            # Calculate Root Mean Squared Error and Write to Summary
            rmse = np.sqrt(mse_sum / float(count))
            rmse_valid = self.sess.run(self.model.summary_valid_rmse, {self.model.rmse_valid_pl: rmse})
            self.summary_writer_valid.add_summary(rmse_valid, global_step=step)

            print('RMSE: ', rmse)
            print('Validation Finished')

            # save checkpoints every x'th epoch
            if (epoch + 1) % self.config.SAVE_CHECKPOINTS_EVERY_EPOCH == 0:
                self.model.save(self.sess)

            self.sess.run(self.model.epoch_increment_op) # increment epoch counter

        self.model.save(self.sess) # save the model after training
