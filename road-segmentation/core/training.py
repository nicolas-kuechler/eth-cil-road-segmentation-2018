import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utility import image_split_and_merge as isam
from PIL import Image

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
            train_init = tf.group(self.model.dataset.init_op_train, self.model.precision_running_vars_initializer, self.model.recall_running_vars_initializer)
            self.sess.run(train_init) # switch to training dataset and reset the accuracy/precision/recall vars

            for i in tqdm(range(self.config.N_BATCHES_PER_EPOCH)):
                step = tf.train.global_step(self.sess, self.model.global_step)

                # train step
                if self.config.LEARNING_RATE_TYPE == 'linear' and step % self.config.LEARNING_RATE_DECAY_STEPS == 0:
                    old_lr = self.sess.run(self.model.lr)
                    self.sess.run(self.model.lr_decay_op)
                    new_lr = self.sess.run(self.model.lr)
                    if old_lr is not new_lr:
                        print('Changed learning rate from {} to {}.'.format(old_lr, new_lr))

                fetches = {
                    'train_op': self.model.train_op,
                    'loss': self.model.loss,
                    'precision_update': self.model.precision_update,
                    'recall_update': self.model.recall_update,
                    'summary': self.model.summary_train
                }

                # write images to summary only ever x'th step
                if step % self.config.SUMMARY_IMAGE_EVERY_STEP == 0:
                    fetches['summary'] =  self.model.summary_train_img

                train_output = self.sess.run(fetches)

                self.summary_writer_train.add_summary(train_output['summary'], global_step=step)

            train_end_summary = self.sess.run(self.model.summary_train_end)
            self.summary_writer_train.add_summary(train_end_summary, global_step=step)


            # Validation
            print('Start Validation...')
            valid_init = tf.group(self.model.dataset.init_op_valid, self.model.precision_running_vars_initializer, self.model.recall_running_vars_initializer)
            self.sess.run(valid_init) # switch to validation dataset and init running vars valid

            if self.config.VALID_METHOD_NAME == 'patch':
                rmse = self.validate_patch(step)
            elif self.config.VALID_METHOD_NAME == 'full':
                rmse = self.validate_full(step)

            valid_end_summary = self.sess.run(self.model.summary_valid_end, {self.model.rmse_valid_pl: rmse})
            self.summary_writer_valid.add_summary(valid_end_summary, global_step=step)
            print('RMSE: ', rmse)
            print('Validation Finished')

            # save checkpoints every x'th epoch
            if (epoch + 1) % self.config.SAVE_CHECKPOINTS_EVERY_EPOCH == 0:
                self.model.save(self.sess)

            self.sess.run(self.model.epoch_increment_op) # increment epoch counter

        self.model.save(self.sess) # save the model after training

    def validate_full(self, step):
        mses = []
        while True:
            try:
                fetches = {
                'mse': self.model.mse,
                'precision_update': self.model.precision_update,
                'recall_update': self.model.recall_update,
                'summary': self.model.summary_valid
                }
                valid_output = self.sess.run(fetches)

                self.summary_writer_valid.add_summary(valid_output['summary'], global_step=step)
                mses.append(valid_output['mse'])

            except tf.errors.OutOfRangeError:
                break # end of dataset

        # Calculate Root Mean Squared Error and Write to Summary
        rmse = np.sqrt(sum(mses) / float(len(mses)))
        return rmse

    def validate_patch(self, step):
        predictions = []
        ids = []
        while True:
            # validation step
            try:
                fetches = {
                'predictions': self.model.predictions,
                'ids': self.model.ids,
                }
                valid_output = self.sess.run(fetches)

                predictions.append(valid_output['predictions'])
                ids.append(valid_output['ids'])

            except tf.errors.OutOfRangeError:
                break # end of dataset

        ids = np.concatenate(ids, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        n_patches_per_image = int(self.config.VALID_N_PATCHES_PER_IMAGE**2)
        n_images = int(predictions.shape[0] / n_patches_per_image)

        # load validation images and gt arrays

        try:
            images = np.load(self.config.VALID_PATH_TO_ARRAYS + 'images.npy')
            gts = np.load(self.config.VALID_PATH_TO_ARRAYS + 'gts.npy')
            arrays_loaded = True
            print('Image and GT array succesfully loaded')
        except IOError:
            arrays_loaded = False
            print('Failed to load Image and/or GT array and hence will create them')
            images = []
            gts = []


        preds = []
        # merge patches back into images
        for i in range(n_images):
            start = i * n_patches_per_image
            end = start + n_patches_per_image

            prediction = isam.merge_into_image_from_flatten(patches_flatten=predictions[start:end, : ,: , :],
                                            index=ids[start:end, :],
                                            image_size=(self.config.VALID_IMAGE_SIZE, self.config.VALID_IMAGE_SIZE, 1),
                                            stride=self.config.VALID_METHOD_STRIDE)
            prediction = prediction.reshape((1, prediction.shape[0], prediction.shape[1], prediction.shape[2]))
            img_id = int(ids[start, 0])

            preds.append(prediction)

            if not arrays_loaded:
                filename = self.config.VALID_IMAGE_NAME_FORMAT.format(img_id)

                # load img
                image = np.asarray(Image.open(self.config.VALID_PATH_TO_DATA + '/' + filename))
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                images.append(image)

                # load gt
                gt = np.asarray(Image.open(self.config.VALID_PATH_TO_GROUNDTRUTH + '/' + filename))
                gt = gt.reshape((1, gt.shape[0], gt.shape[1], 1))
                gt = gt / 255.0
                gt = np.where(gt>self.config.GT_FOREGROUND_THRESHOLD, 1, 0)
                gts.append(gt)

        if not arrays_loaded:
            images = np.concatenate(images, axis=0)
            gts = np.concatenate(gts, axis=0)
            np.save(self.config.VALID_PATH_TO_ARRAYS + 'images', images)
            np.save(self.config.VALID_PATH_TO_ARRAYS + 'gts', gts)
            print('Image and GT array succesfully saved')

        preds = np.concatenate(preds, axis=0)

        fetches = {
            'mse': self.model.mse,
            'precision_update': self.model.precision_update,
            'recall_update': self.model.recall_update,
            'summary': self.model.summary_valid
        }

        feed_dict ={
            self.model.images_pl: images,
            self.model.labels_pl: gts,
            self.model.predictions_pl:preds
        }


        valid_output = self.sess.run(fetches, feed_dict=feed_dict)

        self.summary_writer_valid.add_summary(valid_output['summary'], global_step=step)

        # Calculate Root Mean Squared Error and Write to Summary
        rmse = np.sqrt(valid_output['mse'])

        return rmse
