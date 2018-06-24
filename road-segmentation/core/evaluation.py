import tensorflow as tf
import numpy as np
from PIL import Image
from utility import image_split_and_merge as isam
import os

class Evaluation():

    def __init__(self, sess, config, model):
        self.sess = sess
        self.config = config
        self.model = model

        self.model.load(self.sess) # load the model

    def eval(self):
        print('\nStarting Evaluation...')
        self.sess.run(self.model.dataset.init_op_test) # switch to test dataset

        predictions = []
        ids =  []

        count = 0

        # loop through test dataset and get predictions
        while(True):
            try:
                print('Iter: ', count)
                count += 1
                fetches = {
                    'predictions': self.model.predictions,
                    'ids': self.model.ids
                }

                output = self.sess.run(fetches)

                predictions.append(output['predictions'])
                ids.append(output['ids'])

            except tf.errors.OutOfRangeError:
                break

        ids = np.concatenate(ids, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        # We start with a new submission file, delete previous one if it exists
        submission_file = self.config.TEST_OUTPUT_DIR + 'submission.csv'
        try:
            os.remove(submission_file)
        except OSError:
            pass
        with open(submission_file, 'a') as f:
            f.write('id,prediction\n')


        if self.config.TEST_METHOD_NAME == 'patch':
            # if patch based -> put images back together

            n_patches_per_image = int(self.config.TEST_N_PATCHES_PER_IMAGE**2)
            n_images = int(predictions.shape[0] / n_patches_per_image)

            for i in range(n_images):
                start = i * n_patches_per_image
                end = start + n_patches_per_image

                img = isam.merge_into_image_from_flatten(patches_flatten=predictions[start:end, : ,: , :],
                                                index=ids[start:end, :],
                                                image_size=(self.config.TEST_IMAGE_SIZE, self.config.TEST_IMAGE_SIZE, 1),
                                                stride=self.config.TEST_METHOD_STRIDE)
                img_id = int(ids[start, 0])
                self.append_prediction(submission_file, img[:, :, 0], img_id)


                # scale back
                img = img * 255
                img = Image.fromarray(img[:,:,0].astype('uint8'))
                img.save(self.config.TEST_OUTPUT_DIR + 'out{}.png'.format(img_id))

        elif self.config.TEST_METHOD_NAME == 'full':
            n_images = predictions.shape[0]

            for i in range(n_images):
                img = predictions[i, :, :, 0]
                img_id = int(labels[i])
                self.append_prediction(submission_file, img, img_id)
                img = img * 255
                img = Image.fromarray(img.astype('uint8'))
                img_id = int(ids[i])
                img.save(self.config.TEST_OUTPUT_DIR + 'out{}.png'.format(img_id))
        else:
            raise ValueError('Unknown Test Method Name')

        print('Evaluation Finish   ed: saved {} masks to {}'.format(n_images, self.config.TEST_OUTPUT_DIR))

    def append_prediction(self, submission_file, prediction, id):
        with open(submission_file, 'a') as f:
            f.writelines('{}\n'.format(s) for s in self.prediction_to_submission(prediction, id))

    def prediction_to_submission(self, prediction, id):
        patch_size = 16
        for j in range(0, prediction.shape[1], patch_size):
            for i in range(0, prediction.shape[0], patch_size):
                patch = prediction[i:i + patch_size, j:j + patch_size]
                label = self.patch_to_label(patch)
                yield ("{:03d}_{}_{},{}".format(id, j, i, label))


    # assign a label to a patch
    def patch_to_label(self, patch):
        foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
        df = np.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0
