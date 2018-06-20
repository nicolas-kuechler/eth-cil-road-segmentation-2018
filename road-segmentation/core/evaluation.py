import tensorflow as tf
import numpy as np
from PIL import Image
from utility import image_split_and_merge as isam

class Evaluation():

    def __init__(self, sess, config, model):
        self.sess = sess
        self.config = config
        self.model = model

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)

        self.model.load(sess) # load the model

    def eval(self):
        print('\nStarting Evaluation...')
        self.sess.run(self.model.dataset.init_op_test) # switch to test dataset

        predictions = []
        labels =  []

        count = 0

        # loop through test dataset and get predictions
        while(True):
            try:
                print('Iter: ', count)
                count += 1
                fetches = {
                    'predictions': self.model.predictions,
                    'labels': self.model.labels
                }

                output = self.sess.run(fetches)

                predictions.append(output['predictions'])
                labels.append(output['labels'])

            except tf.errors.OutOfRangeError:
                break

        labels = np.concatenate(labels, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        if self.config.TEST_METHOD_NAME == 'patch':
            # if patch based -> put images back together

            n_patches_per_image = int(self.config.TEST_N_PATCHES_PER_IMAGE**2)
            n_images = int(predictions.shape[0] / n_patches_per_image)

            for i in range(n_images):
                start = i * n_patches_per_image
                end = start + n_patches_per_image

                img = isam.merge_into_image_from_flatten(patches_flatten=predictions[start:end, : ,: , :],
                                                index=labels[start:end, :],
                                                image_size=(self.config.TEST_IMAGE_SIZE, self.config.TEST_IMAGE_SIZE, 1),
                                                stride=self.config.TEST_METHOD_STRIDE)
                img_id = labels[start, 0]

                img = Image.fromarray(img[:,:,0])
                img.save(self.config.TEST_OUTPUT_DIR + 'out{}.png'.format(img_id))

        elif self.config.TEST_METHOD_NAME == 'full':
            n_images = predictions.shape[0]

            for i in range(n_images):
                img = Image.fromarray(predictions[i, :, :, 0].astype('uint8'))
                img_id = labels[i, 0]
                img.save(self.config.TEST_OUTPUT_DIR + 'out{}.png'.format(img_id))
        else:
            raise ValueError('Unknown Test Method Name')

        print('Evaluation Finished: saved {} masks to {}'.format(n_images, self.config.TEST_OUTPUT_DIR))
