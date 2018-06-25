import tensorflow as tf
import numpy as np
from PIL import Image
from utility import image_split_and_merge as isam
import os
from tqdm import tqdm
from utility import util

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

        pbar = tqdm(total=self.config.TEST_N_PATCHES_PER_IMAGE, unit=' pred')

        # loop through test dataset and get predictions
        while(True):
            try:
                fetches = {
                    'predictions': self.model.predictions,
                    'ids': self.model.ids
                }

                output = self.sess.run(fetches)

                predictions.append(output['predictions'])
                ids.append(output['ids'])

                pbar.update(self.config.TEST_BATCH_SIZE)

            except tf.errors.OutOfRangeError:
                break

        pbar.close()

        ids = np.concatenate(ids, axis=0)
        predictions = np.concatenate(predictions, axis=0)

        pred_dict = {}

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
                img = img[:,:,0]

                img = self.invert_augmentation(img)

                pred_dict[img_id] = img

        elif self.config.TEST_METHOD_NAME == 'full':
            n_images = predictions.shape[0]

            for i in range(n_images):
                img = predictions[i, :, :, 0]
                img_id = int(ids[i])

                img = self.invert_augmentation(img)

                pred_dict[img_id] = img
        else:
            raise ValueError('Unknown Test Method Name')

        print('Evaluation Finished')
        return pred_dict

    def invert_augmentation(self, img):
        if self.config.TEST_ROTATION_DEGREE is not 0:
            img = util.to_image(img)
            img = img.rotate(-1 * self.config.TEST_ROTATION_DEGREE, expand=False, resample=Image.BICUBIC)
            img = util.to_array(img)

        return img
