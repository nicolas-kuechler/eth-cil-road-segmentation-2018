import numpy as np
import tensorflow as tf
import Augmentor
from PIL import Image
import random, os
from utility import image_split_and_merge
import time

class Dataset():

    output_type = tf.uint8

    def __init__(self, config):
        self.config = config

    def build_dataset(self, split):
        """ Build the Dataset
        :param split: Which training split to load, must be one of ['train', 'valid', 'test'].
        """

        assert split in ['train', 'valid', 'test']

        if split == 'test':
            image_dir = self.config['test']['path_to_data']
            images = []

            for filename in os.listdir(image_dir):
                # load image
                img = np.asarray(Image.open(image_dir + '/' + filename))
                # reshape image for concatenation
                img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

                # TODO [nku] add option to split into patches
                #patches = image_split_and_merge.split_into_patches(image=img, patch_size=(120, 120, 3), stride=61)
                #patch_flat, flatten_index = image_split_and_merge.flatten_patches(patches)

                images.append(img)

            patches = np.concatenate(images)

            ds = tf.data.Dataset.from_tensor_slices(patches)

        else: # train or valid
            p = self.build_augmentation_pipeline(split)
            ds = tf.data.Dataset.from_generator(lambda: p.tf_generator(), (self.output_type, self.output_type))

        return ds


    def build_augmentation_pipeline(self, split):

        # Create a pipeline
        p = CustomPipeline(self.config[split]['path_to_data'])
        p.ground_truth(self.config[split]['path_to_groundtruth'])
        p.set_seed(self.config[split].get('seed')) # set seed if provided

        # AFFINE TRANSFORMATION AUGMENTATION

        # flip (mirror) the image along either its horizontal or vertical axis.
        if 'flip_random' in self.config:
            args = self.config['flip_random']
            p.flip_random(probability=args['probability'])

        # flip (mirror) the image along its horizontal axis, i.e. from left to right.
        if 'flip_left_right' in self.config:
            args = self.config['flip_left_right']
            p.flip_left_right(probability=args['probability'])

        # flip (mirror) the image along its vertical axis, i.e. from top to bottom.
        if 'flip_top_bottom' in self.config:
            args = self.config['flip_top_bottom']
            p.flip_top_bottom(probability=args['probability'])

        # rotate an image by either 90, 180, or 270 degrees, selected randomly.
        if 'rotate_random_90' in self.config:
            args = self.config['rotate_random_90']
            p.rotate_random_90(probability=args['probability'])

        # rotate an image by an arbitrary amount and crop the largest possible rectangle
        # in practice, angles larger than 25 degrees result in images that do not render correctly, therefore there is a limit of 25 degrees for this function.
        if 'rotate' in self.config:
            args = self.config['rotate']
            p.rotate(probability=args['probability'],
                        max_left_rotation=args['max_left_rotation'],
                        max_right_rotation=args['max_right_rotation'])

        # shear the image by a specified number of degrees. max_shear_left: 1-25, max_shear_right: 1-25
        if 'shear' in self.config:
            args = self.config['shear']
            p.shear(probability=args['probability'],
                        max_shear_left=args['max_shear_left'],
                        max_shear_right=args['max_shear_right'])

        # zooms into an image at a random location within the image.
        if 'zoom_random' in self.config:
            args = self.config['zoom_random']
            p.zoom_random(probability=args['probability'],
                            percentage_area=args['percentage_area'],
                            randomise_percentage_area=args['randomise_percentage_area'])

        # crop a random area of an image, based on the percentage area to be returned.
        if 'crop_random' in self.config:
            args = self.config['crop_random']
            p.crop_random(probability=args['probability'],
                            percentage_area=args['percentage_area'],
                            randomise_percentage_area=args['randomise_percentage_area'])


        # COLOR AUGMENTATION

        # random change brightness of an image
        if 'random_brightness' in self.config:
            args = self.config['random_brightness']
            p.random_brightness(probability=args['probability'],
                                    min_factor=args['min_factor'],
                                    max_factor=args['max_factor'])

        # random change image contrast
        if 'random_contrast' in self.config:
            args = self.config['random_contrast']
            p.random_contrast(probability=args['probability'],
                                min_factor=args['min_factor'],
                                max_factor=args['max_factor'])

        if 'color_pca' in self.config:
            args = self.config['color_pca']
            color_pca = ColorPCA(probability=args['probability'],
                                    evecs=args['evecs'], evals=args['evals'],
                                    mu=args['mu'], sigma=args['sigma'])
            p.add_operation(color_pca)

        # NOISE AUGMENTATION
        # performs a random, elastic gaussian distortion on an image.
        if 'gaussian_distortion' in self.config:
            args = self.config['gaussian_distortion']
            p.gaussian_distortion(probability=args['probability'],
                                    grid_width=args['grid_width'],
                                    grid_height=args['grid_height'],
                                    magnitude=args['magnitude'],
                                    corner=args['corner'],
                                    method=args['method'],
                                    mex=args['mex'], mey=args['mey'],
                                    sdx=args['sdx'], sdy=args['sdy'])

        # Performs a random, elastic distortion on an image. (grid: 2-10 magnitude: 1-10)
        if 'random_distortion' in self.config:
            args = self.config['random_distortion']
            p.random_distortion(probability=args['probability'],
                                    grid_width=args['grid_width'],
                                    grid_height=args['grid_height'],
                                    magnitude=args['magnitude'])
        return p


class CustomPipeline(Augmentor.Pipeline):
    def __init__(self, source_directory=None, output_directory='output', save_format=None):
        Augmentor.Pipeline.__init__(self, source_directory, output_directory, save_format)

    def tf_generator(self):
        while True:
            # Randomly select images for augmentation and yield the augmented images.

            # select random image
            random_image_index = random.randint(0, len(self.augmentor_images)-1)

            # apply pipeline operations to image and groundtruth
            images = self._tf_execute(self.augmentor_images[random_image_index])

            # reshape image
            img_array = np.asarray(images[0])
            w = img_array.shape[0]
            h = img_array.shape[1]
            c = 1 if np.ndim(img_array) == 2 else img_array.shape[2]
            img_array = img_array.reshape(w, h, c)

            # TODO check if necessary?
            # reshape groundtruth
            gt_array = np.asarray(images[1])
            w = gt_array.shape[0]
            h = gt_array.shape[1]
            c = 1 if np.ndim(gt_array) == 2 else gt_array.shape[2]
            gt_array = gt_array.reshape(w, h, c)

            yield (img_array, gt_array)


    def _tf_execute(self, augmentor_image):
        images = []

        if augmentor_image.image_path is not None:
            images.append(Image.open(augmentor_image.image_path))

        if augmentor_image.ground_truth is not None:
            images.append(Image.open(augmentor_image.ground_truth))

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images = operation.perform_operation(images)

        return images


# Custom Operation that performs ColorPCA as described in the AlexNet paper
class ColorPCA(Augmentor.Operations.Operation):

    def __init__(self, probability, evecs, evals, mu, sigma):
        Augmentor.Operations.Operation.__init__(self, probability)

        self.evecs = evecs  # eigenvectors
        self.evals = evals  # eigenvalues
        self.mu = mu        # mean
        self.sigma = sigma  # standard deviation

    def perform_operation(self, images):

        augmented_images = []
        for image in images:

            img_array = np.asarray(image)
            shape = img_array.shape
            if(len(shape) == 3 and shape[2] == self.evals.shape[0]):
                alpha = np.random.normal(self.mu, self.sigma, 3)
                offset = self.evecs @ (alpha * np.sqrt(self.evals))
                img_array = img_array  + offset

                img_array[img_array<0] = 0
                img_array[img_array>255] = 255
                augmented_images.append(Image.fromarray(np.uint8(img_array)))
            else:
                augmented_images.append(image) # skip groundtruth images

        # Return the image so that it can further processed in the pipeline:
        return augmented_images
