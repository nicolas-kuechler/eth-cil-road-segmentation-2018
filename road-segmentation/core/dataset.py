import numpy as np
import tensorflow as tf
import Augmentor, random, os, re
from PIL import Image
from PIL import ImageFilter
from utility import image_split_and_merge as isam

class Dataset():

    OUTPUT_TYPE = tf.uint8

    def __init__(self, config):
        self.config = config

        self.ds_train = self.build_ds_train()
        self.ds_valid = self.build_ds_valid()
        self.ds_test = self.build_ds_test()

        # could do some dataset pre-processing here
        self.pre_process()

        # create iterator

        iter = tf.data.Iterator.from_structure(output_types=(self.OUTPUT_TYPE, self.OUTPUT_TYPE, self.OUTPUT_TYPE), output_shapes=([None, None, None, 3],[None, None, None, 1], [None, None, None, 1]))
        self.img_batch, self.labels, self.id_batch = iter.get_next()

        # create initialization operations
        self.init_op_train = iter.make_initializer(self.ds_train)
        self.init_op_valid = iter.make_initializer(self.ds_valid)
        self.init_op_test = iter.make_initializer(self.ds_test)


    def pre_process(self):
        pass


    def build_ds_train(self):

        p = self.build_augmentation_pipeline(path_to_data=self.config.TRAIN_PATH_TO_DATA,
                                                path_to_groundtruth=self.config.TRAIN_PATH_TO_GROUNDTRUTH,
                                                seed=self.config.TRAIN_SEED)

        # in case of patch based approach adjust pipeline such that a random patch of the image is extracted
        if(self.config.TRAIN_METHOD_NAME == 'patch'):
            # crop a random area of an image, based on the percentage area to be returned
            p.crop_random(probability=1.0, percentage_area=self.config.TRAIN_METHOD_PATCH_SIZE_PERCENTAGE, randomise_percentage_area=False)


        ds = tf.data.Dataset.from_generator(
                lambda: p.tf_generator(),
                (self.OUTPUT_TYPE, self.OUTPUT_TYPE, self.OUTPUT_TYPE))
        ds = ds.batch(self.config.TRAIN_BATCH_SIZE)
        return ds

    def build_ds_valid(self):
        ds = tf.data.Dataset.from_generator(
                lambda: self.tf_generator(image_dir=self.config.VALID_PATH_TO_DATA,
                                            method_name=self.config.VALID_METHOD_NAME,
                                            patch_size=self.config.VALID_METHOD_PATCH_SIZE,
                                            stride=self.config.VALID_METHOD_STRIDE,
                                            gt_dir=self.config.VALID_PATH_TO_GROUNDTRUTH,
                                            gt_foreground_threshold=self.config.GT_FOREGROUND_THRESHOLD),
                (self.OUTPUT_TYPE, self.OUTPUT_TYPE, self.OUTPUT_TYPE))
        ds = ds.batch(self.config.VALID_BATCH_SIZE)
        return ds


    def build_ds_test(self):
        ds = tf.data.Dataset.from_generator(
                lambda: self.tf_generator(image_dir=self.config.TEST_PATH_TO_DATA,
                                            method_name=self.config.TEST_METHOD_NAME,
                                            patch_size=self.config.TEST_METHOD_PATCH_SIZE,
                                            stride=self.config.TEST_METHOD_STRIDE),
                (self.OUTPUT_TYPE, self.OUTPUT_TYPE, self.OUTPUT_TYPE))
        ds = ds.batch(self.config.TEST_BATCH_SIZE)
        return ds


    def tf_generator(self, image_dir:str, method_name:str, patch_size=None, stride=None, gt_dir=None, gt_foreground_threshold=None):
        for filename in os.listdir(image_dir):
            # load image and gt
            img = np.asarray(Image.open(image_dir + '/' + filename))
            gt = 0

            id = int(re.search('\d+', filename).group(0))

            if gt_dir is not None: # has gt
                gt = np.asarray(Image.open(gt_dir + '/' + filename))
                gt = gt.reshape((gt.shape[0], gt.shape[1], 1))

                # process gt such that each pixel is: road=1, background=0
                gt = gt / 255.0
                gt = np.where(gt>gt_foreground_threshold, 1, 0)

            if method_name == 'patch':
                img_patches = isam.split_into_patches(image=img, patch_size=(patch_size,  patch_size, 3), stride=stride)
                img_patches_flatten, ids = isam.flatten_patches(img_patches, id)

                if gt_dir is not None:
                    gt_patches = isam.split_into_patches(image=gt, patch_size=(patch_size, patch_size, 1), stride=stride)
                    gt_patches_flatten, _ = isam.flatten_patches(gt_patches, id)
                else:
                    gt_patches_flatten = np.zeros((img_patches_flatten.shape[0], 1, 1, 1))

                for img_patch, gt_patch, id in zip(img_patches_flatten, gt_patches_flatten, ids):
                    yield(img_patch, gt_patch, id)
            elif method_name == 'full':
                yield(img, gt, id)
            else:
                raise ValueError('Unknown method')


    def build_augmentation_pipeline(self, path_to_data: str, path_to_groundtruth:str , seed = None):
        # Create a pipeline
        p = CustomPipeline(source_directory=path_to_data, gt_foreground_threshold=self.config.GT_FOREGROUND_THRESHOLD)
        p.ground_truth(path_to_groundtruth)
        p.set_seed(seed) # set seed if provided

        # AFFINE TRANSFORMATION AUGMENTATION

        # flip (mirror) the image along either its horizontal or vertical axis.
        p.flip_random(probability=self.config.AUG_FLIP_RANDOM_PROB)

        # flip (mirror) the image along its horizontal axis, i.e. from left to right.
        p.flip_left_right(probability=self.config.AUG_FLIP_LEFT_RIGHT_PROB)

        # flip (mirror) the image along its vertical axis, i.e. from top to bottom.
        p.flip_top_bottom(probability=self.config.AUG_FLIP_TOP_BOTTOM_PROB)

        # rotate an image by either 90, 180, or 270 degrees, selected randomly.
        p.rotate_random_90(probability=self.config.AUG_ROTATE_RANDOM_90_PROB)

        # rotate an image by an arbitrary amount and crop the largest possible rectangle
        # in practice, angles larger than 25 degrees result in images that do not render correctly, therefore there is a limit of 25 degrees for this function.
        p.rotate(probability=self.config.AUG_ROTATE_PROB,
                        max_left_rotation=self.config.AUG_ROTATE_MAX_LEFT_ROTATION,
                        max_right_rotation=self.config.AUG_ROTATE_MAX_RIGHT_ROTATION)

        # shear the image by a specified number of degrees. max_shear_left: 1-25, max_shear_right: 1-25
        p.shear(probability=self.config.AUG_SHEAR_PROB,
                        max_shear_left=self.config.AUG_SHEAR_MAX_SHEAR_LEFT,
                        max_shear_right=self.config.AUG_SHEAR_MAX_SHEAR_RIGHT)

        # zooms into an image at a random location within the image.
        p.zoom_random(probability=self.config.AUG_ZOOM_RANDOM_PROB,
                        percentage_area=self.config.AUG_ZOOM_RANDOM_PERCENTAGE_AREA,
                        randomise_percentage_area=self.config.AUG_ZOOM_RANDOM_RANDOMISE_PERCENTAGE_AREA)

        # COLOR AUGMENTATION

        # random change brightness of an image
        p.random_brightness(probability=self.config.AUG_RANDOM_BRIGHTNESS_PROB,
                                min_factor=self.config.AUG_RANDOM_BRIGHTNESS_MIN_FACTOR,
                                max_factor=self.config.AUG_RANDOM_BRIGHTNESS_MAX_FACTOR)

        # random change image contrast
        p.random_contrast(probability=self.config.AUG_RANDOM_CONTRAST_PROB,
                                min_factor=self.config.AUG_RANDOM_CONTRAST_MIN_FACTOR,
                                max_factor=self.config.AUG_RANDOM_CONTRAST_MAX_FACTOR)

        color_pca = ColorPCA(probability=self.config.AUG_COLOR_PCA_PROB,
                                    evecs=self.config.AUG_COLOR_PCA_EVECS,
                                    evals=self.config.AUG_COLOR_PCA_EVALS,
                                    mu=self.config.AUG_COLOR_PCA_MU,
                                    sigma=self.config.AUG_COLOR_PCA_SIGMA)
        p.add_operation(color_pca)


        # STREET BRIGHTNESS AUGMENTATION
        street_brightness = StreetBrightnessAugmentation(probability=self.config.AUG_STREET_BRIGHTNESS_PROB,
                                                            min_brightness_change = self.config.AUG_STREET_BRIGHTNESS_MIN_CHANGE,
                                                            max_brightness_change = self.config.AUG_STREET_BRIGHTNESS_MAX_CHANGE,
                                                            fg_threshold = self.config.AUG_STREET_BRIGHTNESS_FG_THRESHOLD)
        p.add_operation(street_brightness)

        # GAUSSIAN BLUR
        gaussian = GaussianBlur(probability=self.config.AUG_GAUSSIAN_BLUR_PROB,
                                min_sigma=self.config.AUG_GAUSSIAN_BLUR_MIN_SIGMA,
                                max_sigma=self.config.AUG_GAUSSIAN_BLUR_MAX_SIGMA)
        p.add_operation(gaussian)


        # NOISE AUGMENTATION
        # performs a random, elastic gaussian distortion on an image.
        p.gaussian_distortion(probability=self.config.AUG_GAUSSIAN_DISTORTION_PROB,
                                    grid_width=self.config.AUG_GAUSSIAN_DISTORTION_GRID_WIDTH,
                                    grid_height=self.config.AUG_GAUSSIAN_DISTORTION_GRID_HEIGHT,
                                    magnitude=self.config.AUG_GAUSSIAN_DISTORTION_MAGNITUDE,
                                    corner=self.config.AUG_GAUSSIAN_DISTORTION_CORNER,
                                    method=self.config.AUG_GAUSSIAN_DISTORTION_METHOD,
                                    mex=self.config.AUG_GAUSSIAN_DISTORTION_MEX,
                                    mey=self.config.AUG_GAUSSIAN_DISTORTION_MEY,
                                    sdx=self.config.AUG_GAUSSIAN_DISTORTION_SDX,
                                    sdy=self.config.AUG_GAUSSIAN_DISTORTION_SDY)

        # Performs a random, elastic distortion on an image. (grid: 2-10 magnitude: 1-10)
        p.random_distortion(probability=self.config.AUG_RANDOM_DISTORTION_PROB,
                                    grid_width=self.config.AUG_RANDOM_DISTORTION_GRID_WIDTH,
                                    grid_height=self.config.AUG_RANDOM_DISTORTION_GRID_HEIGHT,
                                    magnitude=self.config.AUG_RANDOM_DISTORTION_MAGNITUDE)
        return p


class CustomPipeline(Augmentor.Pipeline):
    def __init__(self, source_directory, gt_foreground_threshold, output_directory='output', save_format=None):
        self.gt_foreground_threshold = gt_foreground_threshold
        Augmentor.Pipeline.__init__(self, source_directory, output_directory, save_format)

    def tf_generator(self):
        while True:
            # Randomly select images for augmentation and yield the augmented images.

            # select random image
            random_image_index = random.randint(0, len(self.augmentor_images)-1)

            # apply pipeline operations to image and groundtruth
            images = self._tf_execute(self.augmentor_images[random_image_index])

            # reshape image
            img = np.asarray(images[0])
            w = img.shape[0]
            h = img.shape[1]
            c = 1 if np.ndim(img) == 2 else img.shape[2]
            img = img.reshape(w, h, c)

            # reshape groundtruth
            gt = np.asarray(images[1])
            w = gt.shape[0]
            h = gt.shape[1]
            c = 1 if np.ndim(gt) == 2 else gt.shape[2]
            gt = gt.reshape(w, h, c)

            # process gt such that each pixel is: road=1, background=0
            gt = gt / 255.0
            gt = np.where(gt>self.gt_foreground_threshold, 1.0, 0.0)

            yield (img, gt, random_image_index)


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

class StreetBrightnessAugmentation(Augmentor.Operations.Operation):

    def __init__(self, probability, min_brightness_change: int, max_brightness_change: int, fg_threshold: int):
        Augmentor.Operations.Operation.__init__(self, probability)

        self.min_brightness_change = min_brightness_change
        self.max_brightness_change = max_brightness_change
        self.fg_threshold = fg_threshold


    def perform_operation(self, images):

        assert(len(images)==2), 'needs gt'

        img = np.copy(np.asarray(images[0])).astype('int16')
        gt = np.asarray(images[1])

        mask = np.where(gt>self.fg_threshold, 1, 0)

        brightness = random.randint(self.min_brightness_change, self.max_brightness_change)

        img[mask==1] += np.array([brightness]).astype(img.dtype)

        img[img>255] = 255
        img[img<0] = 0

        images[0] = Image.fromarray(img.astype('uint8'))

        return images

class GaussianBlur(Augmentor.Operations.Operation):
    def __init__(self, probability, min_sigma, max_sigma):
        Augmentor.Operations.Operation.__init__(self, probability)

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def perform_operation(self, images):

        augmented_images = []
        for image in images:

            img_array = np.asarray(image)
            shape = img_array.shape
            image = image.filter(ImageFilter.GaussianBlur(np.random.uniform(self.min_sigma, self.max_sigma)))
            augmented_images.append(image)

        # Return the image so that it can further processed in the pipeline:
        return augmented_images