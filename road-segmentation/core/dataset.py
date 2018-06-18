import numpy as np
import tensorflow as tf
import Augmentor, random, os, re
from PIL import Image
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
        iter = tf.data.Iterator.from_structure(output_types=(self.OUTPUT_TYPE, self.OUTPUT_TYPE), output_shapes=([None, None, None, 3],[None, None, None, 1])) #TODO [nku] for training set consider that id's look different maybe?
        self.img_batch, self.labels = iter.get_next()

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
                (self.OUTPUT_TYPE, self.OUTPUT_TYPE))
        ds = ds.batch(self.config.TRAIN_BATCH_SIZE)
        return ds

    def build_ds_valid(self):
        ds = tf.data.Dataset.from_generator(
                lambda: self.tf_generator_gt(image_dir=self.config.VALID_PATH_TO_DATA,
                                                gt_dir=self.config.VALID_PATH_TO_GROUNDTRUTH,
                                                method_name=self.config.VALID_METHOD_NAME,
                                                patch_size=self.config.VALID_METHOD_PATCH_SIZE,
                                                stride=self.config.VALID_METHOD_STRIDE),
                (self.OUTPUT_TYPE, self.OUTPUT_TYPE))
        ds = ds.batch(self.config.VALID_BATCH_SIZE)
        return ds


    def build_ds_test(self):
        ds = tf.data.Dataset.from_generator(
                lambda: self.tf_generator_id(image_dir=self.config.TEST_PATH_TO_DATA,
                                                method_name=self.config.TEST_METHOD_NAME,
                                                patch_size=self.config.TEST_METHOD_PATCH_SIZE,
                                                stride=self.config.TEST_METHOD_STRIDE
                                                ),
                (self.OUTPUT_TYPE, self.OUTPUT_TYPE))
        ds = ds.batch(self.config.TEST_BATCH_SIZE)
        return ds

    def tf_generator_gt(self, image_dir:str, gt_dir:str, method_name:str, patch_size=None, stride=None):
        assert method_name in ['patch', 'full']

        for filename in os.listdir(image_dir):
            # load image and gt
            img = np.asarray(Image.open(image_dir + '/' + filename))
            gt = np.asarray(Image.open(gt_dir + '/' + filename))

            # reshape gt
            gt = gt.reshape((gt.shape[0], gt.shape[1], 1))

            if(method_name=='patch'):
                img_patches = isam.split_into_patches(image=img, patch_size=(patch_size,  patch_size, 3), stride=stride)
                img_patches_flatten, _ = isam.flatten_patches(img_patches, 0)

                gt_patches = isam.split_into_patches(image=gt, patch_size=(patch_size, patch_size, 1), stride=stride)
                gt_patches_flatten, _ = isam.flatten_patches(gt_patches, 0)

                for img_patch, gt_patch in zip(img_patches_flatten, gt_patches_flatten):
                    yield(img_patch, gt_patch)
            else:
                yield(img, gt)

    def tf_generator_id(self, image_dir:str, method_name:str, patch_size=None, stride=None):
        assert method_name in ['patch', 'full']

        for filename in os.listdir(image_dir):
            # load image and extract id from filename
            img = np.asarray(Image.open(image_dir + '/' + filename))
            id = int(re.search('\d+', filename).group(0))

            if(method_name=='patch'):
                img_patches = isam.split_into_patches(image=img, patch_size=(patch_size, patch_size, 3), stride=stride)
                img_patches_flatten, ids = isam.flatten_patches(img_patches, id)

                for img_patch, id in zip(img_patches_flatten, ids): 
                    yield(img_patch, id)
            else:
                yield(img, id)


    def build_augmentation_pipeline(self, path_to_data: str, path_to_groundtruth:str , seed = None):
        # Create a pipeline
        p = CustomPipeline(path_to_data)
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
