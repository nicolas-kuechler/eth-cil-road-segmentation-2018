from abc import ABC
import numpy as np

class AbstractConfig(ABC):

    BASE_DIR = './../'


    '''
      _____         _      _
     |_   _| _ __ _(_)_ _ (_)_ _  __ _
       | || '_/ _` | | ' \| | ' \/ _` |
       |_||_| \__,_|_|_||_|_|_||_\__, |
                                 |___/
    '''
    TRAIN_PATH_TO_DATA = BASE_DIR + 'data/training/images'
    TRAIN_PATH_TO_GROUNDTRUTH = BASE_DIR + 'data/training/groundtruth'
    TRAIN_BATCH_SIZE = 20
    TRAIN_SEED = None

    TRAIN_METHOD_NAME = 'patch' # patch or full
    TRAIN_METHOD_PATCH_SIZE_PERCENTAGE = 0.25  # only for patch, the size of the patch in percentage of the original image size

    N_EPOCHS = 4
    N_BATCHES_PER_EPOCH = 10

    LEARNING_RATE_TYPE = 'fixed' # in ['exponential', 'linear', 'fixed']
    LEARNING_RATE = 1.0
    LEARNING_RATE_DECAY_STEPS = 1000
    LEARNING_RATE_DECAY_RATE = 0.95


    '''
     __   __    _ _    _      _   _
     \ \ / /_ _| (_)__| |__ _| |_(_)___ _ _
      \ V / _` | | / _` / _` |  _| / _ \ ' \
       \_/\__,_|_|_\__,_\__,_|\__|_\___/_||_|
    '''
    VALID_PATH_TO_DATA = BASE_DIR + 'data/validation/images'
    VALID_PATH_TO_GROUNDTRUTH = BASE_DIR + 'data/validation/groundtruth'
    VALID_BATCH_SIZE = 20

    VALID_METHOD_NAME = 'patch'     # patch or full
    VALID_METHOD_PATCH_SIZE = 100   # only for patch
    VALID_METHOD_STRIDE = 100        # only for patch


    '''
      _____       _
     |_   _|__ __| |_
       | |/ -_|_-<  _|
       |_|\___/__/\__|
    '''
    TEST_PATH_TO_DATA = BASE_DIR + 'data/test_images'
    TEST_BATCH_SIZE = 20

    TEST_METHOD_NAME = 'patch'     # patch or full
    TEST_METHOD_PATCH_SIZE = 120   # only for patch
    TEST_METHOD_STRIDE = 61        # only for patch



    '''
      _____                      __ _
     |_   _|__ _ _  ___ ___ _ _ / _| |_____ __ __
       | |/ -_) ' \(_-</ _ \ '_|  _| / _ \ V  V /
       |_|\___|_||_/__/\___/_| |_| |_\___/\_/\_/
    '''
    MAX_CHECKPOINTS_TO_KEEP = 10
    CHECKPOINT_DIR = BASE_DIR + 'output'# TODO [nku] define checkpoint dir
    CHECKPOINT_ID = None # if None, the last checkpoint will be used



    '''
        _                          _        _   _
       /_\ _  _ __ _ _ __  ___ _ _| |_ __ _| |_(_)___ _ _
      / _ \ || / _` | '  \/ -_) ' \  _/ _` |  _| / _ \ ' \
     /_/ \_\_,_\__, |_|_|_\___|_||_\__\__,_|\__|_\___/_||_|
               |___/
    '''
    # flip (mirror) the image along either its horizontal or vertical axis.
    AUG_FLIP_RANDOM_PROB = 0.2

    # flip (mirror) the image along its horizontal axis, i.e. from left to right.
    AUG_FLIP_LEFT_RIGHT_PROB = 0.2

    # flip (mirror) the image along its vertical axis, i.e. from top to bottom.
    AUG_FLIP_TOP_BOTTOM_PROB = 0.2

    # rotate an image by either 90, 180, or 270 degrees, selected randomly.
    AUG_ROTATE_RANDOM_90_PROB = 0.2

    # rotate an image by an arbitrary amount and crop the largest possible rectangle.
    # max_left_rotation: 1- 25, max_right_rotation: 1-25
    AUG_ROTATE_PROB = 0.2
    AUG_ROTATE_MAX_LEFT_ROTATION = 15
    AUG_ROTATE_MAX_RIGHT_ROTATION = 15

    # shear the image by a specified number of degrees.
    # max_shear_left: 1-25, max_shear_right: 1-25
    AUG_SHEAR_PROB = 0.2
    AUG_SHEAR_MAX_SHEAR_LEFT = 15
    AUG_SHEAR_MAX_SHEAR_RIGHT = 15

    # zooms into an image at a random location within the image.
    AUG_ZOOM_RANDOM_PROB = 0.2
    AUG_ZOOM_RANDOM_PERCENTAGE_AREA = 0.5
    AUG_ZOOM_RANDOM_RANDOMISE_PERCENTAGE_AREA = True

    # random change brightness of an image
    # min_factor: 0.0-1.0 black-original, max_factor: 0.0-1.0 black-original
    AUG_RANDOM_BRIGHTNESS_PROB = 0.2
    AUG_RANDOM_BRIGHTNESS_MIN_FACTOR = 0.9
    AUG_RANDOM_BRIGHTNESS_MAX_FACTOR = 0.95

    # random change image contrast
    # min_factor: 0.0-1.0 grey-original, max_factor: 0.0-1.0 grey-original
    AUG_RANDOM_CONTRAST_PROB = 0.2
    AUG_RANDOM_CONTRAST_MIN_FACTOR = 0.9
    AUG_RANDOM_CONTRAST_MAX_FACTOR = 0.95

    # PCA Color Augmentation: (from paper: ImageNet Classification with Deep Convolutional Neural Networks)
    # perform PCA on the set of RGB pixel values throughout the training set.
    # To each training image, we add multiples of the found principal components,
    # with magnitudes proportional to the corresponding eigenvalues
    # times a random variable drawn from a Gaussian with mean and standard deviation
    AUG_COLOR_PCA_PROB = 0.7
    AUG_COLOR_PCA_EVECS = np.array([[-0.59073215, 0.72858809, 0.34669139],
                                    [-0.57144203, -0.07443475, -0.81725973],
                                    [-0.56963982, -0.68089563, 0.46031686]])
    AUG_COLOR_PCA_EVALS = np.array([6927.25308594, 46.78135866, 22.71954328])
    AUG_COLOR_PCA_MU = 0
    AUG_COLOR_PCA_SIGMA = 0.1

    # performs a random, elastic gaussian distortion on an image
    # param see https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Pipeline.py
    AUG_GAUSSIAN_DISTORTION_PROB = 0.2
    AUG_GAUSSIAN_DISTORTION_GRID_WIDTH = 5
    AUG_GAUSSIAN_DISTORTION_GRID_HEIGHT = 5
    AUG_GAUSSIAN_DISTORTION_MAGNITUDE = 3
    AUG_GAUSSIAN_DISTORTION_CORNER = 'bell'
    AUG_GAUSSIAN_DISTORTION_METHOD = 'in'
    AUG_GAUSSIAN_DISTORTION_MEX = 0.5
    AUG_GAUSSIAN_DISTORTION_MEY = 0.5
    AUG_GAUSSIAN_DISTORTION_SDX = 0.05
    AUG_GAUSSIAN_DISTORTION_SDY = 0.05

    # Performs a random, elastic distortion on an image.
    # grid: 2-10, magnitude: 1-10
    AUG_RANDOM_DISTORTION_PROB = 0.2
    AUG_RANDOM_DISTORTION_GRID_WIDTH = 5
    AUG_RANDOM_DISTORTION_GRID_HEIGHT = 5
    AUG_RANDOM_DISTORTION_MAGNITUDE = 3
