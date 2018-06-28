from core.abstract_config import AbstractConfig
import tensorflow as tf

class Config(AbstractConfig):

    def __init__(self, model_name: str, ext):
        super().__init__(model_name, ext)

    # Overwrite any Configurations from Abstract Config
    TRAIN_METHOD_PATCH_SIZE_PERCENTAGE = 0.64

    VALID_METHOD_PATCH_SIZE = 256
    VALID_METHOD_STRIDE = 72
    VALID_N_PATCHES_PER_IMAGE = (400 - VALID_METHOD_PATCH_SIZE)/ VALID_METHOD_STRIDE + 1

    SUB_WRITE_INDIVIDUAL_PREDICTIONS = True

    TEST_METHOD_PATCH_SIZE = [256, 304, 608]   # only for patch
    TEST_METHOD_STRIDE = [88, 152, 608]       # only for patch
    TEST_ROTATION_DEGREE = [0, 90, 180, 270]

    #TEST_N_PATCHES_PER_IMAGE = (608 - TEST_METHOD_PATCH_SIZE)/ TEST_METHOD_STRIDE + 1

    N_EPOCHS = 20
    N_BATCHES_PER_EPOCH = 1000
    TRAIN_BATCH_SIZE = 4

    LEARNING_RATE = 0.01

    LEARNING_RATE_TYPE = 'exponential'
    LEARNING_RATE_DECAY_RATE = 0.95
    LEARNING_RATE_DECAY_STEPS = 1000

    OPTIMIZER = tf.train.AdamOptimizer
    USE_GRADIENT_CLIPPING = False

    SUMMARY_IMAGE_EVERY_STEP = 500

    AUG_SHEAR_PROB = 0.1
    AUG_ROTATE_RANDOM_90_PROB = 0.75
    AUG_ROTATE_PROB = 0.5
    AUG_ROTATE_MAX_LEFT_ROTATION = 25
    AUG_ROTATE_MAX_RIGHT_ROTATION = 25
    AUG_ZOOM_RANDOM_PROB = 0.0000000001

    # Gaussian blur
    AUG_GAUSSIAN_BLUR_PROB = 0.2
    AUG_GAUSSIAN_BLUR_MIN_SIGMA = 0.01
    AUG_GAUSSIAN_BLUR_MAX_SIGMA = 2

    AUG_COLOR_PCA_PROB = 0.5
    AUG_COLOR_PCA_SIGMA = 0.1

    # Define new Configurations for your Model
