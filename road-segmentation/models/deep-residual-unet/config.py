from core.abstract_config import AbstractConfig
import tensorflow as tf

class Config(AbstractConfig):

    def __init__(self, model_name: str, ext):
        super().__init__(model_name, ext)

    # Overwrite any Configurations from Abstract Config
    TRAIN_METHOD_PATCH_SIZE_PERCENTAGE = 0.5

    VALID_METHOD_PATCH_SIZE = 200
    VALID_METHOD_STRIDE = 100

    TEST_METHOD_PATCH_SIZE = 304   # only for patch
    TEST_METHOD_STRIDE = 152       # only for patch
    TEST_N_PATCHES_PER_IMAGE = (608 - TEST_METHOD_PATCH_SIZE)/ TEST_METHOD_STRIDE + 1

    N_EPOCHS = 50
    N_BATCHES_PER_EPOCH = 1000
    TRAIN_BATCH_SIZE = 8

    LEARNING_RATE = 0.001

    LEARNING_RATE_TYPE = 'linear'
    LEARNING_RATE_DECAY_RATE = 0.1
    LEARNING_RATE_DECAY_STEPS = N_BATCHES_PER_EPOCH * 20 # every 20'th epoch

    OPTIMIZER = tf.train.AdamOptimizer
    USE_GRADIENT_CLIPPING = False

    SUMMARY_IMAGE_EVERY_STEP = 500

    AUG_SHEAR_PROB = 0.0000001
    AUG_ROTATE_RANDOM_90_PROB = 0.66
    AUG_ROTATE_PROB = 0.5
    AUG_ROTATE_MAX_LEFT_ROTATION = 25
    AUG_ROTATE_MAX_RIGHT_ROTATION = 25
    AUG_ZOOM_RANDOM_PROB = 0.2

    AUG_COLOR_PCA_PROB = 0.5
    AUG_COLOR_PCA_SIGMA = 0.25

    # Define new Configurations for your Model
