from core.abstract_config import AbstractConfig
import tensorflow as tf

class Config(AbstractConfig):

    def __init__(self, model_name: str):
        super().__init__(model_name)

    VALID_METHOD_NAME = 'full'
    TRAIN_METHOD_NAME = 'full'
    TEST_METHOD_NAME = 'full'

    N_EPOCHS = 50
    N_BATCHES_PER_EPOCH = 1000
    TRAIN_BATCH_SIZE = 4

    TEST_BATCH_SIZE = 4

    LEARNING_RATE = 0.001

    LEARNING_RATE_TYPE = 'linear'
    LEARNING_RATE_DECAY_RATE = 0.1
    LEARNING_RATE_DECAY_STEPS = N_BATCHES_PER_EPOCH * 20 # every 20'th epoch

    OPTIMIZER = tf.train.AdamOptimizer
    USE_GRADIENT_CLIPPING = False

    SUMMARY_IMAGE_EVERY_STEP = 500

    AUG_FLIP_RANDOM_PROB = 0.0000000000000001
    AUG_SHEAR_PROB = 0.00000000000000001
    AUG_ROTATE_RANDOM_90_PROB = 0.75
    AUG_ROTATE_PROB = 0.7
    AUG_ROTATE_MAX_LEFT_ROTATION = 25
    AUG_ROTATE_MAX_RIGHT_ROTATION = 25
    AUG_ZOOM_RANDOM_PROB = 0.2

    AUG_COLOR_PCA_PROB = 0.5
    AUG_COLOR_PCA_SIGMA = 0.25

    # Define new Configurations for your Model
