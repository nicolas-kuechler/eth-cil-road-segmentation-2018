from core.abstract_config import AbstractConfig
import tensorflow as tf

class Config(AbstractConfig):

    def __init__(self, model_name: str):
        super().__init__(model_name)

    # Overwrite any Configurations from Abstract Config
    TRAIN_METHOD_PATCH_SIZE_PERCENTAGE = 0.5

    VALID_METHOD_PATCH_SIZE = 200
    VALID_METHOD_STRIDE = 200

    N_EPOCHS = 30
    N_BATCHES_PER_EPOCH = 1000
    TRAIN_BATCH_SIZE = 8

    LEARNING_RATE = 0.001

    LEARNING_RATE_TYPE = 'linear'
    LEARNING_RATE_DECAY_RATE = 0.1
    LEARNING_RATE_DECAY_STEPS = N_BATCHES_PER_EPOCH * 20 # every 20'th epoch

    # Define new Configurations for your Model
    OPTIMIZER = tf.train.AdamOptimizer
    USE_GRADIENT_CLIPPING = False
