from core.abstract_config import AbstractConfig

class Config(AbstractConfig):

    def __init__(self, model_name: str, ext):
        super().__init__(model_name, ext)


    # Overwrite any Configurations from Abstract Config
    N_EPOCHS = 30
    N_BATCHES_PER_EPOCH = 100
    LEARNING_RATE = 0.0001
    TRAIN_METHOD_NAME = 'full' # patch or full
    VALID_METHOD_NAME = 'full'
    TEST_METHOD_NAME = 'full'
    SUMMARY_IMAGE_EVERY_STEP = 1000
    TRAIN_BATCH_SIZE = 32
    SUMMARY_IMAGE_MAX_OUTPUTS = 5
    SUMMARY_FULL_IMAGE_MAX_OUTPUTS = 5

    # Define new Configurations for your Model

    N_fire_modules = 8
    FILTERS_SQUEEZE1 =  [16, 16, 32, 32, 48, 48, 64, 64]
    FILTERS_EXPAND1 =   [64, 64, 128, 128, 192, 192, 256, 256]
    FILTERS_EXPAND3 =   [64, 64, 128, 128, 192, 192, 256, 256]
    DROPOUTS =  [0, 0, 0, 0, 0, 0, 0, 1]
    MAX_POOLS = [1, 0, 0, 1, 0, 0, 0, 1]

    AUG_GAUSSIAN_BLUR_PROB = 0.00001
    AUG_GAUSSIAN_BLUR_MIN_SIGMA = 0.01
    AUG_GAUSSIAN_BLUR_MAX_SIGMA = 5