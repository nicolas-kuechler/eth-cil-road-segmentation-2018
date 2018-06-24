from core.abstract_config import AbstractConfig

class Config(AbstractConfig):

    def __init__(self, model_name: str):
        super().__init__(model_name)


    # Overwrite any Configurations from Abstract Config
    N_EPOCHS = 50
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

    AUG_SHEAR_PROB = 0.0000001
    AUG_ROTATE_RANDOM_90_PROB = 0.66
    AUG_ROTATE_PROB = 0.5
    AUG_ROTATE_MAX_LEFT_ROTATION = 25
    AUG_ROTATE_MAX_RIGHT_ROTATION = 25
    AUG_ZOOM_RANDOM_PROB = 0.2

    AUG_COLOR_PCA_PROB = 0.5
    AUG_COLOR_PCA_SIGMA = 0.25
