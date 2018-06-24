from core.abstract_config import AbstractConfig

class Config(AbstractConfig):

    def __init__(self, model_name: str):
        super().__init__(model_name)


    # Overwrite any Configurations from Abstract Config
    N_EPOCHS = 1000
    N_BATCHES_PER_EPOCH = 4
    LEARNING_RATE = 0.0001
    TRAIN_BATCH_SIZE = 16
    TRAIN_METHOD_NAME = 'full' # patch or full
    VALID_METHOD_NAME = 'full'
    SUMMARY_IMAGE_EVERY_STEP = 3
    # Define new Configurations for your Model

    VALID_METHOD_NAME = 'full'
    TRAIN_METHOD_NAME = 'full'
    TEST_METHOD_NAME = 'full'
