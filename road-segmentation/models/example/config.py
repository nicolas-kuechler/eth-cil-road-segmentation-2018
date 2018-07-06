from core.abstract_config import AbstractConfig

class Config(AbstractConfig):

    def __init__(self, model_name: str, ext):
        super().__init__(model_name, ext)


    # Overwrite any Configurations from Abstract Config
    N_EPOCHS = 3
    N_BATCHES_PER_EPOCH = 2
    SUMMARY_IMAGE_EVERY_STEP = 4

    # Define new Configurations for your Model
    TEST_METHOD_PATCH_SIZE = [128]
    TEST_METHOD_STRIDE = [60]
    TEST_ROTATION_DEGREE = [0, 90]
