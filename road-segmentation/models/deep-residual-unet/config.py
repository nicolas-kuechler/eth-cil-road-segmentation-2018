from core.abstract_config import AbstractConfig

class Config(AbstractConfig):

    def __init__(self, model_name: str):
        super().__init__(model_name)

    # Overwrite any Configurations from Abstract Config
    TRAIN_METHOD_PATCH_SIZE_PERCENTAGE = 0.5

    VALID_METHOD_PATCH_SIZE = 200
    VALID_METHOD_STRIDE = 200

    N_EPOCHS = 30
    N_BATCHES_PER_EPOCH = 300
    TRAIN_BATCH_SIZE = 2

    LEARNING_RATE = 1.0

    # Define new Configurations for your Model
