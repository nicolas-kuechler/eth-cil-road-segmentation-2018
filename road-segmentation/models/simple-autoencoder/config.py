from core.abstract_config import AbstractConfig

class Config(AbstractConfig):

    def __init__(self, model_name: str):
        super().__init__(model_name)


    # Overwrite any Configurations from Abstract Config
    N_EPOCHS = 100
    N_BATCHES_PER_EPOCH = 2
    TRAIN_METHOD_NAME = 'full' # patch or full

    # Define new Configurations for your Model
