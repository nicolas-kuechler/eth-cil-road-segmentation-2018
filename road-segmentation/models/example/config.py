from core.abstract_config import AbstractConfig

class Config(AbstractConfig):
    # Define a Model Name
    MODEL_NAME = 'example-model'


    # Overwrite any Configurations from Abstract Config
    N_EPOCHS = 1
    N_BATCHES_PER_EPOCH = 2

    # Define new Configurations for your Model
