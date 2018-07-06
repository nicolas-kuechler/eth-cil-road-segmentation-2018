# Config file for the script average.py (used to create a submission)
from core.abstract_config import AbstractConfig

class Config(AbstractConfig):

    def __init__(self, avg_name):
        super().__init__( avg_name, '' )
