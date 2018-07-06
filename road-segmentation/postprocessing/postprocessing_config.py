import sys
from core.abstract_config import AbstractConfig

class Config(AbstractConfig):

    def __init__(self, batch_name):
        super().__init__( batch_name, '' )
        
        self.POST_PREDICTIONS_IN_DIR = self.BASE_DIR + 'output/' + batch_name + '/test_output/predictions/'
        self.TEST_IMAGES_IN_DIR = './../data/test_images/'
        
        self.OUTPUT_DIR = self.BASE_DIR + 'output/' + batch_name + '_post/test_output'
                
		# Overwrite any Configurations from Abstract Config
        self.TEST_OUTPUT_DIR = self.OUTPUT_DIR
        self.TEST_PATH_TO_DATA = self.TEST_IMAGES_IN_DIR