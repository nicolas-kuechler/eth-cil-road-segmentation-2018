import sys
from core.abstract_config import AbstractConfig

class Config(AbstractConfig):

    def __init__(self, batch_name):
        super().__init__( batch_name, '' )
        
        self.PREDICTIONS_IN_DIR = self.POST_DIR + 'to_process/' + batch_name
        self.TEST_IMAGES_IN_DIR = './../data/test_images/'
        
        self.OUTPUT_DIR = self.POST_DIR + 'output/' + batch_name + '/'
        
        self.WRITE_SUBMISSION = True
        
		# Overwrite any Configurations from Abstract Config
        self.TEST_OUTPUT_DIR = self.OUTPUT_DIR + 'submission/' # only used for submissions
        self.TEST_PATH_TO_DATA = self.TEST_IMAGES_IN_DIR
        
        #self.MAX_NUM_PROCESSES = 2
        self.MAX_NUM_PROCESSES = sys.maxsize
        
    POST_DIR = './postprocessing/'