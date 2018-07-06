# Config module of parameter set 2
from core.abstract_config import AbstractConfig

class Config(AbstractConfig):

    def __init__(self, batch_name):
        super().__init__( batch_name, '' )
        
        self.POST_PREDICTIONS_IN_DIR = self.BASE_DIR + 'output/' + batch_name + '/test_output/predictions/'
        self.TEST_IMAGES_IN_DIR = './../data/test_images/'
        
        self.OUTPUT_DIR = self.BASE_DIR + 'output/' + batch_name + '_post_p2/test_output/'
                
		# Overwrite any Configurations from Abstract Config
        self.TEST_OUTPUT_DIR = self.OUTPUT_DIR
        self.TEST_PATH_TO_DATA = self.TEST_IMAGES_IN_DIR

    POST_NUM_INFERENCE_IT = 50
    
    POST_SDIMS_GAUSSIAN_X = 0.05
    POST_SDIMS_GAUSSIAN_Y = 0.05
    POST_COMPAT_GAUSSIAN = 15
    
    POST_SDIMS_BILATERAL_X = 120
    POST_SDIMS_BILATERAL_Y = 120
    POST_SCHAN_BILATERAL_R = 40
    POST_SCHAN_BILATERAL_G = 40
    POST_SCHAN_BILATERAL_B = 40
    POST_COMPAT_BILATERAL = 8
