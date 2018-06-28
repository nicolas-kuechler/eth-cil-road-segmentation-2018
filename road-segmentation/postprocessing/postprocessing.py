import sys
path = "/home/dpakhom1/dense_crf_python/"
sys.path.append(path)
import pydensecrf.densecrf as dcrf
import numpy as np
from utility import util
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

# PyDenseCRF courtesy of https://github.com/lucasb-eyer/pydensecrf:
# Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
# Philipp Krahenbuhl and Vladlen Koltun
# NIPS 2011

class Postprocessing():

    def __init__(self):
    	pass
        
    def crf(self, image, prediction):
        # prepare input for crf
        prediction = prediction.squeeze()
        processed_prediction = np.array([ prediction, 1-prediction ])
        processed_prediction = processed_prediction.reshape(( 2,-1 ))
        
        unary = unary_from_softmax( processed_prediction ) 
        unary = np.ascontiguousarray(unary) # necessary since the library pydensecrf is using a cython wrapper
        
        # create crf
        d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
        d.setUnaryEnergy(unary)
        
        # penalize small pieces of segmentation which are spatially isolated
        pairwise_energy = create_pairwise_gaussian(sdims=(2, 2), shape=image.shape[:2])
        d.addPairwiseEnergy(pairwise_energy, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        # use local color features to refine segmentation
        pairwise_energy = create_pairwise_bilateral(sdims=(2, 2), schan=(1,1,1),
                                                    img=image, chdim=2)
        d.addPairwiseEnergy(pairwise_energy, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        
        # do inference (with 5 iterations)
        Q = d.inference(5)
        result = np.argmin(Q, axis=0)
        processed_prediction = result.reshape((image.shape[0], image.shape[1]))
        #new_prediction = np.array(Q)
        #processed_prediction = new_prediction[0].reshape((image.shape[0], image.shape[1]))
        
        return processed_prediction
        