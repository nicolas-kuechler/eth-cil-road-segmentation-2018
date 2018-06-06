import numpy as np

config = {}

config['batch_size'] = 4

config['path_to_data'] = './training/images'
config['path_to_groundtruth'] = './training/groundtruth'
config['seed'] = None

# AFFINE TRANSFORMATION AUGMENTATION

# flip (mirror) the image along either its horizontal or vertical axis.
config['flip_random'] = {'probability':0.2}

# flip (mirror) the image along its horizontal axis, i.e. from left to right.
config['flip_left_right'] = {'probability':0.2}

# flip (mirror) the image along its vertical axis, i.e. from top to bottom.
config['flip_top_bottom'] = {'probability':0.2}

# rotate an image by either 90, 180, or 270 degrees, selected randomly.
config['rotate_random_90'] = {'probability':0.2}

# rotate an image by an arbitrary amount and crop the largest possible rectangle.
# max_left_rotation: 1- 25, max_right_rotation: 1-25
config['rotate'] = {'probability':0.2, 'max_left_rotation':15, 'max_right_rotation':15}

# shear the image by a specified number of degrees.
# max_shear_left: 1-25, max_shear_right: 1-25
config['shear'] = {'probability':0.2, 'max_shear_left':15, 'max_shear_right':15}

# zooms into an image at a random location within the image.
# TODO: problem: within a batch all images need to be the same size
#config['zoom_random'] = {'probability':0.2, 'percentage_area': 0.5, 'randomise_percentage_area': True}

# crop a random area of an image, based on the percentage area to be returned.
# TODO: problem: within a batch all images need to be the same size
# config['crop_random'] = {'probability':1, 'percentage_area': 0.5, 'randomise_percentage_area': False}

# COLOR AUGMENTATION

# random change brightness of an image
# min_factor: 0.0-1.0 black-original, max_factor: 0.0-1.0 black-original
config['random_brightness'] = {'probability':0.2, 'min_factor': 0.9, 'max_factor': 0.95}

# random change image contrast
# min_factor: 0.0-1.0 grey-original, max_factor: 0.0-1.0 grey-original
config['random_contrast'] = {'probability':0.2, 'min_factor': 0.9, 'max_factor': 0.95}

# PCA Color Augmentation: (from paper: ImageNet Classification with Deep Convolutional Neural Networks)
# perform PCA on the set of RGB pixel values throughout the training set.
# To each training image, we add multiples of the found principal components,
# with magnitudes proportional to the corresponding eigenvalues
# times a random variable drawn from a Gaussian with mean and standard deviation
config['color_pca'] = {
    'probability': 0.7,
    'evecs':    np.array([  [-0.59073215, 0.72858809, 0.34669139],
                            [-0.57144203, -0.07443475, -0.81725973],
                            [-0.56963982, -0.68089563, 0.46031686]]),
    'evals': np.array([6927.25308594, 46.78135866, 22.71954328]),


    'mu': 0,
    'sigma': 0.1
}


# NOISE AUGMENTATION

# performs a random, elastic gaussian distortion on an image
# param see https://github.com/mdbloice/Augmentor/blob/master/Augmentor/Pipeline.py
config['gaussian_distortion'] = {'probability':0.2, 'grid_width': 5, 'grid_height': 5,
                                    'magnitude': 3, 'corner':  'bell', 'method': 'in',
                                    'mex':0.5, 'mey':0.5, 'sdx':0.05, 'sdy':0.05}

# Performs a random, elastic distortion on an image.
# grid: 2-10, magnitude: 1-10
config['random_distortion'] = {'probability':0.2, 'grid_width': 5, 'grid_height': 5, 'magnitude': 3}
