import sys, importlib, os
import tensorflow as tf
import numpy as np
from core.dataset import Dataset
from core.training import Training
from core.evaluation import Evaluation
from core.submission import Submission
import utility.parser as parser
from utility import util


# Parse Argument to Load Model and Config
model_name = parser.args.model_name
mode = parser.args.mode
data = parser.args.data

model_module = importlib.import_module('models.' + model_name + ".model")
Model = model_module.Model

config_module = importlib.import_module('models.' + model_name + ".config")
Config = config_module.Config

model_name = model_name + '_' + data
config = Config(model_name, data)

print('Setting Output Directory: ', config.OUTPUT_DIR)

print('Loading Model: ', config.MODEL_NAME)

print('\nCreating Dataset...')
dataset = Dataset(config)
print('Dataset Created')

print('\nCreating Model...')
model = Model(config, dataset, mode)
print('Created Model with {} parameters'.format(model.n_params))


if mode == 'train':
    sess = tf.Session()
    training = Training(sess, config, model)
    training.train()

elif mode == 'test':

    patch_sizes = util.to_iter(config.TEST_METHOD_PATCH_SIZE)
    strides = util.to_iter(config.TEST_METHOD_STRIDE)
    rotations = util.to_iter(config.TEST_ROTATION_DEGREE)

    predictions = []
    infos = []

    for patch_size, stride in zip(patch_sizes, strides):
        for rotation in rotations:
            info_str = f'patch_size: {patch_size}, stride: {stride}, rotation: {rotation}'
            print(info_str)

            # update config
            config.TEST_METHOD_PATCH_SIZE = patch_size
            config.TEST_METHOD_STRIDE = stride
            config.TEST_N_PATCHES_PER_IMAGE = (config.TEST_IMAGE_SIZE - patch_size)/ stride + 1
            config.TEST_ROTATION_DEGREE = rotation

            # start evaluation
            sess = tf.Session()
            evaluation = Evaluation(sess, config, model)
            pred_dict = evaluation.eval()

            predictions.append(pred_dict)
            infos.append(info_str)

    submission = Submission(config)

    ids = predictions[0].keys()
    n_pred = len(predictions)

    print(f'Start Averaging the {n_pred} Predictions')

    for id in ids:
        avg_prediction = np.zeros((config.TEST_IMAGE_SIZE, config.TEST_IMAGE_SIZE))
        for info, prediction in zip(infos, predictions):
            avg_prediction += prediction[id]

            if config.SUB_WRITE_INDIVIDUAL_PREDICTIONS:
                out_dir = config.TEST_OUTPUT_DIR + 'individual_predictions/'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                util.to_image(prediction[id]).save(out_dir + f'test_{id}_{info}.png')

        avg_prediction = avg_prediction / float(len(predictions))

        # could add postprocessing here
        submission.add(prediction=avg_prediction, img_id=id)

    submission.write()

else:
    raise ValueError('mode "{}" unknown.'.format(mode))
