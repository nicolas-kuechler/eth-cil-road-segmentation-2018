import sys, importlib, os
import tensorflow as tf
from core.dataset import Dataset
from core.training import Training
from core.evaluation import Evaluation


# Parse Argument to Load Model and Config
model_name = str(sys.argv[1])
mode = str(sys.argv[2])

model_module = importlib.import_module('models.' + model_name + ".model")
Model = model_module.Model

config_module = importlib.import_module('models.' + model_name + ".config")
Config = config_module.Config


config = Config(model_name)

print('Setting Output Directory: ', config.OUTPUT_DIR)





print('Loading Model: ', config.MODEL_NAME)

# TODO [nku] create and set output folder in config if not there yet (in models/model_name/output)

# TODO [nku] export the config to some json file

print('\nCreating Dataset...')
dataset = Dataset(config)
print('Dataset Created')

print('\nCreating Model...')
model = Model(config, dataset, mode)
print('Model Created')

sess = tf.Session()

if mode == 'train':
    training = Training(sess, config, model)
    training.train()


elif mode == 'test':
    evaluation = Evaluation(sess, config, model)
    model.load() # load the model
    evaluation.eval()
else:
    raise ValueError('mode "{}" unknown.'.format(mode))
