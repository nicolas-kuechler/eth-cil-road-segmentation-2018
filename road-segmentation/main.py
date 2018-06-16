import sys, importlib, pickle
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


config = Config()
print('Loading Model: ', config.MODEL_NAME)

# TODO [nku] export the config to some json file

# create dataset
dataset = Dataset(config)
print('Created Dataset')

# create model
model = Model(config, dataset)
print('Created Model')

sess = tf.Session()

if mode == 'train':
    training = Training(sess, config, model)
    print('Created Training')
    training.train()
    print('Finished Training')
elif mode == 'test':
    evaluation = Evaluation(sess, config, model)
    evaluation.eval()
else:
    raise ValueError('mode "{}" unknown.'.format(mode))
