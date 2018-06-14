import tensorflow as tf

from dataset.dataset import Dataset
from dataset.ds_config import ds_config
from training.training import Training
from training.training_config import train_config

from models.simple.simple_model import SimpleModel

# Load config
config = {
    'dataset_config': ds_config,
    'model_config': {},
    'training_config' : train_config,
    'evaluation_config' :{}
}

# create dataset
dataset = Dataset(config['dataset_config'])
print('Created Dataset')

# create model
model = SimpleModel(config['model_config'], dataset)
print('Created Model')


# create Training
sess = tf.Session()
training = Training(sess, config['training_config'], model)
print('Created Training')

training.train()
print('Finished Training')


# create Evaluation
#evaluation = evaluation(sess, evaluation_config, model)
#evaluation.eval()
