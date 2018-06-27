import argparse

parser = argparse.ArgumentParser(description='Options for Road Segmentation Models')

# model name
parser.add_argument('model_name', metavar='Model Name', type=str,
                    help='Model name to use for the checkpoint')

# mode
parser.add_argument('mode', metavar='Mode', type=str, choices=['train', 'test'],
                    help='Choose either to train or test the model')

# dataset
parser.add_argument('data', metavar='Data ', type=str, choices=['default', 'ext'],
                    help='Choose either the default or ext dataset')

args = parser.parse_args()
