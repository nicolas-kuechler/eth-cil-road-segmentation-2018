import argparse

parser = argparse.ArgumentParser(description='Options for Road Segmentation Models')

# model name
parser.add_argument('model_name', metavar='Model Name', type=str,
                    help='Model name to use for the checkpoint')

# mode
parser.add_argument('mode', metavar='Mode', type=str, choices=['train', 'test'],
                    help='Choose either to train or test the model')


args = parser.parse_args()