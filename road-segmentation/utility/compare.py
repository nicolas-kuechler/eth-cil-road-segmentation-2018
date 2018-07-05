import os
from PIL import Image
import argparse
import numpy as np
from os import path

'''
    Used to compare the outputs of two trained models. It is required that
    the command main.py model_name test dataset has been run before as
    the overlay images will be used.
    
    Usage:
        compare.py model_folder1 model_folder2
        
    The first model will be represented with a blue mask while the second one will have a yellow one.
    Matching areas will be green.
'''



# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > 0.25:
        return 1
    else:
        return 0

def __build_masks(db):
    patch_size = 16

    patch = {
        0: np.zeros((patch_size,patch_size)),
        1: np.ones((patch_size,patch_size))
    }

    masks = {}
    for (img_id, j, i), label in sorted(db.items()):
        if img_id not in masks:
            masks[img_id] = np.zeros((608, 608))
        masks[img_id][i:i + patch_size, j:j + patch_size] = patch[label]*255
    return masks



output_dir = '../../output/'

parser = argparse.ArgumentParser(description='Comparison script')
parser.add_argument('dir1', metavar='Model 1', type=str,
                    help='')
# this can also be changed
parser.add_argument('dir2', metavar='Model 2', type=str,
                    help='')

args = parser.parse_args()

args.dir1 = path.join(args.dir1, 'test_output', 'masks')
args.dir2 = path.join(args.dir2, 'test_output', 'masks')
images1 = os.listdir(args.dir1)
images2 = os.listdir(args.dir2)
originals = os.listdir('../../data/test_images')

# create directory if it doesn't exist
if not os.path.exists('../../comparisons'):
    os.makedirs('../../comparisons')

for im1, im2, o in zip(images1, images2, originals):
    id = im1.split('_')[1].split('.')[0]
    im1 = Image.open(args.dir1 + '/' + im1)
    im2 = Image.open(args.dir2 + '/' + im2)
    o = Image.open('../../data/test_images/' + o)
    im1 = im1.convert('RGBA')
    im2 = im2.convert('RGBA')
    o = o.convert('RGBA')
    data1 = np.array(im1)
    data2 = np.array(im2)

    data1[:, :, 0] = data1[:, :, 0] * 255
    data2[:, :, 2] = data2[:, :, 2] * 255

    im1 = Image.fromarray(data1)
    im2 = Image.fromarray(data2)

    blended = Image.blend(im1, im2, 0.5)
    blended = Image.blend(o, blended, 0.5)
    blended.save(f'../../comparisons/comp_{id}.png')