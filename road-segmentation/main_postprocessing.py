# Main script to run post-processing on an output of a model.
# Takes as an argument "batch_name" which should correspond to the name the folder in
# ./../output on which one wants to run the post-processing. Expectes the folder to have
# the following structure: <batch_name>/test_output/predictions.
# Dependent on the second parameter "param_set_name", a config module is loaded, expected to
# have to reside in ./postprocessing/pp_config_<param_set_name>
import sys, os
import numpy as np
import argparse, importlib
from PIL import Image
from postprocessing.postprocessing import Postprocessing
from core.submission import Submission
from utility import util


# parse argument for batch name and parameter set name and import corresponding module
parser = argparse.ArgumentParser(description='Options for Postprocessing')
parser.add_argument('batch_name', metavar='Batch Name', type=str,
                    help='Name of batch predictions to be processed')
parser.add_argument('param_set_name', metavar='Parameter Set Name', type=str,
                    help='Name of parameter set')

args = parser.parse_args()
batch_name = args.batch_name
param_set_name = args.param_set_name

# load config module
config_module = importlib.import_module( 'postprocessing.pp_config_' + param_set_name )
Config = config_module.Config

# create Config and Postprocessing
config = Config( batch_name )
post = Postprocessing(config)
submission = Submission(config)

path_to_test_preds = config.POST_PREDICTIONS_IN_DIR
path_to_test_imgs  = config.TEST_IMAGES_IN_DIR
out_dir = config.OUTPUT_DIR
if not os.path.exists( out_dir ):
	os.makedirs( out_dir )

# aux function
def get_id_from_filename( file_name ):
	int_list = list(filter(str.isdigit, f))
	result = ''
	for el in int_list:
		result += str( el )
	return int( result )

# do postprocessing on all files in the selected batch
count = 0
print ( f'Starting postprocessing of {batch_name} with parameter set {param_set_name}...' )
for f in os.listdir( path_to_test_preds ):
	if count == config.POST_MAX_NUM_IMAGES_TOPROCESS:
		break
	ext = os.path.splitext(f)[1]
	if ext.lower() != '.png':
		continue
	pred_img = Image.open( os.path.join( path_to_test_preds, f ) )
	pred_arr = util.to_array( pred_img )

	id = get_id_from_filename( f )
	img = Image.open( path_to_test_imgs + f'test_{id}.png' )
	img_arr = np.asarray( img )
	
	if config.POST_DO_CRFPROCESSING:
		print ( f'  processing {f}...' )
		processed_pred = post.crf( img_arr, pred_arr )
	
		# add to submission
		if config.POST_WRITE_SUBMISSION:
			submission.add(prediction=processed_pred, img_id=id)
		else:
			processed_img = util.to_image( processed_pred )
			processed_img.save( out_dir + f'post_{id}.png' )
		count += 1

print ( f'Postprocessing finished (processed {count} predictions)' )
if config.POST_WRITE_SUBMISSION:
	submission.write()