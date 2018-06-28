import sys, os
import numpy as np
import argparse
from PIL import Image
from postprocessing.postprocessing import Postprocessing
from postprocessing.postprocessing_config import Config
from core.submission import Submission
from utility import util


# parse argument for batch name
parser = argparse.ArgumentParser(description='Options for Postprocessing')
parser.add_argument('batch_name', metavar='Batch Name', type=str,
                    help='Name of batch predictions to be processed')
args = parser.parse_args()
batch_name = args.batch_name

# create Config and Postprocessing
config = Config( batch_name )
post = Postprocessing()
submission = Submission(config)

path_to_test_preds = config.PREDICTIONS_IN_DIR
path_to_test_imgs  = config.TEST_IMAGES_IN_DIR
out_dir = config.OUTPUT_DIR
if not os.path.exists( out_dir ):
	os.makedirs( out_dir )

# aux
def get_id_from_filename( file_name ):
	int_list = list(filter(str.isdigit, f))
	result = ''
	for el in int_list:
		result += str( el )
	return int( result )

count = 0
# do postprocessing on all files in the selected batch
print ( f'Starting postprocessing...' )
for f in os.listdir( path_to_test_preds ):
	if count > config.MAX_NUM_PROCESSES:
		break
	ext = os.path.splitext(f)[1]
	if ext.lower() != '.png':
		continue
	pred_img = Image.open( os.path.join( path_to_test_preds, f ) )
	pred_arr = util.to_array( pred_img )

	#img_id = int(filter(str.isdigit, f))
	id = get_id_from_filename( f )
	img = Image.open( path_to_test_imgs + f'test_{id}.png' )
	img_arr = util.to_array( img )
	processed_pred = post.crf( img_arr, pred_arr )
	
	# add to submission
	if config.WRITE_SUBMISSION:
		submission.add(prediction=processed_pred, img_id=id)
	else:
		processed_img = util.to_image( processed_pred )
		processed_img.save( out_dir + f'post_{id}.png' )
		# for debugging
		#if id == 10:
		#	processed_img.show()
	count += 1

print ( f'Postprocessing finished' )
if config.WRITE_SUBMISSION:
	submission.write()