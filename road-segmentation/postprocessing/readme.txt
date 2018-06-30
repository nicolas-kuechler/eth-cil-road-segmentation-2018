# Postprocessing

To do postprocessing on a batch of predictions obtained by the test images:

	python main_postprocessing <batch_name> <param_set_name>

whereas batch_name should correspond to the name of an output folder in ./../output.
The folder is expected to have the structure: ./../output/<batch_name>/test_output/predictions,
which should contain the prediction .png files.

The param_set_name specifies which parameter set should be used to run the postprocessing.
For an overview, see ./../output/_postprocessing_paramsets.txt.

Example call:

	python main_postprocessing "unet-vanilla" "p3"
	
Which runs postprocessing on the output files of the unet-vanilla model with the parameter set p3.