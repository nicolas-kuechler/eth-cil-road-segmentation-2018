#!/usr/bin/env bash

# vanilla-unet training full dataset
echo "Starting to train the vanilla unet with full extended dataset"
python main.py vanilla-unet test ext-full

echo "Producing predictions for vanilla unet"
python main.py vanilla-unet test ext-full

echo "Postprocessing the vanilla unet results"
python main_postprocessing.py vanilla-unet_ext p1


# squeezenet-encoder
echo "Starting to train squeezenet encoder with half the extended dataset"
python main.py squeezenet-encoder-dropouts train ext-half
python main.py squeezenet-encoder-dropouts test ext-half
python main_postprocessing.py squeezenet-encoder-dropouts_ext-half p1


# average
echo "Averaging post-processed results from both networks"
python average.py squeezenet-encoder-dropouts_ext-half_post_p1 vanilla-unet_ext_post_p1