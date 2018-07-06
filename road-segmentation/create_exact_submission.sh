#!/usr/bin/env bash

# vanilla-unet training full dataset
echo "Producing predictions for vanilla unet"
python main.py vanilla-unet test ext-full

echo "Postprocessing vanilla unet results"
python main_postprocessing.py vanilla-unet_ext-full p1


# squeezenet-encoder
echo "Producing predictions for squeezenet"
python main.py squeezenet-encoder-dropouts test ext-half

echo "Postprocessing squeezenet results"
python main_postprocessing.py squeezenet-encoder-dropouts_ext-half p1


# average
echo "Averaging post-processed results from both networks"
python average.py squeezenet-encoder-dropouts_ext-half_post_p1 vanilla-unet_ext-full_post_p1