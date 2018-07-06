# CIL Semester Project: Road Segmentation


Segmenting an image consists in partitioning an image into multiple segments (formally one has to assign a class label to each pixel). A simple baseline is to partition an image into a set of patches and classify every patch according to some simple features (average intensity). Although this can produce reasonable results for simple images, natural images typically require more complex procedures that reason abut the entire image or very large windows.

For this problem, we provide a set of satellite/aerial images acquired from GoogleMaps. We also provide ground-truth images where each pixel is labeled as {road, background}. Your goal is to train a classifier to segment roads in these images, i.e. assign a label {road=1, background=0} to each pixel.

## Getting started

#### Requirements

Using python 3.6 run the following command to install the dependencies

```
pip install -r requirements.txt
```

Please make sure that the data folder contains the required images for training and testing. Otherwise it is possbile to download them from this repository.

#### Running a model

The following command should be run at the road-segmentation directory level.

```
python main.py model_name mode dataset
```

Where `model_name` is the desired model to use in road-segmentation/models/{model_name}.

`mode` can be either **train** or **test**. The latter generates the predictions for the test images.

`dataset` can be **default** for the originally provided dataset. **ext-half** for using the default and half the additional dataset or **ext-full**
for the complete additional dataset.

The datasets can be found under the data folder. There are validation and training images for each dataset. The additional datasets can be identified by the
suffix *_extension_full* *_extension_half*.

This will generate the model's files corresponding the selected mode in output/{model_name}_{dataset}.

#### Postprocessing

The following command should be run at the road-segmentation directory level.

```
python main_postprocessing.py model_folder config_name
```

Where `model_name_folder` is the desired model to use from output/{model_name}_{dataset}. Note that this has been generated by the previous command using the train mode.

`config_name` is one of the already defined 9 configurations road_segmentation/postprocessing/pp_config_{config_name}.py. The recommended option is p1.
Notice that new configurations could be added by creating a file in the postprocessing folder using the name pp_config_{config_name}.py

This will generate folder with the corresponding files in output/{model_name_folder}_post_{config_name}.

#### Averaging models

The following command should be run at the road-segmentation directory level.

```
python average.py model_folder1 model_folder2
```

Where `model_folder1` and `model_folder2` are the models generated by the main.py train or main_postprocessing.py scripts.

This will generate a folder in output/{model_folder1}--{model_folder2}--avg.


#### Reproducing Kaggle results

As the training and also the evaluation process takes a long time, we recommend running the following commands
on the Leonhard cluster using the following configurations:

```
bsub -n 10 -q "gpu.24h" -R "rusage[mem=10000,ngpus_excl_p=1]" COMMAND
```

Run the following commands to **exactly** reproduce the results (as on Kaggle), using the provided trained models:

```
# We need to be at the road-segmentation level
cd road-segmentation

# vanilla-unet
python main.py vanilla-unet test ext-full
python main_postprocessing.py vanilla-unet_ext-full p1


# squeezenet-encoder
python main.py squeezenet-encoder-dropouts test ext-half
python main_postprocessing.py squeezenet-encoder-dropouts_ext-half p1

# average
python average.py squeezenet-encoder-dropouts_ext-half_post_p1 vanilla-unet_ext-full_post_p1
```
Or run the `create_exact_submission.sh` file.


Run the following commands to reproduce the results by training the models.
**Please notice that this will delete the previous trained model** and it will be
not be possible to exactly reproduce the results from the provided saved models.


```
# We need to be at the road-segmentation level
cd road-segmentation

# We need to remove the previous trained models if any
rm -r ../output/vanilla-unet_ext-full
rm -r ../output/squeezenet-encoder-dropouts_ext-half

# vanilla-unet
python main.py vanilla-unet train ext-full
python main.py vanilla-unet test ext-full
python main_postprocessing.py vanilla-unet_ext-full p1


# squeezenet-encoder
python main.py squeezenet-encoder-dropouts train ext-half
python main.py squeezenet-encoder-dropouts test ext-half
python main_postprocessing.py squeezenet-encoder-dropouts_ext-half p1

# average
python average.py squeezenet-encoder-dropouts_ext-half_post_p1 vanilla-unet_ext-full_post_p1
```
Or run the `create_submission.sh` file.

## Authors

* **Sabina Fischlin** - [inafischlin](https://gitlab.com/inafischlin)
* **Octavio Martínez** - [octmb](https://gitlab.com/octmb)
* **Nicolas Küchler** - [nicolas-kuechler](https://gitlab.com/nicolas-kuechler)