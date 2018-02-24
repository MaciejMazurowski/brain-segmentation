# FLAIR abnormality segmentation in brain MRI

This folder contains an implementation of a FLAIR abnormality segmentation in brain MRI using U-Net like architecture.
It can be used to preprocess MRI images, train the network for tumor segmentation or obtain segmentation predictions of  FLAIR abnormality in a custom dataset (not expected to work outstandingly without prior fine-tuning).


### Usage

#### Preprocessing
You need to have a folder with images preprocessed using provided matlab function `preprocessing3D.m`.
It rescales them to have spacial dimensions 256x256 and performs contrast normalization.
The preprocessing is done for a 3D volume of slices yet for each MRI modality separately.
Refer to the documentation of `preprocessing3D.m` function for more details.
The main requirement for the following steps is to have image names in format `<case_id>_<slice_number>.tif` and corresponding masks named `<case_id>_<slice_number>_mask.tif`.

#### Training
The training script `train.py` has variables defined at the top that you need to set:

- `train_images_path` - folder containing training images
- `valid_images_path` - folder containing validation images
    
Other variables can be changed to adjust some training parameters.
One of them, `init_weights_path` allows to initialize the network with pre-trained weights, e.g. provided by us or from the model trained for skull stripping (in this case make sure to have the same training/test splits for both tasks).
The models described in our paper were trained from scratch.

Then, run the training using
```
python train.py
```

#### Testing
To run the inference, you need to set up some variables defined at the top of the `test.py` script:

- `weights_path` - path to the trained weights
- `train_images_path` - folder containing training images to compute the mean and standard deviation for data normalization; if you pass your own  mean and standard deviation to the `test` function, this variable is not used
- `test_images_path` - folder with test images for prediction; it must contain corresponding mask files as well, however, they can be dummy (all zeros)
- `predictions_path` - folder for saving predicted and ground truth segmentation outlines (will be created if it doesn't exist)

When all variables are set up, run the inference using
```
python test.py
```


### Results

```
TODO
```
