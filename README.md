# Brain segmentation

This is a source code for the deep learning segmentation used in the paper `under review` by `authors`.
It employs a U-Net like network for skull stripping and FLAIR abnormality segmentation.
This repository contains a set of functions for data preprocessing (MatLab), training and inference (Python).
Weights for trained models are provided and can be used for deep learning based skull stripping or fine-tuning on a different dataset.
If you use our model or weights, please cite:

```
under review
```

The repository is divided into two folders.
One for skull stripping and one for the FLAIR abnormality segmentation.
They are based on the same model architecture but can be used separately.

## Prerequisites

- MatLab 2016b for pre-processing
- Python 2 with dependencies listed in the `requirements.txt` file
```
sudo pip install requirements.txt
```

## U-Net architecture

The figure below shows a U-Net architecture implemented in this repository.

![unet](images/unet.png)
