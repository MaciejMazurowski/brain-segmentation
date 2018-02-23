from __future__ import print_function

import numpy as np
import os

from skimage.io import imread
from skimage.transform import rescale
from skimage.transform import rotate

image_rows = 256
image_cols = 256

channels = 1    # refers to neighboring slices; if set to 3, takes previous and next slice as additional channels
modalities = 3  # refers to pre, flair and post modalities; if set to 3, uses all and if set to 1, only flair


def load_data(path):
    """
    Assumes filenames in given path to be in the following format as defined in `preprocessing3D.m`:
    for images: <case_id>_<slice_number>.tif
    for masks: <case_id>_<slice_number>_mask.tif

        Args:
            path: string to the folder with images

        Returns:
            np.ndarray: array of images
            np.ndarray: array of masks
            np.chararray: array of corresponding images' filenames without extensions
    """
    images_list = os.listdir(path)
    total_count = len(images_list) / 2
    images = np.ndarray((total_count, image_rows, image_cols,
                         channels * modalities), dtype=np.uint8)
    masks = np.ndarray((total_count, image_rows, image_cols), dtype=np.uint8)
    names = np.chararray(total_count, itemsize=64)

    i = 0
    for image_name in images_list:
        if 'mask' in image_name:
            continue

        names[i] = image_name.split('.')[0]
        slice_number = int(names[i].split('_')[-1])
        patient_id = '_'.join(names[i].split('_')[:-1])

        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(path, image_name), as_grey=(modalities == 1))
        img_mask = imread(os.path.join(path, image_mask_name), as_grey=True)

        if channels > 1:
            img_prev = read_slice(path, patient_id, slice_number - 1)
            img_next = read_slice(path, patient_id, slice_number + 1)

            img = np.dstack((img_prev, img[..., np.newaxis], img_next))

        elif modalities == 1:
            img = np.array([img])

        img_mask = np.array([img_mask])

        images[i] = img
        masks[i] = img_mask

        i += 1

    images = images.astype('float32')
    masks = masks[..., np.newaxis]
    masks = masks.astype('float32')
    masks /= 255.

    return images, masks, names


def oversample(images, masks):
    """
    Repeats 2 times every slice with nonzero mask

        Args:
            np.ndarray: array of images
            np.ndarray: array of masks

        Returns:
            np.ndarray: array of oversampled images
            np.ndarray: array of oversampled masks
    """
    images_o = []
    masks_o = []
    for i in range(len(masks)):
        if np.max(masks[i]) < 1:
            continue

        for _ in range(2):
            images_o.append(images[i])
            masks_o.append(masks[i])

    images_o = np.array(images_o)
    masks_o = np.array(masks_o)

    return np.vstack((images, images_o)), np.vstack((masks, masks_o))



def read_slice(path, patient_id, slice):
    img = np.zeros((image_rows, image_cols))
    img_name = patient_id + '_' + str(slice) + '.tif'
    img_path = os.path.join(path, img_name)

    try:
        img = imread(img_path, as_grey=(modalities == 1))
    except Exception:
        pass

    return img[..., np.newaxis]
