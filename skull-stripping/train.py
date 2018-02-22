from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from data import load_data
from net import dice_coef
from net import dice_coef_loss
from net import unet

train_images_path = './data/train/'
valid_images_path = './data/valid/'
weights_path = '.'
log_path = '.'

gpu = '0'

epochs = 128
batch_size = 32
base_lr = 1e-5


def train():
    imgs_train, imgs_mask_train, _ = load_data(train_images_path)

    mean = np.mean(imgs_train)
    std = np.std(imgs_train)

    imgs_train -= mean
    imgs_train /= std

    imgs_valid, imgs_mask_valid, _ = load_data(valid_images_path)

    imgs_valid -= mean
    imgs_valid /= std

    model = unet()

    optimizer = Adam(lr=base_lr)
    model.compile(optimizer=optimizer,
                  loss=dice_coef_loss,
                  metrics=[dice_coef])

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    training_log = TensorBoard(log_dir=log_path)

    model.fit(imgs_train, imgs_mask_train,
              validation_data=(imgs_valid, imgs_mask_valid),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              callbacks=[training_log])

    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    model.save_weights(os.path.join(
        weights_path, 'weights_{}.h5'.format(epochs)))


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    if len(sys.argv) > 1:
        gpu = sys.argv[1]
    device = '/gpu:' + gpu

    with tf.device(device):
        train()
