import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

np.set_printoptions(suppress=True)

# disable loading bar
tfds.disable_progress_bar()

logger = tf.get_logger()
logger.setLevel(logging.DEBUG)

# load the dataset from an url
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)
base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
print(base_dir)

# create training and testing dirs
train_dir = os.path.join(base_dir, 'train')
print(train_dir)
