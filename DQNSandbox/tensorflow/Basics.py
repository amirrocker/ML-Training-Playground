import pathlib
import tensorflow as tf
import pandas as pd
import numpy as np

#from tensorflow.data import Dataset

# set the print options for numpy
np.set_printoptions(precision=5)

# basic datastructures
# a Dataset is a python iterable
dataset = tf.data.Dataset.from_tensor_slices([1, 3, 5, 6, 7, 8, 2, 4, 6])
print(dataset)

it = iter(dataset)
print(next(it).numpy())
