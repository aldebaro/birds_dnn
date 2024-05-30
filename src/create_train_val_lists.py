'''Assumes this file structure:
https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
and this Keras pipeline:
https://keras.io/examples/vision/image_classification_from_scratch/

See, for explanation about convnet and filters:
https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convolution2d
and
http://cs231n.github.io/convolutional-networks/
'''
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import tensorflow as tf
from tensorflow import keras

# one can disable the imports below if not plotting / saving
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

epochs = 100  # number of epochs (passing over training set)
numClasses = 8  # total number of labels
# from keras pipeline tutorial:
# should approximately match the actual aspect ratio of the images that is approximately=2
num_rows = 800
num_columns = 1500
image_size = (num_rows, num_columns)
batch_size = 1
dropout_probability = 0.01  # to combat overfitting

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/ak/Work/mayron_bird/train_audio",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "C:/ak/Work/mayron_bird/train_audio",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
print("val_ds", val_ds)
print("train_ds", train_ds)
print("val_ds.cardinality().numpy()=", val_ds.cardinality().numpy())
print("train_ds.cardinality().numpy()=", train_ds.cardinality().numpy())
