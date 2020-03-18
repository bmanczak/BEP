"""Implements the Keras Sequential model."""

import itertools
import multiprocessing.pool
import threading
from functools import partial

import keras
import pandas as pd
from keras import backend as K
from keras import layers, models
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.backend import relu, sigmoid
import numpy as np
import time

import tensorflow as tf


def model_fn(num_classes):
    # Implements the AlexNetArchitecturex

    # Set up the sequential model
    model = keras.Sequential()

    # First layer: Convolutional layer with max pooling and batch normalization.
    model.add(keras.layers.Conv2D(input_shape=(224, 224, 3),
                                  kernel_size=(11, 11),
                                  strides=(4, 4),
                                  padding="valid",
                                  filters=3,
                                  activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding="valid"))
    model.add(keras.layers.BatchNormalization())

    # Second layer: Convolutional layer with max pooling and batch normalization.
    model.add(keras.layers.Conv2D(kernel_size=(11, 11),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=256,
                                  activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding="valid"))
    model.add(keras.layers.BatchNormalization())

    # Third layer: Convolutional layer with batch normalization.
    model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=384,
                                  activation=tf.nn.relu))
    model.add(keras.layers.BatchNormalization())

    # Fourth layer: Convolutional layer with batch normalization.
    model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=384,
                                  activation=tf.nn.relu))
    model.add(keras.layers.BatchNormalization())

    # Fifth layer: Convolutional layer with max pooling and batch normalization.
    model.add(keras.layers.Conv2D(kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding="valid",
                                  filters=256,
                                  activation=tf.nn.relu))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding="valid"))
    model.add(keras.layers.BatchNormalization())

    # Flatten the output to feed it to dense layers
    model.add(keras.layers.Flatten())

    # Sixth layer: fully connected layer with 4096 neurons with 40% dropout and batch normalization.
    model.add(keras.layers.Dense(units=4096,
                                 activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.4))
    model.add(keras.layers.BatchNormalization())

    # Seventh layer: fully connected layer with 4096 neurons with 40% dropout and batch normalization.
    model.add(keras.layers.Dense(units=4096,
                                 activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.4))
    model.add(keras.layers.BatchNormalization())

    # Eigth layer: fully connected layer with 1000 neurons with 40% dropout and batch normalization.
    model.add(keras.layers.Dense(units=1000,
                                 activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.4))
    model.add(keras.layers.BatchNormalization())

    # Output layer: softmax function of 102 classes of the dataset. This integer should be changed to match
    # the number of classes in your dataset if you change from Oxford_Flowers.
    model.add(keras.layers.Dense(units=num_classes,
                                 activation=tf.nn.softmax))
    return model


def compile_model(model):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def read_test_data(path="/mnt/server-home/TUE/20175985/BepDataResNet/npzData/testDataNotProcessedBalanced7Classes3k.npz"):
    start_time = time.time()
    print("Start Read Test Data")
    data = np.load(path)
    print("Test data read --- %s seconds ---" % (time.time() - start_time))
    print(data)
    X_test = data["X_test"]  # TODO
    Y_test = data["Y_test"]
    print("Testing - Total examples per class", np.sum(Y_test, axis=0))
    return [X_test, Y_test]


def read_train_data(path="/mnt/server-home/TUE/20175985/BepDataResNet/npzData/trainDataNotProcessedBalanced7Classes3k.npz"):
    start_time = time.time()
    print("Start Read Train Data")
    data = np.load(path)
    print("Train data read --- %s seconds ---" % (time.time() - start_time))
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    print("Training - Total examples per class", np.sum(Y_train, axis=0))
    return [X_train, Y_train]
