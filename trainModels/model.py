import itertools
import multiprocessing.pool
import threading
from functools import partial

import keras
import pandas as pd
from keras import backend as K
from keras import layers, models
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D, Activation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
import numpy as np
import time
import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras_vggface import utils  # pre-processing
import cv2


def model_fn(num_classes, arch):
    # Implements the AlexNetArchitecturex
    if arch == "alexNet":
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

    elif arch == 'vggFace':
        vggFace = keras.Sequential()
        vggFace.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        vggFace.add(Convolution2D(64, (3, 3), activation='relu'))
        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(64, (3, 3), activation='relu'))
        vggFace.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(128, (3, 3), activation='relu'))
        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(128, (3, 3), activation='relu'))
        vggFace.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(256, (3, 3), activation='relu'))
        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(256, (3, 3), activation='relu'))
        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(256, (3, 3), activation='relu'))
        vggFace.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
        vggFace.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
        vggFace.add(ZeroPadding2D((1, 1)))
        vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
        vggFace.add(MaxPooling2D((2, 2), strides=(2, 2)))

        vggFace.add(Convolution2D(4096, (7, 7), activation='relu'))
        vggFace.add(Dropout(0.4))
        vggFace.add(Convolution2D(4096, (1, 1), activation='relu'))
        vggFace.add(Dropout(0.4))
        vggFace.add(Convolution2D(2622, (1, 1)))
        vggFace.add(Flatten())
        vggFace.add(Activation("softmax"))

        # Load the pre-trained weights
        vggFace.load_weights('/mnt/server-home/TUE/20175985/vgg_face_weights.h5')

        # Freeze all but fully connected layers
        for layer in vggFace.layers[:-7]:
            layer.trainable = False

        # Take output before the softmax
        model_temp = Model(inputs=vggFace.input, outputs=vggFace.layers[-4].output)
        # And adjust the softmax to match the number of classes
        flatten = Flatten()(model_temp.output)
        predictions = Dense(num_classes, activation='softmax')(flatten)
        model = Model(inputs=vggFace.input, outputs=predictions)

    elif arch == "senet50":
        # Convolution Features
        base_senet = VGGFace(model="senet50", include_top=False, input_shape=(
            224, 224, 3), pooling='max')  # pooling: None, avg or max
        last_layer = base_senet.get_layer("avg_pool").output
        x = Flatten(name="flatten")(last_layer)
        x = Dense(4096, activation="relu", name="dense2048")(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation="relu", name="dense1028")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, name="classifer", activation="softmax")(x)

        model = Model(base_senet.input, outputs)
        for layer in model.layers[:-5]:
            layer.trainable = False

    elif arch == "resnet50":
        # Convolution Features
        base_senet = VGGFace(model="resnet50", include_top=False, input_shape=(
            224, 224, 3), pooling='max')  # pooling: None, avg or max
        last_layer = base_senet.get_layer("avg_pool").output
        x = Flatten(name="flatten")(last_layer)
        x = Dense(4096, activation="relu", name="dense2048")(x)
        x = Dropout(0.3)(x)
        x = Dense(4096, activation="relu", name="dense1024")(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, name="classifer", activation="softmax")(x)

        model = Model(base_senet.input, outputs)

        for layer in model.layers[:-5]:
            layer.trainable = False

    return model


def clahe(img_array):
    # Converting image to LAB Color model
    lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # return l
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    #lab[...,0] = clahe.apply(lab[...,0])
    # return lab
    return final


def image_preprocessing(img):
    #pixels = img
    pixels = clahe((img*255).astype(np.uint8))
    pixels_expanded = np.expand_dims(pixels.astype(np.float64), axis=0)
    pre_pro = utils.preprocess_input(pixels_expanded, version=1)  # version 1 for VGG face
    return pre_pro[0]/255


def balanced_class_weights(data):
    # stores the class of instance at each index key =index, value = class
    dic_temp = dict(enumerate(np.where(data[1] == 1)[1]))
    indices_of_classes = {i: [] for i in range(data[1][0].shape[0])}
    # augment_labels = list(class_prop.keys())

    for key, val in dic_temp.items():
        indices_of_classes[val].append(key)

    per_classes = {key: len(val) for key, val in indices_of_classes.items()}
    num_of_samples = sum(val for val in per_classes.values())
    num_of_classes = sum(key for key in per_classes.keys())

    return {key: (num_of_samples / (num_of_classes * val)) for key, val in per_classes.items()}


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
