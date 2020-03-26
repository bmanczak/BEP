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
from keras.utils import np_utils
from keras.backend import relu, sigmoid
from keras.models import Model
import numpy as np
import time
from keras.models import model_from_json



def model_fn(num_classes = 7):
    vggFace = keras.Sequential()
    vggFace.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    vggFace.add(Convolution2D(64, (3, 3), activation='relu'))
    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(64, (3, 3), activation='relu'))
    vggFace.add(MaxPooling2D((2,2), strides=(2,2)))

    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(128, (3, 3), activation='relu'))
    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(128, (3, 3), activation='relu'))
    vggFace.add(MaxPooling2D((2,2), strides=(2,2)))

    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(256, (3, 3), activation='relu'))
    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(256, (3, 3), activation='relu'))
    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(256, (3, 3), activation='relu'))
    vggFace.add(MaxPooling2D((2,2), strides=(2,2)))

    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
    vggFace.add(MaxPooling2D((2,2), strides=(2,2)))

    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
    vggFace.add(ZeroPadding2D((1,1)))
    vggFace.add(Convolution2D(512, (3, 3), activation='relu'))
    vggFace.add(MaxPooling2D((2,2), strides=(2,2)))

    vggFace.add(Convolution2D(4096, (7, 7), activation='relu'))
    vggFace.add(Dropout(0.5))
    vggFace.add(Convolution2D(4096, (1, 1), activation='relu'))
    vggFace.add(Dropout(0.5))
    vggFace.add(Convolution2D(2622, (1, 1)))
    vggFace.add(Flatten())
    vggFace.add(Activation("softmax"))

    # Load the pre-trained weights
    vggFace.load_weights('/Users/blazejmanczak/Desktop/School/Year3/BEP/VGGFace/vgg_face_weights.h5')

    # Freeze all but fully connected layers
    for layer in vggFace.layers[:-7]:
        layer.trainable = False

    # Take output before the softmax
    model_temp = Model(inputs = vggFace.input, outputs = vggFace.layers[-4].output)
    # And adjust the softmax to match the number of classes
    predictions = Dense(num_classes, activation='softmax')(model_temp.output)
    model = Model(inputs = vggFace.input, outputs = predictions)

    return model
