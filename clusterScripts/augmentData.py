import model
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.image as mpimg
import os
import random
import time
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.utils import shuffle


def random_noise(image):
    noise_typ = np.random.choice(["gauss", "s&p", "poisson", "none"], 1, p=[0.1, 0.1, 0.1, 0.7])[0]
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt modex
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="none":
        return image



def augment_data(data, class_prop = {2:3, 3:5, 4:11, 5:15, 6:3}):

    print("Number of input examples:", data[1].shape[0])

    data[0] = data[0][0:10_000]
    data[1] = data[1][0:10_000]
    dic_temp = dict(enumerate(np.where(data[1] == 1)[1])) # stores the class of instance at each index key =index, value = class
    indices_of_classes = {i:[] for i in range(data[1][0].shape[0])}
    augment_labels = list(class_prop.keys())

    for key, val in dic_temp.items():
        indices_of_classes[val].append(key)

    datagen = ImageDataGenerator(zoom_range=[0.7,1], brightness_range=[0.8,1.1], rotation_range = 30, horizontal_flip=True, shear_range = 15,
                            height_shift_range = 0.05, width_shift_range = 0.05, preprocessing_function = random_noise, rescale = 1/255)

    X_train_out = np.ones((1,224,224,3))
    Y_train_out = np.ones((1,7))
    for label in augment_labels:

        data_one_class = [data[0][indices_of_classes[label]]*255, data[1][indices_of_classes[label]] ]
        batch_size = 1024
        #X_train = []
        #Y_train = []
        # prepare iterator
        it = datagen.flow(data_one_class[0], data_one_class[1], batch_size=batch_size)
        batches = 0
        times = class_prop[label]
        class_size = len(indices_of_classes[0])

        for x_batch, y_batch in it:
            #X_train.append(x_batch)
            #Y_train.append(y_batch)
            #print(x_batch.shape)
            X_train_out = np.concatenate((X_train_out, x_batch), axis = 0)
            Y_train_out = np.concatenate((Y_train_out, y_batch), axis = 0)
            batches += 1
            if batches >= (class_size/batch_size)*times:
                break
    print("Number of output examples:", X_train_out[1:].shape[0])
    print("The dataset has been enlarged by {} %".format(round((X_train_out[1:].shape[0]/data[1].shape[0])*100, 2)))
    #print(X_train_out.shape, Y_train_out.shape)
    return X_train_out[1:], Y_train_out[1:]

# Creating a training train set of 7 classes of all images
start = time.time()
full_npz_train_path = "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/trainDataNotProcessedFull7Classes.npz"
train_df = model.read_train_data(path = full_npz_train_path)
out_train = augment_data(train_df)
np.savez_compressed(
    "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/trainDataAugFull7Classes.npz", X_train = out_train[0],
     Y_train = out_train[1])

print("The process took", round(time.time() - start, 3), "seconds")
