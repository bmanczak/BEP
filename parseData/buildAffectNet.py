from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd
import keras
import os
import time
from Preprocessing import *
from mtcnn.mtcnn import MTCNN  # face detector, install with pip install mtcnn
import cv2
import argparse


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def read_data(input_dir,
              output_dir,
              prepro,
              start_ind,
              end_ind):

    # """
    validation_csv = pd.read_csv('/mnt/server-home/TUE/20175985/BepDataResNet/ManuallyAnotatedFileList/validation.csv', sep=',', names=["File_path", 'Face_x', 'Face-y', "Face_width", "Face_height",
                                                                                                                                        'Facial_landmarks', 'Expression', 'Valence', 'Arousal'], index_col="File_path")

    training_csv = pd.read_csv('/mnt/server-home/TUE/20175985/BepDataResNet/ManuallyAnotatedFileList/training.csv', sep=',', names=["File_path", 'Face_x', 'Face-y', "Face_width", "Face_height",
                                                                                                                                    'Facial_landmarks', 'Expression', 'Valence', 'Arousal'], index_col="File_path", header=1)

    # """
    training_csv['Expression'] = pd.to_numeric(training_csv['Expression'])
    validation_csv['Expression'] = pd.to_numeric(validation_csv['Expression'])

    training_csv = training_csv[training_csv['Expression'].isin([0, 1, 2, 3, 4, 5, 6])]
    validation_csv = validation_csv[validation_csv['Expression'].isin([0, 1, 2, 3, 4, 5, 6])]

    train_and_val = pd.concat([training_csv, validation_csv])
    train_and_val['RowNumber'] = range(1, train_and_val.shape[0]+1)

    # train_and_val = train_and_val[train_and_val['Expression'].isin(
    #    [0, 1, 2, 3, 4, 5, 6])]

    found_image_paths = getListOfFiles(input_dir)
    print("Found this many image paths:", len(found_image_paths))
    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    failures = 0  # keeping track of failed imports
    failures_prepro = 0
    class_count = dict(zip(range(0, 7), [0]*7))  # keep track of label counts
    if prepro == True:
        detector = MTCNN()
    start = time.time()
    for path in found_image_paths[start_ind: end_ind]:

        try:  # image is avaliable

            #print("Checking path", path)
            path_for_csv = "/".join(path.split("/")[-2:])
            label_and_rowNum = train_and_val.loc[path_for_csv, ["Expression", "RowNumber"]]
            #print("Path matched to csv")
            img = load_img(path)
            #print("Image loaded")
            x = img_to_array(img).astype(np.uint8)
            #print("Image to array")

            if prepro == True:
                try:
                    x = applyPreprocessing(img_array=x, detector=detector)
                except Exception as e:
                    print("Failed MTCNN preprocessing", e, "with path", path)
                    failures_prepro += 1

            x = cv2.resize(x, (224, 224))
            # faster way of establishing whether row comes from validation set
            if label_and_rowNum[1] > training_csv.shape[0]:
                X_test.append(x)
                Y_test.append(label_and_rowNum[0])
            else:
                X_train.append(x)
                Y_train.append(label_and_rowNum[0])

            class_count[label_and_rowNum[0]] += 1  # keep track of label counts
            #print("Image has been processed with path", path)
        except Exception as e:
            print("Failed to preprocess", path, "with exception", e)
            #print("Failed path", path)
            failures += 1

    X_train = np.asarray(X_train).astype('float16') / 255.  # scaling the images by 255
    Y_train = keras.utils.to_categorical(Y_train, num_classes=7)

    X_test = np.asarray(X_test).astype('float16') / 255.  # scaling the images by 255
    Y_test = keras.utils.to_categorical(Y_test, num_classes=7)

    np.savez_compressed(
        os.path.join("/mnt/server-home/TUE/20175985/BepDataResNet/npzData/", "trainDataProcessed7Classes" + str(start_ind) + "To" + str(end_ind)), X_train=X_train, Y_train=Y_train)
    np.savez_compressed(
        os.path.join("/mnt/server-home/TUE/20175985/BepDataResNet/npzData/", "testDataProcessed7Classes" + str(start_ind) + "To" + str(end_ind)), X_test=X_test, Y_test=Y_test)

    print("There have been {} failures".format(failures))
    print("There have been {} failures in preprocessing".format(failures_prepro))

    sum_all_vals = sum(class_count.values())
    for key in class_count.keys():
        class_count[key] = round(class_count[key] / sum_all_vals, 2)

    print("Test Data size", X_test.shape, Y_test.shape)
    print("Train Data size", X_train.shape, Y_train.shape)

    print("The class distirbution is:", class_count)
    print("The function took", round(time.time() - start, 2), "seconds")


if __name__ == "__main__":
    print("Mains")
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        required=False,
                        type=str,
                        default="/mnt/server-home/TUE/20175985/BepDataResNet/ManuallyAnnotated",
                        help='Directory of images')

    parser.add_argument("--output-dir",
                        required=False,
                        type=str,
                        default="/mnt/server-home/TUE/20175985/BepDataResNet/npzData",
                        help='Directory of to save to ')

    parser.add_argument("--start-ind",
                        required=True,
                        type=int,
                        help='Index to start')

    parser.add_argument("--end-ind",
                        required=True,
                        type=int,
                        help='Index to finish')

    parser.add_argument("--prepro",
                        required=False,
                        type=bool,
                        default=True,
                        help="Do preprocessing?")

    parse_args, unknown = parser.parse_known_args()

    read_data(**parse_args.__dict__)
