from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd
import keras
import os
import time


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


def read_data(input_dir="/mnt/server-home/TUE/20175985/BepDataResNet/ManuallyAnnotated",
              output_dir="/mnt/server-home/TUE/20175985/BepDataResNet/npzData"):

    start = time.time()

    # """
    validation_csv = pd.read_csv('/mnt/server-home/TUE/20175985/BepDataResNet/ManuallyAnotatedFileList/validation.csv', sep=',', names=["File_path", 'Face_x', 'Face-y', "Face_width", "Face_height",
                                                                                                                                        'Facial_landmarks', 'Expression', 'Valence', 'Arousal'], index_col="File_path")

    training_csv = pd.read_csv('/mnt/server-home/TUE/20175985/BepDataResNet/ManuallyAnotatedFileList/training.csv', sep=',', names=["File_path", 'Face_x', 'Face-y', "Face_width", "Face_height",
                                                                                                                                    'Facial_landmarks', 'Expression', 'Valence', 'Arousal'], index_col="File_path", header=1)

    # """
    training_csv['Expression'] = pd.to_numeric(training_csv['Expression'])
    validation_csv['Expression'] = pd.to_numeric(validation_csv['Expression'])

    train_and_val = pd.concat([training_csv, validation_csv])
    train_and_val['RowNumber'] = range(1, train_and_val.shape[0]+1)

    found_image_paths = getListOfFiles(input_dir)
    print("Found this many image paths:", len(found_image_paths))

    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    failures = 0  # keeping track of failed imports
    class_count = dict(zip(range(0, 11), [0]*11))  # keep track of label counts

    for path in found_image_paths:

        try:  # image is avaliable

            #print("Checking path", path)
            path_for_csv = "/".join(path.split("/")[-2:])
            label_and_rowNum = train_and_val.loc[path_for_csv, ["Expression", "RowNumber"]]
            #print("Path matched to csv")
            img = load_img(path, target_size=(224, 224))
            #print("Image loaded")
            x = img_to_array(img)
            #print("Image to array")

            # faster way of establishing whether row comes from validation set
            if label_and_rowNum[1] > training_csv.shape[0]:
                X_test.append(x)
                Y_test.append(label_and_rowNum[0])
            else:
                X_train.append(x)
                Y_train.append(label_and_rowNum[0])

            class_count[label_and_rowNum[0]] += 1  # keep track of label counts
            #print("Image has been processed with path", path)
        except:
            print("Failed path", path)
            failures += 1

    X_train = np.asarray(X_train).astype('float16') / 255.  # scaling the images by 255
    Y_train = keras.utils.to_categorical(Y_train, num_classes=11)

    X_test = np.asarray(X_test).astype('float16') / 255.  # scaling the images by 255
    Y_test = keras.utils.to_categorical(Y_test, num_classes=11)

    np.savez_compressed(
        "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/trainDataNotProcessed", X_train=X_train, Y_train=Y_train)
    np.savez_compressed(
        "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/testDataNotProcessed", X_test=X_test, Y_test=Y_test)

    print("There have been {} failures".format(failures))

    sum_all_vals = sum(class_count.values())
    for key in class_count.keys():
        class_count[key] = round(class_count[key] / sum_all_vals, 2)

    print("Test Data size", X_test.shape, Y_test.shape)
    print("Train Data size", X_train.shape, Y_train.shape)

    print("The class distirbution is:", class_count)
    print("The function took", round(time.time() - start, 2), "seconds")


read_data()
