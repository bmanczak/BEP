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
    """ create a list of files and sub directories """
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


def read_data(img_path,
              txt_path,
              path_bounding_box,
              output_dir,
              prepro):

    data = pd.read_csv(txt_path, sep=' ', header=None)
    data.columns = ["Path", "Expression"]
    data['Status'] = data['Path'].apply(lambda row: row.split("_")[0])
    data['FileName'] = data['Path'].apply(lambda row: row.split(".")[0])

    rafToAffect = {1: 3, 2: 4, 3: 5, 4: 1, 5: 2, 6: 6, 7: 0}
    data['Expression'] = data['Expression'].apply(lambda row: rafToAffect[row])
    data['Path'] = data['Path'].apply(lambda row: os.path.join(img_path, row))
    data.set_index('Path', drop=True, inplace=True)

    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    failures = 0  # keeping track of failed imports
    class_count = dict(zip(range(0, 7), [0]*7))
    detector = MTCNN()

    start = time.time()

    for path in data.index:
        try:
            # print(path)
            img = load_img(path)
            x = img_to_array(img).astype(np.uint8)

            if prepro == True:
                try:
                    x = applyPreprocessing(img_array=x, detector=detector)
                    #print("Processed path ", path)
                except:
                    box = open(os.path.join(path_bounding_box,
                                            data.loc[path, "FileName"] + "_boundingbox.txt"), "r")
                    (X, Y, W, H) = [round(float(i)) for i in box.read().split(" ")[:-1]]
                    x = x[Y:Y+H, X:X+W]
                    #print("failed preprocessing MTCNN. Sucess with bounding box")

            x = cv2.resize(x, (224, 224))

            if data.loc[path, 'Status'] == 'test':
                X_test.append(x)
                Y_test.append(data.loc[path, "Expression"])
            else:
                X_train.append(x)
                Y_train.append(data.loc[path, "Expression"])

            class_count[data.loc[path, "Expression"]] += 1  # keep track of label counts
        except Exception as e:
            print("failed for path", path, "with error", e)
            failures += 1

    X_train = np.asarray(X_train).astype('float16') / 255.  # scaling the images by 255
    Y_train = keras.utils.to_categorical(Y_train, num_classes=7)

    X_test = np.asarray(X_test).astype('float16') / 255.  # scaling the images by 255
    Y_test = keras.utils.to_categorical(Y_test, num_classes=7)

    np.savez_compressed(
        os.path.join(output_dir, "trainDataProcessed"), X_train=X_train, Y_train=Y_train)
    np.savez_compressed(
        os.path.join(output_dir, "testDataProcessed"), X_test=X_test, Y_test=Y_test)

    print("There have been {} failures".format(failures))

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

    parser.add_argument("--img-path",
                        required=False,
                        type=str,
                        default="/mnt/server-home/TUE/20175985/basicRaf/Image/original",
                        help='Directory of images')

    parser.add_argument("--txt-path",
                        required=False,
                        type=str,
                        default="/mnt/server-home/TUE/20175985/basicRaf/EmoLabel/list_patition_label.txt",
                        help='Directory of txt file')

    parser.add_argument("--path_bounding_box",
                        required=False,
                        type=str,
                        default="/mnt/server-home/TUE/20175985/basicRaf/Annotation/boundingbox",
                        help="Directory of a file with bounding box coordinates")

    parser.add_argument("--output-dir",
                        required=False,
                        type=str,
                        default="/mnt/server-home/TUE/20175985/basicRaf/dataNpz",
                        help="Where to save the data")

    parser.add_argument("--prepro",
                        required=False,
                        type=bool,
                        default=True,
                        help="Do preprocessing?")

    parse_args, unknown = parser.parse_known_args()

    read_data(**parse_args.__dict__)
