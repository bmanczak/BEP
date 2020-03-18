import random
from sklearn.utils import shuffle
import model
import numpy as np


def pick_balanced_subset_classes(data:list, classes:list, class_size:int,save_file_name, seed = 1998, do_shuffle = True) -> list:
    """
    Takes data in .npz and outputs the data with classes specifced in classes argument

    Parameters:
    ~~~~~~~~~~
    data - list, required
        has a format[numpy array of images, numpy array of labels
    classes - list, required
        contains the indices of classes to be included. The indices match the ones specifed in data labels
    class_size - int, required
        contains the indices of classes to be included. The indices match the ones specifed in data labels
    seed - int, optional
        fixes the random state
    shuffle - boolean, optional
        whether to shuffle the data before hand
    """

    random.seed(seed) # set seed for reproducability

    if do_shuffle:
        data[0], data[1] = shuffle(data[0], data[1], random_state= seed)

    class_count = {elem:0 for elem in range(0, data[1][0].shape[0])} # initialize a dictionary with a key for each class
    #print(class_count)
    indices = []
    print(data[0].shape, data[1].shape)
    for img in range(0, data[1].shape[0]):
        label = data[1][img]
        #print("label",label)
        for num in classes:
            #print("num",num)
            if label[num] == 1:

                #print("label[num]",label[num])

                if class_count[num] < class_size:
                   # print(class_count[num], class_size)
                    indices.append(img)
                    class_count[num] += 1
                    #print(class_count[num], class_size)
                    #print(len(indices) >= class_size*len(classes))

                    if len(indices) >= class_size*len(classes): # we found all we needed
                        unused_classes = list(set(range(0, len(data[1][0]))) - set(classes))
                        data_subset = [data[0][indices], np.delete(data[1][indices],unused_classes, axis = 1 )]#take care of the unused classes]
                        #data_subset = [data[0][indices], data[1][indices]]
                        print("Success!")
                        return data_subset

                    break # found a 1: since element is at most of one class we do not need to traverse anymore


    unused_classes = list(set(range(0, len(data[1][0]))) - set(classes))
    data_subset = [data[0][indices], np.delete(data[1][indices],unused_classes, axis = 1 )]#take care of the unused classes]
    return data_subset



full_npz_test_path = "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/testDataNotProcessedFull.npz"
test_df = model.read_test_data(path = full_npz_test_path)
out_test = pick_balanced_subset_classes(test_df, [0,1,2,3,4,5,6], 500, save_file_name ="testDataNotProcessedBalanced7Classes3k" )
np.savez_compressed(
    "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/testDataNotProcessedBalanced7Classes3k", X_test = out_test[0],
     Y_test = out_test[1])


full_npz_train_path = "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/trainDataNotProcessedFull.npz"
train_df = model.read_train_data(path = full_npz_train_path)
out_train = pick_balanced_subset_classes(train_df, [0,1,2,3,4,5,6], 3000, save_file_name= "trainDataNotProcessedBalanced7Classes3k")
np.savez_compressed(
    "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/trainDataNotProcessedBalanced7Classes3k", X_train = out_train[0],
     Y_train = out_train[1])
