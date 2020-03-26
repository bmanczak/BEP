"""This code implements a Feed forward neural network using Keras API."""


"""# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="Qs9hrAPUWIusY5UKNebGp1MAN",
                        project_name="bachelor-end-project", workspace="blazejmanczak")
"""
import argparse
import glob
import json
import os
import numpy as np
import keras
from keras.models import load_model
import model
from tensorflow.python.lib.io import file_io
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.keras.backend.clear_session()
CLASS_SIZE = 7

"""
class MaintainMetrics(keras.callbacks.Callback):
    def printMetrics(self):
        print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
            epoch, loss, acc, AFFECTNET_MODEL.metrics_names))
"""

def just_do_it(job_dir,
            num_epochs,
             arch,
             save_path,
             optimizer,
             path_train,
             path_test):

    AFFECTNET_MODEL = model.model_fn(CLASS_SIZE, arch)
    AFFECTNET_MODEL.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=optimizer,
                              metrics=['accuracy'])
    print("Compiled")
    print("Using optimizer", optimizer)
    # Tensorboard logs callback
    tblog = keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    # Implement early stopping
    es = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='min',
        verbose=1,
        patience=10
    )

    #Save the best model
    mc = keras.callbacks.ModelCheckpoint(
        os.path.join(job_dir, save_path),
         monitor='val_accuracy',
         mode='max', verbose=1,
         save_best_only=True
    )

    callbacks = [tblog, mc]

    X_train, Y_train = model.read_train_data(path_train)
    X_test, Y_test = model.read_test_data(path_test)

    print("Using train data", path_train.split("/")[-1])
    print("Using test data", path_test.split("/")[-1])

    datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        featurewise_center = True,
        featurewise_std_normalization = True,
        dtype = "float16")

    AFFECTNET_MODEL.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=128),
        steps_per_epoch = 400, # uncomment if large dataset. For 300, 128*400=51.2sk imgs in one epoch
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=2,
        class_weight= model.balanced_class_weights([X_train, Y_train]),
        validation_data=(X_test,Y_test))


if __name__ == "__main__":
    print("Mains")
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-epochs",
                        required = False,
                        type = int,
                        default = 50,
                        help='Maximum number of epochs on which to train')

    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')

    parser.add_argument('--arch',
                        required=True,
                        type=str,
                        help='What kind of architecture to run')

    parser.add_argument('--save-path',
                        required=False,
                        default = "bestModelAffectNet",
                        type=str,
                        help='Name of the saved model checkpoints')

    parser.add_argument('--optimizer',
                        required=False,
                        default = "rmsprop",
                        type=str,
                        help='Which optimizer to compile the model with')
    parser.add_argument("--path-train",
                        required = False,
                        default = "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/trainDataNotProcessedBalanced7Classes3k.npz",
                        type = str,
                        help = "path to the npz file with the train data")
    parser.add_argument("--path-test",
                        required = False,
                        default = "/mnt/server-home/TUE/20175985/BepDataResNet/npzData/testDataNotProcessedBalanced7Classes3k.npz",
                        type = str,
                        help = "path to the npz file with the train data")

    parse_args, unknown = parser.parse_known_args()

    just_do_it(**parse_args.__dict__)
