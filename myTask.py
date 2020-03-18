"""This code implements a Feed forward neural network using Keras API."""


# import comet_ml in the top of your file
from comet_ml import Experiment

# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="Qs9hrAPUWIusY5UKNebGp1MAN",
                        project_name="bachelor-end-project", workspace="blazejmanczak")

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

CLASS_SIZE = 7

"""
class MaintainMetrics(keras.callbacks.Callback):
    def printMetrics(self):
        print('\nEvaluation epoch[{}] metrics[{:.2f}, {:.2f}] {}'.format(
            epoch, loss, acc, AFFECTNET_MODEL.metrics_names))
"""

def just_do_it(job_dir,
            num_epochs):

    AFFECTNET_MODEL = model.model_fn(CLASS_SIZE)
    AFFECTNET_MODEL.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer='rmsprop',
                              metrics=['accuracy'])
    print("Compiled")
    # Tensorboard logs callback
    tblog = keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        embeddings_freq=0)

    # Implement early stopping
    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=8
    )

    #Save the best model
    mc = keras.callbacks.ModelCheckpoint(
        os.path.join(job_dir, 'best_model_affectNet.hdf5'),
         monitor='val_accuracy',
         mode='max', verbose=1,
         save_best_only=True
    )

    callbacks = [tblog, es, mc]

    X_train, Y_train = model.read_train_data()
    X_test, Y_test = model.read_test_data()

    datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    AFFECTNET_MODEL.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=128),
        # steps_per_epoch=100,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=2,
        class_weight=list(np.sum(X_train, axis=0) / np.sum(X_train, axis=0).sum()),
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

    parse_args, unknown = parser.parse_known_args()

    just_do_it(**parse_args.__dict__)
