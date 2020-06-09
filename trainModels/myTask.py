# import comet_ml in the top of your file
from comet_ml import Experiment

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import model
from keras.models import load_model
import keras
import os
import argparse
from sklearn.model_selection import train_test_split

from numpy.random import seed
seed(1998)
tf.random.set_seed(1998)
# Add the following code anywhere in your machine learning file
experiment = Experiment(api_key="Qs9hrAPUWIusY5UKNebGp1MAN",
                        project_name="bachelor-end-project", workspace="blazejmanczak")


tf.keras.backend.clear_session()
CLASS_SIZE = 7


def just_do_it(job_dir,
               num_epochs,
               arch,
               save_path,
               optimizer,
               path_train,
               path_test,
               type,
               modelpath,
               balance_weights,
               batch_size,
               test_size):

    #AFFECTNET_MODEL = model.model_fn(CLASS_SIZE, arch)

    if type == "fineTune":  # yes if we fine tune with low learning rate

        modelpath = modelpath

        if len(modelpath) <= 1:  # if modelpath is not specified we get intitialize the models from model.py
            AFFECTNET_MODEL = model.model_fn(CLASS_SIZE, arch)
        else:
            print("[INFO] Reading the model with weights from...", modelpath)
            AFFECTNET_MODEL = keras.models.load_model(modelpath)

        AFFECTNET_MODEL.compile(optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
                                loss=keras.losses.categorical_crossentropy,
                                metrics=["accuracy"])

        # Unfreeze all the layers
        for layer in AFFECTNET_MODEL.layers:
            layer.trainable = True

        """for layer in AFFECTNET_MODEL.layers[:-7]:
            layer.trainable = False"""

        AFFECTNET_MODEL.compile(optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
                                loss=keras.losses.categorical_crossentropy,
                                metrics=["accuracy"])

    else:
        AFFECTNET_MODEL = model.model_fn(CLASS_SIZE, arch)
        AFFECTNET_MODEL.compile(loss=keras.losses.categorical_crossentropy,
                                optimizer=optimizer,
                                metrics=['accuracy'])
    print("[INFO] Compiled...")
    print("[INFO] Using model architecture ", arch)
    print("[INFO] Using optimizer", optimizer)
    print("[INFO] Using", type, "type")

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

    # Save the best model
    mc = keras.callbacks.ModelCheckpoint(
        os.path.join(job_dir, save_path),
        monitor='val_accuracy',
        mode='max', verbose=1,
        save_best_only=True
    )

    callbacks = [tblog, mc]

    X_train, Y_train = model.read_train_data(path_train)

    """ # use in case you need to have validation data
    print("[INFO] Splitting the train data into train and validation set with test_size", test_size)
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train, Y_train, test_size=test_size, shuffle=True, random_state=1998)

    print("[INFO] Total examples per class in training set", np.sum(Y_train, axis=0))
    print("[INFO] Total examples per class in validation set", np.sum(Y_valid, axis=0))
    """

    X_test, Y_test = model.read_test_data(path_test)

    print("[INFO] Using train data", path_train.split("/")[-1])
    print("[INFO] Using test data", path_test.split("/")[-1])

    datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.15,
        rotation_range=10,
        horizontal_flip=True,
        #featurewise_center = True,
        #featurewise_std_normalization = True,
        preprocessing_function=model.image_preprocessing,
        dtype="float16")
    #batch_size = 32
    #number_of_test_samples = Y_test.shape[0]

    if balance_weights == False:
        class_weight = None
    else:
        class_weight = model.balanced_class_weights([X_train, Y_train])

    AFFECTNET_MODEL.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=1600,  # uncomment if large dataset. For 300, 128*400=51.2sk imgs in one epoch
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=2,
        class_weight=class_weight,
        validation_data=datagen.flow(X_train, Y_train))

    print("[INFO] Evaluation of the best model...")

    datagenVal = ImageDataGenerator(preprocessing_function=model.image_preprocessing,
                                    dtype="float16")

    """ # use when using validation data to check performance of the best model on the test data
    loaded = keras.models.load_model(os.path.join(job_dir, save_path))
    print("[INFO] Model evaluation results on the test set:",
          loaded.evaluate(datagenVal.flow(X_test, Y_test))) """


if __name__ == "__main__":
    print("Mains")
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-epochs",
                        required=False,
                        type=int,
                        default=50,
                        help='Maximum number of epochs on which to train')

    parser.add_argument("--test-size",
                        required=False,
                        type=float,
                        default=0.05,
                        help='What should the train-validation split be? This chunk of code is originally commented out, so if you use, make sure to uncomment it')

    parser.add_argument("--batch-size",
                        required=False,
                        type=int,
                        default=64,
                        help='What batch size to use?')

    parser.add_argument("--balance-weights",
                        required=False,
                        type=bool,
                        default=True,
                        help='Weight the loss function by the inverse of class support? ')

    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')

    parser.add_argument('--arch',
                        required=True,
                        type=str,
                        help='What kind of architecture to run? See model.py for supported architectures.')

    parser.add_argument('--save-path',
                        required=False,
                        default="bestModelAffectNet",
                        type=str,
                        help='Name of the saved model checkpoints')

    parser.add_argument('--type',
                        required=False,
                        default="noFineTune",
                        type=str,
                        help='If fineTune, then unfreeze all the layers and optimize with smaller learning rate')
    parser.add_argument("--path-train",
                        required=False,
                        default="/mnt/server-home/TUE/20175985/BepDataResNet/npzData/trainDataNotProcessedBalanced7Classes3k.npz",
                        type=str,
                        help="path to the npz file with the train data")
    parser.add_argument("--path-test",
                        required=False,
                        default="/mnt/server-home/TUE/20175985/BepDataResNet/npzData/testDataNotProcessedBalanced7Classes3k.npz",
                        type=str,
                        help="path to the npz file with the test data")

    parser.add_argument('--optimizer',
                        required=False,
                        default="Adam",
                        type=str,
                        help='Which optimizer to compile the model with? Only relevant with no fine-tunning. Uses the default parameter values of the given optimizers.')

    parser.add_argument("--modelpath",
                        required=False,
                        default="/mnt/server-home/TUE/20175985/jobs/vggFaceAffectFullAdamCLAHE",
                        type=str,
                        help="Path to model weights used for fine tunning. Only relevant when type is fine-tune")

    parse_args, unknown = parser.parse_known_args()

    just_do_it(**parse_args.__dict__)
