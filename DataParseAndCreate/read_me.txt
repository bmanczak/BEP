Preprocessing.py: contains a preprocessing class that is responsible for CLAHE image normalization,
                 face detection and landmark extraction using MTCNN detector, face alignment and extraction

buildRAF.py,buildAffectNet.py: parse the directory structure in order to create compressed,
                preprocessed data representation .npz file format. These files are later used for training the model.
