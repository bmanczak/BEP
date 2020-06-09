# USAGE
# python detect_mask_video.py

# import the necessary packages

#from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS
from keras_vggface import utils
import numpy as np
import argparse
import imutils
import time
import cv2
import os
#import tf_lite_runtime.interpreter as tflite
import platform
import tensorflow as tf


def set_input_tensor(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    scale, zero_point = input_details['quantization']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    # The input tensor data must be uint8: within [0, 255].
    input_tensor[:, :] = np.uint8(input / scale + zero_point)


def classify_image(interpreter, input):
    set_input_tensor(interpreter, input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details['index'])
    # Because the model is quantized (uint8 data), we dequantize the results
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)
    return output[0]


EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[
            tflite.load_delegate(EDGETPU_SHARED_LIB,
                                 {'device': device[0]} if device else {})
        ])


def detect_and_predict_mask(frame, faceNet, version):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            pixels_expanded = np.expand_dims(face.astype(np.float64), axis=0)
            face = utils.preprocess_input(pixels_expanded, version=version)[0]
            print(face.shape)
            # add the face and bounding boxes to their respective
            # lists
            faces.append(face/255)
            # print(face)
            locs.append((startX, startY, endX, endY))

    faces = np.array(faces)
    #np.expand_dims(faces[0], axis=0)
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        for face in faces:
            preds.append(classify_image(interpreter, face))
        # print(faces[0])
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        #preds = maskNet.predict(faces)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="/Users/blazejmanczak/Desktop/RealTimeEmo/withResSsd/face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="/Users/blazejmanczak/Desktop/model.tflite",
                help="path to trained tflite emotion model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt.txt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the emotion  model from disk
print("[INFO] loading emotion model...")

print('Loading {} tfLite interpreter.'.format(args["model"]))
# interpreter = make_interpreter(args["model"]) # uncomment for Coral
interpreter = tf.lite.Interpreter(args["model"])
interpreter.allocate_tensors()


#maskNet = load_model(args["model"])
version = 1 if "vgg" in args["model"] else 2

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

count = 0
# loop over the frames from the video stream
while True:
    # while count < 300:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, version)
    # loop over the detected face locations and their corresponding
    # locations

    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        emotions_dic = {0: "neutral", 1: "happy", 2: "sad",
                        3: "suprised", 4: "scared", 5: "disgusted", 6: "angry"}

        #pred_ind = np.argmax(pred)
        #label = "{}".format(emotions_dic[pred_ind])
        # print(pred)
        pred_sorted = np.argsort(pred)
        label_1 = pred_sorted[-1]
        label_2 = pred_sorted[-2]
        # print(label_1)
        emotion = "{} {}%,{}, {}% ".format(emotions_dic[label_1], round(
            pred[label_1] * 100), emotions_dic[label_2], round(pred[label_2] * 100))

        # determine the class label and color we'll use to draw
        # the bounding box and text

        color = (0, 255, 0)  # if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        #label = "{}: {:.2f}%".format(label, pred[pred_ind])

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, emotion, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    count += 1
    fps.update()


fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
