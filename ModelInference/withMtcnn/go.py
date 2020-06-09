from modelInference import applyPreprocessingInference
import imutils
from imutils.video import VideoStream
import numpy as np
import time
import cv2
from mtcnn.mtcnn import MTCNN
import keras
from imutils.video import FPS

detector = MTCNN()

modelpath = "/Users/blazejmanczak/Desktop/senetRafFromAcc64OrgDataAcc877"
model = keras.models.load_model(modelpath)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
count = 0

while True:
    # while count < 300:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    count += 1
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    out = applyPreprocessingInference(frame, detector)

    # print("Applied preprocessing")
    # print(len(out[1]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    emotions_dic = {0: "neutral", 1: "happy", 2: "sad",
                    3: "suprised", 4: "scared", 5: "disgusted", 6: "angry"}

    all_faces = np.array([elem.img_array/255 for elem in out[1]])

    rolling = []

    if all_faces.shape[0] > 0:
        all_faces_preds = model.predict(all_faces)  # make prediction in batches

        for face_ind in range(len(out[1])):
            # model.predict(np.expand_dims(out[1][face_ind].img_array/255, axis=0))
            pred = all_faces_preds[face_ind]
            # print(pred)
            pred_sorted = np.argsort(pred)
            label_1 = pred_sorted[-1]
            label_2 = pred_sorted[-2]
            # print(label_1)
            emotion = "{} {}%,{}, {}% ".format(emotions_dic[label_1], round(
                pred[label_1] * 100), emotions_dic[label_2], round(pred[label_2] * 100))

            (x, y, w, h) = list(out[0].detected[face_ind]['box'])
            # print("Img_array shape", out[0].img_array.shape)
            # print(x, y, w, h)
            # print(frame.shape)
            # print(label)

            cv2.rectangle(frame,  # draw rectangles
                          (x, y),
                          (x + w, y + h),
                          (0, 255, 0),
                          2)

            cv2.putText(frame, emotion, (x, y), font, 0.8, (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Framee", frame)
    key = cv2.waitKey(1) & 0xFF
    cv2.imwrite(
        "/Users/blazejmanczak/Desktop/School/Year3/BEP/figures/mtcnnOut/img{}.png".format(count), frame)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
