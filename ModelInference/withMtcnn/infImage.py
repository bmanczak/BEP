from Preprocessing import *
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
import keras

detector = MTCNN()
img_path = "/Users/blazejmanczak/Desktop/modelTestPhotos/rozpierdol.jpg"
img_array = pyplot.imread(img_path)
#img_array = ((img_array/255)*255).astype(np.uint8)

modelpath = "/Users/blazejmanczak/Desktop/School/Year3/BEP/realTime/vggFaceAffectFullAdamFineTunedAcc64"
model = keras.models.load_model(modelpath)


def applyPreprocessingInference(img, detector):
    """ Applies preprocessing to all the faces in the image

    Parameters:
    --------------
    img: np.array
        an original image

    detector: MTCNN instance
        a detector to be used in the preprocessing function

    Output:
    --------------
    pipe.detected: dict
        contains the location of the faces in the uncropped image.
        Used for drawing rectangles
    output: list
        a list of Preprocessing objects containing cropped faces
    """
    output = []

    pipe = Preprocessing(img_array=img_array, detector=detector,
                         desiredFaceHeight=224, desiredFaceWidth=224)

    pipe.clahe(in_place=True)  # only run in tough lighting conditions
    pipe.run_detector()
    pipe.face_extract(extra=0)

    for face in pipe.faces:
        face.face_align(in_place=True)
        face.resize(in_place=True)
        face.vgg_prepro(in_place=True)
        output.append(face)

    return pipe, output


out = applyPreprocessingInference(img_array, detector)


def output_classifier(out, img_org):
    """Outputs an orginal image with rectangles and predictions drawn next to the face

    Parameters:
    --------------------
    out: list
        list of Preprocessing objects

    img: np.array
        orginal image on which to draw

    Output:
    --------------

    img_org: np.array
        An image with drawn rectangles
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    emotions_dic = {0: "neutral", 1: "happy", 2: "sad",
                    3: "suprised", 4: "scared", 5: "disgusted", 6: "angry"}
    for face_ind in range(len(out[1])):
        pred = model.predict(np.expand_dims(out[1][face_ind].img_array/255, axis=0))
        label = np.argmax(pred)
        emotion = "{}".format(emotions_dic[label])

        (x, y, w, h) = list(out[0].detected[face_ind]['box'])

        cv2.rectangle(img_org,  # draw rectangles
                      (x, y),
                      (x + w, y + h),
                      (0, 155, 255),
                      2)

        cv2.putText(img_org, emotion, (x, y), font, 1, (255, 200, 100), 2)  # write on image

    return img_org


abc = output_classifier(out, img_array)
cv2.imwrite("processedImage.jpg", cv2.cvtColor(abc, cv2.COLOR_RGB2BGR))
