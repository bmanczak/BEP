import math
from PIL import Image
# pre-processing, install with: pip install git+https://github.com/rcmalli/keras-vggface.git
from keras_vggface import utils
import numpy as np
import cv2

"""
Example usage:

detector = Mtcnn()
pipe = Preprocessing(img_array=img_array, detector=detector,
                     desiredFaceHeight=224, desiredFaceWidth=224)

pipe.clahe(in_place=True)  # illumination normalization
pipe.run_detector() # applies detector to an image
pipe.face_extract(extra=0) # extract the faces from an image

for face in pipe.faces:
    if face.detected[0]['confidence'] >= 0.9: # filters out weak predictions
        face.face_align(in_place=True) # aligns the face
        face.resize(in_place=True) # resizes the image
        face.vgg_prepro(in_place=True) # perform preprocessing for VGG-Face/SENet models
        output.append(face) #add Preprocessing object to a list

return pipe, output
"""


class Preprocessing:
    """
    A pipeline for preprocessing of images of a face


    Parametets:
    ------------------
    img_array: numpy array
        Rank 3 image containing a face with pixel values in range 0-255

    detector: an MTCNN detector object
        Detector to fetch bouning face rectangle and facial landmarks

    desiredFaceWidth: int
        Specifices the reshaping width

    desiredFaceHeight: int
        Specifies the reshaping height

    detected: list
        Output of MTCNN classifier



    Attributes:
    --------------
    """

    def __init__(self, img_array, detector, desiredFaceWidth=224, desiredFaceHeight=224, detected=[]):
        self.img_array = img_array
        self.detector = detector
        self.desiredFaceHeight = desiredFaceHeight
        self.desiredFaceWidth = desiredFaceWidth
        self.detected = detected

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def run_detector(self):
        """Executes the detector upon the img_array"""
        self.detected = self.detector.detect_faces(self.img_array)
        self.faces = []

    def face_extract(self, extra=0):
        """
        Extracts a face from an image. Note that it requires run_detector to be ran before

        Parameters:
        ------------------
        extra: float
            Specifies by how many % points to change the detector's bounding box
        """

        # detect faces in the image
        results = self.detected
        face_coor = []
        i = 0  # counter for dic updates
        #output_arrays = []
        if len(results) > 0:
            for face in results:

                face_coor.append(tuple(extend_rect(face['box'], extra=extra)))

        else:
            print("WARNING: no faces detected!")

        for (x, y, w, h) in face_coor:
            #print(x, y, w, h)
            x, y = max(x, 0), max(y, 0)
            fc = self.img_array[y:y + h, x:x + w]
            self.faces.append(Preprocessing(img_array=fc, detector=self.detector,
                                            detected=[update_detection_dic(self.detected[i])]))
            i += 1

    def clahe(self, in_place=False):
        """Performs CLAHE illumination normalization"""
        # Converting image to LAB Color model
        lab = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2LAB)
        # Splitting the LAB image to different channels
        l, a, b = cv2.split(lab)
        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        if in_place == True:
            self.img_array = final
            self.detected = self.detector.detect_faces(self.img_array)  # update the faces

        else:
            return final

    def face_align(self, in_place=False):
        """Aligns the face Note that it requires run_detector to be ran before """

        face = self.detected
        left_eye_center = face[0]['keypoints']['left_eye']
        right_eye_center = face[0]['keypoints']['right_eye']

        left_eye_x = left_eye_center[0]
        left_eye_y = left_eye_center[1]

        right_eye_x = right_eye_center[0]
        right_eye_y = right_eye_center[1]

        if left_eye_y > right_eye_y:  # left eye below the right eye
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
           #print("rotate to clock direction")
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock
            #print("rotate to inverse clock direction")

        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, left_eye_center)
        c = euclidean_distance(right_eye_center, point_3rd)

        cos_a = (b*b + c*c - a*a)/(2*b*c)

        angle = np.arccos(cos_a)  # angle in radians

        angle = (angle * 180) / math.pi

        if direction == -1:
            angle = 90 - angle

        # Perform rotation
        new_img = Image.fromarray(self.img_array)
        new_img = np.array(new_img.rotate(direction * angle))

        if in_place == True:
            self.img_array = new_img
        else:
            return new_img

    def vgg_prepro(self, version=1, in_place=False):
        """
        Preprocesses the image as done in the VGG-Face paper
        Parameters:
        ------------
        version: int
            1 for VGG-Face preprocessing, 2 for resnet50 or senet 50
        """
        pixels_expanded = np.expand_dims(self.img_array.astype(np.float64), axis=0)
        pre_pro = utils.preprocess_input(pixels_expanded, version=version)

        if in_place == True:
            self.img_array = pre_pro[0]
        else:
            return pre_pro[0]

        # return output_arrays

    def resize(self, in_place=False):
        if in_place:
            self.img_array = cv2.resize(
                self.img_array, (self.desiredFaceWidth, self.desiredFaceHeight))
        else:
            return cv2.resize(self.img_array, (self.desiredFaceWidth, self.desiredFaceHeight))


def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def extend_rect(iterable, extra=0):
    """ Has strtucture: x,y,width,height"""
    extra_width = iterable[2]*extra
    extra_height = iterable[3]*extra

    x = iterable[0] - extra_width
    y = iterable[1] - extra_height  # y axis is reversed
    new_width = iterable[2] + extra_width
    new_height = iterable[3] + extra_height
    #out = [x,y,new_width, new_height, extra_width, extra_height]
    out = [x, y, new_width+extra_width, new_height+extra_height]
    return [round(i) for i in out]


def update_detection_dic(old_dic):
    updated = {}
    updated['box'] = [0, 0, old_dic['box'][2], old_dic['box'][3]]
    updated['confidence'] = old_dic['confidence']
    updated['keypoints'] = {}
    for key, val in old_dic['keypoints'].items():
        updated['keypoints'][key] = (val[0] - old_dic['box'][0], val[1] - old_dic['box'][1])

    return updated


def applyPreprocessing(img_array, detector):
    #img_array = pyplot.imread(img_path)
    #img_array = ((img_array/255)*255).astype(np.uint8)

    pipe = Preprocessing(img_array=img_array.astype(np.uint8), detector=detector,
                         desiredFaceHeight=224, desiredFaceWidth=224)

    pipe.clahe(in_place=True)
    pipe.run_detector()
    pipe.face_align(in_place=True)
    pipe.face_extract(extra=0)
    sub_pipe = pipe.faces[0]

    # sub_pipe.resize(in_place=True)
    return sub_pipe.img_array
