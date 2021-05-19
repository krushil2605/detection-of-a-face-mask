import os
import cv2
import numpy as np
from PIL import Image

#recognizer = cv2.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()

path = "dataSet"
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f)
                  for f in os.listdir(path)]  # takes the path
    faceSamples = []
    Ids = []
    for imagePath in imagePaths:

        if(os.path.split(imagePath)[-1].split(".")[-1] != 'jpg'):
            continue

        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(imageNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
    return faceSamples, Ids


faces, Ids = getImagesAndLabels(
    'D:\College\spark internship\Mask detector\my try\dataset')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')
