import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(
    'trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (225, 0, 0), 2)
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        print(Id, conf)

        if(conf < 60):
            if(Id == 1):
                Id = "mask off"
            elif(Id == 2):
                Id = "mask on"

        else:
            Id = "not sure.. :("
        cv2.putText(im, str(Id), (x, y+w), font,
                    2, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.imshow('Face', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
