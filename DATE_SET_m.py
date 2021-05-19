import cv2


face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

id = input("enter user id")
sampleNum=0
while True:
    ret, fream = cap.read()
    gray = cv2.cvtColor(fream, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        sampleNum = sampleNum+1
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(fream, (x,y), (x+w,y+h), (0,255,0),3)
        cv2.waitKey(100)
    cv2.imshow("window", fream)
    cv2.waitKey(1)
    if sampleNum > 100:
       break
cap.release()
cv2.destroyAllWindows()


