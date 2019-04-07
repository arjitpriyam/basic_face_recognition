import cv2
from src.face_recog.face_recog import Names, Ides
faceDetect = cv2.CascadeClassifier('dependencies/haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer\\trainingData.yml')
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX


while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5);

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        if conf < 100:
            id = str(Names[Ides.index(str(id))])
            conf = "  {0}%".format(round(100 - conf))
        else:
            id = "unknown"
            conf = "  {0}%".format(round(100 - conf))
        cv2.putText(img, id, (x, y + h), font, 0.55, (0, 255, 0), 1)
        cv2.putText(img, str(conf), (x+5,y+h-5), font, 1, (255,255,0), 1)
    cv2.imshow('Face', img)
    if(cv2.waitKey(1) == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()

