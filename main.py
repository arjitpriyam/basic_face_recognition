import cv2
import os
import numpy as np
from PIL import Image

class face_recog():

    def __init__(self):
        self.Ides = []
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.faceDetect = cv2.CascadeClassifier('dependencies/haarcascade_frontalface_default.xml')
        self.path = 'dataSet'

    def fetch_data(self,fr):

        no = int(input(' Enter the number of users : '))
        if(no>0):
            for i in range(no):
                id = input('Enter user id : ')
                self.Ides.append(id)
                sampleNum = 0
                cam = cv2.VideoCapture(0)

                while (True):
                    ret, img = cam.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = self.faceDetect.detectMultiScale(gray, 1.3, 5);

                    for (x, y, w, h) in faces:
                        if not os.path.exists('dataSet'):
                            os.mkdir('dataSet')

                        sampleNum = sampleNum + 1
                        cv2.imwrite('dataSet/User.' + str(id) + '.' + str(sampleNum) + '.jpg', gray[y:y + h, x:x + w])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.waitKey(100)
                    cv2.imshow('Face', img)
                    cv2.waitKey(1)
                    if sampleNum > 10:
                        break
            cam.release()
            cv2.destroyAllWindows()
            fr.training(self.path)
            fr.recognition()
        else:
            fr.recognition()


    def training(self,path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        IDs = []
        faces = []

        for imagePath in imagePaths:
            faceImg = Image.open(imagePath).convert('L')
            faceNp = np.array(faceImg, 'uint8')
            ID = int(os.path.split(imagePath)[-1].split('.')[1])
            faces.append(faceNp)
            IDs.append(ID)
            cv2.waitKey(10)

        Ids, Faces = IDs, faces
        self.recognizer.train(Faces, np.array(Ids))

        if not os.path.exists('recognizer'):
            os.mkdir('recognizer')

        self.recognizer.save('recognizer/trainingData.yml')
        cv2.destroyAllWindows()

    def recognition(self):
        cam = cv2.VideoCapture(0)
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.read('recognizer\\trainingData.yml')
        id = 0
        font = cv2.FONT_HERSHEY_SIMPLEX

        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceDetect.detectMultiScale(gray, 1.3, 5);

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, conf = rec.predict(gray[y:y + h, x:x + w])
                cv2.putText(img, str(id), (x, y + h), font, 0.55, (0, 255, 0), 1)

            cv2.imshow('Face', img)
            if (cv2.waitKey(1) == ord('q')):
                break;

        cam.release()
        cv2.destroyAllWindows()

fr=face_recog()
fr.fetch_data(fr)

