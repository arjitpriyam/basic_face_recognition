import cv2
import os

Ides = []
Names = []
faceDetect = cv2.CascadeClassifier('dependencies/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
num = (int)(input("Enter the number of users : "))
for i in range(0, num):
    id = input('Enter user id : ')
    Ides.append(id)
    name = input('Enter the name of the user : ')
    Names.append(name)
    sampleNum = 0

    while True:
        ret, img = cam.read()
        #img = cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

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
