# Real Time Face Recognition

## Installation

### Requirements

  * Python 3.3+ or Python 2.7
  * macOS or Linux (Windows not officially supported, but might work)
  * Numpy 
  * Pillow
  * os

### Installation Options:

#### Installing on Mac or Linux

First, make sure you have dlib already installed with Python bindings:

  * [How to install dlib from source on macOS or Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

Then, install this module from pypi using `pip3` (or `pip2` for Python 2):

```bash
pip3 install face_recognition
```

If you are having trouble with installation, you can also try out this: **Windows Only** 
1. Microsoft Visual Studio 2015 with C/C++ Compiler installed. (Visual C++ 2015 Build Tools didn't work for me and I got into problems in compiling dlib)
2. Boost library version 1.63 or newer. Also, you can use precompiled binaries for specific MSVC you have but I don't suggest. (I've included the compiling procedure of Boost in this tutorial)
3. Of course Python3 (I used Python3.5 x64 but the other versions may work too)
4. CMake for windows and add it to your system environment variables.

## Introduction

This is basically how facial recognition works. 

![kirill-opencv-gif-400](https://user-images.githubusercontent.com/44390802/48006326-431b9000-e13b-11e8-989c-4ada732156c3.gif)

To create a complete project on Face Recognition, we must work on 3 very distinct phases:

1. Face Detection and Data Gathering
2. Train the Recognizer
3. Face Recognition

### 1: Face Detection and Data Gathering

The most basic task on Face Recognition is of course, "Face Detecting". Before anything, you must "capture" a face (Phase 1) in order to recognize it, when compared with a new face captured on future (Phase 3).

The most common way to detect a face (or any objects), is using the ["Haar Cascade classifier"](https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html)

Here we will work with face detection. Initially, the algorithm needs a lot of positive images (images of faces) to train the classifier. Then we need to extract features from it. The good news is that OpenCV comes with a trainer as well as a detector.OpenCV already contains many pre-trained classifiers for face, eyes, smile, etc. which are provided in the "Dependencies" folder.

```python
faceDetect = cv2.CascadeClassifier('dependencies/haarcascade_frontalface_default.xml')
```
This is the line that loads the "classifier" (that must be in a directory named "Dependencies", under your project directory).
Then, we will set our camera and inside the loop, load our input video in grayscale mode (same we saw before).

Now we must call our classifier function, passing it some very important parameters, as scale factor, number of neighbors and minimum size of the detected face.

```python
faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbours=5, minSize=(20, 20))
```
Where,
* **gray** is the input grayscale image.
* **scaleFactor** is the parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid.
* **minNeighbors** is a parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives.
* **minSize** is the minimum rectangle size to be considered a face.

The function will detect faces on the image. Next, we must "mark" the faces in the image, using, for example, a green rectangle. This is done with this portion of the code:
```python
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```
If faces are found, it returns the positions of detected faces as a rectangle with the left up corner (x,y) and having "w" as its Width and "h" as its Height ==> (x,y,w,h).

![face_detection_requests_obama-1024x698-e1485809553817](https://user-images.githubusercontent.com/44390802/48007146-02247b00-e13d-11e8-8106-8cc7d3f3c9f1.jpg)

Let's start the first phase of our project. What we will do here, is starting from last step (Face Detecting), we will simply create a dataset, where we will store for each id, a group of photos in gray with the portion that was used for face detecting.

Next, create a subdirectory where we will store our facial samples and name it "dataset":
```python
for (x, y, w, h) in faces:
            if not os.path.exists('dataSet'):
                os.mkdir('dataSet')
```

On my code, I am capturing 11 samples from each id. You can change it on the last "elif". The number of samples is used to break the loop where the face samples are captured.
Run the Python script and capture a few Ids. You must run the script each time that you want to aggregate a new user (or to change the photos for one that already exists).
```python
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
```
### 2: Train the Recognizer

![phase2_bju1kxulpq](https://user-images.githubusercontent.com/44390802/48008411-a5768f80-e13f-11e8-8b9b-f41c5086a13f.png)

On this second phase, we must take all user data from our dataset and "train" the OpenCV Recognizer. This is done directly by a specific OpenCV function. The result will be a .yml file that will be saved on a "recognizer/" directory.

Confirm if you have the PIL library installed on your Rpi. If not, run the below command in Terminal:
```python
pip install pillow 
```
We will use as a recognizer, the LBPH (LOCAL BINARY PATTERNS HISTOGRAMS) Face Recognizer, included on OpenCV package. We do this in the following line:

```python
recognizer = cv2.face.LBPHFaceRecognizer_create()
```

The function "getImagesAndLabels (path)", will take all photos on directory: "dataset/", returning 2 arrays: "Ids" and "faces". With those arrays as input, we will "train our recognizer":

```python
recognizer.train(faces, np.array(Ids))
```
As a result, a file named "trainer.yml" will be saved in the trainer directory that was previously created by us.

That's it! I included the last print statement where I displayed for confirmation, the number of User's faces we have trained.

Every time that you perform Phase 1, Phase 2 must also be run.

### 3: Recognizer

Here, we will capture a fresh face on our camera and if this person had his face captured and trained before, our recognizer will make a "prediction" returning its id and an index, shown how confident the recognizer is with this match.

![phase3_0qf1izx9hh](https://user-images.githubusercontent.com/44390802/48008873-86c4c880-e140-11e8-85e9-925e035f6766.png)

Next, we will detect a face, same we did before with the haasCascade classifier. Having a detected face we can call the most important function in the above code:

```python
id, conf = rec.predict(gray[y:y+h, x:x+w])
```

The recognizer.predict (), will take as a parameter a captured portion of the face to be analyzed and will return its probable owner, indicating its id and how much confidence the recognizer is in relation with this match.

if the recognizer could predict a face, we put a text over the image with the probable id and how much is the "probability" in % that the match is correct ("probability" = 100 - confidence index). If not, an "unknow" label is put on the face.

![facerecthumbnail](https://user-images.githubusercontent.com/44390802/48009370-9ee91780-e141-11e8-89da-298055823eec.jpg)
