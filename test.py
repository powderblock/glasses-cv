#!/usr/bin/env python2
 
"""
OpenCV example. Show webcam image and detect face.
"""
 
import cv2
 
faceXML = "haarcascade_frontalface_alt.xml"
eyes = "frontalEyes35x16.xml"
DOWNSCALE = 2
 
webcam = cv2.VideoCapture(0)
cv2.namedWindow("preview")
face = cv2.CascadeClassifier(faceXML)
eye = cv2.CascadeClassifier(eyes)
 
 
if webcam.isOpened(): # try to get the first frame
    rval, frame = webcam.read()
else:
    rval = False
 
while rval:
 
    # detect faces and draw bounding boxes
    minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = face.detectMultiScale(miniframe)
    for f in faces:
        x, y, w, h = [ v*DOWNSCALE for v in f ]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
    eyes = eye.detectMultiScale(miniframe)
    for f in eyes:
        x, y, w, h = [ v*DOWNSCALE for v in f ]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
 
    cv2.putText(frame, "Press ESC to close.", (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
    cv2.imshow("preview", frame)
 
    # get next frame
    rval, frame = webcam.read()
 
    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        break
