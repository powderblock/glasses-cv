import cv2, numpy as np
eyeData = "xml/eyes.xml"
faceData = "xml/face.xml"
DOWNSCALE = 4

# make a window
cv2.namedWindow("Facial Features Test")
# data sets
faceClass = cv2.CascadeClassifier(faceData)
eyeClass = cv2.CascadeClassifier(eyeData)

# load the image we want to detect features on
frame = cv2.imread('images/putin.jpg')

# detect face(s)
minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
miniframe = cv2.resize(frame, minisize)
faces = faceClass.detectMultiScale(miniframe)
eyes = eyeClass.detectMultiScale(miniframe)
for face in faces:
    x, y, w, h = [v*DOWNSCALE for v in face]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

for eye in eyes:
    x, y, w, h = [v*DOWNSCALE for v in eye]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)

cv2.imshow("Facial Features Test", frame)

while True:
    # key handling (to close window)
    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        cv2.destroyWindow("Facial Features Test")
        break
