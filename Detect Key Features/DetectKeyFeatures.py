import cv2, numpy as np
eyeData = "eyes.xml"
faceData = "face.xml"
DOWNSCALE = 1

# make a window
cv2.namedWindow("Face Test")
# data sets
faceClass = cv2.CascadeClassifier(faceData)
eyeClass = cv2.CascadeClassifier(eyeData)

# load the image we want to detect features on
frame = cv2.imread('10256963_10203853201583030_6206259209714541315_n.jpg')

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

cv2.imshow("Face Test", frame)

while True:
    # key handling (to close window)
    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        cv2.destroyWindow("Face Test")
        break
