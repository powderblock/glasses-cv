import cv2, numpy as np
2
TRAINSET = "frontalEyes35x16.xml"
DOWNSCALE = 2
 
webcam = cv2.VideoCapture(0)
cv2.namedWindow("preview")
classifier = cv2.CascadeClassifier(TRAINSET)

glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)
 
if webcam.isOpened(): # try to get the first frame
    rval, frame = webcam.read()
else:
    rval = False

while rval:
 
    # detect faces and draw bounding boxes
    minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = classifier.detectMultiScale(miniframe)
    for f in faces:
        x, y, w, h = [ v*DOWNSCALE for v in f ]
        smallglasses = cv2.resize(glasses, (w, h))
        bg = frame[y:y+h, x:x+w]
        bg *= np.atleast_3d(255 - smallglasses[:, :, 3])/255.0
        bg += smallglasses[:, :, 0:3] * np.atleast_3d(smallglasses[:, :, 3])
        frame[y:y+h, x:x+w] = bg
    cv2.imshow("preview", frame)
 
    # get next frame
    rval, frame = webcam.read()
 
    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        cv2.destroyWindow("preview")
        break
