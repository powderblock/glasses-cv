import cv2, numpy as np
eyeData = "eyes.xml"
faceData = "face.xml"
mouthData = "mouth.xml"
DOWNSCALE = 4
 
webcam = cv2.VideoCapture(0)
cv2.namedWindow("preview")
classifier = cv2.CascadeClassifier(eyeData)

glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)
joint = cv2.imread('joint.png', cv2.IMREAD_UNCHANGED)
fedora = cv2.imread('fedora.png', cv2.IMREAD_UNCHANGED)

ratio = glasses.shape[1] / glasses.shape[0]

if webcam.isOpened(): # try to get the first frame
    rval, frame = webcam.read()
else:
    rval = False

while rval:
    # detect eyes and draw glasses
    minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    eye = classifier.detectMultiScale(miniframe)
    for f in eye:
        x, y, w, h = [ v*DOWNSCALE for v in f ]
        h = w / ratio
        y += h / 2
        # resize glasses to a new var called small glasses
        smallglasses = cv2.resize(glasses, (w, h))
        # the area you want to change
        bg = frame[y:y+h, x:x+w]
        bg *= np.atleast_3d(255 - smallglasses[:, :, 3])/255.0
        bg += smallglasses[:, :, 0:3] * np.atleast_3d(smallglasses[:, :, 3])
        # put the changed image back into the scene
        frame[y:y+h, x:x+w] = bg
    cv2.imshow("preview", frame)
 
    # get next frame
    rval, frame = webcam.read()
 
    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        cv2.destroyWindow("preview")
        break
