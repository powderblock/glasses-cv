import cv2, numpy as np
eyeData = "xml/eyes.xml"
faceData = "xml/face.xml"
DOWNSCALE = 4

cv2.namedWindow("Still Image Dank")
eyeClass = cv2.CascadeClassifier(eyeData)
faceClass = cv2.CascadeClassifier(faceData)

glasses = cv2.imread('assets/glasses.png', cv2.IMREAD_UNCHANGED)
fedora = cv2.imread('assets/fedora.png', cv2.IMREAD_UNCHANGED)
frame = cv2.imread('images/putin.jpg')

ratio = glasses.shape[1] / glasses.shape[0]
# detect eyes and draw glasses
minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
miniframe = cv2.resize(frame, minisize)
eyes = eyeClass.detectMultiScale(miniframe)
faces = faceClass.detectMultiScale(miniframe)

print len(eyes)

for eye in eyes:
    x, y, w, h = [v * DOWNSCALE for v in eye]
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

cv2.imshow("Still Image Dank", frame)

while True:
    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        cv2.destroyWindow("Still Image Dank")
        break
