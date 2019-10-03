#Blade Nelson
#Written in August 2019
import cv2
import numpy as np

eyeData = "xml/eyes.xml"
faceData = "xml/face.xml"
DOWNSCALE = 3

#Bools for control
add_face_rect = False
add_objects = False
add_eye_rect = False

#OpenCV boiler plate
webcam = cv2.VideoCapture(0)
cv2.namedWindow("Webcam Facial Tracking")
classifier = cv2.CascadeClassifier(eyeData)
faceClass = cv2.CascadeClassifier(faceData)

#Loading glasses asset
glasses = cv2.imread('assets/glasses.png', cv2.IMREAD_UNCHANGED)

ratio = glasses.shape[1] / glasses.shape[0]

if webcam.isOpened(): # try to get the first frame
    rval, frame = webcam.read()
else:
    rval = False

#Main loop
while rval:
    # detect eyes and draw glasses
    minisize = (frame.shape[1]/DOWNSCALE,frame.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = faceClass.detectMultiScale(miniframe)
    eyes = classifier.detectMultiScale(miniframe)
    
    if add_eye_rect:
        for eye in eyes:
            x, y, w, h = [v * DOWNSCALE for v in eye]

            pts1 = (x, y+h)
            pts2 = (x + w, y)
            # pts1 and pts2 are the upper left and bottom right coordinates of the rectangle
            cv2.rectangle(frame, pts1, pts2, color=(0, 255, 0), thickness=3)

            if add_objects:
                h = w / ratio
                y += h / 2
                # resize glasses to a new var called small glasses
                smallglasses = cv2.resize(glasses, (w, h))
                # the area you want to change
                bg = frame[y:y+h, x:x+w]
                np.multiply(bg, np.atleast_3d(255 - smallglasses[:, :, 3])/255.0, out=bg, casting="unsafe")
                np.add(bg, smallglasses[:, :, 0:3] * np.atleast_3d(smallglasses[:, :, 3]), out=bg)
                # put the changed image back into the scene
                frame[y:y+h, x:x+w] = bg

    if add_face_rect:
        for face in faces:
            x, y, w, h = [v * DOWNSCALE for v in face]

            pts1 = (x, y+h)
            pts2 = (x + w, y)
            # pts1 and pts2 are the upper left and bottom right coordinates of the rectangle
            cv2.rectangle(frame, pts1, pts2, color=(255, 0, 0), thickness=3)

    cv2.imshow("Webcam Glasses Tracking", frame)

    # get next frame
    rval, frame = webcam.read()

    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        cv2.destroyWindow("Webcam Face Tracking")
        break

    #Keyboard input
    if key == ord('1'):
        if add_face_rect:
            add_face_rect = False
        else:
            add_face_rect = True

    if key == ord('2'):
        if add_eye_rect:
            add_eye_rect = False
        else:
            add_eye_rect = True

    if key == ord('3'):
        if add_objects:
            add_objects = False
        else:
            add_objects = True
