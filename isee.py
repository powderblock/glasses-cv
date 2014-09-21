import numpy as np
import cv2

img1 = cv2.imread("glasses.png", 0)
img2 = cv2.imread("test.png", 0)
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
vis[:h1, :w1] = img1
vis[:h2, w1:w1+w2] = img2
vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

cv2.imshow("test", vis)
cv2.waitKey()
