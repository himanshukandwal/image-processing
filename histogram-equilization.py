from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin, pi

green = (0, 255, 0)

def find_ball(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    result = find_ball(image)

    cv2.imshow('frame', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()