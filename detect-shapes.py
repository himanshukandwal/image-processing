from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin, pi

green = (0, 255, 0)

def find_biggest_contour(image_masked, image):
    # Copy
    image_masked = image_masked.copy()
    #input, gives all the contours, contour approximation compresses horizontal, 
    #vertical, and diagonal segments and leaves only their end points. For example, 
    #an up-right rectangular contour is encoded with 4 points.
    #Optional output vector, containing information about the image topology. 
    #It has as many elements as the number of contours.
    #we dont need it
    img, contours, hierarchy = cv2.findContours(image_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    if contours is not None and (len(contours) > 0):
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        area, biggest_contour = max(contour_sizes, key=lambda x: x[0])
        epsilon = 0.01 * cv2.arcLength(biggest_contour, True)
        approx = cv2.approxPolyDP(biggest_contour, epsilon, True)

        x, y, w, h = cv2.boundingRect(biggest_contour)
        radius = int(w/2)
        cv2.circle(image, (int(x + w/2), int(y + h/2)), radius, green, 3)

def find_ball(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    (t, binary) = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    
    (_, contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(image.shape, dtype="uint8")
    
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
        
    cv2.drawContours(image, contours, -1, green, 5)
    return image

def equilizeHistogram(image):
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

    equilizedImage = equilizeHistogram(image)

    result = find_ball(equilizedImage)

    cv2.imshow('frame', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()