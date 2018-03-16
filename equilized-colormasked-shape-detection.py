from __future__ import division
import cv2
#to show the image
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin, pi

green = (0, 255, 0)

def find_biggest_contour(image_masked, image):
    image_masked = image_masked.copy()
    _, contours, _ = cv2.findContours(image_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    if contours is not None and (len(contours) > 0):
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        _, biggest_contour = max(contour_sizes, key=lambda x: x[0])
        
        x, y, w, h = cv2.boundingRect(biggest_contour)
        radius = int(w/2)
        cv2.circle(image, (int(x + w/2), int(y + h/2)), radius, green, 3)
        cv2.rectangle(image, (x, y), (x + w, y + h), green)

def find_ball(equilizedImage, img):
    image = cv2.cvtColor(equilizedImage, cv2.COLOR_BGR2RGB)

    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # Filter by colour
    min_yellow = np.array([20, 50, 100])
    max_yellow = np.array([60, 255, 255])
    mask = cv2.inRange(image_blur_hsv, min_yellow, max_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    cv2.imshow('masked', mask_clean)

    find_biggest_contour(mask_clean, img)
    return (mask_clean, img)

def equilizeHistogram(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, ycrcb)
    return ycrcb

cap = cv2.VideoCapture(0)

while(True):
    ret, image = cap.read()

    equilizedImage = equilizeHistogram(image)

    mask_result, result = find_ball(equilizedImage, image)

    cv2.imshow('frame', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()