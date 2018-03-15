from __future__ import division
import cv2
#to show the image
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # We want to eliminate noise from our image. clean. smooth colors without dots
    # Blurs an image using a Gaussian filter. input, kernel size, how much to filter, empty)
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)

    # It unlike RGB, HSV separates luma, or the image intensity, from chroma or the color information.
    # just want to focus on color, segmentation
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # Filter by colour
    # 0-10 hue
    #minimum yellow amount, max yellow amount
    min_yellow = np.array([20, 100, 100])
    max_yellow = np.array([60, 255, 255])
    #layer
    mask = cv2.inRange(image_blur_hsv, min_yellow, max_yellow)

    # # brightness of a color is hue
    # # 170-180 hue
    # min_yellow2 = np.array([52, 47, 48])
    # max_yellow2 = np.array([60, 100, 97])
    # mask2 = cv2.inRange(image_blur_hsv, min_yellow2, max_yellow2)

    # #looking for what is in both ranges
    # # Combine masks
    # mask = mask1 + mask2

    # Clean up
    #we want to circle our strawberry so we'll circle it with an ellipse
    #with a shape of 15x15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #morph the image. closing operation Dilation followed by Erosion. 
    #It is useful in closing small holes inside the foreground objects, 
    #or small black points on the object.
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #erosion followed by dilation. It is useful in removing noise
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    cv2.imshow('masked', mask_clean)

    # Find biggest strawberry
    #get back list of segmented strawberries and an outline for the biggest one
    find_biggest_contour(mask_clean, image)
    
    #we're done, convert back to original color scheme
    detected_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return detected_image

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