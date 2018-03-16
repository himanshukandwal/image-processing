from __future__ import division
import cv2
#to show the image
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin, pi

green = (0, 255, 0)

def overlay_mask(mask, image):
	#make the mask rgb
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #calculates the weightes sum of two arrays. in our case image arrays
    #input, how much to weight each. 
    #optional depth value set to 0 no need
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    img = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
    return img

def find_biggest_contour(image_masked, image): 
    
    circles = cv2.HoughCircles(image_masked, cv2.HOUGH_GRADIENT, 1, 4, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
    
    if circles is not None and len(circles) > 0:
        print 'here'
        circle_sizes = [(2 * np.pi * circle[2] ** 2, circle) for circle in circles]
        _, biggest_circle = max(circle_sizes, key=lambda x: x[0])
        cv2.circle(image, (biggest_circle[0], biggest_circle[1]), biggest_circle[2], green, 2)

def find_ball(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # We want to eliminate noise from our image. clean. smooth colors without dots
    # Blurs an image using a Gaussian filter. input, kernel size, how much to filter, empty)
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

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
    find_biggest_contour(mask_clean, img)
    
    #we're done, convert back to original color scheme
    detected_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return detected_image

def equilizeHistogram(image):
    img_yuv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2YUV)

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
