import cv2
import numpy as np
import time

class History(object):
    def __init__(self):
        self.lastX, self.lastY = None, None
        self.lastCheckTime = time.time()
        self.status = 'No one is playing the Game'
        self.tracker = []

# constants
green = (0, 255, 0)
ocean_blue = (49, 210, 247)
history = History()
waitTime = 0.002

def printStatus(image):
    cv2.putText(image, history.status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 1)

    for i in xrange(1, len(history.tracker)):
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(image, history.tracker[i - 1], history.tracker[i], ocean_blue, thickness)

def find_biggest_contour(image_masked, image):
    image_masked = image_masked.copy()
    _, contours, _ = cv2.findContours(image_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    if contours is not None and (len(contours) > 0):
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        _, biggest_contour = max(contour_sizes, key=lambda x: x[0])
        
        ((x, y), radius) = cv2.minEnclosingCircle(biggest_contour)
        cx, cy = int(x), int(y)

        M = cv2.moments(biggest_contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        history.tracker.append(center)

        cv2.circle(image, (cx, cy), int(radius), green, 2)

        checkGamePlay(cx, cy, image)

    printStatus(image)


def checkGamePlay(x, y, image):
    if time.time() - history.lastCheckTime > waitTime:
        history.lastCheckTime = time.time()

        if history.lastX is not None:
            # euclidean distance, if changes by 5 pixels, motion detected
            dist = np.linalg.norm(np.array((x, y)) - np.array((history.lastX, history.lastY)))
            history.status = 'Someone playing the Game' if dist >= 15 else 'No one is playing the Game'

        history.lastX, history.lastY = x, y


def find_ball(equilizedImage, img):
    image = cv2.cvtColor(equilizedImage, cv2.COLOR_BGR2RGB)

    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # Filter by colour
    min_yellow = np.array([20, 100, 100])
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
    
    cv2.imshow('equilized', ycrcb)
    return ycrcb


cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()

    # normalize image
    equilizedImage = equilizeHistogram(image)

    # mask + process normalized image
    mask_result, result = find_ball(equilizedImage, image)

    cv2.imshow('frame', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()