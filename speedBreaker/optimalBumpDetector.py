import random

import cv2
import numpy as np


def regionOfInterest(img):
    h, w = img.shape[0], img.shape[1]
    mask = np.zeros_like(img)
    polygon = np.array([[(0, h - 50), (450, 150), (w - 450, 150), (w, h - 50)]])
    cv2.fillPoly(mask, polygon, 255)
    imgBitwise = cv2.bitwise_and(img, mask)
    return imgBitwise


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)


def findHorizantals(img):
    image = np.copy(img)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = imageGray.shape[0], imageGray.shape[1]
    imageROI = regionOfInterest(imageGray)
    imageBlur1 = cv2.GaussianBlur(imageROI, (5, 5), 0)
    imageBlur = cv2.GaussianBlur(imageBlur1, (3, 3), 0)
    imgCanny = cv2.Canny(imageBlur, imageGray.mean() * 0.66, imageGray.mean() * 1.33, apertureSize=3)
    imageLines = cv2.HoughLinesP(imgCanny, 1, np.pi / 180, 50, None, 50, 10)
    # boundaries = np.array(
    #     [[0, h - 50, 450, 150], [450, 150, w - 450, 150], [w - 450, 150, w, h - 50], [w, h - 50, 0, h - 50]])
    if imageLines is not None:
        for i in range(1, len(imageLines)):
            l = imageLines[i][0]
            x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
            slope = abs((y2 - y1) / (x2 - x1))
            distance = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            if (slope > 0 and slope <= 0.4 and distance > 100):
                # print(distance)
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
    return image


cam = cv2.VideoCapture(r'../data/road3.mp4')
while True:
    _, frame = cam.read()
    frame = cv2.resize(frame, (1024, 512))
    road = np.copy(frame)
    result = findHorizantals(road)
    cv2.imshow("Window", result)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
