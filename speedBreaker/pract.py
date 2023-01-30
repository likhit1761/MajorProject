import cv2
import math
import numpy as np


def processContours(contours):
    coordinates = []
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        coordinates.append([x, y, x + w, y + h])
    return coordinates


def regionOfInterest(img):
    h, w = img.shape[0], img.shape[1]
    mask = np.zeros_like(img)
    polygon = np.array([[(0, h), (0, h - 50), (450, 100), (w - 450, 100), (w, h - 50), (w, h)]])
    cv2.fillPoly(mask, polygon, 255)
    imgBitwise = cv2.bitwise_and(img, mask)
    return imgBitwise


def findHorizantals(img):
    image = np.copy(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageROI = regionOfInterest(gray)
    thresh = cv2.threshold(imageROI, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    print(cnts)
    for c in cnts:
        cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
    return image

cam = cv2.VideoCapture(r'../data/road2.mp4')
#
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

##############################################################
# img = cv2.imread(path)
# img = cv2.resize(img, (720, 512))
# imgCopy = img.copy()
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
# imgCannyEdge = cv2.Canny(imgBlur, 50, 150)
# contours, hierarchy = cv2.findContours(imgCannyEdge,
#                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# print("Number of Contours found = " + str(len(contours)))
# arr = processContours(contours)
# print(arr)
# for cnt in contours:
#     if cv2.contourArea(cnt) <= 50:
#         continue
#     x, y, w, h = cv2.boundingRect(cnt)
#     if w > 200:
#         cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     else:
#         cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     center = (x, y)
#     # print(center)
#
# cv2.imshow("window", imgCopy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
##############################################################
# image = cv2.imread('../data/speed_breaker1.jpg')
# result = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
# detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
# cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area>5000:
#         cv2.drawContours(result, [c], -1, (36, 255, 12), 2)
# cv2.imshow('result', result)
# cv2.waitKey()
# #############################################################

# path = r'../data/speed_breaker1.jpg'
# font = cv2.FONT_HERSHEY_COMPLEX
# img = cv2.imread(path)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(gray, 5)
# edges = cv2.Canny(blur, 80, 120)
# lines = cv2.HoughLinesP(edges, 1, math.pi / 2, 2, None, 30, 1)
# print(lines)
# if(lines):
#     for line in lines[0]:
#         pt1 = (line[0], line[1])
#         pt2 = (line[2], line[3])
#         cv2.line(img, pt1, pt2, (0, 0, 255), 3)

# cv2.imshow("window", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
