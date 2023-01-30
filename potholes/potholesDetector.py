import cv2
import numpy as np
import math


def regionOfInterest(img):
    h, w = img.shape[0], img.shape[1]
    mask = np.zeros_like(img)
    polygon = np.array([[(0, h), (0, h - 50), (450, 100), (w - 450, 100), (w, h - 50), (w, h)]])
    cv2.fillPoly(mask, polygon, 255)
    imgBitwise = cv2.bitwise_and(img, mask)
    return imgBitwise


def findHoles(img):
    imgROI = regionOfInterest(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img, [c], -1, (36, 255, 12), 2)
    return img


img = cv2.imread("../data/hole2.jpg")
img = findHoles(img)
cv2.imshow("Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cam = cv2.VideoCapture(r'../data/road2.mp4')
# while True:
#     _, frame = cam.read()
#     frame = cv2.resize(frame, (1024, 512))
#     road = np.copy(frame)
#     result = findHoles(road)
#     cv2.imshow("Window", result)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
# cam.release()
# cv2.destroyAllWindows()

# img = cv2.imread("../data/hole2.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 127, 255, 0)
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# contours, hierarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# M = cv2.moments(cnt)
# print(M)
# perimeter = cv2.arcLength(cnt, True)
# print(perimeter)
# area = cv2.contourArea(cnt)
# print(area)
# epsilon = 0.1 * cv2.arcLength(cnt, True)
# approx = cv2.approxPolyDP(cnt, epsilon, True)
# print(epsilon)
# print(approx)
# for c in contours:
#     rect = cv2.boundingRect(c)
#     if rect[2] < 100 or rect[3] < 100:
#         continue
#     x, y, w, h = rect
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 8)
#     cv2.putText(img, 'Moth Detected', (x + w + 40, y + h), 0, 2.0, (0, 255, 0))
# cv2.imshow("Window", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
