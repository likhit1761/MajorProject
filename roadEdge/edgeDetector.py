import cv2
import numpy as np


def initialPreProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.medianBlur(imgGray, 3)
    imgCanny = cv2.Canny(imgBlur, 50, 150)
    return imgCanny


def regionOfInterest(img):
    h, w = img.shape[0], img.shape[1]
    mask = np.zeros_like(img)
    polygon1 = np.array([[(0, h), (0, 0), (460, 100), (460, 200), (100, h)]])
    polygon2 = np.array([[(w, h), (w, 0), (w - 460, 100), (w - 460, 200), (w - 100, h)]])
    cv2.fillPoly(mask, polygon1, 255)
    cv2.fillPoly(mask, polygon2, 255)
    imgBitwise = cv2.bitwise_and(img, mask)
    return imgBitwise


def displayLines(img, lines):
    lineImg = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImg, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineImg


def roadEdgeLiner(road):
    roadProcessed = initialPreProcess(road)
    roadROI = regionOfInterest(roadProcessed)
    roadEdges = cv2.HoughLinesP(roadROI, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=15)
    roadDrawn = displayLines(road, roadEdges)
    roadCombined = cv2.addWeighted(road, 0.8, roadDrawn, 1, 1)
    return roadCombined


# cam = cv2.VideoCapture(r'../data/road2.mp4')
#
# while True:
#     _, frame = cam.read()
#     frame = cv2.resize(frame, (1024, 512))
#     road = np.copy(frame)
#     result = roadEdgeLiner(road)
#     cv2.imshow("Window", result)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
# cam.release()
# cv2.destroyAllWindows()
