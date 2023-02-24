import cv2
import numpy as np
import math


def regionOfInterest(img):
    h, w = img.shape[0], img.shape[1]
    mask = np.zeros_like(img)
    # polygon = np.array([[(0, h - 50), (0, h - 100), (450, 150), (w - 450, 150), (w, h - 100), (w, h - 50)]])
    polygon = np.array([[(w // 5, h), (w // 5, h // 3), (4 * w // 5, h // 3), (4 * w // 5, h)]])
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


def findHoles(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    imgROI = regionOfInterest(imgBlur)
    # print(imgBlur.mean() * 0.66, imgBlur.mean() * 1.33)
    imgEdged = cv2.Canny(imgROI, 90, 180, apertureSize=3)
    imgErosed = cv2.dilate(imgEdged, np.ones((5, 5), np.uint8), iterations=1)
    cnts = cv2.findContours(imgErosed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, boxes = sort_contours(cnts)
    for (i, c) in enumerate(cnts[2:]):
        # per = cv2.contourArea(c)
        if cv2.contourArea(c) > 200:
            mm = cv2.moments(c)
            cX = int(mm["m10"] / mm["m00"])
            cY = int(mm["m01"] / mm["m00"])
            cv2.drawContours(img, [c], -1, (36, 255, 12), 2)
            # cv2.putText(img, "#{}-{}".format(i + 1, int(per)), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
            #             1.0, (255, 255, 255), 2)
    # print(len(cnts), per, per / len(cnts))
    return img  # cv2.hconcat([cv2.resize(imgGray,(720,512)),cv2.resize(imgROI,(720,512))])


cam = cv2.VideoCapture(r'../data/road2.mp4')
while True:
    _, frame = cam.read()
    frame = cv2.resize(frame, (1024, 512))
    road = np.copy(frame)
    result = findHoles(road)
    cv2.imshow("Window", result)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
