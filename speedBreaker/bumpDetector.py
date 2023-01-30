import cv2
import numpy as np


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
    for c in cnts:
        cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
    return image


# cam = cv2.VideoCapture(r'../data/road2.mp4')
#
# while True:
#     _, frame = cam.read()
#     frame = cv2.resize(frame, (1024, 512))
#     road = np.copy(frame)
#     result = findHorizantals(road)
#     cv2.imshow("Window", result)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
# cam.release()
# cv2.destroyAllWindows()
