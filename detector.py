import cv2
import numpy as np
from roadEdge.edgeDetector import roadEdgeLiner
from speedBreaker.bumpDetector import findHorizantals
from potholes.potholesDetector import findHoles


def detectEntities(img):
    result = roadEdgeLiner(img)
    result = findHorizantals(result)
    result = findHoles(result)
    return result


cam = cv2.VideoCapture(r'data/road2.mp4')

while True:
    _, frame = cam.read()
    frame = cv2.resize(frame, (1024, 512))
    road = np.copy(frame)
    result = detectEntities(road)
    cv2.imshow("Window", result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
