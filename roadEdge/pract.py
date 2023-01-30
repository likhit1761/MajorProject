import cv2
import numpy as np


def initialPreProcess(img):
    # reduce noise and smoothen images and detect edges
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    # when there is a drastic change in the pixel values
    # then we can say that there is an edge
    imgCanny = cv2.Canny(imgBlur, 50, 150)
    return imgCanny


def regionOfInterest(img):
    # takes canny edged images
    h = img.shape[0]
    polygon = np.array([[(0, h), (256, 0), (0, 0)]])
    # polygon = np.array([
    #     [(0, h), (275, h), (345, 340)]])
    # [(700, height), (1600, height), (850, 620)]
    mask = np.zeros_like(img)  # black mask as road image
    cv2.fillPoly(mask, polygon, 255)  # fill this mask with polygon
    imgBitwise = cv2.bitwise_and(img, mask)  # overlap both the image and mask and get the new one
    return imgBitwise


def displayLines(img, lines):
    lineImg = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImg, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineImg


def makeCoOrdinates(img, lineParams):
    slope, intercept = lineParams
    y1 = img.shape[0]
    y2 = int(y1 * (3 / 5))
    x1, x2 = int((y1 - intercept) / slope), int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def averageSlopeIntercept(img, imgLines):
    # finding the best line to describe every line we found
    flag = False
    left_fit, right_fit = [], []  #
    for line in imgLines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = params[0], params[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if len(left_fit) == 0 or len(right_fit) == 0 or (len(left_fit) == 0 and len(right_fit) == 0):
        flag = True
        return np.array([]), flag
    else:
        left_fit_avg = np.average(left_fit, axis=0)
        right_fit_avg = np.average(right_fit, axis=0)
        leftline = makeCoOrdinates(img, left_fit_avg)
        rightline = makeCoOrdinates(img, right_fit_avg)
        return np.array([leftline, rightline]), flag


def roadEdgeLiner(image):
    imgEdged = initialPreProcess(laneImg)
    return image


cap = cv2.VideoCapture(r'../data/road2.mp4')
result = cv2.VideoWriter('../results/processedLane1.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (int(cap.get(3)), int(cap.get(4))))
while True:
    _, img = cap.read()
    laneImg = np.copy(img)
    imgEdged = initialPreProcess(laneImg)
    # imgROI = regionOfInterest(imgEdged)
    # imgLines = cv2.HoughLinesP(imgROI, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    imgLines = cv2.HoughLinesP(imgEdged, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # # we get coordintates of the lines detected
    # # print(imgLines)
    # # to detect straight lines in region of interest and thus identify the lane lines
    # imgAveragedLines, status = averageSlopeIntercept(laneImg, imgLines)
    # if status:
    #     imgDrawn = displayLines(laneImg, imgLines)
    # else:
    #     imgDrawn = displayLines(laneImg, imgAveragedLines)
    imgDrawn = displayLines(laneImg, imgLines)
    imgCombined = cv2.addWeighted(laneImg, 0.8, imgDrawn, 1, 1)
    # displaying the lanes drawn on black images to the real image
    cv2.imshow("window", imgCombined)
    result.write(imgCombined)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
cap.release()
result.release()
cv2.destroyAllWindows()

# img = cv2.imread(r'data/pic')
# laneImg = np.copy(img)
# imgEdged = initialPreProcess(laneImg)
# imgROI = regionOfInterest(imgEdged)
# imgLines = cv2.HoughLinesP(imgROI, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# imgAveragedLines, status = averageSlopeIntercept(laneImg, imgLines)
# if status:
#     imgDrawn = displayLines(laneImg, imgLines)
# else:
#     imgDrawn = displayLines(laneImg, imgAveragedLines)
# imgCombined = cv2.addWeighted(laneImg, 0.8, imgDrawn, 1, 1)
# res = cv2.resize(imgCombined, (500, 500))
# cv2.imshow("result", imgEdged)
# cv2.waitKey(0)
