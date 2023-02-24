# import cv2
# import numpy as np
#
# kernel_size = 5
# low_threshold = 50
# high_threshold = 150
# rho = 1
# theta = np.pi / 180
# threshold = 10
# min_line_length = 10
# max_line_gap = 20
# cap = cv2.VideoCapture(r'../data/road3.mp4')
# while True:
#     _, image = cap.read()
#     image = cv2.resize(image, (1024, 512))
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
#     edges = cv2.Canny(blur, low_threshold, high_threshold)
#     mask = np.zeros_like(edges)
#     ignore_mask_color = 255
#     h, w, _ = image.shape
#     # vertices = np.array([[(0, h), (w // 4, h // 4), (3 * w // 4, h // 4), (w, h)]], dtype=np.int32)
#     vertices = np.array([[(0, h), (0, 0), (460, 100), (460, 200), (100, h)],
#                          [(w, h), (w, 0), (w - 460, 100), (w - 460, 200), (w - 100, h)]], dtype=np.int32)
#     cv2.fillPoly(mask, vertices, ignore_mask_color)
#     masked_edges = cv2.bitwise_and(edges, mask)
#     line_image = np.copy(image) * 0
#     lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
#                             min_line_length, max_line_gap)
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     color_edges = np.dstack((edges, edges, edges))
#     lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
#     cv2.imshow("window", lines_edges)
#     if cv2.waitKey(3) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np

kernel_size = 5
low_threshold = 50
high_threshold = 150
rho = 1
theta = np.pi / 180
threshold = 10
min_line_length = 10
max_line_gap = 20
ignore_mask_color = 255


def get_slope(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return None
    return (y2 - y1) / (x2 - x1)


def partitionLines(imgLines, h, w):
    leftLines, rightLines = [], []  #
    for line in imgLines:
        x1, y1, x2, y2 = line.reshape(4)
        slope = get_slope(x1, y1, x2, y2)
        centre = [(x1 + x2) / 2, (y1 + y2) / 2]
        if slope != None:
            if slope < 0 and centre[0] < w / 2:
                leftLines.append(line)
            else:
                if centre[0] > w / 2:
                    rightLines.append(line)
    return leftLines, rightLines

def displayLines(img, lines):
    lineImg = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(lineImg, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lineImg

cap = cv2.VideoCapture(r'../data/road2.mp4')
while True:
    _, image = cap.read()
    image = cv2.resize(image, (1024, 512))
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    mask = np.zeros_like(edges)
    vertices = np.array([[(0, h), (0, 0), (460, 100), (460, 200), (100, h)],
                         [(w, h), (w, 0), (w - 460, 100), (w - 460, 200), (w - 100, h)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    line_image = np.copy(image) * 0
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    leftLines, rightLines = partitionLines(lines, h, w)
    for line in leftLines+rightLines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    color_edges = np.dstack((edges, edges, edges))
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    cv2.imshow("window", lines_edges)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
