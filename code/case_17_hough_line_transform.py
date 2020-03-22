import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
霍夫直线变换（Hough Line Transform）

用来做直线检测
前提条件：边缘检测已经完成
平面空间到极坐标空间转换

'''


def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    lines = cv.HoughLines(edges, 1, np.pi / 180, 200)

    for line in lines:
        rho, theta = line[0]

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))

        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # 将直线绘制到原图中
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imshow("line_detection", image)


def line_detection_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # 最小线的长度
    minLineLength = 100
    # 最大线的
    maxLineGap = 10

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

    for line in lines:
        print(type(line))  # line 是多维的数组

        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow("hough_lines", image)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/sudoku.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    # line_detection(img)
    line_detection_possible_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
