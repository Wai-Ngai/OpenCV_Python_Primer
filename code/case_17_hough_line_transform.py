import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
霍夫直线变换（Hough Line Transform）

用来做直线检测
前提条件：边缘检测已经完成
平面空间到极坐标空间转换

     cv2.HoughLines(image, rho, theta, threshold)
     
        image：一个二值化图像，所以在进行霍夫变换之前要首先进行二值化，或者进行Canny 边缘检测。
        rho, theta： ρ 和 θ 的精确度。
        threshold：阈值，只有累加其中的值高于阈值时才被认为是一条直线，也可以把它看成能检测到的直线的最短长度（以像素点为单位）。
        
        返回值就是(ρ, θ)。 ρ 的单位是像素， θ 的单位是弧度
        
     cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
        • minLineLength - 线的最短长度。比这个短的线都会被忽略。
        • MaxLineGap - 两条线段之间的最大间隔，如果小于此值，这两条直线就被看成是一条直线。
        
        返回值就是直线的起点和终点
        
    Probabilistic_Hough_Transform 是对霍夫变换的一种优化。它不会对每一个点都进行计算，而是从一幅图像中随机选取一个点集进行计算，对于直线检测来说这已经足够了。
    但是使用这种变换我们必须要降低阈值（总的点数都少了，阈值肯定也要小呀！）
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
    # 最大线的间隔
    maxLineGap = 10

    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)

    for line in lines:
        print(type(line))  # line 是多维的数组

        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv.imshow("hough_lines", image)


def main():
    # 读取图片
    img = cv.imread("../code_images/sudoku.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    line_detection(img)
    line_detection_possible_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
