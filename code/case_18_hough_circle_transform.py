import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
霍夫圆变换（Hough Circle Transform）



'''


def hough_circle_detect(image):
    # 霍夫圆检测对噪声敏感，所以一定要进行噪声消除，也可以用高斯blur
    # 均值迁移，sp，sr为空间域核与像素范围域核半径
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)

    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    # 转换成整数
    circles = np.uint16(np.around(circles))
    print(circles.shape) # (1, 20, 3)

    for i in circles[0, :]:
        # 画出圆
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # 画出圆心
        cv.circle(image, (i[0], i[1]), 2, (0, 255, 0), 2)

    cv.imshow("hough_circle_detect", image)


def main():
    # 读取图片
    img = cv.imread("../code_images/circle1.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    hough_circle_detect(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
