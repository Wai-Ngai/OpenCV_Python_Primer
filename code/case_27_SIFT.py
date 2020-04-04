import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def sift_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()  # 实例化sift算法
    kp = sift.detect(gray, None)  # 得到关键点。返回的关键点是一个带有很多不同属性的特殊结构体，这些属性中包含它的坐标（x，y），有意义的邻域大小，确定其方向的角度等。

    # 绘制关键点的函数，drawKeyPoints()，它可以在关键点的部位绘制一个小圆圈
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 就会绘制代表关键点大小的圆圈甚至可以绘制除关键点的方向。
    img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow("drawKeypoints", img)


def main():
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    sift_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
