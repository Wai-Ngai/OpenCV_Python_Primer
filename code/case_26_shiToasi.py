import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
Shi-Tomasi 角点检测

    goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)
        image：灰度图像
        maxCorners:要检测到的角点数目
        qualityLevel：角点的质量水平，0到1 之间。它代表了角点的最低质量，低于这个数的所有角点都会被忽略
        minDistance：两个角点之间的最短欧式距离
    
'''


def shi_tomasi(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.uint0(corners)
    print(corners.shape)

    for corner in corners:
        x, y = corner.ravel()
        cv.circle(image, (x, y), 3, (0, 0, 255), -1)

    cv.imshow("corner", image)


def main():
    # 读取图片
    img = cv.imread("../code_images/blox.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("imput_image", img)

    shi_tomasi(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
