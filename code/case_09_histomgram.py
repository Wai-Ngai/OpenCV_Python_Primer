import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
图像直方图

一维

二维

'''


# 绘制图像直方图
def plot_demo(image):
    # image.ravel()将图像展开，256为bins数量，[0, 256]为范围
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


# 分别绘制三个通道的直方图
def image_hist(image):
    color = ('blue', 'green', 'red')

    for i, color in enumerate(color):
        hist = cv.calcHist(image, [i], None, [256], [0, 256])
        print(hist.shape)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/Dilraba.png")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("Dilraba", img)

    plot_demo(img)
    image_hist(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
