import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
边缘保留滤波EPF（Edge-Preserving Filter）

1.高斯双边模糊



2.均值迁移

'''

# 高斯双边模糊
def bi_demo(image):
    dst=cv.bilateralFilter(image,0,100,15)
    cv.imshow("bilateralFilter",dst)


# 均值迁移
def mean_shift(image):
    dst = cv.pyrMeanShiftFiltering(image, 10,50)
    cv.imshow("meanShiftFilter", dst)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/example.png")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("example", img)
    bi_demo(img)
    mean_shift(img)



    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
