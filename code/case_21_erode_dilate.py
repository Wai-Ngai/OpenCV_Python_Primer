import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
腐蚀:用最小值替换中心像素
膨胀:用最大值替换中心像素

'''


# 腐蚀
def erode_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # 得到结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.erode(binary, kernel)
    cv.imshow("erode_image", dst)


# 膨胀
def dilate_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # 得到结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(binary, kernel)
    cv.imshow("dilate_image", dst)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/bin2.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    erode_demo(img)
    dilate_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
