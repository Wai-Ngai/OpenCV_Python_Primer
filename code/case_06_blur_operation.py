import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
模糊操作：
均值模糊
中值模糊
自定义模糊

基本原理：
1.基于离散卷积
2.定义好每个卷积核
3.不同卷积核得到不同卷积效果
4.模糊是卷积的一种表象



'''


# 均值模糊
def blur_operation(image):
    # 卷积核大小1*3
    dst = cv.blur(image, (5, 5))
    cv.imshow("blur", dst)


# 中值模糊，用于椒盐噪声
def median_blur_operation(image):
    dst = cv.medianBlur(image, 5)
    cv.imshow("median_blur", dst)

#自定义模糊
def customer_blur_operation(image):

    # kernel=np.ones([5,5],np.float32)/25

    # kernel=np.array([[1,1,1],[1,1,1],[1,1,1]],np.float32)/9

    kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)  # 锐化算子
    dst = cv.filter2D(image, -1,kernel)
    cv.imshow("customer_blur", dst)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/uu.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("uu", img)

    blur_operation(img)

    median_blur_operation(img)

    customer_blur_operation(img)
    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
