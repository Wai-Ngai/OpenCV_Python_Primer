# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
图片色素的数值运算(加减乘除)和逻辑运算(与或非异或)
'''

# 数值运算
def add_operation(image1, image2):
    dst = cv.add(image1, image2)
    cv.imshow("add", dst)


def subtract_operation(image1, image2):
    dst = cv.subtract(image1, image2)
    cv.imshow("subtract", dst)


def divide_operation(image1, image2):
    dst = cv.divide(image1, image2)
    cv.imshow("divide", dst)


def multiply_operation(image1, image2):
    dst = cv.multiply(image1, image2)
    cv.imshow("multiply", dst)


# 计算图像的均值和方差
def mean_std_operation(image1,image2):
    m1=cv.mean(image1)
    m2=cv.mean(image2)
    print(m1)
    print(m2)

    mean,dev=cv.meanStdDev(image1)
    print("mean: %s, std_dev:  %s" %(mean,dev))



# 逻辑运算：AND，OR，NOT，XOR
def logic_operation(image1, image2):
    cv.imshow("im", image2)
    dst_and = cv.bitwise_and(image1, image2)
    cv.imshow("AND", dst_and)

    dst_or = cv.bitwise_or(image1, image2)
    cv.imshow("OR", dst_or)

    dst_not = cv.bitwise_not(image1)
    cv.imshow("NOT", dst_not)

    dst_xor = cv.bitwise_xor(image1, image2)
    cv.imshow("XOR", dst_xor)

    # 使用matplotlib进行统一绘制
    titles = ['linux', 'windows', 'AND', 'OR', 'NOT', 'XOR']
    images = [image1, image2, dst_and, dst_or, dst_not, dst_xor]

    cv.imshow("ss", images[1])

    for i in range(len(images)):
        plt.subplot(3, 2, i + 1)
        plt.imshow(images[i],"gray")


        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()

#



if __name__ == '__main__':
    # 读取图片
    img1 = cv.imread("../code_images/01.jpg")
    img2 = cv.imread("../code_images/02.jpg")

    print(img1.shape)
    print(img2.shape)

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("image", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("linux", img1)
    cv.imshow("windows", img2)

    # 图片数值运算
    # add_operation(img1,img2)
    # subtract_operation(img1,img2)
    # divide_operation(img1,img2)
    # multiply_operation(img1,img2)

    mean_std_operation(img1,img2)

    # 图片逻辑运算
    # logic_operation(img1, img2)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
