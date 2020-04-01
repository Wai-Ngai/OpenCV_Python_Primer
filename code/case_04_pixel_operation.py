# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
图片色素的  数值运算  和  逻辑运算

数值运算:

    相加：add()    
    相减：subtract()    
    相乘：multiply() 
    相除：divide()   
    
    原理就是：通过获取两张（一次只能是两张）图片的同一个位置的色素值来实现运算。
    运算的要求：两张图片的shape要一样。
    
    
    OpenCV的add方法是一种饱和操作，如果像素值超过了255，那就等于255
    Numpy 的加法是一种模操作，如果直接是两张图片相加image1+image2，那就是ndarray的运算，如果像素值超过了255，那就等于 %256，相当于取余
    


逻辑运算:

    与：bitwise_add()   
    或：bitwise_or()   
    非：bitwise_not()  
    异或：bitwise_xor()


图像融合：
    
    addWeighted(src1,alpha,src2,beta,gamma,dst,dtype=-1);
    一共有七个参数：前4个是两张要合成的图片及它们所占比例，第5个double gamma起微调作用，第6个OutputArray dst是合成后的图片，第七个输出的图片的类型（可选参数，默认-1）
    
    有公式得出两个图片加成输出的图片为：dst=src1*alpha+src2*beta+gamma

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
def mean_std_operation(image1, image2):
    m1 = cv.mean(image1)
    m2 = cv.mean(image2)
    print(m1)
    print(m2)

    mean, dev = cv.meanStdDev(image1)
    print("mean: %s, std_dev:  %s" % (mean, dev))


# 图像融合
def image_fusion_demo(image1, image2):
    dst = cv.addWeighted(image1, 0.5, image2, 0.5, 0)
    cv.imshow("image fusion", dst)


# 逻辑运算：AND，OR，NOT，XOR
def logic_operation(image1, image2):
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
        plt.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))

        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()


def main():
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

    # mean_std_operation(img1,img2)

    # 图像融合
    # image_fusion_demo(img1, img2)

    # 图片逻辑运算
    # logic_operation(img1, img2)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
