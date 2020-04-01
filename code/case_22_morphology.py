import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
开闭操作

    开操作:先腐蚀，再膨胀。它被用来去除噪声，消除图像中小的干扰区域。还可以尽量保留了其他的元素没有变化
    闭操作:先膨胀，再腐蚀。它经常被用来填充前景物体中的小洞，或者前景物体上的小黑点。还可以尽量保留了其他的元素没有变化

其他形态学操作：

    顶帽：原图像-开操作  tophat
    黑帽：闭操作-原图像  blackhat

形态学梯度：其实就是一幅图像膨胀与腐蚀的差别。 结果看上去就像前景物体的轮廓
    
    基本梯度：膨胀后-腐蚀
    内部梯度：原图-腐蚀
    外部梯度：膨胀后-原图
'''


# 开操作
def open_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # 得到结构元素
    # 提取竖直线条(1, 15)
    # 提取水平线条(15, 1)
    # (3,3)可以从干扰线中提取数字
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow("open_result", dst)


# 闭操作
def close_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # 得到结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow("close_result", dst)


# 其他形态学操作：
def top_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 得到结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)

    # 下面的步骤主要是为了提升图像的亮度
    c_img = np.array(gray.shape, np.uint8)
    c_img = 100
    dst = cv.add(dst, c_img)

    cv.imshow("tophat_result", dst)


def black_hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 得到结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)

    c_img = np.array(gray.shape, np.uint8)
    c_img = 100
    dst = cv.add(dst, c_img)

    cv.imshow("blackhat_result", dst)


# 基本梯度
def gradient_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 得到结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)

    cv.imshow("gradient_result", dst)


# 内梯度和外梯度
def gradient_demo2(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    dm = cv.dilate(image, kernel)
    em = cv.erode(image, kernel)

    dst1 = cv.subtract(image, em)  # internal gradient
    dst2 = cv.subtract(image, dm)  # external gradient

    cv.imshow("internal gradient", dst1)
    cv.imshow("external gradient", dst2)


def main():
    # 读取图片
    img = cv.imread("../code_images/bin2.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("input_image", img)

    # 开操作
    open_demo(img)

    # 闭操作
    close_demo(img)

    top_hat_demo(img)

    black_hat_demo(img)

    # 基本梯度
    gradient_demo(img)

    # 内梯度和外梯度
    gradient_demo2(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
