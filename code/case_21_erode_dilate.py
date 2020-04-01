import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
形态学操作是根据图像形状进行的简单操作。一般情况下对二值化图像进行的操作。
需要输入两个参数，一个是原始图像，第二个被称为结构化元素或核，它是用来决定操作的性质的。
两个基本的形态学操作是腐蚀和膨胀。

    腐蚀:用最小值替换中心像素。除白噪声很有用，也可以用来断开两个连在一块的物体等
    膨胀:用最大值替换中心像素。增加图像中的白色区域（前景），也可以用来连接两个分开的物体。


结构化元素
    1.使用 Numpy 构建结构化元素，它是正方形的。
    2. cv2.getStructuringElement(shape, ksize, anchor=None)。你只需要告诉他你需要的核的形状和大小。构建一个椭圆形/圆形的核
            - shape:
                cv2.MORPH_RECT      矩形
                cv2.MORPH_ELLIPSE   椭圆
                cv2.MORPH_CROSS     十字
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


def main():
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


if __name__ == '__main__':
    main()
