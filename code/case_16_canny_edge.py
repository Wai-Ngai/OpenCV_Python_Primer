import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
Canny边缘

canny运算步骤：5步
1. 高斯模糊 - GaussianBlur
2. 灰度转换 - cvtColor
3. 计算梯度 - Sobel/Scharr
4. 非极大值抑制
5. 高低阈值输出二值图像
'''


def canny_edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    # canny 要求非浮点数
    grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    # 同时采用x，y方向的梯度提取边缘
    edge_output = cv.Canny(grad_x, grad_y, 50, 150)

    # 也可以直接采用原图进行提取边缘
    # edge_output=cv.Canny(gray,50,150)

    cv.imshow("gray", gray)
    cv.imshow("canny_edge", edge_output)

    dst = cv.bitwise_and(image, image, mask=edge_output)

    cv.imshow("color_edge", dst)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    canny_edge_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
