import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
轮廓发现

是基于图像边缘提取的基础寻找对象轮廓的方法
所以边缘提取的阈值选定会影响最终轮廓发现的结果

利用梯度来避免阈值烦恼
'''


def edge_demo(image):
    bullered = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(bullered, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    edge_output = cv.Canny(grad_x, grad_y, 30, 100)
    return edge_output


def contours_demo(image):
    '''
    获取二值图像的方式一

    # 可以先去除图像噪声
    dst=cv.GaussianBlur(image,(3,3),2)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary_image", binary)
    '''

    # 获取二值图像的方式二
    binary = edge_demo(image)

    # 注意最新的OpenCV返回值只有两个，没有clone_image
    contours, hiersrchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)  # 2为像素大小，-1时填充轮廓
        print(i)

    cv.imshow("detect_contours", image)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/blob.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    contours_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
