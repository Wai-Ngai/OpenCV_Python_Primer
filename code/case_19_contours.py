import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
轮廓发现

是基于图像边缘提取的基础寻找对象轮廓的方法,所以边缘提取的阈值选定会影响最终轮廓发现的结果

轮廓可以简单认为成将连续的点（连着边界）连在一起的曲线，具有相同的颜色或者灰度。轮廓在形状分析和物体的检测和识别中很有用。

利用梯度来避免阈值烦恼

     contours, hiersrchy = cv.findContours(image, mode, method, contours=None, hierarchy=None, offset=None)
        - image：输入图像
        - mode：轮廓检索模式
        - method：轮廓近似方法。轮廓是一个形状具有相同灰度值的边界。它会存贮形状边界上所有的 (x， y) 坐标。
            - cv2.CHAIN_APPROX_NONE：所有的边界点都会被存储。
            - cv2.CHAIN_APPROX_SIMPLE：它会将轮廓上的冗余点都去掉，压缩轮廓，从而节省内存开支。

        - contours：轮廓（包括了内轮廓和外轮廓）。是一个 Python列表，其中存储这图像中的所有轮廓。每一个轮廓都是一个 Numpy 数组，包含对象边界点（ x， y）的坐标。
        - hiersrchy：（轮廓的）层析结构

    cv.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
        -image
        -contours：轮廓，一个 Python 列表
        -contourIdx：轮廓的索引（在绘制独立轮廓是很有用，当设置为 -1 时绘制所有轮廓）
        -color：轮廓的颜色
        -thickness：轮廓的厚度

'''

# 为了更加准确，要使用二值化图像。在寻找轮廓之前，要进行阈值化处理 或者 Canny 边界检测。
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


def main():
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


if __name__ == '__main__':
    main()
