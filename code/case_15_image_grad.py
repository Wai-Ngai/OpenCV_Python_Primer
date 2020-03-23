import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
图像梯度（由x,y方向上的偏导数和偏移构成）

    一阶导数（sobel算子）
    
         cv2.Sobel(src, ddepth, dx, dy, ksize)
            ddepth:图像的深度
            dx和dy分别表示水平和竖直方向
            ksize是Sobel算子的大小
    
    二阶导数（Laplace算子）
    
    
    用于求解图像边缘，一阶的极大值，二阶的零点
    一阶偏导在图像中为一阶差分，再变成算子（即权值）与图像像素值乘积相加，二阶同理
'''


# 图像梯度：sobel算子
def sobel_demo(image):
    # 分别求xy方向的导数
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)

    # 由于算完的图像有正有负，白到黑是正数，黑到白就是负数了，所有的负数会被截断成0，所以对其取绝对值
    # grad_x = cv.convertScaleAbs(grad_x)
    # grad_y = cv.convertScaleAbs(grad_y)

    # 计算两个图像的权值和，dst = src1*alpha + src2*beta + gamma
    gradxy = cv.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    # xy方向的梯度还可以按照下面这样计算
    # 不建议直接计算
    # grad_xy = cv.Sobel(image, cv.CV_32F, 1, 1)

    cv.imshow("gradient_x", grad_x)#颜色变化在水平分层
    cv.imshow("gradient_y", grad_y)#颜色变化在垂直分层
    cv.imshow("gradient", gradxy)


# 图像梯度：scharr算子：增强边缘，sobel算子的增强版本
def scharr_demo(image):
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)

    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)

    cv.imshow("gradient_x", gradx)
    cv.imshow("gradient_y", grady)
    cv.imshow("gradient", gradxy)


# 拉普拉斯算子 默认的是4邻域的算子
# 二阶导数，边缘更细
def laplace_demo(image):
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("laplace", lpls)


# 用户自定义
def custom_laplace(image):
    # opencv中拉普拉斯算子默认的是4邻域的算子
    # kernal=np.array([[0,1,0],[1,-4,1],[0,1,0]])

    kernal = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    dst = cv.filter2D(image, cv.CV_32F, kernal)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("custom_laplace", lpls)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/pie.png")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("imput image", img)


    sobel_demo(img)


    # scharr_demo(img)


    # laplace_demo(img)


    # custom_laplace(img)


    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
