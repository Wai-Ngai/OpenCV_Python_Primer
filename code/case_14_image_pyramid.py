import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
图像金字塔

图像金字塔原理：
    reduce=高斯模糊+降采样
    expand=扩大+卷积


PyrDown降采样
PyrUp还原

高斯金字塔 ( Gaussian pyramid): 用来向下/降采样，主要的图像金字塔
拉普拉斯金字塔(Laplacian pyramid): 用来从金字塔低层图像重建上层未采样图像，在数字图像处理中也即是预测残差，可以对图像进行最大程度的还原，配合高斯金字塔一起使用。

'''


# 高斯金字塔
def pyramid_demo(image):
    # 定义金字塔的层数
    level = 4
    temp = image.copy()
    pyramid_images = []

    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid_down_" + str(i + 1), dst)
        temp = dst.copy()

    return pyramid_images


# 拉普拉斯金字塔
# 要求：拉普拉斯金字塔时，图像大小必须是2的n次方*2的n次方，不然会报错
def laplace_demo(image):
    # 调用高斯金字塔的结果
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)

    for i in range(level - 1, -1, -1):
        if i - 1 < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)  # 最后一层用原图减去
            cv.imshow("laplace_down" + str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i - 1].shape[:2])
            lpls = cv.subtract(pyramid_images[i - 1], expand)
            cv.imshow("laplace_down" + str(i), lpls)


def main():
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    pyramid_demo(img)
    laplace_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
