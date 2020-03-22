import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
二维直方图

图像直方图反向投影


'''


# 创建2D的直方图
def hist2d_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # 计算2个通道的直方图,h通道histSize=180,s通道histSize=256
    hist = cv.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])

    print(hist.shape)
    cv.imshow("hist2D", hist)

    # 插值方式：临近点插值，interpolation="nearest"
    plt.imshow(hist, interpolation="nearest")
    plt.title("2D Histogram")
    plt.show()


def back_projection_demo():
    sample = cv.imread("../code_images/sample3.jpg")
    target = cv.imread("../code_images/target3.jpg")

    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    cv.imshow("sample", sample)
    cv.imshow("target", target)

    #求出样本的直方图
    roi_hist = cv.calcHist([roi_hsv], [0, 1], None, [20, 20], [0, 180, 0, 256])

    # 对样本的直方图进行归一化
    # 归一化后的图像便于显示，归一化后到0,255之间了
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)


    dst = cv.calcBackProject([target_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    cv.imshow("back_projection", dst)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/CharlizeTheron.png")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("CharlizeTheron", img)
    # hist2d_demo(img)

    back_projection_demo()

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
