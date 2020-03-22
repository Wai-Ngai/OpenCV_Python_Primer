import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
二值图像（Binary Image）:就是将灰度图转化成黑白图[0 1]，在一个阈值之前为黑1，之后为白0

基于图像直方图来实现的

全局阈值和局部阈值

OpenCV中图像二值化的方法：
OTSU
Triangle
自动与手动
自适应阈值

'''


# 全局阈值
def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 指定阈值
    # 不能有自动寻找阈值的选项。此时ret = 127,大于127的变成白色
    # ret,binary=cv.threshold(gray,127,255,cv.THRESH_BINARY)

    # 不指定阈值
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    print("threshold_value:", ret)
    cv.imshow("threshold_demo", binary)


def threshold_simple(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)        # 最常用的一个方法，其他都不常用
    ret, thresh2 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)    # 小于127的变成白色，与上面一个方法相反
    ret, thresh3 = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)         # 截断，大于127的就等于127，小于127的就变成黑色
    ret, thresh4 = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO)        # 小于127的全部变成0
    ret, thresh5 = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()


# 自适应阈值（局部阈值）
def threshold_adaptive(image):
    binary = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 中值滤波
    binary = cv.medianBlur(binary, 5)

    ret, th1 = cv.threshold(binary, 175, 255, cv.THRESH_BINARY)

    # 11为Block size(必须是奇数), 2 为C值,局部二值化中有这样一个容差的，可以消除噪声
    # 局部二值化并没有一个全局的阈值，所以返回值只有一个
    th2 = cv.adaptiveThreshold(binary, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    titles = ['Original Image', 'Global Threshold (v = 127)', 'Adaptive Mean Threshold', 'Adaptive Gaussian Threshold']
    images = [binary, th1, th2, th3]

    for i in range(len(images)):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


# 自定义阈值进行分割
def threshold_custom(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 求出图像的高、宽
    h, w = gray.shape[:2]
    # 将其变成一行多列的一维数组
    new_img = np.reshape(gray, [1, w * h])
    # 求出整个灰度图像的平均值
    mean = new_img.sum() / (w * h)
    print("mean:", mean)

    ret, th = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow("threshold_custom", th)


# 超大图像的二值化--分块
# 将大图片拆分成小图片后再用自适应局部阈值比较好
def big_image_demo():
    big_image = cv.imread("../code_images/big_image.jpg")
    print(big_image.shape)

    # 定义一个小块的宽高
    cw = 200
    ch = 200

    h, w = big_image.shape[:2]
    gray = cv.cvtColor(big_image, cv.COLOR_BGR2GRAY)
    cv.imshow("big_image", gray)

    # 将一张图片每隔ch * cw分成一份
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row + ch, col:col + cw]

            # 全局二值化
            # ret,dst=cv.threshold(roi,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)

            # 对于全局二值化，可以设定标准差阈值，进行空白图像过滤，解决全局二值化在有噪声的时候效果不好的问题
            dev = np.std(roi)
            if dev < 15:
                gray[row:row + ch, col:col + cw] = 255
            else:
                ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                gray[row:row + ch, col:col + cw] = dst

            # # 局部阈值
            # dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 127, 30)
            #
            # # 反向修改原图中相应区域
            # gray[row:row + ch, col:col + cw] = dst
            # # 输出均值和方差
            # print(np.mean(dst), np.std(dst))

    cv.imwrite("../code_images/big_image_binary.jpg", gray)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    # 全局阈值
    # threshold_demo(img)

    # threshold_simple(img)

    # 自适应阈值（局部阈值）
    # threshold_adaptive(img)

    # 自定义阈值进行分割
    # threshold_custom(img)

    # 超大图像的二值化
    big_image_demo()

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
