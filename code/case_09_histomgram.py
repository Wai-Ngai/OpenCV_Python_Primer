import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
图像直方图：统计每个像素点的个数

    一维
        cv2.calcHist(images,channels,mask,histSize,ranges)

            - images： 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应用中括号 [] 括来例如[img]
            - channels：同样用中括号括来它会告函数我们统幅图 像的直方图。
               - 如果入图像是灰度图它的值就是 [0]
               - 如果是彩色图像的传入的参数可以是 [0] [1] [2] 它们分别对应着 BGR。
            - mask：掩模图像。统整幅图像的直方图就把它为 None。但是如 果你想统图像某一分的直方图的你就制作一个掩模图像并 使用它。
            - histSize：BIN 的数目。也应用中括号括来
            - ranges：像素值范围常为 [0256]
    
    
    
    二维

'''


# 绘制图像直方图
def plot_demo(image):
    # image.ravel()将图像展开，256为bins数量，[0, 256]为范围
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


# 分别绘制三个通道的直方图
def image_hist(image):
    color = ('blue', 'green', 'red')

    for i, color in enumerate(color):
        hist = cv.calcHist(image, [i], None, [256], [0, 256])
        print(hist.shape)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/Dilraba.png")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("Dilraba", img)

    plot_demo(img)
    image_hist(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
