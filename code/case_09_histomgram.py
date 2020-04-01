import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
图像直方图：统计每个像素点的个数

    一维：只考虑了图像的一个特征：灰度值
        使用 OpenCV 统计直方图 

        cv2.calcHist(images,channels,mask,histSize,ranges)

            - images： 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应用中括号 [] 括来例如[img]
            - channels：同样用中括号括来它会告函数我们统幅图 像的直方图。
               - 如果入图像是灰度图它的值就是 [0]
               - 如果是彩色图像的传入的参数可以是 [0] [1] [2] 它们分别对应着 BGR。
            - mask：掩模图像。统整幅图像的直方图就把它为 None。但是如 果你想统图像某一分的直方图的你就制作一个掩模图像并 使用它。
            - histSize：BIN 的数目。也应用中括号括来
            - ranges：像素值范围常为 [0256]
    
        使用 Numpy 统计直方图
        
        hist,bins = np.histogram(img.ravel(),256,[0,256])
        hist = np.bincount(img.ravel(), minlength=256) 它的运行速度是np.histgram 的十倍。所以对于一维直方图，我们最好使用这个函数。 
        
        OpenCV 的函数要比 np.histgram() 快 40 倍。所以坚持使用OpenCV 函数。
    
    二维：考虑两个图像特征。对于彩色图像的直方图通常情况下我们需要考虑每个的颜色（ Hue）和饱和度（ Saturation）。根据这两个特征绘制 2D 直方图

绘制直方图
    有两种方法来绘制直方图：
    1. Short Way（简单方法）：使用 Matplotlib 中的绘图函数。
        plt.hist(img.ravel(),256,[0,256]);
    
    2. Long Way（复杂方法）：使用 OpenCV 绘图函数

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


def main():
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


if __name__ == '__main__':
    main()
