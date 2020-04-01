import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
二维直方图

    1.将图像的颜色空间从 BGR 转换到 HSV。（记住，计算一维直方图，要从 BGR 转换到 HSV）。
    2.cv2.calcHist(images,channels,mask,histSize,ranges)
        • channels=[0, 1]         因为我们需要同时处理 H 和 S 两个通道。
        • bins=[180, 256]         H 通道为 180， S 通道为 256。
        • range=[0, 180, 0, 256]  H 的取值范围在 0 到 180， S 的取值范围在 0 到 256。
        
    Numpy 中2D直方图
        hist, xbins, ybins = np.histogram2d(h.ravel(),s.ravel(),[180,256],[[0,180],[0,256]])


    绘制 2D 直方图
        方法 1：使用 cv2.imshow() 我们得到结果是一个 180x256 的两维数组。所以我们可以使用函数 cv2.imshow() 来显示它。但是这是一个灰度图，除非我们知道不同颜色 H 通道的值，否则我们根本就不知道那到底代表什么颜色。
        方法 2：使用 Matplotlib() 我们还可以使用函数 matplotlib.pyplot.imshow()来绘制 2D 直方图，再搭配上不同的颜色图（ color_map）。这样我们会对每个点所代表的数值大小有一个更直观的认识。但是跟前面的问题一样，你还是不知道那个数代表的颜色到底是什么。


图像直方图反向投影

    它可以用来做图像分割，或者在图像中找寻我们感兴趣的部分。
    
    简单来说，它会输出与输入图像（待搜索）同样大小的图像，其中的每一个像素值代表了输入图像上对应点属于目标对象的概率。
    用更简单的话来解释，输出图像中像素值越高（越白）的点就越可能代表我们要搜索的目标（在输入图像所在的位置）
    
    calcBackProject(images, channels, hist, ranges, scale, dst=None)
    
    
'''


# 创建2D的直方图
def hist2d_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # 计算2个通道的直方图,h通道histSize=180,s通道histSize=256
    hist = cv.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])

    print(hist.shape)
    cv.imshow("hist2D", hist)

    # 插值方式：临近点插值，interpolation="nearest"
    plt.imshow(hist, interpolation="nearest")  # X 轴显示 S 值， Y 轴显示 H 值。
    plt.title("2D Histogram")
    plt.show()


# 图像直方图反向投影
def back_projection_demo():
    sample = cv.imread("../code_images/sample3.jpg")
    target = cv.imread("../code_images/target3.jpg")

    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    cv.imshow("sample", sample)
    cv.imshow("target", target)

    # 求出样本的直方图
    roi_hist = cv.calcHist([roi_hsv], [0, 1], None, [20, 20], [0, 180, 0, 256])

    # 对样本的直方图进行归一化
    # 归一化后的图像便于显示，归一化后到0,255之间了
    # 归一化：原始图像，结果图像，映射到结果图像中的最小值，最大值，归一化类型
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    dst = cv.calcBackProject([target_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    cv.imshow("back_projection", dst)


def main():
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


if __name__ == '__main__':
    main()
