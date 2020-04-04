import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
霍夫圆变换（Hough Circle Transform）

    cv.HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)
        - image： 8位，单通道灰度图像.
        - method： 变换方式，目前只支持CV_HOUGH_GRADIENT 霍夫梯度
        - dp：累加器图像的分辨率。这个参数允许创建一个比输入图像分辨率低的累加器。（这样做是因为有理由认为图像中存在的圆会自然降低到与图像宽高相同数量的范畴）。
            如果dp设置为1，则分辨率是相同的；
            如果设置为更大的值（比如2），累加器的分辨率受此影响会变小（此情况下为一半）。
            dp的值不能比1小。
        - minDist：圆心之间的最小距离，如果检测到的两个圆心之间距离小于该值，则认为它们是同一个圆心
        - circles=None ： 为输出圆向量，每个向量包括三个浮点型的元素——圆心横坐标，圆心纵坐标和圆半径
        - param1=None：为边缘检测时使用Canny算子的高阈值
        - param2=None：为步骤1.5和步骤2.5中所共有的阈值
        - minRadius=None：最小圆半径。
        - maxRadius=None：最大圆半径。

'''


def hough_circle_detect(image):
    # 霍夫圆检测对噪声敏感，所以一定要进行噪声消除，也可以用高斯blur
    # 均值迁移，sp，sr为空间域核与像素范围域核半径
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)

    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
    print(circles.shape) # (1, 20, 3)
    # 转换成整数
    circles = np.uint16(np.around(circles))
    print(circles.shape) # (1, 20, 3)
    print(circles) # (1, 20, 3)

    for i in circles[0, :]:
        # 画出圆
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # 画出圆心
        cv.circle(image, (i[0], i[1]), 2, (0, 255, 0), 2)

    cv.imshow("hough_circle_detect", image)


def main():
    # 读取图片
    img = cv.imread("../code_images/circle1.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    hough_circle_detect(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
