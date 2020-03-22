import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)





    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
