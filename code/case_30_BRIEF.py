import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
BRIEF算法

在OpenCV中CenSurE检测器叫做STAR

'''


def BRIEF_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 初始化STAR检测器
    star=cv.xfeatures2d.StarDetector_create()
    # 初始化BRIEF特征提取器
    brief=cv.xfeatures2d.BriefDescriptorExtractor_create()
    # 使用STAR寻找特征点
    kp=star.detect(gray,None)
    # 计算特征描述符
    kp,des=brief.compute(gray,kp)

    image=cv.drawKeypoints(gray,kp,image,(0,0,255))

    cv.imshow("BRIEF",image)


def main():
    # 读取图片
    img = cv.imread("../code_images/blox.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    BRIEF_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
