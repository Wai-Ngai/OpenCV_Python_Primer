import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
ORB算法


'''


def ORB_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    orb=cv.ORB_create()
    kp=orb.detect(gray,None)

    kp,des=orb.compute(gray,kp)

    image=cv.drawKeypoints(gray,kp,image,(0,0,255))

    cv.imshow("ORB",image)


def main():
    # 读取图片
    img = cv.imread("../code_images/blox.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)
    ORB_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
