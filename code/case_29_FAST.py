import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
FAST算法


'''


def FAST_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 使用默认参数实例化一个FAST
    fast = cv.FastFeatureDetector_create()

    kp = fast.detect(gray, None)
    image = cv.drawKeypoints(gray, kp, image, (0, 0, 255))

    cv.imshow("fast", image)

    # 查看默认参数
    print("Threshold: ", fast.getThreshold())
    print("nonmaxSuppression: ", fast.getNonmaxSuppression())
    print("neighborhood: ", fast.getType())
    print("Total Keypoints with nonmaxSuppression: ", len(kp))


    # 关闭非极大值抑制
    fast.setNonmaxSuppression(0)

    kp = fast.detect(gray, None)
    image = cv.drawKeypoints(gray, kp, image, (0, 0, 255))

    cv.imshow("fast without nonmaxSuppression", image)


def main():
    # 读取图片
    img = cv.imread("../code_images/blox.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    FAST_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
