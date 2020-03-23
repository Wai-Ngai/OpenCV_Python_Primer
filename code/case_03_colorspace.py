import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
'''
颜色空间

HSV色彩空间说明：
     H：0-180  S: 0-255 V： 0-255
'''

# 颜色空间转换，从bgr到gray，hsv，yuv，ycrcb
def color_space_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)

    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)

    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow("ycrcb", ycrcb)


# 从视频中提取指定颜色范围，并将其置为白，其余置为黑
def extract_object_demo():
    capture = cv.VideoCapture("../code_images/cxk_playBB.mp4")

    while True:
        ret, frame = capture.read()

        # 如果没有获取到视频帧则返回false
        if ret is False:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # hsv中h，s，v的最小值
        lower_hsv = np.array([0, 43, 46])
        # hsv中h，s，v的最大值
        upper_hsv = np.array([50, 255, 255])
        # 提取指定范围颜色，保留指定范围颜色, 其余置为黑(0)
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)

        cv.imshow("video", frame)
        cv.imshow("mask", mask)

        c = cv.waitKey(40)
        if c == 27:
            break


# 通道分离、合并，修改某一通道
def channels_split_merge(image):
    b, g, r = cv.split(image)
    cv.imshow("blue", b)
    cv.imshow("green", g)
    cv.imshow("red", r)

    changed_image = image.copy()

    # 将r通道颜色全部置为0
    changed_image[:, :, 2] = 0
    cv.imshow("changed_image", changed_image)

    # 合并通道
    merge_image = cv.merge([b, g, r])
    cv.imshow("merge_image", merge_image)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)
    # color_space_demo(img)

    # extract_object_demo()

    channels_split_merge(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
