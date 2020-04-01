import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
分水岭算法

基于距离变换，找到种子点

基于距离的分水岭分割流程：（需要去除噪声）
输入图像->灰度化->二值化->距离变换->寻找种子->生成marker->分水岭变换->输出图像
'''


def watershed_demo(image):
    # remove noise
    blured = cv.pyrMeanShiftFiltering(image, 10, 100)

    # gray/binary
    gray = cv.cvtColor(blured, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # morphology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)  # 连续两次开操作
    sure_bg = cv.dilate(mb, kernel, iterations=3)  # 连续三次膨胀操作
    cv.imshow("morphology operation", sure_bg)

    # Distance Transform
    # 第二个参数 0,1,2 分别表示 CV_DIST_L1, CV_DIST_L2 , CV_DIST_C
    dist = cv.distanceTransform(mb, 1, 5)
    dis_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow("distance", dis_output + 50)

    ret, surface = cv.threshold(dist, 0.6 * dist.max(), 255, cv.THRESH_BINARY)
    cv.imshow("sufrace", surface)

    # Finding unknown region
    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)

    # markers
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)

    # watershed transform
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]
    cv.imshow("result", image)


def main():
    # 读取图片
    img = cv.imread("../code_images/circle.png")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("input_image", img)

    watershed_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
