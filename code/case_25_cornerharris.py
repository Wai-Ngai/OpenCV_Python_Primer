import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
角点检测


'''


# Harris角点检测
def corner_harris_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 输入图像必须是float32，最后一个参数在0.04 到0.05 之间
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    # 用红色在原图上标记出角点
    image[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv.imshow("dst", image)


# 亚像素级精确度的角点
def corner_subpix_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # 返回值由角点坐标组成的一个数组（而非图像）
    corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    # np.int0 可以用来省略小数点后面的数字（非四舍五入）。
    res = np.int0(res)

    image[res[:, 1], res[:, 0]] = [0, 0, 255]
    image[res[:, 3], res[:, 2]] = [0, 255, 0]

    cv.imshow("dst", image)


def main():
    # 读取图片
    img = cv.imread("../code_images/blox.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    # cv.imshow("lena", img)
    # corner_harris_demo(img)
    corner_subpix_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
