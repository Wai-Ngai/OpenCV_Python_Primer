import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
对象测量

    弧长与面积：发现轮廓，计算每个轮廓的弧长与面积，像素单位
    
        轮廓面积    cv2.contourArea()
        轮廓周长    cv2.arcLength() 
    
    多边形拟合
    
    几何矩计算：图像的矩可以帮助我们计算图像的质心，面积等
        cv2.moments() 会将计算得到的矩以一个字典的形式返回

'''


# 对象测量
def measure_object(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value:", ret)
    cv.imshow("binary image", binary)

    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

    # 发现轮廓
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        # 用黄色线条画出轮廓
        cv.drawContours(image, contours, i, (0, 255, 255), 1)

        # 计算轮廓面积
        area = cv.contourArea(contour)
        print("contour area:", area)

        # 计算轮廓周长
        perimeter = cv.arcLength(contour, True)
        print("contour perimeter:", perimeter)

        # 用矩阵框出轮廓
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 计算矩形宽高比
        rate = min(w, h) / max(w, h)
        print("rectangle rate:", rate)

        # 计算几何矩
        # 会将计算得到的几何矩以一个字典的形式返回
        mm = cv.moments(contour)
        # 计算对象重心
        cx = mm['m10'] / mm['m00']
        cy = mm['m01'] / mm['m00']

        # 用实心圆画出重心
        cv.circle(image, (np.int(cx), np.int(cy)), 2, (0, 255, 255), -1)
        cv.circle(dst, (np.int(cx), np.int(cy)), 2, (0, 255, 255), -1)

        # 多边形拟合，识别出不同形状的图形，三角形，圆形，矩形
        approx_curve = cv.approxPolyDP(contour, 4, True)
        print(approx_curve.shape)

        if approx_curve.shape[0] > 6:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)
        if approx_curve.shape[0] == 4:
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)
        if approx_curve.shape[0] == 3:
            cv.drawContours(dst, contours, i, (255, 0, 0), 2)

    cv.imshow("measure_object", image)
    cv.imshow("measure_object_2", dst)


def main():
    # 读取图片
    img = cv.imread("../code_images/blob.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)
    measure_object(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
