import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
高斯模糊

'''


# 保证每个通道的像素值大小在0~255之间
def clamp(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv


# 加高斯噪声
def gaussian_noise(image):
    h, w, c = image.shape

    # 产生随机数，分别加到图像的三个通道
    for row in range(h):
        for col in range(w):
            # normal(loc=0.0, scale=1.0, size=None),均值，标准差，大小
            s = np.random.normal(0, 20, 4)

            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]

            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(b + s[1])
            image[row, col, 2] = clamp(b + s[2])

    cv.imshow("gaussian_noise", image)

    return image


# 高斯模糊
def gaussian_blur_operation(image):
    # GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
    # ksize表示卷积核大小，sigmaX/Y表示x，y方向上的标准差，这两者只需一个即可，并且ksize为大于0的奇数
    dst = cv.GaussianBlur(image, (5, 5), 0)
    cv.imshow("gaussian_blur", dst)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    t1=cv.getTickCount()
    # 为原图加上高斯噪声
    new_image = gaussian_noise(img)
    t2 = cv.getTickCount()
    time=(t2-t1)/cv.getTickFrequency()
    print("为图像添加高斯噪声用时： %s ms" %(time*1000))
    # 高斯模糊
    gaussian_blur_operation(new_image)
    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
