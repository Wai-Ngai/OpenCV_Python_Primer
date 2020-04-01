import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
numpy操作数组输出图片

'''


# 用numpy创建一张图片
def creat_image():
    # 创建一个三维数组[400,400,3],数组元素全部是0
    img = np.zeros([400, 400, 3], np.uint8)
    # 将图片0通道的值全部赋值为400*255，由于上面已经定义的数组元素的类型为uint8,所以存在溢出，实际赋值为255
    # 0通道为B，所以图片显示为蓝色
    img[:, :, 0] = np.ones([400, 400]) * 255

    cv.imshow("creat_image", img)


# 获取图片参数，自定义图片反转
def access_pixels(image):
    print(image.shape)

    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]

    print("width: %s,height: %s,channel: %s" % (width, height, channel))

    # 复制图片
    new_image = image.copy()

    # 图像反转，3层循环，这样时间复杂度为O(n^3)
    for row in range(height):
        for col in range(width):
            for ch in range(channel):
                pv = image[row, col, ch]
                new_image[row, col, ch] = 256 - pv

    cv.imshow("new_image", new_image)


# 图片反转，采用cv自带方法，该方法底层为C++，所以执行效率要高得多
def inverse_image(image):
    # 按位取反
    dst = cv.bitwise_not(image)
    cv.imshow("inverse_image", dst)

# 使用numpy操作图像的像素点
def numpy_test(image):
    px=image[100,100]
    print(px)
    blue=image[100,100,0]
    print(blue)

    image[100,100]=[255,255,255]
    print(image[100,100])

    print(image.item(10,10,2))
    image.itemset((10,10,2),100)
    print(image.item(10,10,2))


def main():
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    # 用numpy创建一张图片
    creat_image()

    # 获取当前电脑时钟
    t1 = cv.getTickCount()

    access_pixels(img)
    inverse_image(img)

    t2 = cv.getTickCount()
    # cv2.getTickFrequency 返回时钟频率
    time = (t2 - t1) / cv.getTickFrequency()
    print("time: %s ms" % (time * 1000))

    numpy_test(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
