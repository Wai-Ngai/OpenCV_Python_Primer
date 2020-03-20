import cv2 as cv
import numpy as np


"""
图像、视频的加载和保存
图像的基础知识
你将要学习如下函数：cv.imread()，cv.imshow()，cv.imwrite()，cv.VideoCapture()
"""


# 获取图片信息
def get_image_info(image):
    print("图片类型：", type(image))
    print("图片形状（长,宽,通道数）：", image.shape)
    print("图片大小（长×宽×通道数）：", image.size)
    print("图片像素数据类型：", image.dtype)

    # 将图片转换成数组,是一个3维数组
    pixel_data = np.array(image)
    print("像素大小：", pixel_data)


# 保存图片
def save_image(image):
    # 将图片转换为灰度图
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 保存图片
    cv.imwrite("new.png", gray)


# 读取视频
def video_read():
    # 读取摄像头视频
    capture = cv.VideoCapture(0)

    # 读取本地文件视频
    # capture = cv.VideoCapture("../code_images/slow.mp4")
    print("类型", type(capture))

    while True:
        # 获取相机图像，返回ret(结果为True/False)，和每一帧图片
        ret, frame = capture.read()
        print("ret:", ret)

        # 将图片水平翻转，竖直翻转为0
        frame = cv.flip(frame, 1)

        # 将每一帧图片放入video窗口
        cv.imshow("video", frame)
        # 等有键输入(这里指c=Esc键)或者50ms后自动将窗口消除
        c = cv.waitKey(50)
        # esc的ASCII编码为27
        if c == 27:
            break


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    get_image_info(img)
    # save_image(img)

    # video_read()

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
