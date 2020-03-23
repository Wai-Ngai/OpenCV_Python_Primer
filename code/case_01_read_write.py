import cv2 as cv
import numpy as np

"""
图像、视频的加载和保存
图像的基础知识

    cv.imread(filename, flags=None)
        - filename:图片地址
        - flags:
            cv2.IMREAD_COLOR：彩色图像
            cv2.IMREAD_GRAYSCALE：灰度图像
    
    cv.imshow(winname, mat)
        - winname：窗口名称
        - mat：要显示的图片
    
    cv.imwrite(filename, img, params=None)
        -filename
        -img
        -params=None
    
    cv.VideoCapture(*args, **kwargs)
        - 打开摄像头，0代表的是设备id，如果有多个摄像头，可以设置其他数值
        - 也可以是视频文件地址，调用视频文件，如果要播放要设置帧的循环
    
    cv.waitKey(delay=None)
        - delay : 窗口显示时间，单位：毫秒
             k=0: （也可以是小于0的数值）一直显示，键盘上按下一个数字键即会消失
             k>0:显示多少毫秒
    
    cv.namedWindow(winname, flags=None)
        - winname：窗口名称
        - flags
            为0或cv.WINDOW_NORMAL：可以改变窗口大小
            不写或cv.WINDOW_AUTOSIZE则不可改变大小
          
    cv.destroyAllWindows()
        删除建立的全部窗口，释放资源
"""


# 获取图片信息
def get_image_info(image):
    print("图片类型：", type(image))  # <class 'numpy.ndarray'>
    print("图片形状（长,宽,通道数）：", image.shape)  # <class 'tuple'>
    print("图片大小（长×宽×通道数）：", image.size)
    print("图片像素数据类型：", image.dtype)  # uint8

    # 将图片转换成数组,是一个3维数组
    pixel_data = np.array(image)
    print("像素数组：", pixel_data)


# 保存图片
def save_image(image):
    # 将图片转换为灰度图
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 保存图片
    cv.imwrite("../code_images/new.png", gray)


# 读取视频
def video_read():
    # 读取摄像头视频，用数字来控制不同的设备，例如0,1。
    # capture = cv.VideoCapture(0)

    # 读取本地文件视频
    capture = cv.VideoCapture("../code_images/cxk_playBB.gif")
    print("类型", type(capture))  # <class 'cv2.VideoCapture'>

    while True:
        # 获取相机图像，返回ret(结果为True/False)，和每一帧图片frame
        ret, frame = capture.read()
        print("ret:", ret)

        # 将图片水平翻转，竖直翻转为0
        frame = cv.flip(frame, 1)

        # 将每一帧图片放入video窗口
        cv.imshow("video", frame)

        # 等有键输入(这里指c=Esc键)或者50ms后自动将窗口消除
        c = cv.waitKey(100)
        # esc的ASCII编码为27
        if c == 27:
            break


if __name__ == '__main__':
    # 读取图片,可以通过设置第二个参数来将一张彩色图像读取为一张灰度图
    img = cv.imread("../code_images/lena.jpg", cv.IMREAD_GRAYSCALE)

    # 创建窗口，窗口尺寸自动调整
    cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    get_image_info(img)
    # save_image(img)

    video_read()

    # 等待时间，毫秒级，1000ms后自动将窗口消除,0表示任意键终止
    cv.waitKey(1000)
    cv.destroyAllWindows()
